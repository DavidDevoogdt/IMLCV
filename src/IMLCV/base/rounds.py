from __future__ import annotations

import os
import shutil
import time
from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import ase
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from equinox import Partial
from filelock import FileLock
from IMLCV.base.bias import Bias
from IMLCV.base.bias import CompositeBias
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvTrans
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CV import padded_pmap
from IMLCV.base.CV import SystemParams
from IMLCV.base.MdEngine import MDEngine
from IMLCV.base.MdEngine import StaticMdInfo
from IMLCV.base.MdEngine import TrajectoryInfo
from IMLCV.configs.bash_app_python import bash_app_python
from jax import Array
from jax.random import choice
from jax.random import PRNGKey
from jax.random import split
from molmod.constants import boltzmann
from parsl.data_provider.files import File


@dataclass
class TrajectoryInformation:
    ti: TrajectoryInfo
    round: int
    num: int
    folder: Path
    name_bias: str | None = None
    valid: bool = True
    finished: bool = False

    def get_bias(self) -> Bias | None:
        try:
            assert self.name_bias is not None
            return Bias.load(self.folder / self.name_bias)
        except Exception as e:
            print(f"unable to load bias {e=}")
            return None


@dataclass
class RoundInformation:
    round: int
    valid: bool
    num: int
    num_vals: Array
    tic: StaticMdInfo
    folder: Path
    name_bias: str | None = None
    name_md: str | None = None

    def get_bias(self) -> Bias:
        assert self.name_bias is not None
        return Bias.load(self.folder / self.name_bias)

    def get_engine(self) -> MDEngine:
        assert self.name_md is not None
        return MDEngine.load(self.folder / self.name_md)


class Rounds(ABC):
    def __init__(
        self,
        folder: str | Path = "output",
        copy=True,
        new_folder=True,
    ) -> None:
        """
        this class saves all relevant info in a hdf5 container. It is build as follows:
        root
            cv_0
                round 0
                    attrs:
                        - name_bias
                        - name_md
                        - valid
                        - num
                    static_trajectory info
                        data, see static_trajectory info._save
                    0:
                        attrs:
                            - valid
                            - num
                        trajectory_info
                            data, see trajectory info._save
                    1:
                        ..
                    ...
                round 1
                    ...
                ...
            cv_1
                ...
            ...
        """

        folder = Path(folder)

        # only consider folder if it has reults file in it
        if (folder / "results.h5").exists() and new_folder:
            # look for first avaialble folder
            i = 0
            while True:
                p = folder.parent / (f"{folder.name}_{i:0>3}")
                if (p / "results.h5").exists():
                    i += 1
                else:
                    break

            if copy:
                shutil.copytree(folder, p)
        else:
            p = folder

        if not p.exists():
            p.mkdir(parents=True)

        self.folder = p
        self.lock = FileLock(self.h5filelock_name)

        self._make_file()
        self.recover()

    ######################################
    #             IO                     #
    ######################################

    def _make_file(self):
        # todo: make backup
        if (p := Path(self.h5file_name)).exists():
            os.remove(p)

        self.h5file = h5py.File(self.h5file_name, mode="w")

    @property
    def h5file_name(self):
        return self.full_path("results.h5")

    @property
    def h5filelock_name(self):
        return self.full_path("results.h5.lock")

    def full_path(self, name: str | Path) -> str:
        return str((self.folder / name).resolve())

    def rel_path(self, name: str | Path):
        return str(Path(name).relative_to(self.folder))

    def path(self, c=None, r=None, i=None) -> Path:
        p = Path(self.folder)
        if c is not None:
            p /= f"cv_{c}"
            if r is not None:
                p /= f"round_{r}"
                if i is not None:
                    p /= f"md_{i}"

        return p

    @staticmethod
    def load(folder: str | Path, copy=False):
        return Rounds(folder=folder, copy=copy)

    def write_xyz(
        self,
        c: int | None = None,
        r: int | None = None,
        num: int = 1,
        repeat=None,
        minkowski_reduce=True,
        r_cut=None,
    ):
        from ase.io.extxyz import write_extxyz

        if c is None:
            c = self.cv

        for i, (atoms, round, trajejctory) in enumerate(
            self.iter_ase_atoms(
                c=c,
                r=r,
                num=num,
                minkowski_reduce=minkowski_reduce,
                r_cut=r_cut,
            ),
        ):
            with open(
                self.path(c=c, r=round.round, i=trajejctory.num) / "trajectory.xyz",
                mode="w",
            ) as f:
                if repeat is not None:
                    atoms = [a.repeat(repeat) for a in atoms]

                write_extxyz(f, atoms)

    ######################################
    #             storage                #
    ######################################

    def recover(self):
        f = self.h5file

        for cv_c in self.path().glob("cv_*"):
            c = cv_c.parts[-1][3:]

            self.add_cv(c=c)

            for round_r in cv_c.glob("round_*"):
                r = round_r.parts[-1][6:]

                if r not in f[f"{c}"].keys():
                    if not (p := self.path(c=c, r=r) / "static_trajectory_info.h5").exists():
                        print(f"could not find {p}")
                        continue

                    self.add_round(c=c, r=r)

                for md_i in round_r.glob("md_*"):
                    i = md_i.parts[-1][3:]

                    if i in f[f"{c}/{r}"].keys():
                        continue
                    if not (md_i / "trajectory_info.h5").exists():
                        continue

                    self.add_md(c=c, i=i, r=r)

    def add_cv(self, c=None, attr=None):
        if c is None:
            c = self.cv + 1

        if attr is None:
            attr = {}

        dir = self.path(c=c)
        if not dir.exists():
            dir.mkdir(parents=True)

        with self.lock:
            f = self.h5file
            f.create_group(f"{c}")

            if "valid" not in attr:
                if (p := (self.path(c=c) / "invalid")).exists():
                    attr["valid"] = False
                else:
                    attr["valid"] = True

            if "name_cv" not in attr:
                if (p := (self.path(c=c) / "cv.json")).exists():
                    attr["name_cv"] = self.rel_path(p)
                elif (p := (self.path(c=c) / "cv")).exists():
                    attr["name_cv"] = self.rel_path(p)
                else:
                    raise

            for key in attr:
                if attr[key] is not None:
                    f[f"{c}"].attrs[key] = attr[key]

            f[f"{c}"].attrs["num"] = 0
            f[f"{c}"].attrs["num_vals"] = np.array([], dtype=np.int32)

    def add_cv_from_cv(self, cv: CollectiveVariable):
        c = self.cv + 1

        attr = {}

        directory = self.path(c=c)
        if not os.path.isdir(directory):
            os.mkdir(directory)

        cv.save(self.path(c=c) / "cv.json")

        attr["name_cv"] = self.rel_path(self.path(c=c) / "cv.json")
        self.add_cv(attr=attr, c=c)

    def add_round(self, stic: StaticMdInfo | None = None, c=None, r=None, attr=None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c) + 1

        if attr is None:
            attr = {}

        dir = self.path(c=c, r=r)
        if not dir.exists():
            dir.mkdir(parents=True)

        if not (p := self.path(c=c, r=r) / "static_trajectory_info.h5").exists():
            assert stic is not None
            stic.save(p)

        with self.lock:
            f = self.h5file
            f[f"{c}"].create_group(f"{r}")

            if "valid" not in attr:
                if (p := (self.path(c=c, r=r) / "invalid")).exists():
                    attr["valid"] = False
                else:
                    attr["valid"] = True

            if "name_bias" not in attr:
                if (p := (self.path(c=c, r=r) / "bias.json")).exists():
                    attr["name_bias"] = self.rel_path(p)
                elif (p := (self.path(c=c, r=r) / "bias")).exists():
                    attr["name_bias"] = self.rel_path(p)

            if "name_md" not in attr:
                if (p := (self.path(c=c, r=r) / "engine.json")).exists():
                    attr["name_md"] = self.rel_path(p)
                elif (p := (self.path(c=c, r=r) / "engine")).exists():
                    attr["name_md"] = self.rel_path(p)

            for key in attr:
                if attr[key] is not None:
                    f[f"{c}/{r}"].attrs[key] = attr[key]

            f[f"{c}/{r}"].attrs["num"] = 0
            f[f"{c}/{r}"].attrs["num_vals"] = np.array([], dtype=np.int32)

            # update c
            f[f"{c}"].attrs["num"] += 1
            f[f"{c}"].attrs["num_vals"] = np.append(f[f"{c}"].attrs["num_vals"], int(r))

            self.h5file.flush()

    def add_round_from_md(self, md: MDEngine, cv: int | None = None, r: int | None = None):
        if cv is None:
            c = self.cv
        else:
            c = cv

        if r is None:
            r = self._get_attr(c=c, name="num")

        # r = self.round + 1

        assert c != -1, "run add_cv first"

        directory = self.path(c=c, r=r)
        if not os.path.isdir(directory):
            os.mkdir(directory)

        name_md = directory / "engine.json"
        name_bias = directory / "bias.json"
        md.save(name_md)
        md.bias.save(name_bias)
        md.bias.collective_variable

        attr = {}

        attr["name_md"] = self.rel_path(name_md)
        attr["name_bias"] = self.rel_path(name_bias)

        self.add_round(attr=attr, stic=md.static_trajectory_info, r=r, c=c)

    def add_md(
        self,
        i,
        d: TrajectoryInfo | None = None,
        attrs=None,
        bias: str | None = None,
        r=None,
        c=None,
    ):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        if bias is None:
            if (p := (self.path(c=c, r=r, i=i) / "new_bias.json")).exists():
                bias = self.rel_path(p)
            elif (p := (self.path(c=c, r=r, i=i) / "new_bias")).exists():
                bias = self.rel_path(p)
            elif (p := (self.path(c=c, r=r, i=i) / "bias.json")).exists():
                bias = self.rel_path(p)
            elif (p := (self.path(c=c, r=r, i=i) / "bias")).exists():
                bias = self.rel_path(p)

        if not (p := self.path(c=c, r=r, i=i) / "trajectory_info.h5").exists():
            assert d is not None
            d.save(filename=p)

        with self.lock:
            f = self.h5file
            if f"{i}" not in f[f"{c}/{r}"]:
                f.create_group(f"{c}/{r}/{i}")

            # check if in recover mode

            if attrs is None:
                attrs = {}

            if "valid" not in attrs:
                if (p := self.path(c=c, r=r, i=i) / "invalid").exists():
                    attrs["valid"] = False
                else:
                    attrs["valid"] = True

            if "finished" not in attrs:
                if not (p := self.path(c=c, r=r, i=i) / "finished").exists():
                    attrs["finished"] = False
                else:
                    attrs["finished"] = True

            if bias is not None:
                attrs["name_bias"] = bias

            # copy
            for key, val in attrs.items():
                if val is not None:
                    f[f"{c}/{r}/{i}"].attrs[key] = val

            f[f"{c}/{r}"].attrs["num"] += 1
            f[f"{c}/{r}"].attrs["num_vals"] = np.append(
                f[f"{c}/{r}"].attrs["num_vals"],
                int(i),
            )

            self.h5file.flush()

    ######################################
    #             retrieval              #
    ######################################
    def iter(
        self,
        start=None,
        stop=None,
        num=3,
        ignore_invalid=False,
        only_finshed=True,
        c=None,
        md_trajs: list[int] | None = None,
        print_timings=False,
    ) -> Iterable[tuple[RoundInformation, TrajectoryInformation]]:
        t_0 = time.time()
        load_r_time = 0
        load_i_time = 0
        yield_time = 0

        if c is None:
            c = self.cv

        if stop is None:
            stop = self._get_attr(c=c, name="num") - 1

        if start is None:
            start = 0

        if md_trajs is not None:
            assert num == 1

        for r0 in range(max(stop - (num - 1), start), stop + 1):
            t_r = time.time()
            _r = self.round_information(c=c, r=r0)
            load_r_time += time.time() - t_r

            if not _r.valid and not ignore_invalid:
                continue

            if md_trajs is not None:
                rn = list(set(md_trajs).intersection(set(_r.num_vals)))
            else:
                rn = _r.num_vals

            rn.sort()

            for i in rn:
                t_i = time.time()
                try:
                    _r_i = self.get_trajectory_information(c=c, r=r0, i=i)
                except Exception as e:
                    print(f"could not load {c=} {r0=} {i=} {e=}, skipping")
                    continue

                load_i_time += time.time() - t_i

                if not _r_i.valid and not ignore_invalid:
                    continue

                if (not _r_i.finished) and only_finshed:
                    continue
                # no points in collection
                if _r_i.ti._size <= 0:
                    continue

                t_y = time.time()
                yield _r, _r_i
                yield_time += time.time() - t_y

        t_1 = time.time()

        if print_timings:
            print(f"{'iter stats':-^16}")
            print(f"load round time {load_r_time}")
            print(f"load traj  time {load_i_time}")
            print(f"yield time {yield_time}")
            print(f"total iter time {t_1 - t_0}")
            print(f"{'end iter stats':-^16}")

    @dataclass(repr=False)
    class data_loader_output:
        sp: list[SystemParams]
        nl: list[NeighbourList] | None
        cv: list[CV]
        sti: StaticMdInfo
        ti: list[TrajectoryInfo]
        collective_variable: CollectiveVariable
        time_series: bool = False
        bias: list[Bias] | None = None

        def __iter__(self):
            for spi, nli, cvi, ti in zip(self.sp, self.nl, self.cv, self.ti):
                yield Rounds.data_loader_output(
                    sp=[spi],
                    nl=[nli],
                    cv=[cvi],
                    ti=[ti],
                    sti=self.sti,
                    collective_variable=self.collective_variable,
                )

        def __add__(self, other):
            assert isinstance(other, Rounds.data_loader_output)
            return Rounds.data_loader_output(
                sp=[*self.sp, *other.sp],
                nl=[*self.nl, *other.nl] if self.nl is not None else None,
                cv=[*self.cv, *other.cv],
                ti=[*self.ti, *other.ti],
                sti=self.sti,
                collective_variable=self.collective_variable,
            )

        def weights(self, norm: str | None = None):
            beta = 1 / (self.sti.T * boltzmann)

            weights = []

            for ti_i in self.ti:
                if ti_i.e_bias is None:
                    weights.append(None)
                    print("no e_bias")
                    continue

                u = beta * ti_i.e_bias

                if norm is None:
                    pass
                elif norm == "max":
                    u -= jnp.max(u)
                else:
                    raise

                w = jnp.exp(u)

                if norm:
                    if (n := np.sum(w)) != 0:
                        w /= n
                weights.append(w)

            return weights

    def data_loader(
        self,
        num=4,
        out=-1,
        split_data=False,
        new_r_cut=-1,
        cv_round=None,
        filter_bias=False,
        filter_energy=False,
        ignore_invalid=False,
        energy_threshold=None,
        md_trajs: list[int] | None = None,
        start: int | None = None,
        stop: int | None = None,
        time_series: bool = False,
        T_max_over_T=50,
        chunk_size=None,
        get_colvar=True,
        min_traj_length=None,
        recalc_cv=False,
        get_bias_list=False,
        num_cv_rounds=1,
        only_finished=True,
    ) -> data_loader_output:
        weights = []

        if new_r_cut == -1:
            new_r_cut = self.round_information(c=cv_round).tic.r_cut

        sti: StaticMdInfo | None = None
        sp: list[SystemParams] = []
        cv: list[CV] = []
        ti: list[TrajectoryInfo] = []
        if get_bias_list:
            bias_list: list[Bias] = []

        min_energy = None

        out = int(out)

        cvrnds = []

        if num_cv_rounds != 1:
            if cv_round is None:
                cv_round = self.cv

            cvrnds = range(max(0, cv_round - num_cv_rounds), cv_round + 1)
            recalc_cv = True
        else:
            cvrnds.append(cv_round)

        if get_colvar or recalc_cv:
            colvar = self.get_collective_variable(c=cv_round)
        else:
            colvar = None

        for cv_round in cvrnds:
            for round, traj in self.iter(
                start=start,
                stop=stop,
                num=num,
                c=cv_round,
                ignore_invalid=ignore_invalid,
                md_trajs=md_trajs,
                only_finshed=only_finished,
            ):
                if min_traj_length is not None:
                    if traj.ti._size < min_traj_length:
                        # print(f"skipping trajectyory because it's not long enough {traj.ti._size}<{min_traj_length}")
                        continue
                    # else:
                    # print("adding trajectory")

                if sti is None:
                    sti = round.tic

                ti.append(traj.ti)
                sp0 = traj.ti.sp

                if (cv0 := traj.ti.CV) is None:
                    print("calculating CV")

                    if colvar is None:
                        bias = traj.get_bias()
                        colvar = bias.collective_variable

                    nlr = (
                        sp0.get_neighbour_list(
                            r_cut=round.tic.r_cut,
                            z_array=round.tic.atomic_numbers,
                            chunk_size=chunk_size,
                        )
                        if round.tic.r_cut is not None
                        else None
                    )

                    cv0, _ = colvar.compute_cv(sp=sp0, nl=nlr)

                e = None

                if filter_bias:
                    if (b0 := traj.ti.e_bias) is None:
                        # map cvs
                        bias = traj.get_bias()

                        b0, _ = bias.compute_from_cv(cvs=cv0)

                    e = b0

                if filter_energy:
                    if (e0 := traj.ti.e_pot) is None:
                        raise ValueError("e_pot is None")

                    if e is None:
                        e = e0
                    else:
                        e = e + e0

                if energy_threshold is not None:
                    if (e0 := traj.ti.e_pot) is None:
                        raise ValueError("e_pot is None")

                    if min_energy is None:
                        min_energy = e0.min()
                    else:
                        min_energy = min(min_energy, e0.min())

                sp.append(sp0)
                cv.append(cv0)
                if get_bias_list:
                    bias_list.append(traj.get_bias())

                if e is None:
                    weights.append(None)
                else:
                    beta = 1 / (round.tic.T * boltzmann)
                    weight = -beta * e

                    # weight = jnp.exp(-beta * e)
                    weights.append(weight)

        assert sti is not None
        assert len(sp) != 0

        def choose(key, probs: Array, len: int):
            if len is None:
                len = probs.shape[0]

            key, key_return = split(key, 2)

            indices = choice(
                key=key,
                a=len,
                shape=(int(out),),
                p=probs,
                replace=out > len,
            )

            return key_return, indices

        key = PRNGKey(0)

        if energy_threshold is not None:
            raise
            sp_new: list[SystemParams] = []
            cv_new: list[CV] = []
            ti_new: list[TrajectoryInfo] = []
            new_weights = []

            for n in range(len(sp)):
                indices = jnp.where(ti[n].e_pot - min_energy < energy_threshold)[0]

                if len(indices) == 0:
                    continue

                if time_series:
                    print("energy threshold surpassed in time_series, removing the data")
                    continue

                sp_new.append(sp[n][indices])
                cv_new.append(cv[n][indices])
                ti_new.append(ti[n][indices])
                new_weights.append(weights[n])

            sp = sp_new
            ti = ti_new
            cv = cv_new
            weights = new_weights

        if T_max_over_T is not None:
            sp_new: list[SystemParams] = []
            cv_new: list[CV] = []
            ti_new: list[TrajectoryInfo] = []
            new_weights = []
            if get_bias_list:
                new_bias_list = []

            for n in range(len(sp)):
                indices = jnp.where(ti[n].T > sti.T * T_max_over_T)[0]

                if len(indices) != 0:
                    print(f"temperature threshold surpassed in time_series {n=}, removing the data")
                    continue

                sp_new.append(sp[n])

                cv_new.append(cv[n])
                ti_new.append(ti[n])
                new_weights.append(weights[n])
                if get_bias_list:
                    new_bias_list.append(bias_list[n])

            sp = sp_new
            ti = ti_new
            cv = cv_new
            weights = new_weights
            if get_bias_list:
                bias_list = new_bias_list

        out_sp: list[SystemParams] = []
        out_cv: list[CV] = []
        out_ti = []

        if time_series:
            if out == -1:
                skip_step = 1
            else:
                n_points = sum([a.shape[0] for a in sp])

                skip_step = n_points // out

                if skip_step == 0:
                    skip_step = 1
                    print(f"not enough data, returning {n_points=} ")

            for sp_n, cv_n, ti_n in zip(sp, cv, ti):
                out_sp.append(sp_n[::skip_step])
                out_cv.append(cv_n[::skip_step])
                out_ti.append(ti_n[::skip_step])

        else:
            if split_data:
                for n, wi in enumerate(weights):
                    if wi is None:
                        probs = None
                    else:
                        wi -= jnp.mean(wi)
                        wi = jnp.exp(-wi)
                        probs = wi / jnp.sum(wi)
                    key, indices = choose(key, probs, len=sp[n].shape[0])

                    out_sp.append(sp[n][indices])
                    out_cv.append(cv[n][indices])
                    out_ti.append(ti[n][indices])

            else:
                if weights[0] is None:
                    probs = None
                else:
                    ws = jnp.hstack(weights)
                    ws -= jnp.mean(ws)
                    probs = jnp.exp(-ws)
                    probs /= jnp.sum(probs)

                key, indices = choose(key, probs, len=sum([sp_n.shape[0] for sp_n in sp]))

                indices = jnp.sort(indices)

                count = 0

                sp_trimmed = []
                cv_trimmed = []
                ti_trimmed = []

                for n, (sp_n, cv_n, ti_n) in enumerate(zip(sp, cv, ti)):
                    n_i = sp_n.shape[0]

                    index = indices[jnp.logical_and(count <= indices, indices < count + n_i)] - count
                    sp_trimmed.append(sp_n[index])

                    cv_trimmed.append(cv_n[index])
                    ti_trimmed.append(ti_n[index])

                    count += n_i

                count = 0

                # t0 = time.time()

                out_sp.append(SystemParams.stack(*sp_trimmed))
                out_cv.append(CV.stack(*cv_trimmed))
                out_ti.append(TrajectoryInfo.stack(*ti_trimmed))

                # print(f"{'stack time':-^20}={t0- time.time()}")
                bias = None

        out_nl = None
        if new_r_cut is not None:
            # much faster if stacked
            if len(out_sp) >= 2:
                out_sp_merged = SystemParams.stack(*out_sp)
            else:
                out_sp_merged = out_sp[0]

            out_nl_stacked = out_sp_merged.get_neighbour_list(
                r_cut=new_r_cut,
                z_array=sti.atomic_numbers,
                chunk_size=chunk_size,
            )

            if len(out_sp) >= 2:
                ind = 0
                out_nl = []
                for osp in out_sp:
                    out_nl.append(out_nl_stacked[ind : ind + osp.shape[0]])
                    ind += osp.shape[0]
            else:
                out_nl = [out_nl_stacked]

        if recalc_cv:
            # print(f"old cvs {out_cv=}")

            if len(out_sp) >= 2:
                out_sp_merged = SystemParams.stack(*out_sp)
                out_nl_merged = NeighbourList.stack(*out_nl)
            else:
                out_sp_merged = out_sp[0]
                out_nl_merged = out_nl[0]

            out_cv_stacked = padded_pmap(Partial(colvar.compute_cv, chunk_size=chunk_size))(
                out_sp_merged,
                out_nl_merged,
            )[0]

            if len(out_sp) >= 2:
                out_cv = out_cv_stacked.replace(_stack_dims=[a.shape[0] for a in out_sp]).unstack()
            else:
                out_cv = [out_cv_stacked]

            # print(f"new cvs {out_cv=}")

        return Rounds.data_loader_output(
            sp=out_sp,
            nl=out_nl,
            cv=out_cv,
            ti=out_ti,
            sti=sti,
            collective_variable=colvar,
            time_series=time_series,
            bias=bias_list if get_bias_list else None,
        )

    def copy_from_previous_round(
        self,
        num_copy=2,
        out=-1,
        cv_trans: None | CvTrans = None,
        chunk_size=None,
        filter_bias=True,
        filter_energy=True,
        split_data=True,
        md_trajs: list[int] | None = None,
        dlo: Rounds.data_loader_output | None = None,
        new_cvs: list[CV] | None = None,
        invalidate: bool = True,
        cv_round=None,
    ):
        if cv_round is None:
            cv_round = self.cv - 1

        current_round = self.round_information()
        # bias = self.get_bias()
        col_var = self.get_collective_variable()

        if dlo is None:
            dlo = self.data_loader(
                num=num_copy,
                out=out,
                filter_bias=filter_bias,
                filter_energy=filter_energy,
                split_data=split_data,
                new_r_cut=current_round.tic.r_cut,
                md_trajs=md_trajs,
                cv_round=cv_round,
            )

        if new_cvs is None:
            new_cvs = []
            for i, (sp, nl, cv, traj_info) in enumerate(zip(dlo.sp, dlo.nl, dlo.cv, dlo.ti)):
                # # TODO: tranform biasses probabilitically or with jacobian determinant
                sp: SystemParams
                nl: NeighbourList
                cv: CV
                traj_info: TrajectoryInfo

                if cv_trans is not None:
                    new_cvs.append(cv_trans.compute_cv_trans(x=cv, nl=nl, chunk_size=chunk_size)[0])
                else:
                    new_cvs.append(col_var.compute_cv(sp=sp, nl=nl, chunk_size=chunk_size)[0])

        for i, (sp, nl, cv, traj_info, new_cv) in enumerate(zip(dlo.sp, dlo.nl, dlo.cv, dlo.ti, new_cvs)):
            # # TODO: tranform biasses probabilitically or with jacobian determinant
            sp: SystemParams
            nl: NeighbourList
            cv: CV
            traj_info: TrajectoryInfo
            new_cv: CV

            round_path = self.path(c=self.cv, r=0, i=i)
            round_path.mkdir(parents=True, exist_ok=True)

            # bias.save(round_path / "bias.json")
            traj_info

            new_traj_info = TrajectoryInfo(
                _positions=traj_info.positions,
                _cell=traj_info.cell,
                _charges=traj_info.charges,
                _e_pot=traj_info.e_pot,
                _e_pot_gpos=traj_info.e_pot_gpos,
                _e_pot_vtens=traj_info.e_pot_vtens,
                _e_bias=None,
                _e_bias_gpos=None,
                _e_bias_vtens=None,
                _cv=new_cv.cv,
                _T=traj_info._T,
                _P=traj_info._P,
                _err=traj_info._err,
                _t=traj_info._t,
                _capacity=traj_info._capacity,
                _size=traj_info._size,
            )

            self.add_md(
                i=i,
                d=new_traj_info,
                attrs=None,
                bias=None,  # self.rel_path(round_path / "bias.json"),
            )

            if invalidate:
                self.invalidate_data(c=self.cv, r=self.round, i=i)

    def iter_ase_atoms(
        self,
        r: int | None = None,
        c: int | None = None,
        num: int = 3,
        r_cut=None,
        minkowski_reduce=True,
    ):
        from molmod import angstrom

        for round, trajejctory in self.iter(stop=r, c=c, num=num, ignore_invalid=True):
            # traj = trajejctory.ti

            sp = trajejctory.ti.sp

            if minkowski_reduce:
                _, op = sp[0].canonicalize(qr=True)

                sp = jax.vmap(SystemParams.apply_canonicalize, in_axes=(0, None))(sp, op)

            pos_A = sp.coordinates / angstrom
            pbc = sp.cell is not None
            if pbc:
                cell_A = sp.cell / angstrom

                atoms = [
                    ase.Atoms(
                        numbers=round.tic.atomic_numbers,
                        masses=round.tic.masses,
                        positions=pos,
                        pbc=pbc,
                        cell=cell,
                    )
                    for pos, cell in zip(pos_A, cell_A)
                ]

            else:
                atoms = [
                    ase.Atoms(
                        numbers=round.tic.atomic_numbers,
                        masses=round.tic.masses,
                        positions=positions,
                    )
                    for positions in pos_A
                ]

            yield atoms, round, trajejctory

    def get_trajectory_information(
        self,
        r: int,
        i: int,
        c: int | None = None,
    ) -> TrajectoryInformation:
        if c is None:
            c = self.cv

        with self.lock:
            f = self.h5file
            d = f[f"{c}/{r}/{i}"]

            ti = TrajectoryInfo.load(self.path(c=c, r=r, i=i) / "trajectory_info.h5")
            r_attr = {key: d.attrs[key] for key in d.attrs}

            self.h5file.flush()

        return TrajectoryInformation(
            ti=ti,
            **r_attr,
            round=r,
            num=i,
            folder=self.folder,
        )

    def round_information(
        self,
        c: int | None = None,
        r: int | None = None,
    ) -> RoundInformation:
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        with self.lock:
            f = self.h5file

            folder = self.path(c=c, r=r)

            stic = StaticMdInfo.load(folder / "static_trajectory_info.h5")

            d = f[f"{c}/{r}"].attrs
            r_attr = {key: d[key] for key in d}

        return RoundInformation(
            round=int(r),
            folder=self.folder,
            tic=stic,
            **r_attr,
        )

    ######################################
    #           Properties               #
    ######################################
    def _set_attr(self, name, value, c=None, r=None, i=None):
        with self.lock:
            f = self.h5file
            if c is None:
                c = self.cv

            f = f[f"{c}"]
            if r is not None:
                f = f[f"{r}"]

            if i is not None:
                assert r is not None, "also provide round"
                f = f[f"{i}"]

            f.attrs[name] = value

            self.h5file.flush()

    def _get_attr(self, name, c=None, r=None, i=None):
        with self.lock:
            f = self.h5file
            if c is None:
                c = self.cv

            f = f[f"{c}"]

            if r is not None:
                f2 = f[f"{r}"]
            else:
                f2 = f

            if i is not None:
                assert r is not None, "also provide round"
                f2 = f[f"/{i}"]

            return f2.attrs[name]

    @property
    def T(self):
        return self.round_information().tic.T

    @property
    def P(self):
        return self.round_information().tic.P

    @property
    def round(self):
        c = self.cv
        with self.lock:
            f = self.h5file
            l = [int(i) for i in f[f"{c}"]]
            l.sort()
            return l[-1]

    def get_round(self, c=None):
        return self._get_attr(c=c, name="num") - 1

    @property
    def cv(self):
        try:
            with self.lock:
                f = self.h5file
                l = [int(i) for i in f]
                l.sort()
                return l[-1]
        except Exception:
            return -1

    def n(self, c=None, r=None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)
        return self.round_information(r=r).num

    def invalidate_data(self, c=None, r=None, i=None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        if not (p := self.path(c=c, r=r, i=i) / "invalid").exists():
            if not p.parent.exists():
                p.parent.mkdir(parents=True)

            with open(p, "w+"):
                pass

        self._set_attr(name="valid", value=False, c=c, r=r, i=i)

    def finish_data(self, c=None, r=None, i=None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        if not (p := self.path(c=c, r=r, i=i) / "finished").exists():
            if not p.parent.exists():
                p.parent.mkdir(parents=True)

            with open(p, "w+"):
                pass

        self._set_attr(name="finished", value=True, c=c, r=r, i=i)

    def is_valid(self, c=None, r=None, i=None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        return (self.path(c=c, r=r, i=i) / "invalid").exists()

    def is_finished(self, c=None, r=None, i=None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        return (self.path(c=c, r=r, i=i) / "finished").exists()

    def get_collective_variable(
        self,
        c=None,
    ) -> CollectiveVariable:
        bias = self.get_bias(c=c)
        return bias.collective_variable

    def get_bias(self, c=None, r=None, i=None) -> Bias:
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        bn = self._get_attr("name_bias", c=c, r=r, i=i)

        return Bias.load(self.full_path(bn))

    def get_engine(self, c=None, r=None) -> MDEngine:
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        name = self._get_attr("name_md", c=c, r=r)
        return MDEngine.load(self.full_path(name), filename=None)

    ######################################
    #          MD simulations            #
    ######################################

    def run(self, bias, steps):
        self.run_par([bias], steps)

    def run_par(
        self,
        biases: Iterable[Bias],
        steps,
        plot=True,
        KEY=42,
        sp0: SystemParams | None = None,
        filter_e_pot=False,
        correct_previous_bias=False,
        ignore_invalid=False,
        md_trajs: list[int] | None = None,
        cv_round: int | None = None,
        wait_for_plots=False,
        min_traj_length=None,
        recalc_cv=False,
        only_finished=True,
    ):
        if cv_round is None:
            cv_round = self.cv

        round = self.get_round(c=cv_round)

        if isinstance(KEY, int):
            KEY = jax.random.PRNGKey(KEY)

        with self.lock:
            f = self.h5file
            common_bias_name = self.full_path(
                f[f"{cv_round}/{round}"].attrs["name_bias"],
            )
            common_md_name = self.full_path(
                f[f"{cv_round}/{round}"].attrs["name_md"],
            )
        from parsl.dataflow.dflow import AppFuture

        tasks: list[tuple[int, AppFuture]] | None = None
        plot_tasks = []
        md_engine = MDEngine.load(common_md_name)

        sp0_provided = sp0 is not None

        if not sp0_provided:
            data = self.data_loader(
                num=2,
                out=1000,
                split_data=False,
                filter_bias=correct_previous_bias,
                filter_energy=filter_e_pot,
                new_r_cut=None,
                ignore_invalid=ignore_invalid,
                md_trajs=md_trajs,
                cv_round=cv_round,
                min_traj_length=min_traj_length,
                recalc_cv=recalc_cv,
                only_finished=only_finished,
            )

            cv_stack = data.cv[0]
            sp_stack = data.sp[0]
            # print(f"{cv_stack=}")
        else:
            assert sp0.shape[0] == len(
                biases,
            ), f"The number of initials cvs provided {sp0.shape[0]} does not correspond to the number of biases {len(biases)}"

        for i, bias in enumerate(biases):
            path_name = self.path(c=cv_round, r=round, i=i)
            if not os.path.exists(path_name):
                os.mkdir(path_name)

            b = CompositeBias.create([Bias.load(common_bias_name), bias])

            b_name = path_name / "bias.json"
            b_name_new = path_name / "bias_new.json"
            b.save(b_name)

            traj_name = path_name / "trajectory_info.h5"

            if not sp0_provided:
                biases, _ = bias.compute_from_cv(cvs=cv_stack)
                biases -= jnp.min(biases)

                probs = jnp.exp(
                    -biases / (md_engine.static_trajectory_info.T * boltzmann),
                )
                probs = probs / jnp.sum(probs)

                KEY, k = jax.random.split(KEY, 2)
                index = jax.random.choice(
                    a=probs.shape[0],
                    key=k,
                    p=probs,
                )

                spi = sp_stack[index]
                # spi = spi.unbatch()
                # cvi = cv_stack[index]

            else:
                spi = sp0[i]
                # cvi = "unknown"
                spi = spi.unbatch()

            # cvi_recalc, _ = md_engine.bias.collective_variable.compute_cv(
            #     spi, spi.get_neighbour_list(data.sti.r_cut, data.sti.atomic_numbers)
            # )

            # print(f"starting trajectory {i} in {cvi=} {spi=}  ")

            future = bash_app_python(Rounds.run_md, pass_files=True, executors=["reference"])(
                sp=spi,  # type: ignore
                inputs=[File(common_md_name), File(str(b_name))],
                outputs=[File(str(b_name_new)), File(str(traj_name))],
                steps=int(steps),
                execution_folder=path_name,
            )

            if plot:
                plot_file = path_name / "plot.pdf"

                plot_fut = bash_app_python(Rounds.plot_md_run, pass_files=True, executors=["default"])(
                    traj=future,
                    st=md_engine.static_trajectory_info,
                    inputs=[future.outputs[0]],
                    outputs=[File(str(plot_file))],
                    execution_folder=path_name,
                )

                plot_tasks.append(plot_fut)

            if tasks is None:
                tasks = [(i, future)]
            else:
                tasks.append((i, future))

        assert tasks is not None

        # wait for tasks to finish
        for i, future in tasks:
            try:
                d = future.result()
                self.add_md(d=d, bias=self.rel_path(Path(future.outputs[0].filename)), i=i, c=cv_round)
            except Exception as e:
                print(f"got exception {e} while collecting md {i}, round {round}, cv {cv_round}, continuing anyway")

        # wait for plots to finish
        if plot and wait_for_plots:
            for i, future in enumerate(plot_tasks):
                try:
                    d = future.result()
                except Exception as e:
                    print(f"got exception {e} while trying to collect plot of {i}, continuing ")

    def continue_run(
        self,
        steps: int,
        cv_round: int | None = None,
        round: int | None = None,
        plot=True,
        wait_for_plots=True,
    ):
        if cv_round is None:
            cv_round = self.cv

        if round is None:
            round = self.get_round(c=cv_round)

        with self.lock:
            f = self.h5file
            common_md_name = self.full_path(
                f[f"{cv_round}/{round}"].attrs["name_md"],
            )

        md_engine = MDEngine.load(common_md_name)

        from parsl.dataflow.dflow import AppFuture

        tasks: list[tuple[int, AppFuture]] | None = None
        plot_tasks = []

        ri = self.round_information(c=cv_round, r=round)

        for i in ri.num_vals:
            path_name = self.path(c=cv_round, r=round, i=i)

            b_name = path_name / "bias.json"
            b_name_new = path_name / "bias_new.json"

            if not b_name.exists():
                print(f"skipping {i=}, {b_name_new=}not found")
                continue

            if b_name_new.exists():
                print(f"skipping {i=}, new bias found")
                continue

            traj_name = path_name / "trajectory_info.h5"

            future = bash_app_python(Rounds.run_md, pass_files=True, executors=["reference"])(
                sp=None,  # type: ignore
                inputs=[File(common_md_name), File(str(b_name))],
                outputs=[File(str(b_name_new)), File(str(traj_name))],
                steps=int(steps),
                execution_folder=path_name,
            )

            if plot:
                plot_file = path_name / "plot.pdf"

                plot_fut = bash_app_python(Rounds.plot_md_run, pass_files=True, executors=["default"])(
                    traj=future,
                    st=md_engine.static_trajectory_info,
                    inputs=[future.outputs[0]],
                    outputs=[File(str(plot_file))],
                    execution_folder=path_name,
                )

                plot_tasks.append(plot_fut)

            if tasks is None:
                tasks = [(i, future)]
            else:
                tasks.append((i, future))

        assert tasks is not None

        # wait for tasks to finish
        for i, future in tasks:
            try:
                d = future.result()
                self.add_md(d=d, bias=self.rel_path(Path(future.outputs[0].filename)), i=i, c=cv_round)
            except Exception as e:
                print(f"got exception {e} while collecting md {i}, round {round}, cv {cv_round}, continuing anyway")

        # wait for plots to finish
        if plot and wait_for_plots:
            for i, future in enumerate(plot_tasks):
                try:
                    d = future.result()
                except Exception as e:
                    print(f"got exception {e} while trying to collect plot of {i}, continuing ")

    @staticmethod
    def run_md(
        steps: int,
        sp: SystemParams | None,
        inputs=[],
        outputs=[],
    ) -> TrajectoryInfo:
        bias = Bias.load(inputs[1].filepath)

        kwargs = dict(
            bias=bias,
            trajectory_file=outputs[1].filepath,
        )
        if sp is not None:
            kwargs["sp"] = sp
        md = MDEngine.load(inputs[0].filepath, **kwargs)

        if sp is not None:
            # assert md.sp == sp
            print(f"will start with {sp=}")

        md.run(steps)
        bias.save(outputs[0].filepath)
        d = md.get_trajectory()
        return d

    @staticmethod
    def plot_md_run(
        st: StaticMdInfo,
        traj: TrajectoryInfo,
        inputs=[],
        outputs=[],
    ):
        bias = Bias.load(inputs[0].filepath)

        if st.equilibration is not None:
            if traj._t is not None:
                traj = traj[traj._t > st.equilibration]

        cvs = traj.CV
        if cvs is None:
            sp = traj.sp
            nl = sp.get_neighbour_list(
                r_cut=st.r_cut,
                z_array=st.atomic_numbers,
            )
            cvs, _ = bias.collective_variable.compute_cv(sp=sp, nl=nl)

        bias.plot(name=outputs[0].filepath, traj=[cvs])
