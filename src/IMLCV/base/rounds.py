from __future__ import annotations

import os
import shutil
import time
from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import ase
import jax
import jax.numpy as jnp
import numpy as np
from equinox import Partial
from IMLCV.base.bias import Bias
from IMLCV.base.bias import CompositeBias
from IMLCV.implementations.bias import RbfBias
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV, CvMetric
from IMLCV.base.CV import CvTrans, CvFlow
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
from IMLCV.configs.config_general import DEFAULT_LABELS, REFERENCE_LABELS, TRAINING_LABELS
from typing import Callable
import scipy.linalg
from jax import vmap
from functools import partial
from IMLCV.implementations.bias import _clip
from IMLCV.base.bias import BiasModify
from IMLCV.base.CV import chunk_map
from IMLCV.base.bias import NoneBias
from IMLCV.base.CVDiscovery import Transformer
from molmod.units import kjmol


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
        if self.name_bias is None:
            return None

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


@dataclass
class Rounds(ABC):
    """
    Class that bundle all the information in a folder structure. The structure is shown below. Files within parentheses are not always present optional.


    folder/
        cv_0/
            cv.json
            round 0/
                static_trajectory info.h5
                bias.json
                engine.json
                (invalid)
                md_0/
                    trajectory_info.h5
                    bias.json
                    (bias_new.json)
                    (invalid)
                    (finished)

                md_1
                    ..
                ...
            round 1/
                ...
            ...
        cv_1/
            ...
        ...



    """

    folder: Path

    @staticmethod
    def create(
        folder: str | Path = "output",
        copy=True,
        new_folder=True,
    ) -> Rounds:
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

        folder = p

        rnds = Rounds(folder=p)

        return rnds

    ######################################
    #             IO                     #
    ######################################

    def __getstate__(self):
        return self.folder.absolute()

    def __setstate__(self, state):
        self.folder = state

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
        only_finished=False,
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
                only_finished=only_finished,
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

    def _c_vals(self):
        cvs = []

        for cv_c in self.path().glob("cv_*"):
            c = cv_c.parts[-1][3:]

            cvs.append(int(c))

        cvs.sort()

        if len(cvs) == 0:
            cvs.append(-1)

        return cvs

    def _r_vals(self, c=None):
        if c is None:
            c = self.cv

        rounds = []

        for round_r in self.path(c=c).glob("round_*"):
            r = round_r.parts[-1][6:]

            if not (p := self.path(c=c, r=r) / "static_trajectory_info.h5").exists():
                print(f"could not find {p}")
                continue

            rounds.append(int(r))

        rounds.sort()

        if len(rounds) == 0:
            rounds.append(-1)

        return rounds

    def _i_vals(self, c=None, r=None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        i_s = []

        for md_i in self.path(c=c, r=r).glob("md_*"):
            i = md_i.parts[-1][3:]

            if not (md_i / "trajectory_info.h5").exists():
                continue

            i_s.append(int(i))

        i_s.sort()

        return i_s

    def _name_md(self, c, r):
        if (p := (self.path(c=c, r=r) / "engine.json")).exists():
            return self.rel_path(p)
        elif (p := (self.path(c=c, r=r) / "engine")).exists():
            return self.rel_path(p)

        return None

    def _name_bias(self, c, r, i=None):
        if (p := (self.path(c=c, r=r, i=i) / "bias_new.json")).exists():
            return self.rel_path(p)
        elif (p := (self.path(c=c, r=r, i=i) / "bias_new")).exists():
            return self.rel_path(p)
        elif (p := (self.path(c=c, r=r, i=i) / "bias.json")).exists():
            return self.rel_path(p)
        elif (p := (self.path(c=c, r=r, i=i) / "bias")).exists():
            return self.rel_path(p)

        return None

    def _num_vals(self, c, r=None):
        if r is not None:
            return len(self._i_vals(c, r))

        return len(self._r_vals(c))

    def add_cv(self, c=None):
        if c is None:
            c = self.cv + 1

        dir = self.path(c=c)
        if not dir.exists():
            dir.mkdir(parents=True)

    def add_cv_from_cv(self, cv: CollectiveVariable, c=None):
        if c is None:
            c = self.cv + 1

        # attr = {}

        directory = self.path(c=c)
        if not os.path.isdir(directory):
            os.mkdir(directory)

        cv.save(self.path(c=c) / "cv.json")

        # attr["name_cv"] = self.rel_path(self.path(c=c) / "cv.json")
        self.add_cv(
            # (attr=attr,
            c=c
        )

    def add_round(self, stic: StaticMdInfo | None = None, c=None, r=None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c) + 1

        dir = self.path(c=c, r=r)
        if not dir.exists():
            dir.mkdir(parents=True)

        if not (p := self.path(c=c, r=r) / "static_trajectory_info.h5").exists():
            assert stic is not None
            stic.save(p)

    def add_round_from_md(self, md: MDEngine, cv: int | None = None, r: int | None = None):
        if cv is None:
            c = self.cv
        else:
            c = cv

        if r is None:
            r = self._r_vals(c=c)[-1] + 1

        assert c != -1, "run add_cv first"

        directory = self.path(c=c, r=r)
        if not os.path.isdir(directory):
            os.mkdir(directory)

        name_md = directory / "engine.json"
        name_bias = directory / "bias.json"
        md.save(name_md)
        md.bias.save(name_bias)

        self.add_round(stic=md.static_trajectory_info, r=r, c=c)

    def add_md(
        self,
        i,
        d: TrajectoryInfo | None = None,
        bias: str | None = None,
        r=None,
        c=None,
    ):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        # if bias is None:
        #     if (p := (self.path(c=c, r=r, i=i) / "new_bias.json")).exists():
        #         bias = self.rel_path(p)
        #     elif (p := (self.path(c=c, r=r, i=i) / "new_bias")).exists():
        #         bias = self.rel_path(p)
        #     elif (p := (self.path(c=c, r=r, i=i) / "bias.json")).exists():
        #         bias = self.rel_path(p)
        #     elif (p := (self.path(c=c, r=r, i=i) / "bias")).exists():
        #         bias = self.rel_path(p)

        if not (p := self.path(c=c, r=r, i=i) / "trajectory_info.h5").exists():
            assert d is not None
            d.save(filename=p)

    ######################################
    #             retrieval              #
    ######################################
    def iter(
        self,
        start=None,
        stop=None,
        num=3,
        ignore_invalid=False,
        only_finished=True,
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
            stop = self._num_vals(c=c) - 1

        if start is None:
            start = 0

        if md_trajs is not None:
            assert num == 1

        low = max(stop - (num - 1), start)
        high = stop + 1

        for r0 in range(low, high):
            t_r = time.time()
            _r = self._round_information(c=c, r=r0)
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
                    _r_i = self._trajectory_information(c=c, r=r0, i=i)
                except Exception as e:
                    print(f"could not load {c=} {r0=} {i=} {e=}, skipping")
                    continue

                load_i_time += time.time() - t_i

                if not _r_i.valid and not ignore_invalid:
                    continue

                if (not _r_i.finished) and only_finished:
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

    def data_loader(
        self,
        num=4,
        out=-1,
        split_data=False,
        new_r_cut=-1,
        cv_round=None,
        ignore_invalid=False,
        md_trajs: list[int] | None = None,
        start: int | None = None,
        stop: int | None = None,
        time_series: bool = False,
        T_max_over_T=50,
        chunk_size=None,
        get_colvar=True,
        min_traj_length=None,
        recalc_cv=False,
        get_bias_list=True,
        num_cv_rounds=1,
        only_finished=True,
        uniform=False,
        lag_n=1,
        colvar=None,
        check_dtau=True,
        verbose=False,
        weight=True,
        T_scale=3,
        macro_chunk=10000,
    ) -> data_loader_output:
        # weights = []

        if cv_round is None:
            cv_round = self.cv

        sti = self._round_information(c=cv_round).tic

        if new_r_cut == -1:
            new_r_cut = sti.r_cut

        sp: list[SystemParams] = []
        cv: list[CV] = []
        ti: list[TrajectoryInfo] = []
        weights: list[Array] = []

        if not time_series:
            lag_n = 0

        if get_bias_list:
            bias_list: list[Bias] = []

        out = int(out)

        cvrnds = []

        if num_cv_rounds != 1:
            cvrnds = range(max(0, cv_round - num_cv_rounds), cv_round + 1)
            recalc_cv = True

        else:
            cvrnds.append(cv_round)

        try:
            ground_bias = self.get_bias(c=cv_round, r=stop)
        except Exception as e:
            print(f"could not load ground bias {e=}")
            ground_bias = None

        if colvar is not None:
            recalc_cv = True

        if get_colvar or recalc_cv:
            if colvar is None:
                colvar = self.get_collective_variable(c=cv_round)

        if verbose:
            print("obtaining raw data")

        for cvi in cvrnds:
            sti_c: StaticMdInfo | None = None
            sp_c: list[SystemParams] = []
            cv_c: list[CV] = []
            ti_c: list[TrajectoryInfo] = []

            weight_c = weight

            try:
                ground_bias_c = self.get_bias(c=cvi, r=stop)
            except Exception as e:
                print(f"could not load ground bias {e=}")
                ground_bias_c = None
                weight_c = False

            try:
                colvar_c = self.get_collective_variable(c=cvi)
            except Exception as e:
                print(f"could not load collective variable {e=}")
                colvar_c = None
                weight_c = False

            for round_info, traj_info in self.iter(
                start=start,
                stop=stop,
                num=num,
                c=cvi,
                ignore_invalid=ignore_invalid,
                md_trajs=md_trajs,
                only_finished=only_finished,
            ):
                if min_traj_length is not None:
                    if traj_info.ti._size < min_traj_length or traj_info.ti._size <= lag_n:
                        # print(f"skipping trajectyory because it's not long enough {traj.ti._size}<{min_traj_length}")
                        continue
                    # else:
                    # print("adding traweights=jectory")

                if sti_c is None:
                    sti_c = round_info.tic

                ti_c.append(traj_info.ti)

                sp0 = traj_info.ti.sp
                cv0 = traj_info.ti.CV

                if sp0.shape[0] != cv0.shape[0]:
                    print(
                        f"shapes do not match {sp0.shape=} {cv0.shape=} {traj_info.ti._size=} {traj_info.ti._positions.shape=} {traj_info.ti._cell.shape=}  {traj_info.ti._cv.shape=} "
                    )

                    raise

                if cv0 is None:
                    if colvar is None:
                        bias = traj_info.get_bias()
                        colvar = bias.collective_variable

                    nlr = (
                        sp0.get_neighbour_list(
                            r_cut=round_info.tic.r_cut,
                            z_array=round_info.tic.atomic_numbers,
                            chunk_size=chunk_size,
                        )
                        if round_info.tic.r_cut is not None
                        else None
                    )

                    cv0, _ = colvar.compute_cv(sp=sp0, nl=nlr)

                sp_c.append(sp0)
                cv_c.append(cv0)
                if get_bias_list:
                    bias_list.append(traj_info.get_bias())

            if len(sp_c) == 0:
                continue

            if weight_c:
                if verbose:
                    print(f"getting weights for cv_round {cvi}")

                dlo = data_loader_output(
                    sp=sp_c,
                    cv=cv_c,
                    ti=ti_c,
                    sti=sti_c,
                    nl=None,
                    collective_variable=colvar_c,
                    # time_series=time_series,
                    ground_bias=ground_bias_c,
                )
                # select points according to free energy divided by histogram count
                wc = dlo.weights(
                    use_ground_bias=True,
                    correct_U=False,
                    T_scale=T_scale,
                    chunk_size=chunk_size,
                )

                n = sum([len(wi) for wi in wc])
                wc = [wc[i] * n for i in range(len(wc))]

            else:
                wc = [jnp.ones((spi.shape[0])) for spi in sp_c]

            sp.extend(sp_c)
            cv.extend(cv_c)
            ti.extend(ti_c)
            weights.extend(wc)

        assert len(sp) != 0, "no data found"

        def choose(key, probs: Array, out: int, len: int):
            if uniform:
                return key, jnp.arange(0, len, round(len / out))

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

        if T_max_over_T is not None:
            sp_new: list[SystemParams] = []
            cv_new: list[CV] = []
            ti_new: list[TrajectoryInfo] = []
            weights_new: list[Array] = []
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
                if get_bias_list:
                    new_bias_list.append(bias_list[n])

                weights_new.append(weights[n])

            sp = sp_new
            ti = ti_new
            cv = cv_new
            weights = weights_new
            if get_bias_list:
                bias_list = new_bias_list

        print(f"len(sp) = {len(sp)}")

        for j, k in zip(sp, cv):
            if j.shape[0] != k.shape[0]:
                print(f"shapes do not match {j.shape=} {k.shape=}")

        # select the requested number of points and time lagged data

        out_sp: list[SystemParams] = []
        out_cv: list[CV] = []
        out_ti: list[TrajectoryInfo] = []

        if time_series:
            out_sp_t = []
            out_cv_t = []
            out_ti_t = []

        if get_bias_list:
            out_biases = []

        total = sum([max(a.shape[0] - lag_n, 0) for a in sp])

        if out == -1:
            out = total

        if out > total:
            print(f"not enough data, returning {total} data points instead of {out}")
            out = total

        if verbose:
            print(f"total data points {total}, selecting {out}")

        if split_data:
            frac = out / total

            for n, wi in enumerate(weights):
                if wi is None:
                    probs = None
                else:
                    if lag_n != 0:
                        wi = wi[:-lag_n]

                    probs = wi / jnp.sum(wi)

                ni = int(frac * (sp[n].shape[0] - lag_n))
                if ni == 0 and (sp[n].shape[0] - lag_n) > 0:
                    ni = 1

                key, indices = choose(
                    key,
                    probs,
                    out=ni,
                    len=sp[n].shape[0] - lag_n,
                )

                out_sp.append(sp[n][indices])
                out_cv.append(cv[n][indices])
                out_ti.append(ti[n][indices])

                if time_series:
                    out_sp_t.append(sp[n][indices + lag_n])
                    out_cv_t.append(cv[n][indices + lag_n])
                    out_ti_t.append(ti[n][indices + lag_n])

        else:
            if weights[0] is None:
                probs = None
            else:
                if lag_n == 0:
                    wi = jnp.hstack(weights)
                else:
                    wi = jnp.hstack([wi[:-lag_n] for wi in weights])

                probs = wi / jnp.sum(wi)

                print(f"probs = {probs.shape}  {out=} {total=} ")

            key, indices = choose(key, probs, out=int(out), len=total)

            # indices = jnp.sort(indices)

            count = 0

            sp_trimmed = []
            cv_trimmed = []
            ti_trimmed = []

            out_biases = []

            if time_series:
                sp_trimmed_t = []
                cv_trimmed_t = []
                ti_trimmed_t = []

            for n, (sp_n, cv_n, ti_n) in enumerate(zip(sp, cv, ti)):
                n_i = sp_n.shape[0] - lag_n

                index = indices[jnp.logical_and(count <= indices, indices < count + n_i)] - count

                if len(index) == 0:
                    count += n_i
                    continue

                sp_trimmed.append(sp_n[index])

                cv_trimmed.append(cv_n[index])
                ti_trimmed.append(ti_n[index])

                if time_series:
                    sp_trimmed_t.append(sp_n[index + lag_n])
                    cv_trimmed_t.append(cv_n[index + lag_n])
                    ti_trimmed_t.append(ti_n[index + lag_n])

                if get_bias_list:
                    out_biases.append(bias_list[n])

                count += n_i

            out_sp = sp_trimmed
            out_cv = cv_trimmed
            out_ti = ti_trimmed

            if time_series:
                out_sp_t = sp_trimmed_t
                out_cv_t = cv_trimmed_t
                out_ti_t = ti_trimmed_t

            if get_bias_list:
                bias_list = out_biases

            # bias = None

        out_nl = None

        if time_series:
            out_nl_t = None

        # return data
        if time_series:
            tau = None

            # consistency check
            for tii, ti_ti in zip(out_ti, out_ti_t):
                tii: TrajectoryInfo
                ti_ti: TrajectoryInfo

                dt = ti_ti.t - tii.t

                if tau is None:
                    tau = dt[0]

                if not jnp.allclose(dt, tau):
                    print(f"dt = {dt}, tau = {tau}  ")

                    print("WARNING:time steps are not equal")

            from molmod.units import femtosecond

            print(f"tau = {tau/femtosecond:.2f} fs, lag_time*timestep = {lag_n* sti.timestep/ femtosecond:.2f} fs")

            dlo = data_loader_output(
                sp=out_sp,
                nl=out_nl,
                cv=out_cv,
                ti=out_ti,
                sti=sti,
                collective_variable=colvar,
                time_series=time_series,
                bias=bias_list if get_bias_list else None,
                sp_t=out_sp_t,
                nl_t=out_nl_t,
                cv_t=out_cv_t,
                ti_t=out_ti_t,
                tau=tau,
                ground_bias=ground_bias,
            )
        else:
            dlo = data_loader_output(
                sp=out_sp,
                nl=out_nl,
                cv=out_cv,
                ti=out_ti,
                sti=sti,
                collective_variable=colvar,
                time_series=time_series,
                bias=bias_list if get_bias_list else None,
                ground_bias=ground_bias,
            )

        if new_r_cut is not None:
            if verbose:
                print("getting Neighbourr List")

            dlo = dlo.calc_neighbours(
                r_cut=new_r_cut,
                chunk_size=chunk_size,
                verbose=verbose,
                macro_chunk=macro_chunk,
            )

        if recalc_cv:
            if verbose:
                print("recalculating CV")
            dlo = dlo.recalc(
                chunk_size=chunk_size,
                verbose=verbose,
                macro_chunk=macro_chunk,
            )

        return dlo

    def iter_ase_atoms(
        self,
        r: int | None = None,
        c: int | None = None,
        num: int = 3,
        r_cut=None,
        minkowski_reduce=True,
        only_finished=False,
        ignore_invalid=False,
    ):
        from molmod import angstrom

        for round, trajejctory in self.iter(
            stop=r,
            c=c,
            num=num,
            ignore_invalid=ignore_invalid,
            only_finished=only_finished,
        ):
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

    def _trajectory_information(
        self,
        r: int,
        i: int,
        c: int | None = None,
    ) -> TrajectoryInformation:
        if c is None:
            c = self.cv

        ti = TrajectoryInfo.load(self.path(c=c, r=r, i=i) / "trajectory_info.h5")

        return TrajectoryInformation(
            ti=ti,
            valid=self.is_valid(c=c, r=r, i=i),
            finished=self.is_finished(c=c, r=r, i=i),
            name_bias=self._name_bias(c=c, r=r, i=i),
            round=r,
            num=i,
            folder=self.folder,
        )

    def _round_information(
        self,
        c: int | None = None,
        r: int | None = None,
    ) -> RoundInformation:
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        folder = self.path(c=c, r=r)
        stic = StaticMdInfo.load(folder / "static_trajectory_info.h5")

        mdi = self._i_vals(c, r)

        return RoundInformation(
            round=int(r),
            folder=self.folder,
            tic=stic,
            num_vals=np.array(mdi, dtype=np.int32),
            num=len(mdi),
            valid=self.is_valid(c=c, r=r),
            name_bias=self._name_bias(c=c, r=r),
            name_md=self._name_md(c=c, r=r),
        )

    ######################################
    #           Properties               #
    ######################################

    @property
    def T(self):
        return self._round_information().tic.T

    @property
    def P(self):
        return self._round_information().tic.P

    @property
    def round(self):
        return self.get_round()

    def get_round(self, c=None):
        if c is None:
            c = self.cv

        try:
            return self._r_vals(c)[-1]
        except Exception:
            return -1

    @property
    def cv(self):
        try:
            return self._c_vals()[-1]
        except Exception:
            return -1

    def n(self, c=None, r=None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)
        return self._round_information(r=r).num

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

    def is_valid(self, c=None, r=None, i=None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        return not (self.path(c=c, r=r, i=i) / "invalid").exists()

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

        bn = self._name_bias(c=c, r=r, i=i)

        return Bias.load(self.full_path(bn))

    def get_engine(self, c=None, r=None) -> MDEngine:
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        name = self._name_md(c=c, r=r)
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
        ignore_invalid=False,
        md_trajs: list[int] | None = None,
        cv_round: int | None = None,
        wait_for_plots=False,
        min_traj_length=None,
        recalc_cv=False,
        only_finished=True,
        profile=False,
    ):
        if cv_round is None:
            cv_round = self.cv

        round = self.get_round(c=cv_round)

        if isinstance(KEY, int):
            KEY = jax.random.PRNGKey(KEY)

        common_bias_name = self.full_path(self._name_bias(c=cv_round, r=round))
        common_md_name = self.full_path(self._name_md(c=cv_round, r=round))
        from parsl.dataflow.dflow import AppFuture

        tasks: list[tuple[int, AppFuture]] | None = None
        plot_tasks = []
        md_engine = MDEngine.load(common_md_name)

        sp0_provided = sp0 is not None

        if not sp0_provided:
            dlo_data = self.data_loader(
                num=4,
                out=10000,
                split_data=False,
                new_r_cut=None,
                ignore_invalid=ignore_invalid,
                md_trajs=md_trajs,
                cv_round=cv_round,
                min_traj_length=min_traj_length,
                recalc_cv=recalc_cv,
                only_finished=only_finished,
                get_bias_list=False,
                weight=True,
                T_scale=5,
            )

            sp_stack = SystemParams.stack(*dlo_data.sp)
        else:
            assert (
                sp0.shape[0] == len(biases)
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
                # reweigh data points according to new bias
                probs = jnp.hstack(dlo_data.weights(ground_bias=bias, sign=-1, correct_U=False))

                probs = probs / jnp.sum(probs)

                KEY, k = jax.random.split(KEY, 2)
                index = jax.random.choice(
                    a=probs.shape[0],
                    key=k,
                    p=probs,
                )

                spi = sp_stack[index]

            else:
                spi = sp0[i]
                # cvi = "unknown"
                spi = spi.unbatch()

            future = bash_app_python(
                Rounds.run_md,
                pass_files=True,
                executors=REFERENCE_LABELS,
                profile=profile,
            )(
                sp=spi,  # type: ignore
                inputs=[File(common_md_name), File(str(b_name))],
                outputs=[File(str(b_name_new)), File(str(traj_name))],
                steps=int(steps),
                execution_folder=path_name,
            )

            if plot:
                plot_file = path_name / "plot.pdf"

                plot_fut = bash_app_python(Rounds.plot_md_run, pass_files=True, executors=DEFAULT_LABELS)(
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

            except Exception as e:
                print(f"got exception {e} while collecting md {i}, round {round}, cv {cv_round}, continuing anyway")

                # raise e
                continue

            self.add_md(
                d=d,
                bias=self.rel_path(Path(future.outputs[0].filename)),
                i=i,
                c=cv_round,
            )

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

        common_md_name = self.full_path(self._name_md(c=cv_round, r=round))

        md_engine = MDEngine.load(common_md_name)

        from parsl.dataflow.dflow import AppFuture

        tasks: list[tuple[int, AppFuture]] | None = None
        plot_tasks = []

        ri = self._round_information(c=cv_round, r=round)

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

            future = bash_app_python(Rounds.run_md, pass_files=True, executors=REFERENCE_LABELS)(
                sp=None,  # type: ignore
                inputs=[File(common_md_name), File(str(b_name))],
                outputs=[File(str(b_name_new)), File(str(traj_name))],
                steps=int(steps),
                execution_folder=path_name,
            )

            if plot:
                plot_file = path_name / "plot.pdf"

                plot_fut = bash_app_python(Rounds.plot_md_run, pass_files=True, executors=DEFAULT_LABELS)(
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
                self.add_md(
                    d=d,
                    bias=self.rel_path(Path(future.outputs[0].filename)),
                    i=i,
                    c=cv_round,
                )
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

    ######################################
    #          CV transformations        #
    ######################################

    def update_CV(
        self,
        md: MDEngine,
        transformer: Transformer,
        dlo_kwargs=None,
        dlo: data_loader_output | None = None,
        chunk_size=None,
        plot=True,
        new_r_cut=None,
        save_samples=True,
        save_multiple_cvs=False,
        jac=jax.jacrev,
        cv_round_from=None,
        cv_round_to=None,
        test=False,
        max_bias=None,
        transform_bias=True,
        samples_per_bin=100,
        min_samples_per_bin=20,
        percentile=1e-1,
        use_executor=True,
        n_max=30,
        vmax=100 * kjmol,
        macro_chunk=10000,
    ):
        if cv_round_from is None:
            cv_round_from = self.cv

        if dlo_kwargs is None:
            dlo_kwargs = {}

        if "chunk_size" in dlo_kwargs:
            chunk_size = dlo_kwargs.pop("chunk_size")

        if "macro_chunk" in dlo_kwargs:
            macro_chunk = dlo_kwargs.pop("macro_chunk")

        if cv_round_from is not None:
            dlo_kwargs["cv_round"] = cv_round_from

        if chunk_size is not None:
            dlo_kwargs["chunk_size"] = chunk_size

        if cv_round_to is None:
            cv_round_to = self.cv + 1

        kw = dict(
            rounds=self,
            md=md,
            transformer=transformer,
            dlo_kwargs=dlo_kwargs,
            dlo=dlo,
            chunk_size=chunk_size,
            plot=plot,
            new_r_cut=new_r_cut,
            save_samples=save_samples,
            save_multiple_cvs=save_multiple_cvs,
            jac=jac,
            cv_round_from=cv_round_from,
            cv_round_to=cv_round_to,
            test=test,
            max_bias=max_bias,
            transform_bias=transform_bias,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            percentile=percentile,
            n_max=n_max,
            vmax=vmax,
            macro_chunk=macro_chunk,
        )

        if use_executor:
            return bash_app_python(Rounds._update_CV, executors=TRAINING_LABELS)(
                execution_folder=self.path(c=cv_round_to), **kw
            ).result()

        return Rounds._update_CV(**kw)

    @staticmethod
    def _update_CV(
        rounds: Rounds,
        md: MDEngine,
        transformer: Transformer,
        dlo_kwargs={},
        dlo: data_loader_output | None = None,
        chunk_size=None,
        macro_chunk=10000,
        plot=True,
        new_r_cut=None,
        save_samples=True,
        save_multiple_cvs=False,
        jac=jax.jacrev,
        cv_round_from=None,
        cv_round_to=None,
        test=False,
        max_bias=None,
        transform_bias=True,
        samples_per_bin=100,
        min_samples_per_bin=20,
        percentile=1e-1,
        n_max=30,
        vmax=100 * kjmol,
    ):
        if dlo is None:
            dlo = rounds.data_loader(**dlo_kwargs, macro_chunk=macro_chunk, verbose=True)

        cvs_new, new_collective_variable, new_bias = transformer.fit(
            dlo=dlo,
            chunk_size=chunk_size,
            plot=plot,
            plot_folder=rounds.path(c=cv_round_to),
            jac=jac,
            test=test,
            max_fes_bias=max_bias,
            transform_FES=transform_bias,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            percentile=percentile,
            n_max=n_max,
            cv_titles=[f"{cv_round_from}", f"{cv_round_to}"],
            vmax=vmax,
            macro_chunk=macro_chunk,
        )

        # update state
        return rounds.__update_CV(
            md=md,
            new_collective_variable=new_collective_variable,
            new_bias=new_bias,
            cv_round_from=cv_round_from,
            cvs_new=cvs_new,
            dlo=dlo,
            new_r_cut=new_r_cut,
            save_samples=save_samples,
            save_multiple_cvs=save_multiple_cvs,
        )

    def transform_CV(
        self,
        md: MDEngine,
        cv_trans: CvTrans,
        dlo_kwargs=None,
        dlo: data_loader_output | None = None,
        chunk_size=None,
        cv_round_from=None,
        cv_round_to=None,
        new_r_cut=None,
        plot=True,
        vmax=100 * kjmol,
        verbose=True,
    ):
        if cv_round_from is None:
            cv_round_from = self.cv

        if cv_round_to is None:
            cv_round_to = self.cv + 1

        if dlo is None:
            if dlo_kwargs is None:
                dlo_kwargs = {}

            dlo_kwargs["new_r_cut"] = None
            dlo_kwargs["cv_round"] = cv_round_from
            dlo_kwargs["weight"] = False

            dlo = self.data_loader(**dlo_kwargs, verbose=True)

        bias = dlo.transform_FES(trans=cv_trans, max_bias=vmax)
        collective_variable = bias.collective_variable

        x, _ = dlo.apply_cv_trans(cv_trans=cv_trans, chunk_size=chunk_size, verbose=verbose)

        x = CV.stack(*x)

        if plot:
            Transformer.plot_app(
                collective_variables=[md.bias.collective_variable, collective_variable],
                cv_data=[CV.stack(*dlo.cv), x],
                duplicate_cv_data=True,
                T=dlo.sti.T,
                plot_FES=True,
                weight=dlo.weights(),
                name=self.path(c=cv_round_to) / "transformed_FES.png",
                cv_titles=[cv_round_from, cv_round_to],
                data_titles=[cv_round_from, cv_round_to],
                vmax=vmax,
            )

        md = self.__update_CV(
            md=md,
            new_collective_variable=collective_variable,
            new_bias=bias,
            cv_round_from=cv_round_from,
            cvs_new=x,
            dlo=dlo,
            new_r_cut=new_r_cut,
        )

        return md

    def __update_CV(
        rounds: Rounds,
        md: MDEngine,
        new_collective_variable: CollectiveVariable,
        new_bias: Bias,
        cv_round_from: int,
        cvs_new: list[CV],
        dlo: data_loader_output,
        new_r_cut=None,
        save_samples=True,
        save_multiple_cvs=False,
    ):
        if new_r_cut is None:
            new_r_cut = dlo.sti.r_cut

        rounds.add_cv_from_cv(new_collective_variable, c=cv_round_from + 1)
        md.bias = new_bias
        md.static_trajectory_info.r_cut = new_r_cut

        rounds.add_round_from_md(md)

        if save_samples:
            first = True

            if save_multiple_cvs:
                for dlo_i, cv_new_i in zip(iter(dlo), CV.unstack(cvs_new)):
                    if not first:
                        md.bias = NoneBias.create(new_collective_variable)
                        rounds.add_cv_from_cv(new_collective_variable)
                        md.static_trajectory_info.r_cut = new_r_cut
                        rounds.add_round_from_md(md)

                    rounds._copy_from_previous_round(
                        dlo=dlo_i,
                        new_cvs=[cv_new_i],
                        cv_round=cv_round_from,
                    )
                    rounds.add_round_from_md(md)

                    first = False

            else:
                rounds._copy_from_previous_round(
                    dlo=dlo,
                    new_cvs=CV.unstack(cvs_new),
                    cv_round=cv_round_from,
                )
                rounds.add_round_from_md(md)
        return md

    def _copy_from_previous_round(
        self,
        dlo: data_loader_output,
        new_cvs: list[CV],
        invalidate: bool = False,
        cv_round: int | None = None,
    ):
        if cv_round is None:
            cv_round = self.cv - 1

        for i in range(len(dlo.cv)):
            round_path = self.path(c=self.cv, r=0, i=i)
            round_path.mkdir(parents=True, exist_ok=True)

            # bias.save(round_path / "bias.json")
            traj_info = dlo.ti[i]

            new_traj_info = TrajectoryInfo.create(
                positions=traj_info.positions,
                cell=traj_info.cell,
                charges=traj_info.charges,
                e_pot=traj_info.e_pot,
                e_pot_gpos=traj_info.e_pot_gpos,
                e_pot_vtens=traj_info.e_pot_vtens,
                e_bias=None,
                e_bias_gpos=None,
                e_bias_vtens=None,
                cv=new_cvs[i].cv,
                T=traj_info._T,
                P=traj_info._P,
                err=traj_info._err,
                t=traj_info._t,
                capacity=traj_info._capacity,
                size=traj_info._size,
            )

            self.add_md(
                i=i,
                d=new_traj_info,
                bias=None,
            )

            self.finish_data(c=self.cv, r=0, i=i)

            if invalidate:
                self.invalidate_data(c=self.cv, r=self.round, i=i)


@dataclass(repr=False)
class data_loader_output:
    sp: list[SystemParams]
    nl: list[NeighbourList] | None
    cv: list[CV]
    sti: StaticMdInfo
    ti: list[TrajectoryInfo]
    collective_variable: CollectiveVariable
    sp_t: list[SystemParams] | None = None
    nl_t: list[NeighbourList] | None = None
    cv_t: list[CV] | None = None
    ti_t: list[TrajectoryInfo] | None = None
    time_series: bool = False
    tau: float | None = None
    bias: list[Bias] | None = None
    ground_bias: Bias | None = None

    def __iter__(self):
        for i in range(len(self.sp)):
            d = dict(
                sti=self.sti,
                time_series=self.time_series,
                tau=self.tau,
                ground_bias=self.ground_bias,
                collective_variable=self.collective_variable,
            )

            if self.sp is not None:
                d["sp"] = [self.sp[i]]
            if self.nl is not None:
                d["nl"] = [self.nl[i]]
            if self.cv is not None:
                d["cv"] = [self.cv[i]]
            if self.ti is not None:
                d["ti"] = [self.ti[i]]

            if self.time_series:
                if self.sp_t is not None:
                    d["sp_t"] = [self.sp_t[i]]
                if self.nl_t is not None:
                    d["nl_t"] = [self.nl_t[i]]
                if self.cv_t is not None:
                    d["cv_t"] = [self.cv_t[i]]
                if self.ti_t is not None:
                    d["ti_t"] = [self.ti_t[i]]

            if self.bias is not None:
                d["bias"] = [self.bias[i]]

            yield data_loader_output(**d)

    def __add__(self, other):
        assert isinstance(other, data_loader_output)

        assert self.time_series == other.time_series
        assert (
            self.collective_variable == other.collective_variable
        ), "dlo cannot be added because the collective variables are different"
        if self.tau is not None:
            assert self.tau == other.tau
        else:
            assert other.tau is None

        kwargs = dict(
            sti=self.sti,
            time_series=self.time_series,
            tau=self.tau,
            ground_bias=None,
            collective_variable=self.collective_variable,
        )

        kwargs["sp"] = [*self.sp, *other.sp]
        kwargs["nl"] = [*self.nl, *other.nl] if self.nl is not None else None
        kwargs["cv"] = [*self.cv, *other.cv]
        kwargs["ti"] = [*self.ti, *other.ti]

        if self.time_series:
            kwargs["sp_t"] = [*self.sp_t, *other.sp_t] if self.sp_t is not None else None
            kwargs["nl_t"] = [*self.nl_t, *other.nl_t] if self.nl_t is not None else None
            kwargs["cv_t"] = [*self.cv_t, *other.cv_t] if self.cv_t is not None else None
            kwargs["ti_t"] = [*self.ti_t, *other.ti]

        if self.bias is not None:
            assert other.bias is not None
            kwargs["bias"] = [*self.bias, *other.bias]

        if self.ground_bias is not None and other.ground_bias is not None:
            if self.ground_bias == other.ground_bias:
                kwargs["ground_bias"] = self.ground_bias

            else:
                print("ground_bias not the same, omitting")

        return data_loader_output(**kwargs)

    @staticmethod
    def _histogram(
        collective_variable: CollectiveVariable,
        n_grid=40,
        grid_bounds=None,
        chunk_size=None,
        chunk_size_mid=1,
    ):
        _, _, cv_mid = collective_variable.metric.grid(n=n_grid, bounds=grid_bounds)

        def closest(data, mid):
            @vmap
            def _closest(data):
                return jnp.argmin(jnp.sum((mid - data) ** 2, axis=1))

            if chunk_size is not None:
                _closest = chunk_map(_closest, chunk_size)

            return _closest(data)

        def get_histo(data_nums, nums, weights=None):
            @vmap
            def _f(num):
                @vmap
                def __f(a, b=None):
                    if b is not None:
                        return jnp.sum((a == num) * b)
                    return jnp.sum(a == num)

                if chunk_size is not None:
                    __f = chunk_map(__f, chunk_size)

                return jnp.sum(__f(data_nums, weights))

            if chunk_size_mid is not None:
                _f = chunk_map(_f, chunk_size_mid)

            return _f(nums)

        nums = closest(cv_mid.cv, cv_mid.cv)

        return cv_mid, nums, closest, get_histo

    def weights(
        self,
        correct_U=False,
        samples_per_bin=30,
        n_max=50,
        ground_bias=None,
        use_ground_bias=True,
        sign=1,
        time_series=False,
        hist_eps=1e-15,
        T_scale=1,
        chunk_size=None,
    ) -> list[jax.Array]:
        # TODO:https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00867

        data = CV.stack(*self.cv)
        beta = 1 / (self.sti.T * boltzmann)

        if ground_bias is None:
            ground_bias = self.ground_bias

        if correct_U:
            energies = []

            for ti_i in self.ti:
                energies.append(ti_i.e_pot)

            energies = jnp.hstack(energies)
            energies -= jnp.min(energies)

            w_u = jnp.exp(-beta * energies)
            w_u /= jnp.mean(w_u)

        else:
            w_u = jnp.ones((data.shape[0],))

        # norm based on FES

        n = CvMetric.get_n(samples_per_bin=samples_per_bin, samples=data.shape[0], n_dims=data.shape[1])

        if n > n_max:
            n = n_max
            # print(f"reducing {n=} to {n_max=}")

        grid_bounds, mask = CvMetric.bounds_from_cv(data, margin=0.1, chunk_size=chunk_size)

        cv_mid, nums, closest, get_histo = data_loader_output._histogram(
            collective_variable=self.collective_variable,
            n_grid=n,
            grid_bounds=grid_bounds,
            chunk_size=chunk_size,
        )

        grid_nums = closest(data.cv, cv_mid.cv)
        hist = get_histo(grid_nums, nums, w_u)

        if use_ground_bias:
            p_grid = jnp.exp(sign * beta * ground_bias.compute_from_cv(cvs=cv_mid, chunk_size=chunk_size)[0] / T_scale)
        else:
            p_grid = jnp.ones((cv_mid.cv.shape[0],))

        p_grid = jnp.where(hist <= hist_eps, 0, p_grid / hist)

        p = w_u * p_grid[grid_nums]
        p /= jnp.sum(p)

        weights = [d.cv.reshape((-1,)) for d in data.replace(cv=jnp.expand_dims(p, 1)).unstack()]

        return weights

    @staticmethod
    def _transform(cv, nl, _, pi, add_1, q):
        cv_v = (cv.cv - pi) @ q

        if add_1:
            cv_v = jnp.hstack([cv_v, jnp.array([1])])

        return cv.replace(cv=cv_v)

    @staticmethod
    def _whiten(
        cv_0: CV,
        cv_tau=None,
        symmetric=False,
        w=None,
        add_1=False,
        eps=1e-10,
        max_features=2000,
    ):
        if w is None:
            w = jnp.ones((cv_0.shape[0],))
            w /= jnp.sum(w)

        if symmetric:
            assert cv_tau is not None

        x = cv_0.cv

        if not symmetric:
            pi = jnp.einsum("ni,n->i", x, w)
            C0 = jnp.einsum("ni,n,nj->ij", x, w, x) - jnp.einsum("i,j->ij", pi, pi)
        else:
            x_t = cv_tau.cv
            pi = jnp.einsum("ni,n->i", 0.5 * (x + x_t), w)
            C0 = 0.5 * (jnp.einsum("ni,n,nj->ij", x, w, x) + jnp.einsum("ni,n,nj->ij", x_t, w, x_t)) - jnp.einsum(
                "i,j->ij",
                pi,
                pi,
            )

        # TODO: tends to hang here due to parsl forking. see https://github.com/google/jax/issues/1805#issuecomment-561244991 and https://github.com/Parsl/parsl/issues/2343
        # l, q = jnp.linalg.eigh(C0)
        l, q = scipy.linalg.eigh(C0, subset_by_value=(eps, np.inf))
        l, q = jnp.array(l), jnp.array(q)

        mask = l >= eps
        print(f"{jnp.sum(mask)}/{mask.shape[0]} eigenvalues larger than {eps=}")

        if sum(mask) == 0:
            print(f"all eigenvalues are smaller than {eps=}, keeping features")
            print(f"{l=}")
            mask = jnp.ones_like(mask, dtype=bool)

        q = q[:, mask] @ jnp.diag(l[mask] ** (-1 / 2))

        if max_features is not None:
            if jnp.sum(mask) > max_features:
                print(f"reducing to {max_features=}")
                q = q[:, -max_features:]

        transform_maf = CvTrans.from_cv_function(
            data_loader_output._transform,
            static_argnames=["add_1"],
            add_1=add_1,
            q=q,
            pi=pi,
        )

        if cv_tau is not None:
            cv_0, _, _ = transform_maf.compute_cv_trans(cv_0)
            cv_tau = transform_maf.compute_cv_trans(cv_tau)[0]
            return cv_0, cv_tau, transform_maf, pi, q

        cv_0 = transform_maf.compute_cv_trans(cv_0)[0]

        return cv_0, transform_maf, pi, q

    @staticmethod
    def _get_covariance(cv_0: CV, cv_1: CV = None, w=None, calc_pi=False, symmetric=True):
        if w is None:
            w = jnp.ones((cv_0.shape[0],))
            w /= jnp.sum(w)

        X = cv_0.cv
        Y = cv_1.cv

        if symmetric:
            pi = None
            if calc_pi:
                pi = jnp.einsum("ni,n->i", 0.5 * (X + Y), w)

                X = X - pi
                Y = Y - pi

            C0 = 0.5 * (jnp.einsum("ni,n,nj->ij", X, w, X) + jnp.einsum("ni,n,nj->ij", Y, w, Y))
            C1 = 0.5 * (jnp.einsum("ni,n,nj->ij", X, w, Y) + jnp.einsum("ni,n,nj->ij", Y, w, X))

            return C0, C1, pi

        pi_0 = None
        pi_1 = None

        if calc_pi:
            pi_0 = jnp.einsum("ni,n->i", X, w)
            pi_1 = jnp.einsum("ni,n->i", Y, w)

            X = X - pi_0
            Y = Y - pi_1

        C00 = jnp.einsum("ni,n,nj->ij", X, w, X)
        C01 = jnp.einsum("ni,n,nj->ij", X, w, Y)
        C11 = jnp.einsum("ni,n,nj->ij", Y, w, Y)

        return C00, C01, C11, pi_0, pi_1

    def koopman_weights(
        self,
        cv_0: CV = None,
        cv_tau: CV = None,
        w: list[jax.Array] | None = None,
        eps=1e-10,
        max_features=2000,
        add_1=True,
        test=False,
    ):
        # see  https://publications.imp.fu-berlin.de/1997/1/17_JCP_WuEtAl_KoopmanReweighting.pdf
        # https://pubs.aip.org/aip/jcp/article-abstract/146/15/154104/152394/Variational-Koopman-models-Slow-collective?redirectedFrom=fulltext

        assert self.time_series

        if cv_0 is None:
            cv_0 = self.cv
        cv_0 = CV.stack(*cv_0)

        if cv_tau is None:
            cv_tau = self.cv_t

        if cv_tau is not None:
            cv_tau = CV.stack(*cv_tau)

        if w is None:
            w = jnp.ones((cv_0.shape[0],))
        else:
            w = jnp.hstack(w)

        w /= jnp.sum(w)

        cv0_white, cv_tau_white, transform_maf, pi, q = data_loader_output._whiten(
            cv_0=cv_0,
            cv_tau=cv_tau,
            w=w,
            add_1=add_1,
            eps=eps,
            max_features=max_features,
        )
        # cv_tau_white = transform_maf.compute_cv_trans(cv_tau)[0]

        C_00, C_01, C_11, pi_0, pi_1 = data_loader_output._get_covariance(
            cv_0=cv0_white,
            cv_1=cv_tau_white,
            w=w,
            symmetric=False,
            calc_pi=False,
        )

        # eigenvalue of K.T
        eigval, u = jnp.linalg.eig(a=C_01.T)

        p = jnp.abs(eigval) > 1 + 1e-10
        in_bounds = jnp.all(p)

        if not in_bounds:
            print(f" {jnp.sum(p)}/ {p.shape[0]} eigvals too large {eigval[p]=}.")

        if add_1:
            idx = jnp.argsort(jnp.abs(eigval - 1))[0]
            assert jnp.abs(eigval[idx] - 1) < 1e-8, f"{eigval[idx]=}, but should be 1"
        else:
            idx = jnp.argmax(jnp.real(eigval))
            print(f"largest eigval = {eigval[idx]}")
            assert jnp.allclose(jnp.imag(eigval[idx]), 0)

        print(f"idx = {idx}, {eigval[idx]=}, {u[:, idx]=}")

        w_k = jnp.einsum("ni,n,i->n", cv0_white.cv, w, u[:, idx])

        if not (w_k > 0).all():
            print(
                f"  { jnp.sum( w_k<= 0 ) }/{w_k.shape[0]} negative weights,  mean {  jnp.mean( w_k[w_k<= 0]) }, min { jnp.min( w_k[w_k<= 0] ) }. Setting to zero and reestimating",
            )

        w_k /= jnp.sum(w_k)

        return w_k

    def koopman_model(
        self,
        cv_0: list[CV] = None,
        cv_tau: list[CV] = None,
        method="tica",
        koopman_weight=True,
        symmetric_tica=True,
        w: list[jax.Array] | None = None,
        eps=1e-10,
        max_features=2000,
        out_dim=None,
        add_1=True,
    ):
        # TODO: https://www.mdpi.com/2079-3197/6/1/22
        assert method in ["tica", "tcca"]

        if cv_0 is None:
            cv_0 = self.cv
        cv_0 = CV.stack(*cv_0)

        if cv_tau is None:
            cv_tau = self.cv_t

        if cv_tau is not None:
            cv_tau = CV.stack(*cv_tau)

        if w is not None:
            w = jnp.hstack(w)

        print(f"{cv_0.shape=}, {cv_tau.shape=}")

        if koopman_weight:
            w = self.koopman_weights(cv_0=cv_0, cv_tau=cv_tau, w=w, eps=eps)

        if method == "tica":
            (
                cv0_white,
                cv_tau_white,
                transform_maf,
                pi,
                q,
            ) = data_loader_output._whiten(
                cv_0=cv_0,
                cv_tau=cv_tau,
                symmetric=symmetric_tica,
                w=w,
                add_1=add_1,
                eps=eps,
                max_features=max_features,
            )

            _, C_01, _ = data_loader_output._get_covariance(
                cv_0=cv0_white,
                cv_1=cv_tau_white,
                w=w,
                calc_pi=False,
                symmetric=symmetric_tica,
            )

            if symmetric_tica:
                k, u = jnp.linalg.eigh(C_01)
            else:
                k, u = jnp.linalg.eig(C_01)

            # reverse order
            k = k[::-1]
            u = u[:, ::-1]

            if add_1:
                assert jnp.allclose(k[0], 1, atol=1e-6), f"{k[0]=}, but should be 1"
                M = q @ (u[:-1,][:, 1:])
                k = k[1:]
            else:
                M = q @ u

            if out_dim is not None:
                M = M[:, :out_dim]

            f = CvTrans.from_cv_function(
                data_loader_output._transform,
                static_argnames=["add_1"],
                add_1=False,
                q=M,
                pi=pi,
            )

            cv_0_out = f.compute_cv_trans(cv_0)[0]
            cv_tau_out = f.compute_cv_trans(cv_tau)[0]

            return k, f, pi, M, cv_0_out, cv_tau_out

        if method == "tcca":
            (
                cv0_white,
                transform_maf_0,
                pi_0,
                q_0,
            ) = data_loader_output._whiten(
                cv_0=cv_0,
                w=w,
                add_1=add_1,
                eps=eps,
                max_features=max_features,
            )

            (
                cv_tau_white,
                transform_maf_1,
                pi_1,
                q_1,
            ) = data_loader_output._whiten(
                cv_0=cv_tau,
                w=w,
                add_1=add_1,
                eps=eps,
                max_features=max_features,
            )

            C_00, C_01, C_11, _, _ = data_loader_output._get_covariance(
                cv_0=cv0_white,
                cv_1=cv_tau_white,
                w=w,
                symmetric=False,
                calc_pi=False,
            )

            K = C_01

            U, s, Vh = jnp.linalg.svd(K)

            if add_1:
                U = U[:-1, :][:, 1:]
                Vh = Vh[:, :-1][1:, :]
                s = s[1:]

            msk = s >= 1
            if jnp.any(msk):
                print(f"{jnp.sum(msk)}/{msk.shape[0]} singular values larger than 1. Removing them.")

                U = U[:, ~msk]
                Vh = Vh[~msk, :]
                s = s[~msk]

            q_0 = q_0 @ U
            q_1 = q_1 @ jnp.transpose(Vh)
            if out_dim is not None:
                q_0 = q_0[:, :out_dim]
                q_1 = q_1[:, :out_dim]

            f = CvTrans.from_cv_function(
                data_loader_output._transform,
                static_argnames=["add_1"],
                add_1=False,
                q=q_0,
                pi=pi_0,
            )

            g = CvTrans.from_cv_function(
                data_loader_output._transform,
                static_argnames=["add_1"],
                add_1=False,
                q=q_1,
                pi=pi_1,
            )

            cv_0_out = f.compute_cv_trans(cv_0)[0]
            cv_tau_out = f.compute_cv_trans(cv_tau)[0]

            return s, f, g, pi_0, q_0, pi_1, q_1, cv_0_out, cv_tau_out

    def filter_nans(self, x: list[CV] | None = None, x_t: list[CV] | None = None) -> tuple[data_loader_output, CV, CV]:
        if x is None:
            x = self.cv

        x = CV.stack(*x)

        if x_t is None:
            x_t = self.cv_t

        if self.time_series:
            x_t = CV.stack(*x_t)

        nanmask = jnp.logical_and(jax.vmap(jnp.any)(jnp.isnan(x.cv)), jax.vmap(jnp.any)(jnp.isinf(x.cv)))

        if x_t is not None:
            nanmask = jnp.logical_or(
                nanmask, jnp.logical_and(jax.vmap(jnp.any)(jnp.isnan(x_t.cv)), jax.vmap(jnp.any)(jnp.isinf(x_t.cv)))
            )

        mask = None

        if not jnp.any(nanmask):
            return self, x.unstack(), x_t.unstack() if x_t is not None else None

        print(f"found {jnp.sum(nanmask)}/{len(nanmask)} nans or infs in the cv data, removing")
        x = x[~nanmask]
        if x_t is not None:
            x_t = x_t[~nanmask]

        mask = [m.cv for m in CV(cv=~nanmask, _stack_dims=x.stack_dims).unstack()]

        dlo = data_loader_output(
            sp=[sp_i[m_i] for sp_i, m_i in zip(self.sp, mask)],
            nl=[nl_i[m_i] for nl_i, m_i in zip(self.nl, mask)],
            sp_t=[sp_i[m_i] for sp_i, m_i in zip(self.sp_t, mask)] if self.sp_t is not None else None,
            nl_t=[nl_i[m_i] for nl_i, m_i in zip(self.nl_t, mask)] if self.nl_t is not None else None,
            cv=[cv_i[m_i] for cv_i, m_i in zip(self.cv, mask)],
            cv_t=[cv_i[m_i] for cv_i, m_i in zip(self.cv_t, mask)] if self.cv_t is not None else None,
            time_series=self.time_series,
            tau=self.tau,
            bias=self.bias,
            ti=[ti_i[m_i] for ti_i, m_i in zip(self.ti, mask)] if self.ti is not None else None,
            ti_t=[ti_t_i[m_i] for ti_t_i, m_i in zip(self.ti_t, mask)] if self.ti_t is not None else None,
            collective_variable=self.collective_variable,
            sti=self.sti,
            ground_bias=self.ground_bias,
        )

        return dlo, x.unstack(), x_t.unstack() if x_t is not None else None

    def apply_cv_trans(
        self,
        cv_trans: CvTrans,
        x: list[CV] | None = None,
        x_t: list[CV] | None = None,
        chunk_size=None,
        macro_chunk=10000,
        pmap=True,
        verbose=False,
    ) -> tuple[list[CV], list[CV] | None]:
        if x is None:
            x = self.cv

        if x_t is None:
            x_t = self.cv_t

        def f(x, nl):
            return cv_trans.compute_cv_trans(x, nl, chunk_size=chunk_size)[0]

        if pmap:
            f = padded_pmap(f)

        if self.time_series:
            y = [*x, *x_t]
            nl = [*self.nl, *self.nl_t] if self.nl is not None else None
        else:
            y = [*x]
            nl = [*self.nl] if self.nl is not None else None

        z = data_loader_output._macro_chunk(
            f,
            CV.stack,
            y,
            nl,
            macro_chunk=macro_chunk,
            verbose=verbose,
        )

        if self.time_series:
            z, z_t = z[0 : len(self.cv)], z[len(self.cv) :]
        else:
            z_t = None

        return z, z_t

    def apply_cv_flow(
        self,
        flow: CvFlow,
        x: list[SystemParams] | None = None,
        x_t: list[SystemParams] | None = None,
        chunk_size=None,
        macro_chunk=10000,
        pmap=True,
        verbose=False,
    ) -> tuple[list[CV], list[CV] | None]:
        if x is None:
            x = self.sp

        if x_t is None:
            x_t = self.sp_t

        def f(x, nl):
            return flow.compute_cv_flow(x, nl, chunk_size=chunk_size)[0]

        if pmap:
            f = padded_pmap(f)

        # f: Callable[[SystemParams, NeighbourList], tuple[CV, Any]]

        if verbose:
            print(f"apply_cv_func: stacking {len(x)} {len(x_t) if x_t is not None else 0} ")

        if self.time_series:
            y = [*x, *x_t]
            nl = [*self.nl, *self.nl_t] if self.nl is not None else None
        else:
            y = [*x]
            nl = [*self.nl] if self.nl is not None else None

        z = data_loader_output._macro_chunk(
            f,
            SystemParams.stack,
            y,
            nl,
            macro_chunk=macro_chunk,
            verbose=verbose,
        )

        if self.time_series:
            z, z_t = z[0 : len(self.sp)], z[len(self.sp) :]
        else:
            z_t = None

        return z, z_t

    @staticmethod
    def _macro_chunk(
        f: Callable[[SystemParams | CV, NeighbourList], CV],
        op: SystemParams.stack | CV.stack,
        y: list[SystemParams | CV],
        nl: list[NeighbourList] | None,
        macro_chunk=10000,
        verbose=False,
    ):
        # helper method to apply a function to list of SystemParams or CVs, chunked in groups of macro_chunk

        if macro_chunk is None:
            stack_dims = [nli.shape[0] for nli in nl]

            z = f(SystemParams.stack(*y), NeighbourList.stack(*nl))

            z = z.replace(_stack_dims=stack_dims).unstack()

        else:
            n = 0

            z = []

            y_chunk = []
            nl_chunk = [] if nl is not None else None

            tot_chunk = 0
            stack_dims_chunk = []

            while n < len(y):
                s = y[n].shape[0]

                y_chunk.append(y[n])
                nl_chunk.append(nl[n]) if nl is not None else None

                tot_chunk += s
                stack_dims_chunk.append(s)

                if tot_chunk > macro_chunk or n == len(y) - 1:
                    if verbose:
                        print(f"apply_cv_func: chunk {n}/{len(y)}, {tot_chunk=}")

                    z_chunk = f(
                        op(*y_chunk),
                        NeighbourList.stack(*nl_chunk) if nl is not None else None,
                    )

                    z_chunk = z_chunk.replace(_stack_dims=stack_dims_chunk).unstack()

                    z.extend(z_chunk)

                    y_chunk = []
                    nl_chunk = [] if nl is not None else None
                    stack_dims_chunk = []

                    tot_chunk = 0

                n += 1

        print(f"{len(z)=}")

        return z

    @staticmethod
    def _get_fes_bias_from_weights(
        T,
        weights: list[jax.Array],
        collective_variable: CollectiveVariable,
        cv: list[CV],
        samples_per_bin=10,
        min_samples_per_bin: int | None = 3,
        n_max=60,
        n_grid=None,
        max_bias=None,
    ) -> RbfBias:
        beta = 1 / (T * boltzmann)

        cv = CV.stack(*cv)
        p = jnp.hstack(weights)

        if n_grid is None:
            n_grid = CvMetric.get_n(samples_per_bin=samples_per_bin, samples=cv.shape[0], n_dims=cv.shape[1])

        if n_grid > n_max:
            # print(f"reducing n_grid from {n_grid} to {n_max}")
            n_grid = n_max

        if n_grid <= 1:
            return RbfBias.create(
                cvs=collective_variable,
                cv=cv,
                kernel="thin_plate_spline",
                vals=jnp.zeros((cv.shape[0],)),
            )

        grid_bounds, mask = CvMetric.bounds_from_cv(cv, margin=0.1)

        # # do not update periodic bounds
        grid_bounds = jnp.where(
            collective_variable.metric.periodicities,
            collective_variable.metric.bounding_box,
            grid_bounds,
        )

        cv_mid, nums, closest, get_histo = data_loader_output._histogram(
            n_grid=n_grid, grid_bounds=grid_bounds, collective_variable=collective_variable
        )
        grid_nums = closest(cv.cv, cv_mid.cv)

        @partial(vmap, in_axes=(None, 0, None))
        def fes_weight(data_num, num, p):
            b = data_num == num
            p = jnp.where(b, p, 0)
            return jnp.sum(b), jnp.sum(p)

        num_grid, p_grid = fes_weight(
            grid_nums,
            nums,
            p,
        )

        if min_samples_per_bin is not None:
            mask = num_grid <= min_samples_per_bin
            p_grid = p_grid.at[mask].set(0)

        mask = p_grid != 0

        p_grid = p_grid[mask]

        fes_grid = -jnp.log(p_grid) / beta
        fes_grid -= jnp.min(fes_grid)

        bias = RbfBias.create(
            cvs=collective_variable,
            cv=cv_mid[mask],
            kernel="thin_plate_spline",
            vals=-fes_grid,
        )

        if max_bias is None:
            max_bias = jnp.max(fes_grid)

        return BiasModify.create(
            bias=bias,
            fun=_clip,
            kwargs={"a_min": -max_bias, "a_max": 0.0},
        )

    def get_transformed_fes(
        self,
        new_cv: list[CV],
        new_colvar: CollectiveVariable,
        samples_per_bin=50,
        min_samples_per_bin: int = 5,
        chunk_size=1,
        smoothing=0.0,
        max_bias=None,
        pmap=True,
        n_grid_old=50,
        n_grid_new=30,
    ) -> RbfBias:
        old_cv = CV.stack(*self.cv)
        new_cv = CV.stack(*new_cv)

        # get bins for new CV
        grid_bounds_new, _ = CvMetric.bounds_from_cv(new_cv, margin=0.1)
        cv_mid_new, nums_new, closest_new, get_histo_new = data_loader_output._histogram(
            n_grid=n_grid_new,
            grid_bounds=grid_bounds_new,
            collective_variable=new_colvar,
        )
        grid_nums_new = closest_new(new_cv.cv, cv_mid_new.cv)

        # get bins for old CV
        grid_bounds_old, _ = CvMetric.bounds_from_cv(old_cv, margin=0.1)
        cv_mid_old, nums_old, closest_old, get_histo_old = data_loader_output._histogram(
            n_grid=n_grid_old,
            grid_bounds=grid_bounds_old,
            collective_variable=self.collective_variable,
        )
        grid_nums_old = closest_old(old_cv.cv, cv_mid_old.cv)

        # get old FES weights
        beta = 1 / (self.sti.T * boltzmann)
        p_grid_old = jnp.exp(
            beta
            * self.ground_bias.compute_from_cv(
                cvs=cv_mid_old,
                chunk_size=chunk_size,
            )[0]
        )
        p_grid_old /= jnp.sum(p_grid_old)

        @partial(vmap, in_axes=(None, 0))
        def f(b0, old_num):
            b = jnp.logical_and(grid_nums_old == old_num, b0)
            return jnp.sum(b)

        def prob(new_num, grid_nums_new, nums_old, p_grid_old):
            b = grid_nums_new == new_num
            bins = f(b, nums_old)

            bins_sum = jnp.sum(bins)
            p_bins = bins * jnp.where(bins_sum == 0, 0, 1 / bins_sum)

            return bins_sum, jnp.sum(p_bins * p_grid_old)

        # takes a lot of memory somehow

        prob = Partial(prob, grid_nums_new=grid_nums_new, nums_old=nums_old, p_grid_old=p_grid_old)
        prob = vmap(prob)

        if chunk_size is not None:
            prob = chunk_map(prob, chunk_size=chunk_size)

        if pmap:
            prob = padded_pmap(prob)

        num_grid_new, p_grid_new = prob(nums_new)

        # mask = num_grid_new >= min_samples_per_bin

        # print(f"{jnp.sum(mask)}/{mask.shape[0]} bins with samples")

        # p_grid_new = p_grid_new  # [mask]

        fes_grid = -jnp.log(p_grid_new) / beta
        fes_grid -= jnp.min(fes_grid)

        bias = RbfBias.create(
            cvs=new_colvar,
            cv=cv_mid_new,  # [mask],
            kernel="thin_plate_spline",
            vals=-fes_grid,
            smoothing=smoothing,
        )

        if max_bias is None:
            max_bias = jnp.max(fes_grid)

        bias = BiasModify.create(
            bias=bias,
            fun=_clip,
            kwargs={"a_min": -max_bias, "a_max": 0.0},
        )

        return bias

    def transform_FES(
        self,
        trans: CvTrans,
        T: float | None = None,
        max_bias=100 * kjmol,
        n_grid=25,
    ):
        if T is None:
            T = self.sti.T

        _, cv_grid, _ = self.collective_variable.metric.grid(n=n_grid)
        new_cv_grid, _, log_det = trans.compute_cv_trans(cv_grid, log_Jf=True)

        FES_bias_vals, _ = self.ground_bias.compute_from_cv(cv_grid)

        new_FES_bias_vals = FES_bias_vals + T * boltzmann * log_det
        new_FES_bias_vals -= jnp.max(new_FES_bias_vals)

        weight = jnp.exp(new_FES_bias_vals / (T * boltzmann))
        weight /= jnp.sum(weight)

        bounds, _ = self.collective_variable.metric.bounds_from_cv(
            new_cv_grid,
            weights=weight,
            percentile=1e-5,
            # margin=0.1,
        )

        print(f"{bounds=}")

        new_collective_variable = CollectiveVariable(
            f=self.collective_variable.f * trans,
            metric=CvMetric.create(
                bounding_box=bounds,
                periodicities=self.collective_variable.metric.periodicities,
            ),
        )

        new_FES_bias = RbfBias.create(
            cvs=new_collective_variable,
            vals=new_FES_bias_vals,
            cv=new_cv_grid,
            kernel="thin_plate_spline",
        )

        if max_bias is None:
            max_bias = -jnp.min(new_FES_bias_vals)

        return BiasModify.create(
            bias=new_FES_bias,
            fun=_clip,
            kwargs={"a_min": -max_bias, "a_max": 0.0},
        )

    def recalc(self, chunk_size=None, macro_chunk=10000, pmap=True, verbose=False) -> data_loader_output:
        x, x_t = self.apply_cv_flow(
            self.collective_variable.f,
            chunk_size=chunk_size,
            pmap=pmap,
            macro_chunk=macro_chunk,
            verbose=verbose,
        )

        return data_loader_output(
            sp=self.sp,
            nl=self.nl,
            cv=x,
            sti=self.sti,
            ti=self.ti,
            collective_variable=self.collective_variable,
            sp_t=self.sp_t,
            nl_t=self.nl_t,
            cv_t=x_t,
            time_series=self.time_series,
            tau=self.tau,
            bias=self.bias,
            ground_bias=self.ground_bias,
        )

    def calc_neighbours(self, r_cut, chunk_size=None, macro_chunk=10000, verbose=False) -> data_loader_output:
        if self.time_series:
            y = [*self.sp, *self.sp_t]
        else:
            y = [*self.sp]

        n = 0

        nl = []

        y_chunk = []

        tot_chunk = 0
        stack_dims_chunk = []

        while n < len(y):
            s = y[n].shape[0]

            y_chunk.append(y[n])

            tot_chunk += s
            stack_dims_chunk.append(s)

            if tot_chunk > macro_chunk or n == len(y) - 1:
                if verbose:
                    print(f"get nl: [sp] {n}/{len(y)}, {tot_chunk=}")

                nl_chunk = SystemParams.stack(*y_chunk).get_neighbour_list(
                    r_cut=r_cut,
                    r_skin=0,
                    z_array=self.sti.atomic_numbers,
                    chunk_size=chunk_size,
                    verbose=verbose,
                )

                ind = 0
                for sdc in stack_dims_chunk:
                    nl.append(nl_chunk[ind : ind + sdc])
                    ind += sdc

                y_chunk = []
                stack_dims_chunk = []
                tot_chunk = 0

            n += 1

        if self.time_series:
            nl, nl_t = nl[0 : len(self.sp)], nl[len(self.sp) :]
        else:
            nl_t = None

        return data_loader_output(
            sp=self.sp,
            nl=nl,
            cv=self.cv,
            sti=self.sti,
            ti=self.ti,
            collective_variable=self.collective_variable,
            sp_t=self.sp_t,
            nl_t=nl_t,
            cv_t=self.cv_t,
            time_series=self.time_series,
            tau=self.tau,
            bias=self.bias,
            ground_bias=self.ground_bias,
        )
