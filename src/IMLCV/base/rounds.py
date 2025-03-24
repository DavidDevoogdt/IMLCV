from __future__ import annotations

import os
import shutil
import time
from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import h5py
import jax
import jax.experimental
import jax.experimental.sparse
import jax.experimental.sparse.bcoo
import jax.numpy as jnp
import numpy as np
from jax import Array, vmap
from jax.random import PRNGKey, choice, split
from jax.tree_util import Partial
from molmod.constants import boltzmann
from molmod.units import kjmol
from parsl.data_provider.files import File

from IMLCV.base.bias import Bias, BiasModify, CompositeBias, NoneBias
from IMLCV.base.CV import (
    CV,
    CollectiveVariable,
    CvFlow,
    CvMetric,
    CvTrans,
    NeighbourList,
    NeighbourListInfo,
    NeighbourListUpdate,
    ShmapKwargs,
    SystemParams,
    macro_chunk_map,
    padded_shard_map,
    padded_vmap,
)
from IMLCV.base.CVDiscovery import Transformer
from IMLCV.base.MdEngine import MDEngine, StaticMdInfo, TrajectoryInfo
from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.configs.config_general import Executors
from IMLCV.implementations.bias import GridBias, RbfBias, _clip


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

    def is_zipped(self, c, r) -> bool | None:
        p = self.path(c=c, r=r)

        if p.exists():
            return False

        if p.with_suffix(".zip").exists():
            return True

        return None

    def zip_cv_round(self, c, r):
        p = self.path(c=c, r=r)

        if not p.exists():
            print(f"{p} does not exist")
            return

        if p.with_suffix(".zip").exists():
            print(f"{p}.zip already exitst, moving to .old.zip")
            shutil.move(p.with_suffix(".zip"), p.with_suffix(".old.zip"))

        shutil.make_archive(p, "zip", p)
        shutil.rmtree(p)

    def unzip_cv_round(self, c, r):
        p = self.path(c=c, r=r)

        if p.exists():
            print(f"{p} already exists")
            return

        if not p.with_suffix(".zip").exists():
            print(f"{p}.zip does not exist")
            return

        shutil.unpack_archive(p.with_suffix(".zip"), p)
        os.remove(p.with_suffix(".zip"))

    def zip_cv(self, cv):
        print(f"zipping {cv=}")
        for r in self._r_vals(c=cv):
            if r == -1:
                continue

            print(f"zipping {r=}")
            p = self.path(c=cv, r=r)

            if p.with_suffix(".zip").exists():
                print(f"{p}.zip already exitst, moving to .old.zip")
                shutil.move(p.with_suffix(".zip"), p.with_suffix(".old.zip"))

            shutil.make_archive(p, "zip", p)
            shutil.rmtree(self.path(c=cv, r=r))

    def unzip_cv(self, cv):
        print(f"unzipping {cv=}")

        for round_r in self.path(c=cv).glob("round_*.zip"):
            r = round_r.parts[-1][6:-4]

            if r.endswith(".old"):
                continue

            print(f"unzipping {r=}")

            self.unzip_cv_round(cv, r)

    def zip_cv_rounds(self, begin=1, end=-2):
        _c_vals = self._c_vals()

        n_cv = len(_c_vals)

        if n_cv == 1:
            return

        if end is None:
            end = n_cv

        if end <= 0:
            end = n_cv + end

        if end <= begin:
            return

        for cv in _c_vals[begin:end]:
            print(f"zipping {cv=}")
            for r in self._r_vals(c=cv):
                if r == -1:
                    continue

                print(f"zipping {r=}")
                p = self.path(c=cv, r=r)

                if p.with_suffix(".zip").exists():
                    print(f"{p}.zip already exitst, moving to .old.zip")
                    shutil.move(p.with_suffix(".zip"), p.with_suffix(".old.zip"))

                shutil.make_archive(p, "zip", p)
                shutil.rmtree(self.path(c=cv, r=r))

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
        ext="xyz",
    ):
        # if format == "xyz":
        #     from ase.io.extxyz import write_extxyz

        #     def writer(file, atoms):
        #         write_extxyz(file, atoms)
        # elif format == "cif":
        #     from ase.io.cif import write_cif

        #     def writer(file, atoms):
        #         write_cif(file, atoms)

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
            print(f"writing {i} {round.round=} {trajejctory.num=} {ext=}")
            # with open(
            #     self.path(c=c, r=round.round, i=trajejctory.num) / "trajectory.xyz",
            #     mode="w" if format == "xyz" else "wb",
            # ) as f:
            if repeat is not None:
                atoms = [a.repeat(repeat) for a in atoms]

                print(f"repeating {repeat=} {atoms[0].get_positions().shape=}")

            from ase.io import write

            write(
                filename=self.path(c=c, r=round.round, i=trajejctory.num) / f"trajectory.{ext}",
                images=atoms,
                format=ext,
            )

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
            if round_r.suffix == ".zip":
                continue

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
            return len(self._r_vals(c, r))

        return len(self._r_vals(c))

    def add_cv(self, collective_variable: CollectiveVariable, c=None):
        if c is None:
            c = self.cv + 1

        directory = self.path(c=c)
        if not os.path.isdir(directory):
            os.mkdir(directory)

        collective_variable.save(self.path(c=c) / "cv.json")

    def add_round(self, bias: Bias, stic: StaticMdInfo | None = None, mde=None, c=None, r=None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c) + 1

        if stic is None:
            stic = self.static_trajectory_information(c=c)

        if mde is None:
            mde = self.get_engine(c=c)

        mde.static_trajectory_info = stic
        mde.bias = bias

        dir = self.path(c=c, r=r)
        if not dir.exists():
            dir.mkdir(parents=True)

        stic.save(self.path(c=c, r=r) / "static_trajectory_info.h5")
        bias.save(self.path(c=c, r=r) / "bias.json")
        mde.save(self.path(c=c, r=r) / "engine.json")

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
            start = 1 if self.get_round(c=c) > 1 else 0

        if md_trajs is not None:
            assert num == 1

        low = max(stop - (num - 1), start)
        high = stop + 1

        print(f"iterating {low=} {high=} {num=}  {start=} {stop=}")

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
        get_bias_list=False,
        num_cv_rounds=1,
        only_finished=True,
        uniform=True,
        lag_n=1,
        colvar=None,
        check_dtau=True,
        verbose=False,
        weight=True,
        T_scale=1,
        macro_chunk=2000,
        macro_chunk_nl=5000,
        only_update_nl=False,
        n_max=50,
        wham=True,
        scale_times=False,
        reweight_to_fes=False,
        reweight_inverse_bincount=True,
        output_FES_bias=False,
        weighing_method="WHAM",
        samples_per_bin=10,
        min_samples_per_bin=3,
    ) -> data_loader_output:
        if cv_round is None:
            cv_round = self.cv

        if uniform and T_scale != 1:
            print(f"WARNING: uniform and {T_scale=} are not compatible")

        # if uniform and divide_by_histogram:
        #     print("WARNING: uniform and divide_by_histogram are not compatible")

        sti = self._round_information(c=cv_round).tic

        if new_r_cut == -1:
            new_r_cut = sti.r_cut

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

        if scale_times and not time_series:
            scale_times = False

        ###################

        if verbose:
            print("obtaining raw data")

        sp: list[SystemParams] = []
        cv: list[CV] = []
        ti: list[TrajectoryInfo] = []

        weights: list[Array] = []
        p_select: list[Array] = []
        grid_nums: list[Array] = []

        # if reweight_to_fes:
        #     fes_weights: list[Array] = []

        if reweight_inverse_bincount:
            bincount: list[Array] = []

        if scale_times:
            time_scalings: list[Array] = []

        sti_c: StaticMdInfo | None = None

        nl_info: NeighbourListInfo | None = None
        update_info: NeighbourListUpdate | None = None

        # n_bin = 0

        n_tot = 0

        FES_biases = []

        for cvi in cvrnds:
            # sti_c: StaticMdInfo | None = None
            sp_c: list[SystemParams] = []
            cv_c: list[CV] = []
            ti_c: list[TrajectoryInfo] = []

            if get_bias_list:
                bias_c: list[Bias] | None = []

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

                if new_r_cut is not None:
                    if nl_info is None:
                        nl_info = NeighbourListInfo.create(
                            r_cut=new_r_cut,
                            r_skin=0.0,
                            z_array=sti_c.atomic_numbers,
                        )

                        b, nn, new_nxyz, _ = Partial(
                            SystemParams._get_neighbour_list,
                            info=nl_info,
                            chunk_size=chunk_size,
                            chunk_size_inner=10,
                            shmap=False,
                            only_update=True,
                            update=update_info,
                        )(
                            self=sp0,
                        )

                        update_info = NeighbourListUpdate.create(
                            num_neighs=nn,
                            nxyz=new_nxyz,
                        )

                        print(f"initializing neighbour list with {nn=} {new_nxyz=}")

                    else:
                        # TODO
                        b, nn, new_nxyz, _ = jax.jit(
                            SystemParams._get_neighbour_list,
                            static_argnames=[
                                "info",
                                "chunk_size",
                                "chunk_size_inner",
                                "shmap",
                                "only_update",
                                "update",
                                "verbose",
                                "shmap_kwargs",
                            ],
                        )(
                            self=sp0,
                            info=nl_info,
                            chunk_size=chunk_size,
                            chunk_size_inner=10,
                            shmap=False,
                            only_update=True,
                            update=update_info,
                        )

                        if new_nxyz is not None:
                            if jnp.any(jnp.array(new_nxyz) > jnp.array([2, 2, 2])):
                                print(f"neighbour list is too big {new_nxyz=}")
                                continue

                            if (
                                jnp.any(jnp.array(new_nxyz) > jnp.array(update_info.nxyz))
                                or nn > update_info.num_neighs
                            ):
                                print(f"updating nl with {nn=} {new_nxyz=}")

                                update_info = NeighbourListUpdate.create(
                                    num_neighs=nn,
                                    nxyz=new_nxyz,
                                )

                print(".", end="", flush=True)
                n_tot += 1

                if n_tot % 100 == 0:
                    print("")

                if cv0 is None:
                    if colvar is None:
                        bias = traj_info.get_bias()
                        colvar = bias.collective_variable

                    info = NeighbourListInfo.create(
                        r_cut=round_info.tic.r_cut,
                        z_array=round_info.tic.atomic_numbers,
                    )

                    nlr = (
                        sp0.get_neighbour_list(
                            info=info,
                            chunk_size=chunk_size,
                        )
                        if round_info.tic.r_cut is not None
                        else None
                    )

                    cv0, _ = colvar.compute_cv(sp=sp0, nl=nlr)

                sp_c.append(sp0)
                cv_c.append(cv0)

                if get_bias_list:
                    bias_c.append(traj_info.get_bias())

            print("")

            if len(sp_c) == 0:
                continue

            if weight_c:
                if verbose:
                    print(f"getting weights for cv_round {cvi} {len(sp_c)} trajectories")

                dlo = data_loader_output(
                    sp=sp_c,
                    cv=cv_c,
                    ti=ti_c,
                    sti=sti_c,
                    nl=None,
                    collective_variable=colvar_c,
                    bias=bias_c if get_bias_list else None,
                    # time_series=time_series,
                    ground_bias=ground_bias_c,
                )

                # select points according to free energy divided by histogram count

                if weighing_method == "WHAM":
                    weight_output = dlo.wham_weight(
                        chunk_size=chunk_size,
                        n_max=n_max,
                        verbose=verbose,
                        return_bias=output_FES_bias,
                        samples_per_bin=samples_per_bin,
                        min_samples=min_samples_per_bin,
                        output_free_energy=reweight_to_fes,
                        output_time_scaling=scale_times,
                    )
                elif weighing_method == "DHAMed":
                    weight_output = dlo.dhamed_weight(
                        chunk_size=chunk_size,
                        n_max=n_max,
                        verbose=verbose,
                        return_bias=output_FES_bias,
                    )

                w_c = weight_output.weights
                p_c = weight_output.p_select
                gn_c = weight_output.grid_nums

                # if reweight_to_fes:
                #     w_fes_c = weight_output.free_energy

                if reweight_inverse_bincount:
                    bincount_c = weight_output.bin_counts
                    # n_bin = weight_output.n_bins

                if output_FES_bias:
                    FES_biases.append(weight_output.FES_bias)

                if scale_times:
                    time_scalings.extend(weight_output.time_scaling)

                # n = sum([len(wi) for wi in w_c])
                # w_c = [w_c[i] * n for i in range(len(w_c))]

                # if reweight_to_fes:
                #     w_fes_c = [w_fes_c[i] * n for i in range(len(w_fes_c))]

            else:
                print("setting weights to one!")

                w_c = [jnp.ones((spi.shape[0])) for spi in sp_c]
                p_c = [jnp.ones((spi.shape[0])) for spi in sp_c]

                gn_c = [CV(cv=jnp.zeros((spi.shape[0], 1), dtype=jnp.integer)) for spi in sp_c]

                # if reweight_to_fes:
                #     w_fes_c = [jnp.ones((spi.shape[0])) for spi in sp_c]

                if reweight_inverse_bincount:
                    bincount_c = [jnp.ones((spi.shape[0])) for spi in sp_c]

                if output_FES_bias:
                    FES_biases.append(None)

            sp.extend(sp_c)
            cv.extend(cv_c)
            ti.extend(ti_c)

            weights.extend(w_c)
            p_select.extend(p_c)
            grid_nums.extend(gn_c)

            # if reweight_to_fes:
            #     fes_weights.extend(w_fes_c)

            if reweight_inverse_bincount:
                bincount.extend(bincount_c)

            if get_bias_list:
                bias_list.extent(bias_c)

        ###################
        if verbose:
            print("Checking data")

        assert len(sp) != 0, "no data found"

        key = PRNGKey(0)

        if T_max_over_T is not None:
            sp_new: list[SystemParams] = []
            cv_new: list[CV] = []
            ti_new: list[TrajectoryInfo] = []

            weights_new: list[Array] = []
            p_select_new: list[Array] = []
            grid_nums_new: list[Array] = []

            # if reweight_to_fes:
            #     fes_weights_new: list[Array] = []

            if reweight_inverse_bincount:
                bincount_new: list[Array] = []

            if scale_times:
                time_scalings_new: list[Array] = []

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
                p_select_new.append(p_select[n])
                grid_nums_new.append(grid_nums[n])

                # if reweight_to_fes:
                #     fes_weights_new.append(fes_weights[n])

                if reweight_inverse_bincount:
                    bincount_new.append(bincount[n])

                if scale_times:
                    time_scalings_new.append(time_scalings[n])

            sp = sp_new
            ti = ti_new
            cv = cv_new

            weights = weights_new
            p_select = p_select_new
            grid_nums = grid_nums_new

            # if reweight_to_fes:
            #     fes_weights = fes_weights_new

            if reweight_inverse_bincount:
                bincount = bincount_new

            if scale_times:
                time_scalings = time_scalings_new

            if get_bias_list:
                bias_list = new_bias_list

        print(f"len(sp) = {len(sp)}")

        for j, k in zip(sp, cv):
            if j.shape[0] != k.shape[0]:
                print(f"shapes do not match {j.shape=} {k.shape=}")

        for w_i, p_i, spi in zip(weights, p_select, sp):
            assert w_i.shape[0] == spi.shape[0], f"weights and sp shape are different: {w_i.shape=} {spi.shape=}"

            assert p_i.shape[0] == spi.shape[0], f"p_select and sp shape are different: {p_i.shape=} {spi.shape=}"

        c_list = []
        lag_indices = []
        percentage_list = [] if (weights is not None and scale_times) and (lag_n != 0) else None

        if weights is not None:
            if lag_n != 0 and verbose:
                print(f"getting lag indices for {len(ti)} trajectories")

            timestep = sti_c.timestep

            @jax.jit
            @partial(jax.vmap, in_axes=(None, 0, None))
            def get_lag_idx(dt, n, u):
                # u = jnp.log(w)/beta

                # u -= u[n]  # set bias to 0 at time 0

                def sinhc(x):
                    return jnp.where(x < 1e-10, 1, jnp.sinh(x) / x)

                # m_u = jnp.exp(beta * (u[1:] + u[1:]) / 2)  # opprox e^(\beta( U ))
                # d_u = sinhc(beta * (u[:-1] - u[1:]) / 2)  # approx 1 if du is small

                integral = jnp.zeros((u.shape[0] + 1))
                integral = integral.at[1:].set(u)
                integral = jnp.cumsum(integral)
                integral -= integral[n]  # set time to zero at n

                integral *= timestep

                _index_k = jax.lax.top_k(-jnp.abs(integral - dt), 1)[1][0]  # find closest matching index

                # first larger index

                b = integral[_index_k] > dt

                index_0 = jnp.where(b, _index_k - 1, _index_k)
                index_1 = index_0 + 1

                # index_0, index_1 = jnp.where(
                #     integral[index_0] > dt,
                #     (index_0 - 1, index_0),
                #     (index_0, index_0 + 1),
                # )
                indices = jnp.array([index_0, index_1])

                indices = jnp.sort(indices)
                values = integral[indices]

                b = (
                    (indices[1] < u.shape[0] - 1)
                    * (indices[0] < u.shape[0] - 1)  # if the values are the same, the values might be ordered wrong
                    * (indices[0] >= n)
                    * (indices[1] - indices[0] == 1)  # consitency check
                )

                # make sure that percentage is not if int is zero
                percentage = jnp.where(
                    (values[1] - values[0]) < 1e-20,
                    0.0,
                    (dt - values[0]) / (values[1] - values[0]),
                )

                return indices[0], indices[1], b, percentage

            for n, ti_i in enumerate(ti):
                if lag_n != 0:
                    if scale_times:
                        scales = time_scalings[n]

                        lag_indices_max, _, bools, p = get_lag_idx(
                            lag_n * sti_c.timestep,
                            jnp.arange(scales.shape[0]),
                            scales,
                        )

                        c = jnp.sum(jnp.logical_and(scales[bools] != 0, scales[lag_indices_max[bools]] != 0))

                        if c == 0:
                            continue

                        percentage_list.append(p[bools])
                        lag_idx = lag_indices_max[bools]

                    else:
                        c = ti_i.e_bias.shape[0] - lag_n
                        lag_idx = jnp.arange(c) + lag_n

                    c_list.append(c)
                    lag_indices.append(lag_idx)

                else:
                    c = ti_i._size

                    lag_indices.append(jnp.arange(c))
                    c_list.append(c)

        else:
            for a in sp:
                c = a.shape[0] - lag_n
                lag_indices.append(jnp.arange(c))
                c_list.append(c)

        if scale_times and lag_n != 0:
            print(f" {jnp.hstack(c_list)=}")

            ll = []

            for li, pi in zip(lag_indices, percentage_list):
                ni = jnp.arange(li.shape[0])
                dn = li - ni

                dnp = dn * pi + (dn + 1) * (1 - pi)

                ll.append(jnp.mean(dnp))

            ll = jnp.hstack(ll)

            print(f"average lag index { ll } {jnp.mean(ll)=}")

        total = sum(c_list)

        if out == -1:
            out = total

        if out > total:
            print(f"not enough data, returning {total} data points instead of {out}")
            out = total

        ###################

        if verbose:
            print(f"total data points {total}, selecting {out}")

        def choose(
            key,
            weight: list[Array],
            p_select: list[Array],
            grid_nums: list[Array],
            # fes_weight: list[Array] | None,
            bin_count: list[Array] | None,
            out: int,
        ):
            key, key_return = split(key, 2)

            # if reweight_inverse_bincount and bin_count is not None:
            #     selection = jnp.where(jnp.hstack(bin_count) == 0, 0, 1 / jnp.hstack(bin_count))

            # #     # p is constructed such that the samples are uniformly samples but no more samples than the numbers of samples per bin is taken

            # #     bc_stack = jnp.hstack(bin_count)

            # #     # it is possible that the bin count is zero, because the bin has been discarded
            # #     bc_stack_inv = jnp.where(bc_stack == 0, 0, 1 / bc_stack)

            # #     full = jnp.full(bc_stack.shape, False)

            # #     out_p = jnp.ones_like(bc_stack_inv)

            # #     while True:
            # #         print(".")

            # #         _select = out - jnp.sum(full)

            # #         _n_bins = jnp.sum(bc_stack_inv[~full])

            # #         select_per_bin = _select / _n_bins

            # #         # the odds to select a data point from a given bin
            # #         p = select_per_bin * bc_stack_inv[~full]
            # #         _full = p >= 1

            # #         # none of the remaining bins is overfilled
            # #         if jnp.sum(_full) == 0:
            # #             print("all data fits in bins  ")
            # #             out_p = out_p.at[~full].set(p)

            # #             break

            # #         # all the bins are  overfilled.
            # #         if jnp.sum(~_full) == 0:
            # #             print("all bins are full")

            # #             break

            # #         full = full.at[~full].set(_full)

            # #     p_corr = out_p
            # #     # p_corr_inv = jnp.where(p_corr == 0, 0, 1 / p_corr)

            # #     ni = 0

            # #     # weight_new = []
            # #     p_select_new = []
            # #     selection = []

            # #     for wi, bi, pi in zip(weight, bin_count, p_select):
            # #         out_p_i = p_corr[ni : ni + bi.shape[0]]
            # #         # out_p_i_inv = p_corr_inv[ni : ni + bi.shape[0]]

            # #         selection.append(out_p_i)
            # #         # p_select_new.append(pi * out_p_i_inv)  # rho is saved separately, no need to do

            # #         ni += bi.shape[0]

            # #     # weight = weight_new
            # #     # p_select = p_select_new

            # #     selection = jnp.hstack(selection)

            # else:
            selection = jnp.ones_like(jnp.hstack(p_select))

            if (ns := jnp.sum(jnp.isnan(selection))) != 0:
                print(f"found {ns=} nan in p_select")

                selection = jnp.where(jnp.isnan(selection), 0, selection)

            n = jnp.sum(selection != 0)

            if out > n:
                print(f"requested {out=} data points, but only {n=} are available")
                out = n

            indices = choice(
                key=key,
                a=selection.shape[0],
                shape=(int(out),),
                p=selection,
                replace=False,
            )

            # print(
            #     f"{jnp.sum(selection[indices])/jnp.sum( selection):.1%} of weight selected with {out/selection.shape[0]:.1%} of the samples"
            # )

            # reweight each bin such that prob sums to 1 per bin

            # this should be 1 (approx due to lag time)

            grid_nums_stack = CV.stack(*grid_nums)
            nn = int(jnp.max(grid_nums_stack.cv)) + 1

            # print(f"{grid_nums_stack.shape=} {nn=}  {jnp.hstack(p_select).shape=} ")

            rho_grid = data_loader_output.get_histo(
                [grid_nums_stack],
                [selection],
                macro_chunk=macro_chunk,
                nn=nn,
            )

            # print(f"{rho_grid=}")

            rho_grid_new = data_loader_output.get_histo(
                [grid_nums_stack[indices]],
                [selection],
                macro_chunk=macro_chunk,
                nn=nn,
            )

            print(
                f"{jnp.sum(rho_grid_new)/jnp.sum(rho_grid):.1%} of weight selected with {out/selection.shape[0]:.1%} of the samples"
            )

            print(f"selection reweighing")

            p_select_new = []

            for grid_num_i, p_select_i in zip(grid_nums, p_select):
                # print(f"{grid_num_i.shape=} {p_select_i.shape=}")

                gi = grid_num_i.cv.reshape((-1,))

                f = jnp.where(rho_grid_new[gi] == 0, 0, rho_grid[gi] / rho_grid_new[gi])

                p_select_new.append(p_select_i * f)

            p_select = p_select_new

            return key_return, indices, weight, p_select

        def remove_lag(w, c):
            w = w[:c]
            return w

        out_indices = []

        n_list = []

        if split_data:
            frac = out / total

            out_reweights = []
            out_rhos = []

            for n, (w_i, ps_i, c_i, gn_i) in enumerate(zip(weights, p_select, c_list, grid_nums)):
                # if w_i is not None:
                w_i = remove_lag(
                    w_i,
                    c_i,
                )

                ps_i = remove_lag(
                    ps_i,
                    c_i,
                )

                gn_i = remove_lag(
                    gn_i,
                    c_i,
                )

                if reweight_inverse_bincount:
                    bincount_i = remove_lag(
                        bincount[n],
                        c_i,
                    )

                ni = int(frac * c_i)

                key, indices, reweight, rerho = choose(
                    key=key,
                    weight=[w_i],
                    p_select=[ps_i],
                    grid_nums=[gn_i],
                    # fes_weight=[w_fes_i] if reweight_to_fes else None,
                    bin_count=[bincount_i] if reweight_inverse_bincount else None,
                    out=ni,
                )

                out_indices.append(indices)
                out_reweights.extend(reweight)
                out_rhos.extend(rerho)
                n_list.append(n)

        else:
            # w_fes = None
            # w_bincount = None

            if weights[0] is None:
                w = None
                p_select = None
                grid_nums = None

            else:
                w = []
                ps = []

                gn = []

                # if reweight_to_fes:
                #     w_fes = []

                if reweight_inverse_bincount:
                    w_bincount = []

                # print(f"{weights=}")
                # print(f"{fes_weights=}")

                for n, (w_i, ps_i, c_i, gn_i) in enumerate(zip(weights, p_select, c_list, grid_nums)):
                    w_i = remove_lag(
                        w_i,
                        c_i,
                    )

                    w.append(w_i)

                    ps_i = remove_lag(
                        ps_i,
                        c_i,
                    )

                    ps.append(ps_i)

                    gn_i = remove_lag(
                        gn_i,
                        c_i,
                    )

                    gn.append(gn_i)

                    # if reweight_to_fes:
                    #     w_fes_i = remove_lag(
                    #         fes_weights[n],
                    #         c_i,
                    #     )

                    #     w_fes.append(w_fes_i)

                    if reweight_inverse_bincount:
                        w_bincount_i = remove_lag(
                            bincount[n],
                            c_i,
                        )

                        w_bincount.append(w_bincount_i)

            key, indices, out_reweights, out_rhos = choose(
                key=key,
                weight=w,
                p_select=ps,
                grid_nums=gn,
                # fes_weight=w_fes,
                bin_count=w_bincount if reweight_inverse_bincount else None,
                out=int(out),
            )

            print(f"selected {len(indices)} out of {total} data points {len(out_reweights)=} {len(out_rhos)=}")

            count = 0

            for n, n_i in enumerate(c_list):
                indices_full = indices[jnp.logical_and(count <= indices, indices < count + n_i)]
                index = indices_full - count

                if len(index) == 0:
                    count += n_i

                    continue

                out_indices.append(index)

                n_list.append(n)

                count += n_i

        ###################
        # storing data    #
        ###################

        out_sp: list[SystemParams] = []
        out_cv: list[CV] = []
        out_ti: list[TrajectoryInfo] = []
        out_weights: list[Array] = []
        out_rho: list[Array] = []

        if time_series:
            out_sp_t = []
            out_cv_t = []
            out_ti_t = []
            out_weights_t = []
            out_rho_t = []

        if get_bias_list:
            out_biases = []
        else:
            out_biases = None

        if verbose and time_series and percentage_list is not None:
            print("interpolating data")

        for n, indices_n, reweights_n, rho_n in zip(n_list, out_indices, out_reweights, out_rhos):
            out_sp.append(sp[n][indices_n])
            out_cv.append(cv[n][indices_n])

            ti_n = ti[n][indices_n]
            out_ti.append(ti_n)
            out_weights.append(reweights_n[indices_n])
            out_rho.append(rho_n[indices_n])

            if time_series:
                idx_t = lag_indices[n][indices_n]
                idx_t_p = idx_t + 1

                # print(f"{idx_t=} {idx_t_p=}")

                if percentage_list is None:
                    out_sp_t.append(sp[n][idx_t])
                    out_cv_t.append(cv[n][idx_t])
                    out_ti_t.append(ti[n][idx_t])
                    # TODO: is this correct? seems to go out of bounds
                    out_weights_t.append(reweights_n[idx_t])
                    out_rho_t.append(rho_n[idx_t])
                else:
                    # print(f"interpolating {n=}")

                    percentage = percentage_list[n][indices_n]

                    # print(f"{percentage.shape=} {sp[n][idx_t].shape=} {sp[n][idx_t_p].shape=}")

                    # this performs a lineat interpolation between the two points
                    def interp(x, y, p):
                        return jax.vmap(
                            lambda xi, yi, pi: jax.tree_util.tree_map(
                                lambda xii, yii: (1 - pi) * xii + pi * yii,
                                xi,
                                yi,
                            ),
                            in_axes=(0, 0, 0),
                        )(x, y, p)

                    # TODO: make interp a CV function and use apply

                    out_sp_t.append(interp(sp[n][idx_t], sp[n][idx_t_p], percentage))
                    out_cv_t.append(interp(cv[n][idx_t], cv[n][idx_t_p], percentage))

                    ti_t_n = interp(ti[n][idx_t], ti[n][idx_t_p], percentage)

                    # times need to be compensated
                    ti_t_n.t = ti_n.t + lag_n * sti.timestep

                    out_ti_t.append(ti_t_n)

                    out_weights_t.append(interp(reweights_n[idx_t], reweights_n[idx_t_p], percentage))
                    out_rho_t.append(interp(rho_n[idx_t], rho_n[idx_t_p], percentage))

            if get_bias_list:
                out_biases.append(bias_list[n])

        # print(f"c05 {out_weights=} ")

        s = 0
        for ow in out_weights:
            s += jnp.sum(ow)

        out_weights = [ow / s for ow in out_weights]

        s_rho = 0
        for rho in out_rho:
            s_rho += jnp.sum(rho)

        out_rho = [rho / s_rho for rho in out_rho]

        if time_series:
            s = 0
            for ow in out_weights_t:
                s += jnp.sum(ow)

            out_weights_t = [ow / s for ow in out_weights_t]

            s_rho = 0
            for rho in out_rho_t:
                s_rho += jnp.sum(rho)

            out_rho_t = [rho / s_rho for rho in out_rho_t]

        # print(f"c06 {out_weights=}")

        print(f"len(out_sp) = {len(out_sp)} ")

        ###################

        if verbose:
            print("Checking data")

        out_nl = (
            NeighbourList(
                info=nl_info,
                update=update_info,
                sp_orig=None,
            )
            if nl_info is not None
            else None
        )

        if time_series:
            out_nl_t = out_nl

        if time_series:
            tau = None

            arr = []

            for tii, ti_ti in zip(out_ti, out_ti_t):
                tii: TrajectoryInfo
                ti_ti: TrajectoryInfo

                dt = ti_ti.t - tii.t

                tau = jnp.median(dt) if tau is None else tau

                mask = jnp.allclose(dt, tau)

                if not mask.all():
                    arr.append(jnp.sum(jnp.logical_not(mask)))

            if tau is None:
                print("WARNING: tau None")
            else:
                if len(arr) != 0:
                    print(
                        f"WARNING:time steps are not equal, {jnp.array(arr)} out of { out   } trajectories have different time steps"
                    )

                from molmod.units import femtosecond

                print(f"tau = {tau/femtosecond:.2f} fs, lag_time*timestep = {lag_n* sti.timestep/ femtosecond:.2f} fs")

        ###################

        dlo_kwargs = dict(
            sp=out_sp,
            nl=out_nl,
            cv=out_cv,
            ti=out_ti,
            sti=sti,
            collective_variable=colvar,
            time_series=time_series,
            bias=out_biases,
            ground_bias=ground_bias,
            _weights=out_weights,
            scaled_tau=scale_times,
            _rho=out_rho,
        )

        if time_series:
            dlo_kwargs.update(
                sp_t=out_sp_t,
                nl_t=out_nl_t,
                cv_t=out_cv_t,
                ti_t=out_ti_t,
                tau=tau,
                _weights_t=out_weights_t,
                _rho_t=out_rho_t,
            )

        dlo = data_loader_output(
            **dlo_kwargs,
        )

        ###################

        # if new_r_cut is not None:
        #     if verbose:
        #         print("getting Neighbour List")

        #     dlo.calc_neighbours(
        #         r_cut=new_r_cut,
        #         chunk_size=chunk_size,
        #         verbose=verbose,
        #         macro_chunk=macro_chunk_nl,
        #         only_update=only_update_nl,
        #     )

        ###################

        if recalc_cv:
            if verbose:
                print("recalculating CV")
            dlo.recalc(
                chunk_size=chunk_size,
                verbose=verbose,
                macro_chunk=macro_chunk,
            )

        if output_FES_bias:
            return dlo, FES_biases

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
        import ase
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

    def static_trajectory_information(self, c=None, r=None) -> StaticMdInfo:
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        folder = self.path(c=c, r=r)
        return StaticMdInfo.load(folder / "static_trajectory_info.h5")

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
            num_vals=np.array(mdi, dtype=np.int64),
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

        if i is None:
            p = self.path(c=c, r=r, i=i) / "static_trajectory_info.h5"
        else:
            p = self.path(c=c, r=r, i=i) / "trajectory_info.h5"

        if not p.exists():
            print(f"cannot invalidate data for {c=} {r=} {i=} because traj file does not exist")
            return

        n = "invalid" if i is None else "_invalid"

        try:
            with h5py.File(p, "r+") as hf:
                hf.attrs[n] = True
        except Exception:
            print(f"could not invalidate {c=} {r=} {i=}, writing file")

            with open(self.path(c=c, r=r, i=i) / "invalid", "w"):
                pass

    def finish_data(self, c=None, r=None, i=None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        p = self.path(c=c, r=r, i=i) / "trajectory_info.h5"

        if not p.exists():
            print(f"cannont finish data for {c=} {r=} {i=} because traj file does not exist")

        with h5py.File(p, "r+") as hf:
            hf.attrs["_finished"] = True

    def is_valid(self, c=None, r=None, i=None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        if i is None:
            p = self.path(c=c, r=r, i=i) / "static_trajectory_info.h5"
        else:
            p = self.path(c=c, r=r, i=i) / "trajectory_info.h5"

        n = "invalid" if i is None else "_invalid"

        if (p1 := self.path(c=c, r=r, i=i) / "invalid").exists():
            print("replacing invalid file with h5py")
            with h5py.File(p, "r+") as hf:
                hf.attrs[n] = True

            p1.unlink()

            return False

        with h5py.File(p, "r+") as hf:
            if n in hf.attrs:
                return not hf.attrs[n]
            else:
                print("adding invalid=False file with h5py")
                hf.attrs[n] = False
        return True

    def is_finished(self, c=None, r=None, i=None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        p = self.path(c=c, r=r, i=i) / "trajectory_info.h5"

        if (p1 := self.path(c=c, r=r, i=i) / "finished").exists():
            print("replacing finished file with h5py")
            with h5py.File(p, "r+") as hf:
                hf.attrs["_finished"] = True

            p1.unlink()

            return True

        with h5py.File(p, "r") as hf:
            if "_finished" in hf.attrs:
                return hf.attrs["_finished"]

        return False

    def get_collective_variable(
        self,
        c=None,
    ) -> CollectiveVariable:
        if c is None:
            c = self.cv

        return CollectiveVariable.load(self.path(c=c) / "cv.json")

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
        chunk_size=None,
        T_scale=10,
        macro_chunk=2000,
        lag_n=20,
    ):
        if cv_round is None:
            cv_round = self.cv

        r = self.get_round(c=cv_round)

        common_bias_name = self.full_path(self._name_bias(c=cv_round, r=r))
        common_md_name = self.full_path(self._name_md(c=cv_round, r=r))

        pn = self.path(c=cv_round, r=r)

        out = bash_app_python(
            Rounds._get_init,
            executors=Executors.training,
        )(
            rounds=self,
            KEY=KEY,
            ignore_invalid=ignore_invalid,
            only_finished=only_finished,
            min_traj_length=min_traj_length,
            recalc_cv=recalc_cv,
            T_scale=T_scale,
            chunk_size=chunk_size,
            md_trajs=md_trajs,
            cv_round=cv_round,
            biases=biases,
            sp0=sp0,
            common_bias_name=common_bias_name,
            r=r,
            execution_folder=pn,
            macro_chunk=macro_chunk,
            lag_n=lag_n,
        ).result()

        from parsl.dataflow.dflow import AppFuture

        tasks: list[tuple[int, AppFuture]] | None = None
        plot_tasks = []

        for i, (spi, bi, traj_name, b_name, b_name_new, path_name) in enumerate(zip(*out)):
            future = bash_app_python(
                Rounds.run_md,
                pass_files=True,
                executors=Executors.reference,
                profile=profile,
            )(
                sp=spi,  # type: ignore
                inputs=[File(common_md_name), File(str(b_name))],
                outputs=[File(str(b_name_new)), File(str(traj_name))],
                steps=int(steps),
                execution_folder=path_name,
            )

            if plot:
                plot_file = path_name / "plot.png"

                plot_fut = bash_app_python(Rounds.plot_md_run, pass_files=True)(
                    rnds=self,
                    fut=future,
                    c=cv_round,
                    r=r,
                    i=i,
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
                future.result()

            except Exception as _:
                print(f"got exception  while collecting md {i}, round {r}, cv {cv_round}, marking as invalid")

                self.invalidate_data(c=cv_round, r=r, i=i)
                # raise e
                continue

        # wait for plots to finish
        if plot and wait_for_plots:
            for i, future in enumerate(plot_tasks):
                try:
                    future.result()
                except Exception as _:
                    print(f"got exception  while trying to collect plot of {i},round {r}, cv {cv_round}, continuing ")

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

            if (path_name / "invalid").exists():
                print(f"skipping {i=}, invalid")
                continue

            # if b_name_new.exists():
            #     print(f"skipping {i=}, new bias found")
            #     continue

            traj_name = path_name / "trajectory_info.h5"

            future = bash_app_python(Rounds.run_md, pass_files=True, executors=Executors.reference)(
                sp=None,  # type: ignore
                inputs=[File(common_md_name), File(str(b_name))],
                outputs=[File(str(b_name_new)), File(str(traj_name))],
                steps=int(steps),
                execution_folder=path_name,
            )

            if plot:
                plot_file = path_name / "plot.png"

                plot_fut = bash_app_python(Rounds.plot_md_run, pass_files=True)(
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
                future.result()

            except Exception as e:
                print(f"got exception {e} while collecting md {i}, round {round}, cv {cv_round}, marking as invalid")

                self.invalidate_data(c=cv_round, r=round, i=i)

        # wait for plots to finish
        if plot and wait_for_plots:
            for i, future in enumerate(plot_tasks):
                try:
                    future.result()
                except Exception as e:
                    print(f"got exception {e} while trying to collect plot of {i}, continuing ")

    @staticmethod
    def _get_init(
        rounds: Rounds,
        KEY,
        common_bias_name: Bias,
        ignore_invalid=False,
        only_finished=True,
        min_traj_length=None,
        recalc_cv=False,
        T_scale=10,
        chunk_size=None,
        md_trajs=None,
        cv_round=None,
        biases=None,
        sp0=None,
        r=None,
        macro_chunk=1000,
        lag_n=20,
        out=20000,
        use_energies=False,
        # divide_by_histogram=True,
    ):
        sps = []
        bs = []
        traj_names = []
        b_names = []
        b_names_new = []
        path_names = []

        sp0_provided = sp0 is not None

        common_bias = Bias.load(common_bias_name)

        if not sp0_provided:
            dlo_data = rounds.data_loader(
                num=2,
                out=out,
                split_data=False,
                new_r_cut=None,
                ignore_invalid=ignore_invalid,
                md_trajs=md_trajs,
                cv_round=cv_round,
                min_traj_length=min_traj_length,
                recalc_cv=recalc_cv,
                only_finished=only_finished,
                weight=False,
                T_scale=T_scale,
                time_series=False,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                verbose=True,
                # lag_n=lag_n,
            )

            # use_energies = True
            # energies = []

            # for ti in dlo_data.ti:
            #     # e = ti.ti.e_pot
            #     if e is None:
            #         print("not using energies because not available")
            #         use_energies = False
            #         break

            #     energies.append(e)

            # if use_energies:
            #     energies = jnp.hstack(energies)

            beta = 1 / (dlo_data.sti.T * boltzmann)

            # get the weights of the points

            sp_stack = SystemParams.stack(*dlo_data.sp)
            cv_stack = CV.stack(*dlo_data.cv)

            # get  weights, and correct for ground state bias.
            # this corrects for the fact that the samples are not uniformly distributed

            # w = jnp.hstack(dlo_data._weights)

            # w_init = w
            # w_init = w_init / jnp.mean(w_init)

            # print(f"initial weights {w_init=}")

        else:
            assert (
                sp0.shape[0] == len(biases)
            ), f"The number of initials cvs provided {sp0.shape[0]} does not correspond to the number of biases {len(biases)}"

        if isinstance(KEY, int):
            KEY = jax.random.PRNGKey(KEY)

        for i, bias in enumerate(biases):
            path_name = rounds.path(c=cv_round, r=r, i=i)
            if not os.path.exists(path_name):
                os.mkdir(path_name)

            b = CompositeBias.create([common_bias, bias])

            b_name = path_name / "bias.json"
            b_name_new = path_name / "bias_new.json"
            b.save(b_name)

            traj_name = path_name / "trajectory_info.h5"

            if not sp0_provided:
                # reweigh data points according to new bias

                ener = bias.compute_from_cv(cvs=cv_stack, chunk_size=chunk_size)[0]

                # if use_energies:
                #     ener += energies

                ener -= jnp.min(ener)

                probs = jnp.exp(-ener * beta)  # * w_init
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
                spi = spi.unbatch()

            sps.append(spi)
            bs.append(b)
            traj_names.append(traj_name)
            b_names.append(b_name)
            b_names_new.append(b_name_new)
            path_names.append(path_name)

        return sps, bs, traj_names, b_names, b_names_new, path_names

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

    @staticmethod
    def plot_md_run(
        rnds: Rounds,
        i: int,
        r: int,
        c: int,
        fut=None,
        inputs=[],
        outputs=[],
    ):
        rnd = rnds._round_information(c=c, r=r)
        traj = rnds._trajectory_information(c=c, r=r, i=i)

        bias = traj.get_bias()

        cvs = traj.ti.CV

        if cvs is None:
            sp = traj.sp

            nl = sp.get_neighbour_list(
                info=rnd.tic.neighbour_list_info,
            )
            cvs, _ = bias.collective_variable.compute_cv(sp=sp, nl=nl)

        # bias.plot(
        #     name=outputs[0].filepath,
        #     traj=[cvs],
        #     offset=True,
        #     margin=0.1,
        # )

    ######################################
    #          CV transformations        #
    ######################################

    def update_CV(
        self,
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
        samples_per_bin=5,
        min_samples_per_bin=1,
        percentile=1e-1,
        use_executor=True,
        n_max=50,
        vmax=100 * kjmol,
        macro_chunk=1000,
        macro_chunk_nl: int = 5000,
        verbose=False,
    ):
        if cv_round_from is None:
            cv_round_from = self.cv

        if dlo_kwargs is None:
            dlo_kwargs = {}

        if "chunk_size" in dlo_kwargs:
            chunk_size = dlo_kwargs.pop("chunk_size")

        if "macro_chunk" in dlo_kwargs:
            macro_chunk = dlo_kwargs.pop("macro_chunk")

        if "macro_chunk_nl" in dlo_kwargs:
            macro_chunk_nl = dlo_kwargs.pop("macro_chunk_nl")

        if "verbose" in dlo_kwargs:
            verbose = dlo_kwargs.pop("verbose")

        if "samples_per_bin" in dlo_kwargs:
            samples_per_bin = dlo_kwargs.pop("samples_per_bin")

        if "min_samples_per_bin" in dlo_kwargs:
            min_samples_per_bin = dlo_kwargs.pop("min_samples_per_bin")

        if cv_round_from is not None:
            dlo_kwargs["cv_round"] = cv_round_from

        if chunk_size is not None:
            dlo_kwargs["chunk_size"] = chunk_size

        if cv_round_to is None:
            cv_round_to = self.cv + 1

        kw = dict(
            rounds=self,
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
            macro_chunk_nl=macro_chunk_nl,
            verbose=verbose,
        )

        if use_executor:
            bash_app_python(
                Rounds._update_CV,
                executors=Executors.training,
                remove_stderr=False,
                remove_stdout=False,
            )(execution_folder=self.path(c=cv_round_from), **kw).result()

            return

        Rounds._update_CV(**kw)

    @staticmethod
    def _update_CV(
        rounds: Rounds,
        transformer: Transformer,
        dlo_kwargs={},
        dlo: data_loader_output | None = None,
        chunk_size=None,
        macro_chunk=1000,
        macro_chunk_nl: int = 5000,
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
        samples_per_bin=5,
        min_samples_per_bin=1,
        percentile=1e-1,
        n_max=50,
        vmax=100 * kjmol,
        verbose=True,
    ):
        plot_folder = rounds.path(c=cv_round_to)
        cv_titles = [f"{cv_round_from}", f"{cv_round_to}"]
        max_fes_bias = max_bias

        if dlo is None:
            dlo, fb = rounds.data_loader(
                **dlo_kwargs,
                macro_chunk=macro_chunk,
                verbose=verbose,
                macro_chunk_nl=macro_chunk_nl,
                min_samples_per_bin=min_samples_per_bin,
                samples_per_bin=samples_per_bin,
                output_FES_bias=True,
            )

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
            verbose=verbose,
        )

        # update state
        rounds.__update_CV(
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

        x, _ = dlo.apply_cv(cv_trans=cv_trans, x=dlo.cv, chunk_size=chunk_size, verbose=verbose)

        #
        if plot:
            Transformer.plot_app(
                collective_variables=[dlo.collective_variable, collective_variable],
                cv_data=[dlo.cv, x],
                biases=[bias],
                duplicate_cv_data=True,
                T=dlo.sti.T,
                plot_FES=True,
                # weight=dlo.weights(),
                name=self.path(c=cv_round_to) / "transformed_FES.png",
                cv_titles=[cv_round_from, cv_round_to],
                data_titles=[cv_round_from, cv_round_to],
                vmax=vmax,
            )

        self.__update_CV(
            new_collective_variable=collective_variable,
            new_bias=bias,
            cv_round_from=cv_round_from,
            cvs_new=x,
            dlo=dlo,
            new_r_cut=new_r_cut,
        )

    def __update_CV(
        rounds: Rounds,
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

        rounds.add_cv(new_collective_variable, c=cv_round_from + 1)

        stic = rounds.static_trajectory_information(c=cv_round_from)
        stic.r_cut = new_r_cut

        rounds.add_round(
            bias=new_bias,
            stic=stic,
            mde=rounds.get_engine(c=cv_round_from),
        )

        # make sure it doesn't take to much space
        rounds.zip_cv_rounds()

        if save_samples:
            first = True

            if save_multiple_cvs:
                for dlo_i, cv_new_i in zip(iter(dlo), cvs_new):
                    if not first:
                        rounds.add_cv(new_collective_variable)
                        rounds.add_round(bias=NoneBias.create(new_collective_variable), stic=stic)

                    rounds._copy_from_previous_round(
                        dlo=dlo_i,
                        new_cvs=[cv_new_i],
                        cv_round=cv_round_from,
                    )
                    rounds.add_round(bias=new_bias, stic=stic)

                    first = False

            else:
                rounds._copy_from_previous_round(
                    dlo=dlo,
                    new_cvs=cvs_new,
                    cv_round=cv_round_from,
                )

                rounds.add_round(bias=new_bias, stic=stic)

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

            traj_info = dlo.ti[i]

            new_traj_info = TrajectoryInfo.create(
                positions=traj_info.positions,
                cell=traj_info.cell,
                charges=traj_info.charges,
                e_pot=traj_info.e_pot,
                e_bias=traj_info.e_bias,
                cv=new_cvs[i].cv,
                cv_orig=dlo.cv[i].cv,
                w=dlo._weights[i],
                rho=dlo._rho[i],
                T=traj_info._T,
                P=traj_info._P,
                err=traj_info._err,
                t=traj_info._t,
                capacity=traj_info._capacity,
                size=traj_info._size,
                finished=True,
            )

            new_traj_info.save(round_path / "trajectory_info.h5")

            if invalidate:
                self.invalidate_data(c=self.cv, r=self.round, i=i)


@dataclass(repr=False)
class WeightOutput:
    weights: list[Array]  # e^( beta U_i - F_i  )

    p_select: list[Array]  # p_select = e^( -( beta U_i - F_i)   ) / sum_i e^( beta U_i - F_i )

    time_scaling: list[Array] | None = None

    # free_energy: list[Array] | None = None
    # bin_counts_i: list[Array] | None = None
    bin_counts: list[Array] | None = None
    grid_nums: list[Array] | None = None

    # n_bins: int | None = None

    FES_bias: Bias | None = None
    FES_bias_std: Bias | None = None

    # log_space: bool = False


@dataclass(repr=False)
class data_loader_output:
    sp: list[SystemParams]

    cv: list[CV]
    sti: StaticMdInfo
    ti: list[TrajectoryInfo]
    collective_variable: CollectiveVariable
    nl: list[NeighbourList] | NeighbourList | None = None
    sp_t: list[SystemParams] | None = None
    nl_t: list[NeighbourList] | NeighbourList | None = None
    cv_t: list[CV] | None = None
    ti_t: list[TrajectoryInfo] | None = None
    time_series: bool = False
    tau: float | None = None
    bias: list[Bias] | None = None
    ground_bias: Bias | None = None
    _weights: list[Array] | None = None
    _weights_t: list[Array] | None = None
    _rho: list[Array] | None = None
    _rho_t: list[Array] | None = None
    scaled_tau: bool = False

    def __iter__(self):
        for i in range(len(self.sp)):
            d = dict(
                sti=self.sti,
                time_series=self.time_series,
                tau=self.tau,
                ground_bias=self.ground_bias,
                collective_variable=self.collective_variable,
                scaled_tau=self.scaled_tau,
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
        # assert (
        #     self.collective_variable == other.collective_variable
        # ), "dlo cannot be added because the collective variables are different"
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
    def get_histo(
        data_nums: list[CV],
        weights: None | list[Array] = None,
        log_w=False,
        macro_chunk=1000,
        verbose=False,
        shape_mask=None,
        nn=-1,
    ):
        def f_func(x, nl):
            return x

        if shape_mask is not None:
            nn = jnp.sum(shape_mask) + 1  # +1 for bin -1

        nn_range = jnp.arange(nn)

        if shape_mask is not None:
            nn_range = nn_range.at[-1].set(-1)

        # @jax.jit
        def _get_histo(x, cvi: CV, _, weights, weights_t):
            alpha_factors, h = x

            cvi = cvi.cv[:, 0]

            if not log_w:
                wi = jnp.ones_like(cvi) if weights is None else weights
                h = h.at[cvi].add(wi)

                return (alpha_factors, h)

            if alpha_factors is None:
                alpha_factors = jnp.full_like(h, -jnp.inf)

            # this log of hist. all calculations are done in log space to avoid numerical issues
            w_log = jnp.zeros_like(cvi) if weights is None else weights

            @partial(jax.vmap, in_axes=(0, None, None, 0))
            def get_max(ref_num, data_num, log_w, alpha):
                val = jnp.where(data_num == ref_num, log_w, -jnp.inf)

                max_new = jnp.max(val)
                max_tot = jnp.max(jnp.array([alpha, max_new]))

                return max_tot, max_new > alpha, max_tot - alpha

            alpha_factors, alpha_factors_changed, d_alpha = get_max(nn_range, cvi, w_log, alpha_factors)

            # shift all probs by the change in alpha
            h = jnp.where(alpha_factors_changed, h - d_alpha, h)

            m = alpha_factors != -jnp.inf

            p_new = jnp.zeros_like(h)
            p_new = p_new.at[cvi].add(jnp.exp(w_log - alpha_factors[cvi]))

            h = jnp.where(m, jnp.log(jnp.exp(h) + p_new), h)

            return (alpha_factors, h)

        h = jnp.zeros((nn,))

        alpha_factors, h = macro_chunk_map(
            f=f_func,
            op=data_nums[0].stack,
            y=data_nums,
            macro_chunk=macro_chunk,
            verbose=verbose,
            chunk_func=_get_histo,
            chunk_func_init_args=(None, h),
            w=weights,
            jit_f=True,
        )

        if log_w:
            h += alpha_factors

        if shape_mask is not None:
            # print(f"{h[-1]=}")
            h = h[:-1]

        return h

    @staticmethod
    def _histogram(
        metric: CvMetric,
        n_grid=40,
        grid_bounds=None,
        chunk_size=None,
        chunk_size_mid=1,
    ):
        bins, _, cv_mid, _ = metric.grid(n=n_grid, bounds=grid_bounds)

        @partial(CvTrans.from_cv_function, mid=cv_mid)
        def closest_trans(cv: CV, _nl, _, shmap, shmap_kwargs, mid: CV):
            m = jnp.argmin(jnp.sum((mid.cv - cv.cv) ** 2, axis=1), keepdims=True)

            return cv.replace(cv=m)

        nums = closest_trans.compute_cv_trans(cv_mid, chunk_size=chunk_size)[0].cv

        return cv_mid, nums, bins, closest_trans, partial(data_loader_output.get_histo, nn=cv_mid.shape[0])

    @staticmethod
    def _unstack_weights(stack_dims, weights: Array) -> list[Array]:
        return [
            d.cv.reshape((-1,))
            for d in CV(
                cv=jnp.expand_dims(weights, 1),
                _stack_dims=stack_dims,
            ).unstack()
        ]

    @staticmethod
    def norm_w(w_stacked, ret_n=False):
        n = jnp.sum(w_stacked)

        if ret_n:
            return w_stacked / n, n

        return w_stacked / n

    @staticmethod
    def check_w(w_stacked):
        if jnp.any(jnp.isnan(w_stacked)):
            print(f"WARNING: w_stacked has {jnp.sum(jnp.isnan(w_stacked))} nan values")
            w_stacked = jnp.where(jnp.isnan(w_stacked), 0, w_stacked)

        if jnp.any(jnp.isinf(w_stacked)):
            print(f"WARNING: w_stacked has {jnp.sum(jnp.isinf(w_stacked))} inf values")
            w_stacked = jnp.where(jnp.isinf(w_stacked), 0, w_stacked)

        if jnp.any(w_stacked < 0):
            print(f"WARNING: w_stacked has {jnp.sum(w_stacked<0)} neg values")
            w_stacked = jnp.where(w_stacked < 0, 0, w_stacked)

        if jnp.sum(w_stacked) < 1e-16:
            print("WARNING: all w_Stacked values are zero")
            raise

        if len(w_stacked) == 0:
            print("WARNING: len w_Stacked is zero")

        return w_stacked

    def koopman_weight(
        self,
        w=None,
        samples_per_bin=2,
        n_max_koopman=40,
        out_dim=-1,
        chunk_size=None,
        indicator_CV=True,
        koopman_eps=1e-10,
        koopman_eps_pre=1e-10,
        cv_0: list[CV] | None = None,
        cv_t: list[CV] | None = None,
        macro_chunk=1000,
        verbose=False,
        max_features_koopman=5000,
        margin=0.1,
        add_1=False,
        only_diag=False,
        calc_pi=False,
        sparse=False,
        output_w_corr=False,
        correlation=True,
        return_km=False,
    ):
        if cv_0 is None:
            cv_0 = self.cv

        sd = [a.shape[0] for a in cv_0]
        tot_samples = sum(sd)

        ndim = cv_0[0].shape[1]

        if cv_t is None:
            cv_t = self.cv_t

        def unstack_w(w_stacked, stack_dims=None):
            if stack_dims is None:
                stack_dims = sd
            return self._unstack_weights(stack_dims, w_stacked)

        if w is None:
            assert self._weights is not None
            w = self._weights

        assert self._weights_t is not None

        # print("using precomputed weights")
        # w_stacked = jnp.hstack(self._weights)
        # w_stacked = self.check_w(w_stacked)
        # w_stacked = self.norm_w(w_stacked)

        # if self.time_series:
        #     w_stacked_t = jnp.hstack(self._weights_t)
        #     w_stacked_t = self.check_w(w_stacked_t)
        #     w_stacked_t = self.norm_w(w_stacked_t)

        n_hist = CvMetric.get_n(
            samples_per_bin=samples_per_bin,
            samples=tot_samples,
            n_dims=ndim,
        )

        if n_hist > n_max_koopman:
            n_hist = n_max_koopman

        if verbose:
            print(f"using {n_hist=}")

        if verbose:
            print("koopman weights")

        if indicator_CV:
            print("getting bounds")
            grid_bounds, _, constants = CvMetric.bounds_from_cv(
                cv_0,
                margin=margin,
                chunk_size=chunk_size,
                n=20,
            )

            if constants:
                print("not performing koopman weighing because of constants in cv")
                # koopman = False
                skip = True
            else:
                cv_mid, nums, bins, closest, get_histo = data_loader_output._histogram(
                    metric=self.collective_variable.metric,
                    n_grid=n_hist,
                    grid_bounds=grid_bounds,
                    chunk_size=chunk_size,
                )

                grid_nums, grid_nums_t = self.apply_cv(
                    closest, cv_0, cv_t, chunk_size=chunk_size, macro_chunk=macro_chunk
                )

                w_pos = [(a > 0) * 1.0 for a in w]

                hist = get_histo(grid_nums, w_pos)

                hist_t = get_histo(grid_nums_t, w_pos)

                mask = jnp.argwhere(jnp.logical_and(hist > 0, hist_t > 0)).reshape(-1)

                @partial(CvTrans.from_cv_function, mask=mask)
                def get_indicator(cv: CV, nl, _, shmap, shmap_kwargs, mask):
                    out = jnp.zeros((hist.shape[0],))
                    out = out.at[cv.cv].set(1)
                    out = jnp.take(out, mask)

                    return cv.replace(cv=out)

                cv_km = grid_nums
                cv_km_t = grid_nums_t

                tr = get_indicator

        else:
            cv_km = cv_0
            cv_km_t = cv_t

            tr = None

        if verbose:
            print("constructing koopman model")

        km = self.koopman_model(
            cv_0=cv_km,
            cv_tau=cv_km_t,
            nl=None,
            w=w,
            # w_t=self._weights_t,
            rho=self._rho,
            # rho_t=self._rho_t,
            add_1=add_1,
            method="tcca",
            symmetric=False,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            # out_dim=None,
            verbose=verbose,
            eps=koopman_eps,
            eps_pre=koopman_eps_pre,
            trans=tr,
            max_features=max_features_koopman,
            max_features_pre=max_features_koopman,
            only_diag=only_diag,
            calc_pi=calc_pi,
            scaled_tau=self.scaled_tau,
            sparse=sparse,
            out_dim=out_dim,
            correlation=correlation,
        )

        w_unstacked, w_corr, b = km.koopman_weight(
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            verbose=verbose,
        )

        if not b:
            print("koopman reweighing failed")

        w_stacked = jnp.hstack(w_unstacked)
        w_stacked = self.check_w(w_stacked)
        w_stacked = self.norm_w(w_stacked)

        out = [self._unstack_weights(sd, w_stacked)]

        if output_w_corr:
            w_corr_stacked = jnp.hstack(w_corr)
            w_corr_stacked = self.check_w(w_corr_stacked)
            w_corr_stacked = self.norm_w(w_corr_stacked)

            out.append(self._unstack_weights(sd, w_corr_stacked))

        if return_km:
            out.append(km)

        return out

    def dhamed_weight(
        self,
        samples_per_bin=5,
        n_max=30,
        chunk_size=None,
        wham_eps=1e-7,
        cv_0: list[CV] | None = None,
        cv_t: list[CV] | None = None,
        macro_chunk=1000,
        verbose=False,
        margin=0.1,
        bias_cutoff=100 * kjmol,  # bigger values can lead to nans, about 10^-17
        # min_f_i=1e-30,
        log_sum_exp=True,
        return_bias=False,
    ):
        if cv_0 is None:
            cv_0 = self.cv

        if cv_t is None:
            cv_t = self.cv_t

        # https://pubs.acs.org/doi/full/10.1021/acs.jctc.7b00373

        u_unstacked = []
        beta = 1 / (self.sti.T * boltzmann)

        # u_masks = []

        for ti_i in self.ti:
            e = ti_i.e_bias

            if e is None:
                if verbose:
                    print("WARNING: no bias enerrgy found")
                e = jnp.zeros((ti_i.sp.shape[0],))

            u = beta * e  # - u0

            u_unstacked.append(u)

        sd = [a.shape[0] for a in cv_0]
        tot_samples = sum(sd)

        ndim = cv_0[0].shape[1]

        def unstack_w(w_stacked, stack_dims=None):
            if stack_dims is None:
                stack_dims = sd
            return self._unstack_weights(stack_dims, w_stacked)

        # prepare histo

        n_hist = CvMetric.get_n(
            samples_per_bin=samples_per_bin,
            samples=tot_samples,
            n_dims=ndim,
        )

        if n_hist > n_max:
            n_hist = n_max

        if verbose:
            print(f"using {n_hist=}")

        print("getting bounds")
        grid_bounds, _, constants = CvMetric.bounds_from_cv(
            cv_0,
            margin=margin,
            chunk_size=chunk_size,
            n=20,
        )

        print("getting histo")
        cv_mid, nums, bins, closest, get_histo = data_loader_output._histogram(
            metric=self.collective_variable.metric,
            n_grid=n_hist,
            grid_bounds=grid_bounds,
            chunk_size=chunk_size,
        )

        # assert self.bias is not None

        if verbose:
            print("step 1 dham")

        grid_nums, _ = self.apply_cv(
            closest,
            cv_0,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
        )

        if verbose:
            print("getting histo")

        log_hist = get_histo(
            grid_nums,
            None,
            log_w=True,
            verbose=True,
            # macro_chunk=macro_chunk,
        )

        hist_mask = log_hist > jnp.log(5)
        print(f"{jnp.sum(hist_mask)=}")

        # get histos and expectations values e^-beta U_a

        len_a = len(u_unstacked)
        len_i = int(jnp.sum(hist_mask))

        log_b_ai = jnp.full((len_a, len_i), -jnp.inf)

        for a in range(len_a):
            if verbose:
                print(".", end="", flush=True)
                if (a + 1) % 100 == 0:
                    print("")
            log_hist_a_weights = get_histo(
                [grid_nums[a]],
                [-u_unstacked[a]],
                log_w=True,
            )[hist_mask]

            log_hist_a_num = get_histo(
                [grid_nums[a]],
                None,
                log_w=True,
            )[hist_mask]

            # e^(-beta U'_alpha) =< e^(-beta U_alpha) delta_i > _alpha
            log_b_a = jnp.where(
                log_hist_a_weights == -jnp.inf,
                -jnp.inf,
                log_hist_a_num - log_hist_a_weights,
            )
            log_b_ai = log_b_ai.at[a, :].set(log_b_a)

        idx = jnp.arange(hist_mask.shape[0])[hist_mask]
        idx_inv = jnp.zeros(hist_mask.shape[0], dtype=jnp.int_)
        idx_inv = idx_inv.at[hist_mask].set(jnp.arange(jnp.sum(hist_mask)))

        # find transitions counts, make sparse arrays

        list_unique = []
        list_counts = []
        list_unique_all = []

        diag_unique = []
        diag_counts = []

        for a, g_i in enumerate(grid_nums):
            u, c = jnp.unique(
                jnp.hstack([idx_inv[g_i.cv[1:]], idx_inv[g_i.cv[:-1]]]),
                axis=0,
                return_counts=True,
            )

            list_unique_all.append(jnp.unique(u))

            diag = u[:, 0] == u[:, 1]

            diag_vals = c[diag]

            diag_unique.append(jnp.hstack([jnp.full((diag_vals.shape[0], 1), a), u[diag, 0].reshape(-1, 1)]))
            diag_counts.append(diag_vals)

            c = c.at[diag].set(0)

            # print(u)

            u = jnp.hstack([jnp.full((u.shape[0], 1), a), u])

            list_unique.append(u)
            list_counts.append(c)

        list_unique = jnp.vstack(list_unique)
        list_counts = jnp.hstack(list_counts)

        diag_unique = jnp.vstack(diag_unique)
        diag_counts = jnp.hstack(diag_counts)

        from jax.experimental.sparse import BCOO, sparsify

        N_a_ik = BCOO((list_counts, list_unique), shape=(len_a, len_i, len_i))
        # t_a_k: sum_i  N_a_ik
        # in prev, diqg elems are removed
        t_a_k = sparsify(lambda x: jnp.sum(x, axis=(1)))(N_a_ik) + BCOO(
            (diag_counts, diag_unique), shape=(len_a, len_i)
        )

        _counts_out = sparsify(lambda x: jnp.sum(x, axis=(0, 1)))(N_a_ik).todense()
        _counts_in = sparsify(lambda x: jnp.sum(x, axis=(0, 2)))(N_a_ik).todense()

        log_denom_k = jnp.log(_counts_out)

        max_bins_per_run = max([a.shape[0] for a in list_unique_all])
        print(f"{max_bins_per_run=}")

        # cahnge dat structure. instead of storing all i and k, ze store used s and tm per md run a

        bin_list = jnp.full((len_a, max_bins_per_run), -1)
        log_bias_a_t = jnp.full((len_a, max_bins_per_run), -jnp.inf)

        t_a_t = jnp.full((len_a, max_bins_per_run), -1)
        N_a_st = jnp.full((len_a, max_bins_per_run, max_bins_per_run), -1)

        @jax.jit
        def upd(bin_list, log_bias_a_t, t_a_t, N_a_st, a, n):
            bin_list = bin_list.at[a, : len(n)].set(n)
            log_bias_a_t = log_bias_a_t.at[a, : len(n)].set(log_b_ai[a, n])
            t_a_t = t_a_t.at[a, : len(n)].set(t_a_k[a, n].todense())

            N_a_st = N_a_st.at[a, : len(n), : len(n)].set(N_a_ik[a, n, :][:, n].todense())

            return bin_list, log_bias_a_t, t_a_t, N_a_st

        for a, n in enumerate(list_unique_all):
            print(".", flush=True, end="")

            bin_list, log_bias_a_t, t_a_t, N_a_st = upd(bin_list, log_bias_a_t, t_a_t, N_a_st, a, n)

        # to sum over all i != k, ze need to create index

        @jax.vmap
        def _idx_k(k):
            return jnp.sum(bin_list == k)

        max_k_in_runs = jnp.max(_idx_k(jnp.arange(len_i)))

        assert max_k_in_runs <= len_a

        print(f"{max_k_in_runs=}")

        @jax.vmap
        def _arg_k(k):
            return jnp.argwhere(bin_list == k, size=max_k_in_runs, fill_value=-1)

        k_pos = _arg_k(jnp.arange(len_i))

        # define fixed point fun

        @jax.jit
        def T(log_p):
            @partial(jax.vmap, in_axes=0, out_axes=0)  # a
            @partial(jax.vmap, in_axes=(0, 1, None, 0, None, 0, None, 0), out_axes=0)  # s
            @partial(jax.vmap, in_axes=(0, 0, 0, None, 0, None, 0, None), out_axes=0)  # t
            def _nom(N_st, N_ts, t_t, t_s, u_t, u_s, log_p_t, log_p_s):
                use = jnp.logical_and(N_ts != -1, N_st != -1)

                nom = jnp.log(N_st + N_ts) + u_t + jnp.log(t_t)

                m = jnp.max(jnp.array([u_t - log_p_t, u_s - log_p_s]))

                m = jnp.where(m == -jnp.inf, 0, m)

                denom = jnp.log(t_t * jnp.exp(u_t - log_p_t - m) + t_s * jnp.exp(u_s - log_p_s - m)) + m

                return jnp.where(use, nom - denom, -jnp.inf)

            @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
            def _log_p(log_p, bin_list_a):
                return log_p[bin_list_a]

            log_p_t = _log_p(log_p, bin_list)

            log_nom_a_st = _nom(
                N_a_st,
                N_a_st,
                t_a_t,
                t_a_t,
                log_bias_a_t,
                log_bias_a_t,
                log_p_t,
                log_p_t,
            )

            @partial(jax.vmap, in_axes=0, out_axes=0)
            @partial(jax.vmap, in_axes=1, out_axes=0)  # sum over s (i) axis
            def _out_a_t(out_s):
                m = jnp.nanmax(out_s)

                m = jnp.where(m == -jnp.inf, 0, m)
                return jnp.log(jnp.nansum(jnp.exp(out_s - m))) + m

            log_nom_a_t = _out_a_t(log_nom_a_st)
            # log_denom_a_t = _out_a_t(log_denom_a_st)

            @partial(jax.vmap, in_axes=(0, None))
            def _get_k(k_pos, arr):
                @jax.vmap
                def _get_k_n(k_pos_n):
                    b = jnp.all(k_pos_n == jnp.array([-1, -1]))
                    return jnp.where(b, -jnp.inf, arr[k_pos_n[0], k_pos_n[1]])

                log_out = _get_k_n(k_pos)

                m = jnp.nanmax(log_out)
                m = jnp.where(m == -jnp.inf, 0, m)

                return jnp.log(jnp.nansum(jnp.exp(log_out - m))) + m

            log_nom_k = _get_k(k_pos, log_nom_a_t)

            log_p = log_nom_k - log_denom_k

            m = jnp.max(log_p)
            log_p -= jnp.log(jnp.sum(jnp.exp(log_p - m))) + m

            return log_p

        # get fixed point
        import jaxopt

        solver = jaxopt.FixedPointIteration(
            fixed_point_fun=T,
            maxiter=100,
            tol=wham_eps,
        )

        print("solving dham")

        log_p = jnp.log(jnp.ones((len_i)) / len_i)

        # with jax.debug_nans():
        out = solver.run(log_p)

        print("dham done")

        log_p = out.params

        # if verbose:
        print(f" {out.state.iter_num=} {out.state.error=} {log_p=}")

        # get per sample weights
        @jax.jit
        def w_k(log_p):
            @partial(jax.vmap, in_axes=0, out_axes=0)  # a
            @partial(jax.vmap, in_axes=(0, 1, None, 0, None, 0, None, 0), out_axes=0)  # s
            @partial(jax.vmap, in_axes=(0, 0, 0, None, 0, None, 0, None), out_axes=0)  # t
            def _nom(N_st, N_ts, t_t, t_s, u_t, u_s, log_p_t, log_p_s):
                use = jnp.logical_and(N_ts != -1, N_st != -1)

                nom = jnp.log(N_st + N_ts) + u_t  # ommited, t_t is count and e^{beta U} is

                m = jnp.max(jnp.array([u_t - log_p_t, u_s - log_p_s]))

                m = jnp.where(m == -jnp.inf, 0, m)

                denom = jnp.log(t_t * jnp.exp(u_t - log_p_t - m) + t_s * jnp.exp(u_s - log_p_s - m)) + m

                return jnp.where(use, nom - denom, -jnp.inf)

            @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
            def _log_p(log_p, bin_list_a):
                return log_p[bin_list_a]

            log_p_t = _log_p(log_p, bin_list)

            log_nom_a_st = _nom(
                N_a_st,
                N_a_st,
                t_a_t,
                t_a_t,
                log_bias_a_t,
                log_bias_a_t,
                log_p_t,
                log_p_t,
            )

            @partial(jax.vmap, in_axes=0)
            @partial(jax.vmap, in_axes=1)  # sum over s (i) axis
            def _out_a_t(out_s):
                m = jnp.nanmax(out_s)

                m = jnp.where(m == -jnp.inf, 0, m)
                return jnp.log(jnp.nansum(jnp.exp(out_s - m))) + m

            log_nom_a_t = _out_a_t(log_nom_a_st)

            @partial(jax.vmap, in_axes=(0, 0))
            def _out_a_k(log_nom_a_t, bin_t):
                x = jnp.full((len_i,), -jnp.inf)
                x = x.at[bin_t].set(log_nom_a_t)
                x -= log_denom_k

                return x

            return _out_a_k(log_nom_a_t, bin_list)

        x_a_k = w_k(log_p)
        out_weights = []

        s = 0

        for a in range(len_a):
            grid_nums_a = grid_nums[a]

            w = jnp.exp(x_a_k[a, idx_inv[grid_nums_a.cv]]).reshape(-1)

            s += jnp.sum(w)

            out_weights.append(w)

        out_weights = [oi / s for oi in out_weights]

        if not return_bias:
            return out_weights

        # divide by grid spacing?
        fes_grid = -log_p / beta
        fes_grid -= jnp.min(fes_grid)

        print("computing rbf")

        from IMLCV.implementations.bias import RbfBias

        bias = RbfBias.create(
            cvs=self.collective_variable,
            cv=cv_mid[hist_mask],
            kernel="multiquadric",
            vals=-fes_grid,
            # epsilon=1 / (0.815 * jnp.array([b[1] - b[0] for b in bins])),
        )

        return out_weights, bias

    def wham_weight(
        self,
        samples_per_bin=10,
        n_max=100,
        chunk_size=None,
        wham_eps=1e-7,
        cv_0: list[CV] | None = None,
        cv_t: list[CV] | None = None,
        macro_chunk=1000,
        verbose=False,
        margin=0.1,
        bias_cutoff=100 * kjmol,  # bigger values can lead to nans, about 10^-17
        min_f_i=1e-30,
        log_sum_exp=True,
        return_bias=False,
        return_std_bias=False,
        min_samples=3,
        lagrangian=True,
        max_sigma=5 * kjmol,
        sparse_inverse=True,
        inverse_sigma_weighting=False,
        output_bincount=True,
        output_free_energy=True,
        output_time_scaling=True,
        smooth_bias=True,
        max_bias_margin=0.1,
    ) -> WeightOutput:
        if cv_0 is None:
            cv_0 = self.cv

        if cv_t is None:
            cv_t = self.cv_t

        # TODO:https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00867

        # get raw rescaling
        u_unstacked = []
        beta = 1 / (self.sti.T * boltzmann)

        for ti_i in self.ti:
            e = ti_i.e_bias

            if e is None:
                if verbose:
                    print("WARNING: no bias enerrgy found")
                e = jnp.zeros((ti_i.sp.shape[0],))

            u = beta * e  # - u0

            u_unstacked.append(u)

        sd = [a.shape[0] for a in cv_0]
        tot_samples = sum(sd)

        ndim = cv_0[0].shape[1]

        def unstack_w(w_stacked, stack_dims=None):
            if stack_dims is None:
                stack_dims = sd
            return self._unstack_weights(stack_dims, w_stacked)

        # prepare histo

        n_hist = CvMetric.get_n(
            samples_per_bin=samples_per_bin,
            samples=tot_samples,
            n_dims=ndim,
        )

        if n_hist > n_max:
            n_hist = n_max

        if verbose:
            print(f"using {n_hist=}")

        print("getting bounds")
        grid_bounds, _, constants = CvMetric.bounds_from_cv(
            cv_0,
            margin=margin,
            chunk_size=chunk_size,
            n=20,
        )

        print(f"{grid_bounds=}")

        print("getting histo")
        cv_mid, nums, bins, closest, get_histo = data_loader_output._histogram(
            metric=self.collective_variable.metric,
            n_grid=n_hist,
            grid_bounds=grid_bounds,
            chunk_size=chunk_size,
        )

        # assert self.bias is not None

        if verbose:
            print("step 1 wham")

        grid_nums, _ = self.apply_cv(
            closest,
            cv_0,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
        )

        if verbose:
            print("getting histo")

        log_hist = get_histo(
            grid_nums,
            None,
            log_w=True,
            # verbose=True,
        )

        # print(f"{jnp.exp(hist)=}")
        hist_mask = log_hist >= jnp.log(min_samples)

        n_hist_mask = jnp.sum(hist_mask)
        print(f"{n_hist_mask=} {jnp.sum( jnp.logical_and(hist_mask< jnp.log(min_samples), log_hist != -jnp.inf))=}")

        # lookup to convert grid num to new grid num wihtout empty bins
        idx_inv = jnp.full(hist_mask.shape[0], -1)
        idx_inv = idx_inv.at[hist_mask].set(jnp.arange(jnp.sum(hist_mask)))

        # print(f"{idx_inv=}")

        grid_nums_mask = [g.replace(cv=idx_inv[g.cv]) for g in grid_nums]

        # print(f"{grid_nums_mask=}")

        m_log_b_ik = jnp.full((len(u_unstacked), jnp.sum(hist_mask)), -jnp.inf)

        H_k = jnp.zeros((jnp.sum(hist_mask),))

        N_i = []

        len_i = len(u_unstacked)

        for i in range(len_i):
            if verbose:
                print(".", end="", flush=True)
                if (i + 1) % 100 == 0:
                    print("")
            # log sum exp(-u_i)
            log_hist_i_weights = get_histo(
                [grid_nums_mask[i]],
                [u_unstacked[i]],
                log_w=True,
                shape_mask=hist_mask,
            )

            log_hist_i_num = get_histo(
                [grid_nums_mask[i]],
                None,
                log_w=True,
                shape_mask=hist_mask,
            )

            # mean of e^(-beta U)
            log_b_k = jnp.where(
                log_hist_i_weights == -jnp.inf,
                -jnp.inf,
                log_hist_i_num - log_hist_i_weights,
            )

            H_k += jnp.exp(log_hist_i_num)
            N_i.append(jnp.sum(jnp.exp(log_hist_i_num)))

            m_log_b_ik = m_log_b_ik.at[i, :].set(log_b_k)

            # samples_in_bin.append(
            #     jnp.where(
            #         grid_nums_mask[i].cv.reshape((-1)) == -1, 0, log_hist_i_num[grid_nums_mask[i].cv.reshape((-1))]
            #     )
            # )

        print(f"{jnp.max(m_log_b_ik)=}")

        log_H_k = jnp.log(H_k)

        a_k = jnp.ones((m_log_b_ik.shape[1],))
        a_k /= jnp.sum(a_k)

        N_i = jnp.array(N_i)

        log_N_tot = jnp.log(jnp.sum(N_i))

        log_a_k = jnp.log(a_k)
        log_N_i = jnp.log(N_i)

        if log_sum_exp:
            # m_log_b_ik = jnp.log(b_ik)

            print(f"{jnp.min(m_log_b_ik)=} {jnp.max(m_log_b_ik)=}")

            def log_sum_exp_safe(*x):
                x = jnp.sum(jnp.stack(x, axis=0), axis=0)

                x_max = jnp.max(x)

                e = jnp.sum(jnp.exp(x - x_max))

                # safegaurd is needed to prevent infities in error for kl divergence
                # e_safe = jnp.where(e < jnp.log(min_f_i), jnp.log(min_f_i), e)

                return x_max + jnp.log(e)

            @jax.jit
            def T(log_a_k, log_x):
                m_log_b_ik, log_N_i, log_H_k = log_x

                # get log f_i
                log_f_i = -vmap(log_sum_exp_safe, in_axes=(None, 0))(log_a_k, m_log_b_ik)

                # get a_k from f_i
                log_a_k = log_H_k - vmap(log_sum_exp_safe, in_axes=(None, None, 1))(log_N_i, log_f_i, m_log_b_ik)

                # norm a_k
                log_a_k_norm = log_sum_exp_safe(log_a_k)

                return log_a_k - log_a_k_norm

            @jax.jit
            def norm(log_a_k, log_x):
                log_a_k_p = T(log_a_k, log_x)

                return 0.5 * jnp.sum((jnp.exp(log_a_k) - jnp.exp(log_a_k_p)) ** 2)

            @jax.jit
            def kl_div(log_a_k, log_x):
                log_a_k_p = T(log_a_k, log_x)

                return jnp.sum(jnp.exp(log_a_k) * (log_a_k - log_a_k_p))

            import jaxopt

            @jax.jit
            def A(log_a_k, log_x):
                m_log_b_ik, log_N_i, log_H_k = log_x

                return jnp.sum(
                    jnp.exp(log_N_i - log_N_tot) * vmap(log_sum_exp_safe, in_axes=(None, 0))(log_a_k, m_log_b_ik)
                ) - jnp.sum(jnp.exp(log_H_k - log_N_tot) * log_a_k)

            if lagrangian:
                solver = jaxopt.LBFGS(
                    fun=A,
                    tol=1e-12,
                    implicit_diff=True,
                    maxiter=10000,
                )
                # print("solving wham")

            else:
                # every action is performed in log space. For summation in real space, the largest log value is subtracted from all values. The norm is then calculated in real space

                solver = jaxopt.AndersonAcceleration(
                    fixed_point_fun=T,
                    history_size=5,
                    tol=1e-5,
                    implicit_diff=True,
                    maxiter=10000,
                )

            print("solving wham")
            log_x = (m_log_b_ik, log_N_i, log_H_k)

            out = solver.run(log_a_k, log_x=log_x)

            print(f"wham done! {out.state.error=}")

            print("solved")

            log_a_k = out.params
            log_a_k -= log_sum_exp_safe(log_a_k)  # not normed for lagrangian approach

            log_f_i = -vmap(log_sum_exp_safe, in_axes=(None, 0))(log_a_k, m_log_b_ik)

            if verbose:
                n, k = norm(log_a_k, log_x), kl_div(log_a_k, log_x)
                print(f"wham err={n}, kl divergence={k} {out.state.iter_num=} {out.state.error=} ")

        else:
            b_ik = jnp.exp(m_log_b_ik)

            @jax.jit
            def T(a_k, x):
                b_ik, N_i, H_k = x

                f_inv_i = jnp.einsum("k,ik->i", a_k, b_ik)
                f_i_inv_safe = jnp.where(f_inv_i < min_f_i, min_f_i, f_inv_i)
                a_k_new_inv = jnp.einsum("i,ik->k", N_i / f_i_inv_safe, b_ik)

                a_k_new_inv_safe = jnp.where(a_k_new_inv < min_f_i, min_f_i, a_k_new_inv)
                a_k_new = H_k / a_k_new_inv_safe
                a_k_new = a_k_new / jnp.sum(a_k_new)

                return a_k_new, 1 / f_i_inv_safe

            def norm(a_k, x):
                a_k_p = T(a_k, x)[0]

                if log_sum_exp:
                    a_k = jnp.exp(a_k)
                    a_k_p = jnp.exp(a_k_p)

                return 0.5 * jnp.sum((a_k - a_k_p) ** 2) / jnp.sum(a_k > 0)

            def kl_div(a_k, x):
                a_k_p, f = T(a_k, x)

                a_k = jnp.where(a_k >= min_f_i, a_k, min_f_i)
                a_k_p = jnp.where(a_k_p >= min_f_i, a_k_p, min_f_i)

                return jnp.sum(a_k * (jnp.log(a_k) - jnp.log(a_k_p)))

            import jaxopt

            solver = jaxopt.ProjectedGradient(
                fun=norm,
                projection=jaxopt.projection.projection_simplex,  # prob space is simplex
                maxiter=2000,
                tol=wham_eps,
            )

            print("solving wham")

            out = solver.run(a_k, x=(b_ik, N_i, H_k))

            print("wham done")

            a_k = out.params

            if verbose:
                n, k = norm(a_k, (b_ik, N_i)), kl_div(a_k, (b_ik, N_i))
                print(f"wham err={n}, kl divergence={k} {out.state.iter_num=} {out.state.error=} ")

            _, f = T(a_k, (b_ik, N_i))

            log_f_i = jnp.log(f)
            log_a_k = jnp.log(a_k)

        if return_std_bias or inverse_sigma_weighting or smooth_bias:
            nk = log_a_k.shape[0]
            ni = log_N_i.shape[0]

            # see thermolib derivation

            print(f"test: {nk=} {ni=}")

            def _s(*x):
                return jnp.sum(jnp.stack(x, axis=0), axis=0)

            A = [
                [
                    jnp.diag(
                        jax.vmap(
                            lambda log_b_i, log_a: jnp.sum(jnp.exp(log_a + log_N_i + log_f_i + log_b_i)),
                            in_axes=(1, 0),
                        )(m_log_b_ik, log_a_k)
                    ),
                    -jnp.exp(
                        jax.vmap(
                            jax.vmap(_s, in_axes=(0, None, None, 0)),  # k
                            in_axes=(None, 0, 0, 0),  # i
                        )(log_a_k, log_N_i, log_f_i, m_log_b_ik)
                    ).T,
                    -jnp.exp(
                        jax.vmap(
                            jax.vmap(_s, in_axes=(0, None, None, 0)),  # k
                            in_axes=(None, 0, 0, 0),  # i
                        )(log_a_k, log_N_i, log_f_i, m_log_b_ik)
                    ).T,
                    -jnp.exp(log_a_k + log_N_tot).reshape((nk, 1)),
                ],
                [
                    -jnp.exp(
                        jax.vmap(
                            jax.vmap(
                                _s,
                                in_axes=(None, 0, 0, 0),
                            ),  # i
                            in_axes=(0, None, None, 1),  # k
                        )(log_a_k, log_N_i, log_f_i, m_log_b_ik)
                    ).T,
                    jnp.diag(jnp.exp(log_N_i)),
                    jnp.diag(jnp.exp(log_N_i)),
                    jnp.zeros((ni, 1)),
                ],
                [
                    -jnp.exp(
                        jax.vmap(
                            jax.vmap(
                                _s,
                                in_axes=(None, 0, 0, 0),
                            ),  # i
                            in_axes=(0, None, None, 1),  # k
                        )(log_a_k, log_N_i, log_f_i, m_log_b_ik)
                    ).T,
                    jnp.diag(jnp.exp(log_N_i)),
                    jnp.zeros((ni, ni)),
                    jnp.zeros((ni, 1)),
                ],
                [
                    -jnp.exp(log_a_k + log_N_tot).reshape((1, nk)),
                    jnp.zeros((1, ni)),
                    jnp.zeros((1, ni)),
                    jnp.zeros((1, 1)),
                ],
            ]

            F = jnp.block(A)

            if sparse_inverse:
                from scipy.sparse import csc_matrix

                F_sparse = csc_matrix((F.__array__()).__array__())

                from scipy.sparse.linalg import splu

                B = splu(
                    F_sparse,
                    permc_spec="NATURAL",
                )

                print(f"{B.shape=} {B.nnz=}")

                Bx = B.solve(
                    jnp.eye(F.shape[0]).__array__(),
                )  # only interested in the first nk + ni columns

                cov_sigma = jnp.diag(jnp.array(Bx))

            else:
                l, V = jnp.linalg.eigh(F)

                l_inv = jnp.where(jnp.abs(l) < 1e-12, 0, 1 / l)

                # print(f"{l=}")

                cov_sigma = jnp.diag(V @ jnp.diag(l_inv) @ V.T)

            # print(f"{cov_sigma=}")

            sigma_ak = cov_sigma[:nk]
            sigma_fi = cov_sigma[nk : nk + ni]

            # print(f"{sigma_ak=}")

            sigma_ak = jnp.where(sigma_ak < 0, 0, jnp.sqrt(sigma_ak))
            sigma_fi = jnp.where(sigma_fi < 0, 0, jnp.sqrt(sigma_fi))

            print(f"{jnp.mean(sigma_ak/beta/kjmol)=} {jnp.max(sigma_ak/beta/kjmol)=}  ")
            print(f"{jnp.mean(sigma_fi/beta/kjmol)=}")

        log_denom_k = -vmap(log_sum_exp_safe, in_axes=(None, None, 1))(log_N_i, log_f_i, m_log_b_ik)
        log_denom_k = jnp.hstack([log_denom_k, jnp.array([-jnp.inf])])
        log_H_k_ext = jnp.hstack([log_H_k, jnp.array([-jnp.inf])])

        # out_weights = []

        # if output_free_energy:
        #     free_energy = []
        #     log_a_k_ext = jnp.hstack([log_a_k, jnp.array([-jnp.inf])])

        if output_time_scaling:
            time_scaling = []

        samples_in_bin = []

        # # only integratesof bins k if part of simulation i
        @partial(jax.vmap, in_axes=(None, 0, 0))  # i
        def _s(log_a_k, log_f_i, m_log_b_ik):
            @jax.vmap  # k
            def _s_i(log_a_k, m_log_b_ik):
                log_nom = log_a_k + m_log_b_ik

                log_denom = jnp.where(m_log_b_ik == -jnp.inf, -jnp.inf, log_a_k)

                return log_nom, log_denom

            log_nom, log_denom = _s_i(log_a_k, m_log_b_ik)

            m = jnp.max(log_nom)
            log_nom = jnp.log(jnp.sum(jnp.exp(log_nom - m))) + m

            m = jnp.max(log_denom)
            log_denom = jnp.log(jnp.sum(jnp.exp(log_denom - m))) + m

            return log_nom - log_denom

        log_c_i = -_s(log_a_k, log_f_i, m_log_b_ik)

        s = 0

        # w_ind = []

        w_out = []
        p_select_out = []

        for i, (_u_i, _log_f_i) in enumerate(zip(u_unstacked, log_f_i)):
            gi = grid_nums_mask[i].cv.reshape(-1)

            w = jnp.exp(_u_i - _log_f_i)

            p_select = jnp.exp(m_log_b_ik[i, gi] + _log_f_i + log_denom_k[gi])

            # selection criterium should be high enough to prevent numerical issues
            w = jnp.where(p_select <= 1e-16, 0, w)

            w_out.append(w)
            p_select_out.append(p_select)

            s += jnp.sum(w * p_select)

            samples_in_bin.append(jnp.exp(log_H_k_ext[gi]))

            if output_time_scaling:
                time_scaling.append(jnp.exp(-(_u_i - log_c_i[i])))
                # time_scaling.append(1 / w)

        # if output_free_energy:
        #     free_energy.append(jnp.exp(log_a_k_ext[gi]))

        output_weight_kwargs = {
            "weights": w_out,
            "p_select": p_select_out,
            "grid_nums": grid_nums,
        }

        output_weight_kwargs["bin_counts"] = samples_in_bin
        # output_weight_kwargs["n_bins"] = log_a_k.shape[0]

        if output_time_scaling:
            output_weight_kwargs["time_scaling"] = time_scaling

        print(f"{s=}")

        # if output_free_energy:
        #     output_weight_kwargs["free_energy"] = free_energy

        if return_bias or output_free_energy:
            # divide by grid spacing?
            fes_grid = -log_a_k / beta
            # fes_grid -= jnp.min(fes_grid)
            # fes_grid = -fes_grid

            cv_grid = cv_mid[hist_mask]

            if (nn := jnp.sum(jnp.isnan(fes_grid))) != 0:
                print(f" found {nn=} nans in  ")

                raise

            if (nn := jnp.sum(jnp.isinf(fes_grid))) != 0:
                print(f" found {nn=} infs in  ")

                raise

            if smooth_bias:
                if (nn := jnp.sum(jnp.isnan(sigma_ak))) != 0:
                    print(f" found {nn=} nans in sigma_ak")

                    raise

                if (nn := jnp.sum(jnp.isinf(sigma_ak))) != 0:
                    print(f" found {nn=} infs in sigma_ak")

                    raise

            fes_grid -= jnp.min(fes_grid)

            # mask = fes_grid > bias_cutoff

            # if jnp.sum(mask) > 0:
            #     print(f"found {jnp.sum(mask)} bins with bias above cutoff")
            #     cv_grid = cv_grid[mask]
            #     fes_grid = fes_grid[mask]

            print("computing rbf, including smoothing")

            bias = RbfBias.create(
                cvs=self.collective_variable,
                cv=cv_grid,
                kernel="multiquadric",
                vals=-fes_grid,
                epsilon=1 / (0.815 * jnp.array([b[1] - b[0] for b in bins])),
                # smoothing=(sigma_ak / beta) ** 2 if smooth_bias else 0,
            )

            if bias_cutoff is None:
                max_bias = jnp.max(fes_grid) * (1 + max_bias_margin)
            else:
                max_bias = bias_cutoff

            bias = BiasModify.create(
                bias=bias,
                fun=_clip,
                kwargs={"a_min": -max_bias, "a_max": 0.0},
            )

            output_weight_kwargs["FES_bias"] = bias

        # if output_free_energy:
        #     free_energies = bias.apply(
        #         cv_0,
        #     )

        #     output_weight_kwargs["free_energy"] = [
        #         jnp.exp(beta * f) for f in free_energies
        #     ]

        if return_std_bias:
            print("log_exp_slice")

            sigma_bias = RbfBias.create(
                cvs=self.collective_variable,
                cv=cv_mid[hist_mask],
                kernel="multiquadric",
                vals=sigma_ak / beta,
                slice_exponent=2.0,
                log_exp_slice=False,
                slice_mean=True,
                epsilon=1 / (0.815 * jnp.array([b[1] - b[0] for b in bins])),
            )

            output_weight_kwargs["FES_bias_std"] = sigma_bias

        return WeightOutput(**output_weight_kwargs)

    def get_bincount(
        self,
        cv_0: list[CV] | None = None,
        n_hist=40,
        margin=0.1,
        chunk_size=None,
        use_w=False,
    ):
        if cv_0 is None:
            cv_0 = self.cv

        grid_bounds, _, constants = CvMetric.bounds_from_cv(
            cv_0,
            margin=margin,
            chunk_size=chunk_size,
            n=20,
        )

        if constants:
            print("not performing koopman weighing because of constants in cv")
            # koopman = False
            return None

        cv_mid, nums, bins, closest, get_histo = data_loader_output._histogram(
            metric=self.collective_variable.metric,
            n_grid=n_hist,
            grid_bounds=grid_bounds,
            chunk_size=chunk_size,
        )

        grid_nums, _ = self.apply_cv(
            closest,
            cv_0,
            chunk_size=chunk_size,
            macro_chunk=1000,
        )

        if use_w:
            w = self._weights
        else:
            w = None

        hist = get_histo(grid_nums, w)

        bin_counts = [hist[i.cv].reshape((-1,)) for i in grid_nums]

        return bin_counts

    @staticmethod
    def _transform(
        cv,
        nl,
        _,
        shmap,
        shmap_kwargs,
        argmask: jnp.Array | None = None,
        pi: jnp.Array | None = None,
        add_1: bool = False,
        add_1_pre: bool = False,
        q: jnp.Array | None = None,
        l: jnp.Array | None = None,
    ):
        x = cv.cv

        if argmask is not None:
            x = x[argmask]

        if pi is not None:
            x = x - pi

        if add_1_pre:
            x = jnp.hstack([x, jnp.array([1])])

        if q is not None:
            x = x @ q

        if l is not None:
            x = x * l

        if add_1:
            x = jnp.hstack([x, jnp.array([1])])

        return cv.replace(cv=x, _combine_dims=None)

    def koopman_model(
        self,
        cv_0: list[CV] | None = None,
        cv_tau: list[CV] | None = None,
        nl: NeighbourList | None = None,
        nl_t: NeighbourList | None = None,
        method="tcca",
        koopman_weight=False,
        only_return_weights=False,
        symmetric=False,
        rho: list[jax.Array] | None = None,
        # rho_t: list[jax.Array] | None = None,
        w: list[jax.Array] | None = None,
        # w_t: list[jax.Array] | None = None,
        eps=1e-12,
        eps_pre=1e-12,
        max_features=5000,
        max_features_pre=5000,
        out_dim=-1,
        add_1=False,
        chunk_size=None,
        macro_chunk=1000,
        verbose=False,
        trans=None,
        T_scale=1,
        only_diag=False,
        calc_pi=True,
        scaled_tau=None,
        sparse=True,
        correlation=False,
    ) -> KoopmanModel:
        # TODO: https://www.mdpi.com/2079-3197/6/1/22

        assert method in ["tica", "tcca"]

        if scaled_tau is None:
            scaled_tau = self.scaled_tau

        if cv_0 is None:
            cv_0 = self.cv

        if cv_tau is None:
            cv_tau = self.cv_t

        if rho is None:
            if self._rho is not None:
                rho = self._rho
            else:
                print("W koopman model not given, will use uniform weights")

                t = sum([cvi.shape[0] for cvi in cv_0])
                rho = [jnp.ones((cvi.shape[0],)) / t for cvi in cv_0]

        # if rho_t is None:
        #     if self._rho_t is not None:
        #         rho_t = self._rho_t
        #     else:
        #         print("W_t koopman model not given, will use w")

        #         rho_t = rho

        if w is None:
            if self._weights is not None:
                w = self._weights
            else:
                print("W koopman model not given, will use uniform weights")

                t = sum([cvi.shape[0] for cvi in cv_0])
                w = [jnp.ones((cvi.shape[0],)) / t for cvi in cv_0]

        # if w_t is None:
        #     if self._weights_t is not None:
        #         w_t = self._weights_t
        #     else:
        #         print("W_t koopman model not given, will use w")

        #         w_t = w

        return KoopmanModel.create(
            w=w,
            # w_t=w_t,
            rho=rho,
            # rho_t=rho_t,
            cv_0=cv_0,
            cv_tau=cv_tau,
            nl=nl,
            add_1=add_1,
            eps=eps,
            eps_pre=eps_pre,
            method=method,
            symmetric=symmetric,
            out_dim=out_dim,
            koopman_weight=koopman_weight,
            max_features=max_features,
            max_features_pre=max_features_pre,
            tau=self.tau,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            verbose=verbose,
            trans=trans,
            T_scale=T_scale,
            only_diag=only_diag,
            calc_pi=calc_pi,
            scaled_tau=scaled_tau,
            sparse=sparse,
            only_return_weights=only_return_weights,
            correlation_whiten=correlation,
        )

    def filter_nans(
        self,
        x: list[CV] | None = None,
        x_t: list[CV] | None = None,
        macro_chunk=1000,
    ) -> tuple[data_loader_output, CV, CV]:
        if x is None:
            x = self.cv

        if x_t is None:
            x_t = self.cv_t

        def _check_nan(x: CV, nl, cv, shmap):
            return x.replace(cv=jnp.isnan(x.cv) | jnp.isinf(x.cv))

        check_nan = CvTrans.from_cv_function(_check_nan)

        nan_x, nan_x_t = self.apply_cv(
            check_nan,
            x,
            x_t,
            macro_chunk=macro_chunk,
            shmap=False,
            verbose=True,
        )

        nan = CV.stack(*nan_x).cv

        if nan_x_t is not None:
            nan = jnp.logical_or(nan, CV.stack(*nan_x_t).cv)

        assert not jnp.any(nan), f"found {jnp.sum(nan)}/{len(nan)} nans or infs in the cv data"

        if jnp.any(nan):
            raise

    @staticmethod
    def apply_cv(
        # self,
        f: CvTrans | CvFlow,
        x: list[CV] | list[SystemParams] | None = None,
        x_t: list[CV] | list[SystemParams] | None = None,
        nl=None,
        nl_t=None,
        chunk_size=None,
        macro_chunk=1000,
        shmap=True,
        shmap_kwargs=ShmapKwargs.create(),
        verbose=False,
        print_every=10,
        jit_f=True,
    ) -> tuple[list[CV], list[CV] | None]:
        # if isinstance(f, CvTrans):
        #     _f = f.compute_cv_trans
        # elif isinstance(f, CvFlow):
        #     _f = f.compute_cv_flow
        # else:
        #     raise ValueError(f"{f=} must be CvTrans or CvFlow")
        _f = f.compute_cv

        def f(x, nl):
            return _f(
                x,
                nl,
                chunk_size=chunk_size,
                shmap=False,
            )[0]

        if shmap:
            f = jax.jit(f)
            f = padded_shard_map(f, kwargs=shmap_kwargs)  # (pmap=True))

        return data_loader_output._apply(
            x=x,
            x_t=x_t,
            nl=nl,
            nl_t=nl_t,
            f=f,
            macro_chunk=macro_chunk,
            verbose=verbose,
            jit_f=not shmap and jit_f,
            print_every=print_every,
        )

    @staticmethod
    def _apply(
        x: CV | SystemParams,
        f: Callable,
        x_t: CV | SystemParams | None = None,
        nl: NeighbourList | NeighbourListUpdate | None = None,
        nl_t: NeighbourList | NeighbourListUpdate | None = None,
        macro_chunk=1000,
        verbose=False,
        jit_f=True,
        print_every=10,
    ):
        print("inside _apply")
        out = macro_chunk_map(
            f=f,
            op=x[0].__class__.stack,
            y=x,
            y_t=x_t,
            nl=nl,
            nl_t=nl_t,
            macro_chunk=macro_chunk,
            verbose=verbose,
            jit_f=jit_f,
            print_every=print_every,
        )
        print("outside _apply")
        # jax.debug.inspect_array_sharding(out[0], callback=print)

        if x_t is not None:
            z, z_t = out
        else:
            z = out
            z_t = None

        for i in range(len(z)):
            assert (
                z[i].shape[0] == x[i].shape[0]
            ), f" shapes do not match {[zi.shape[0] for  zi in z ]} != {[xi.shape[0] for xi in x]}"

            if x_t is not None:
                assert z[i].shape[0] == z_t[i].shape[0]

        return z, z_t

    @staticmethod
    def _apply_bias(
        x: list[CV],
        bias: Bias,
        chunk_size=None,
        macro_chunk=1000,
        verbose=False,
        shmap=False,
        shmap_kwargs=ShmapKwargs.create(),
    ) -> list[Bias]:
        def f(x, _):
            b = bias.compute_from_cv(
                x,
                chunk_size=chunk_size,
                shmap=False,
            )[0]

            return x.replace(cv=b.reshape((-1, 1)))

        if shmap:
            # TODO 0909 09:41:06.108200 3141823 collective_ops_utils.h:306] This thread has been waiting for 5000ms for and may be stuck: participant AllReduceParticipantData{rank=27, element_count=1, type=PRED, rendezvous_key=RendezvousKey{run_id=RunId: 5299332, global_devices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], num_local_participants=32, collective_op_kind=cross_module, op_id=3}} waiting for all participants to arrive at rendezvous RendezvousKey{run_id=RunId: 5299332, global_devices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], num_local_participants=32, collective_op_kind=cross_module, op_id=3}
            f = jax.jit(f)
            f = padded_shard_map(f, kwargs=shmap_kwargs)  # (pmap=True))

        out, _ = data_loader_output._apply(
            x=x,
            f=f,
            macro_chunk=macro_chunk,
            verbose=verbose,
            jit_f=not shmap,
        )

        return [a.cv[:, 0] for a in out]

    def apply_bias(self, bias: Bias, chunk_size=None, macro_chunk=1000, verbose=False, shmap=False) -> list[Bias]:
        return data_loader_output._apply_bias(
            x=self.cv,
            bias=bias,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            verbose=verbose,
            shmap=shmap,
        )

    def get_point(
        self,
        point: CV,
        key=jax.random.PRNGKey(0),
    ) -> tuple[CV, SystemParams, NeighbourList | None, TrajectoryInfo]:
        import jax.numpy as jnp

        from IMLCV.implementations.bias import HarmonicBias

        center_bias = HarmonicBias.create(cvs=self.collective_variable, k=50.0, q0=point)
        center = self.apply_bias(
            bias=center_bias,
        )

        p = jnp.exp(-jnp.hstack(center) / (300 * kjmol))
        p /= jnp.sum(p)

        n = []

        for i, x in enumerate(self.cv):
            n.append(jnp.stack([jnp.full(x.shape[0], i), jnp.arange(x.shape[0])], axis=1))

        n = jnp.vstack(n)

        idx = jax.random.choice(key, a=n, p=p, shape=(), replace=True)

        return (
            self.cv[idx[0]][idx[1]],
            self.sp[idx[0]][idx[1]],
            self.nl[idx[0]][idx[1]] if self.nl is not None else self.sti.neighbour_list_info,
            self.ti[idx[0]],
        )

    @staticmethod
    def get_fes_bias_from_weights(
        T,
        weights: list[jax.Array],
        rho: list[jax.Array],
        collective_variable: CollectiveVariable,
        cv: list[CV],
        samples_per_bin=5,
        min_samples_per_bin: int | None = 1,
        n_max=40,
        n_grid=None,
        max_bias=None,
        chunk_size=None,
        macro_chunk=1000,
        max_bias_margin=0.2,
        rbf_bias=True,
        kernel="multiquadric",
    ) -> RbfBias:
        beta = 1 / (T * boltzmann)

        samples = sum([cvi.shape[0] for cvi in cv])

        if n_grid is None:
            n_grid = CvMetric.get_n(
                samples_per_bin=samples_per_bin,
                samples=samples,
                n_dims=cv[0].shape[1],
            )

        if n_grid > n_max:
            # print(f"reducing n_grid from {n_grid} to {n_max}")
            n_grid = n_max
        # else:
        #     print(f"{n_grid=}")

        if n_grid <= 1:
            raise

        grid_bounds, _, _ = CvMetric.bounds_from_cv(
            cv,
            # weights=weights,
            margin=0.2,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
            # percentile=1e-10,
        )

        # # do not update periodic bounds

        grid_bounds = jax.vmap(
            lambda x, y: jnp.where(collective_variable.metric.periodicities, x, y),
            in_axes=(1, 1),
            out_axes=1,
        )(
            collective_variable.metric.bounding_box,
            grid_bounds,
        )

        cv_mid, nums, bins, closest, get_histo = data_loader_output._histogram(
            n_grid=n_grid,
            grid_bounds=grid_bounds,
            metric=collective_variable.metric,
        )

        print(f"{cv_mid.shape=}")

        grid_nums, _ = data_loader_output.apply_cv(
            x=cv,
            f=closest,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
        )

        w_rho_grid = get_histo(
            grid_nums,
            [wi * rho for wi, rho in zip(weights, rho)],
            macro_chunk=macro_chunk,
        )

        # w_rho = get_histo(
        #     grid_nums,
        #     rho,
        #     macro_chunk=macro_chunk,
        # )

        mask = w_rho_grid > 0

        print(f"{w_rho_grid.shape=} {jnp.sum(mask)=}")

        p_grid = w_rho_grid[mask]  # / w_rho[mask]

        # rho_grid = get_histo(
        #     grid_nums,
        #     rho,
        #     macro_chunk=macro_chunk,
        # )

        # mask = jnp.logical_and(rho_grid > 0, w_rho_grid > 0)

        # print(f"{w_rho_grid.shape=} {rho_grid.shape=} {jnp.sum(mask)=}")

        # p_grid = w_rho_grid[mask] / rho_grid[mask]

        # p_grid = jnp.where( rho_grid >= 1e-16, w_rho_grid / rho_grid, 0)

        # mask = jnp.logical_and(p_grid > 0, n_grid >= min_samples_per_bin)

        # # print(f"{p_grid=}  {p_grid[p_grid<0]} {jnp.sum(p_grid>0)} ")

        # p_grid = p_grid[mask]

        fes_grid = -jnp.log(p_grid) / beta
        fes_grid -= jnp.min(fes_grid)

        print(f"{cv_mid[mask].shape=}")

        if rbf_bias:
            bias = RbfBias.create(
                cvs=collective_variable,
                cv=cv_mid[mask],
                kernel=kernel,
                vals=-fes_grid,
                epsilon=1 / (0.815 * jnp.array([b[1] - b[0] for b in bins])),
            )
        else:
            vals = jnp.full((n_grid,) * collective_variable.n, jnp.nan)
            vals = vals.at[mask].set(-fes_grid)
            raise NotImplementedError
            bias = GridBias()

        if max_bias is None:
            max_bias = jnp.max(fes_grid) * (1 + max_bias_margin)

        return BiasModify.create(
            bias=bias,
            fun=_clip,
            kwargs={"a_min": -max_bias, "a_max": 0.0},
        )

    def get_transformed_fes(
        self,
        new_cv: list[CV],
        new_colvar: CollectiveVariable,
        samples_per_bin=5,
        min_samples_per_bin: int = 1,
        chunk_size=1,
        smoothing=0.0,
        max_bias=None,
        shmap=False,
        n_grid_old=50,
        n_grid_new=30,
    ) -> RbfBias:
        old_cv = CV.stack(*self.cv)
        new_cv = CV.stack(*new_cv)

        # get bins for new CV
        grid_bounds_new, _, _ = CvMetric.bounds_from_cv(new_cv, margin=0.1)
        cv_mid_new, nums_new, _, closest_new, get_histo_new = data_loader_output._histogram(
            n_grid=n_grid_new,
            grid_bounds=grid_bounds_new,
            metric=new_colvar.metric,
        )
        grid_nums_new = closest_new.compute_cv_trans(new_cv, chunk_size=chunk_size)[0].cv

        # get bins for old CV
        grid_bounds_old, _, _ = CvMetric.bounds_from_cv(old_cv, margin=0.1)
        cv_mid_old, nums_old, _, closest_old, get_histo_old = data_loader_output._histogram(
            n_grid=n_grid_old,
            grid_bounds=grid_bounds_old,
            metric=self.collective_variable.metric,
        )
        grid_nums_old = closest_old.compute_cv_trans(old_cv, chunk_size=chunk_size)[0].cv

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

        prob = partial(prob, grid_nums_new=grid_nums_new, nums_old=nums_old, p_grid_old=p_grid_old)
        prob = padded_vmap(prob, chunk_size=chunk_size)

        if shmap:
            prob = padded_shard_map(prob)

        num_grid_new, p_grid_new = prob(nums_new)

        # mask = num_grid_new >= min_samples_per_bin

        # print(f"{jnp.sum(mask)}/{mask.shape[0]} bins with samples")

        # p_grid_new = p_grid_new  # [mask]

        fes_grid = -jnp.log(p_grid_new) / beta
        fes_grid -= jnp.min(fes_grid)

        bias = RbfBias.create(
            cvs=new_colvar,
            cv=cv_mid_new,  # [mask],
            kernel="multiquadric",
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

        _, cv_grid, _, _ = self.collective_variable.metric.grid(n=n_grid)
        new_cv_grid, _, log_det = trans.compute_cv_trans(cv_grid, log_Jf=True)

        FES_bias_vals, _ = self.ground_bias.compute_from_cv(cv_grid)

        new_FES_bias_vals = FES_bias_vals + T * boltzmann * log_det
        new_FES_bias_vals -= jnp.max(new_FES_bias_vals)

        weight = jnp.exp(new_FES_bias_vals / (T * boltzmann))
        weight /= jnp.sum(weight)

        bounds, _, _ = self.collective_variable.metric.bounds_from_cv(
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
            kernel="multiquadric",
        )

        if max_bias is None:
            max_bias = -jnp.min(new_FES_bias_vals)

        return BiasModify.create(
            bias=new_FES_bias,
            fun=_clip,
            kwargs={"a_min": -max_bias, "a_max": 0.0},
        )

    def recalc(self, chunk_size=None, macro_chunk=1000, shmap=False, verbose=False):
        x, x_t = self.apply_cv(
            self.collective_variable.f,
            self.cv,
            self.cv_t,
            self.nl,
            self.nl_t,
            chunk_size=chunk_size,
            shmap=shmap,
            macro_chunk=macro_chunk,
            verbose=verbose,
        )

        self.cv = x
        self.cv_t = x_t

    def calc_neighbours(
        self,
        r_cut,
        chunk_size=None,
        macro_chunk=1000,
        verbose=False,
        only_update=False,
        chunk_size_inner=10,
        max=(2, 2, 2),
    ):
        if self.time_series:
            y = [*self.sp, *self.sp_t]
        else:
            y = [*self.sp]

        from molmod.units import angstrom

        nl_info = NeighbourListInfo.create(
            r_cut=r_cut,
            r_skin=0 * angstrom,
            z_array=self.sti.atomic_numbers,
        )

        @partial(jax.jit, static_argnames=["update"])
        def _f(sp, update):
            return sp._get_neighbour_list(
                info=nl_info,
                chunk_size=chunk_size,
                chunk_size_inner=chunk_size_inner,
                shmap=False,
                only_update=only_update,
                update=update,
            )

        if only_update:
            # it might be possible that some neihgbour lsit are too small
            # remove those trjaectories

            def f(nl_update: NeighbourListUpdate | None, sp, *_):
                if nl_update is None:
                    b, nn, n_xyz, _ = sp._get_neighbour_list(
                        info=nl_info,
                        chunk_size=chunk_size,
                        chunk_size_inner=chunk_size_inner,
                        shmap=False,
                        only_update=only_update,
                    )

                    return NeighbourListUpdate.create(
                        nxyz=n_xyz,
                        num_neighs=nn,
                        stack_dims=None,
                    )

                b, new_nn, new_xyz, _ = _f(sp, nl_update)

                if not jnp.all(b):
                    n_xyz, nn = nl_update.nxyz, nl_update.num_neighs

                    nl_update = NeighbourListUpdate.create(
                        nxyz=tuple([max(a, int(b)) for a, b in zip(n_xyz, new_xyz)]),
                        num_neighs=max(nn, int(new_nn)),
                        stack_dims=None,
                    )

                    print(f"updating neighbour list {nl_update=}")

                return nl_update

            nl_update = macro_chunk_map(
                f=lambda x, y: x,
                op=SystemParams.stack,
                y=y,
                nl=None,
                macro_chunk=macro_chunk,
                verbose=verbose,
                jit_f=False,
                chunk_func=f,
            )

            self.nl = NeighbourList(
                info=nl_info,
                update=nl_update,
                sp_orig=None,
            )

            self.nl_t = self.nl if self.time_series else None

            return

        def f(sp: SystemParams, nl=None):
            b, _, _, nl = sp._get_neighbour_list(
                info=nl_info,
                chunk_size=chunk_size,
                chunk_size_inner=chunk_size_inner,
                shmap=False,
                only_update=only_update,
            )

            assert jnp.all(b)

            return nl

        nl = macro_chunk_map(
            f=f,
            op=SystemParams.stack,
            y=y,
            nl=None,
            macro_chunk=macro_chunk,
            verbose=verbose,
            jit_f=True,
        )

        if self.time_series:
            nl, nl_t = nl[0 : len(self.sp)], nl[len(self.sp) :]
        else:
            nl_t = None

        self.nl = nl
        self.nl_t = nl_t


@partial(dataclass, frozen=False, eq=False)
class KoopmanModel:
    s: jax.Array

    cov: Covariances
    W0: jax.Array
    W1: jax.Array

    # U: jax.Array
    # VT: jax.Array

    # argmask_0: jax.Array | None
    # argmask_1: jax.Array | None
    argmask: jax.Array | None

    shape: int

    cv_0: list[CV]
    cv_tau: list[CV]
    nl: list[NeighbourList] | NeighbourList | None
    nl_t: list[NeighbourList] | NeighbourList | None

    w: list[jax.Array] | None = None
    # w_t: list[jax.Array] | None = None

    rho: list[jax.Array] | None = None
    # rho_t: list[jax.Array] | None = None

    eps: float = 1e-10
    eps_pre: float = 1e-10

    only_diag: bool = False
    calc_pi: bool = True

    scaled_tau: bool = False

    add_1: bool = False
    max_features: int = 5000
    max_features_pre: int = 5000
    out_dim: int | None = None
    method: str = "tcca"
    correlation_whiten: bool = True

    tau: float | None = None
    T_scale: float = 1.0

    trans: CvTrans | CvFlow | None = None

    @staticmethod
    def create(
        w: list[jax.Array] | None,
        # w_t: list[jax.Array] | None,
        rho: list[jax.Array] | None,
        # rho_t: list[jax.Array] | None,
        cv_0: list[CV],
        cv_tau: list[CV],
        nl: list[NeighbourList] | NeighbourList | None = None,
        nl_t: list[NeighbourList] | NeighbourList | None = None,
        add_1=True,
        eps=1e-14,
        eps_pre=1e-14,
        method="tcca",
        symmetric=False,
        out_dim=-1,  # maximum dimension for koopman model
        koopman_weight=False,
        max_features=5000,
        max_features_pre=5000,
        tau=None,
        macro_chunk=1000,
        chunk_size=None,
        verbose=False,
        trans: CvTrans | None = None,
        T_scale=1,
        only_diag=False,
        calc_pi=True,
        use_scipy=False,
        auto_cov_threshold=None,
        sparse=True,
        # n_modes=10,
        scaled_tau=False,
        only_return_weights=False,
        correlation_whiten=False,
        out_eps=None,
    ):
        #  see Optimal Data-Driven Estimation of Generalized Markov State Models
        if verbose:
            print("getting covariances")

        print(f"{out_dim=}")

        cov = Covariances.create(
            cv_0=cv_0,
            cv_1=cv_tau,
            nl=nl,
            nl_t=nl_t,
            w=[wi * rhoi for wi, rhoi in zip(w, rho)],
            w_t=[wi * rhoi for wi, rhoi in zip(w, rho)],
            calc_pi=calc_pi or add_1,
            only_diag=only_diag,
            symmetric=symmetric and not koopman_weight,
            T_scale=T_scale,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            trans_f=trans,
            trans_g=trans,
        )

        if not calc_pi and add_1:
            cov.C00 += jnp.outer(cov.pi_0, cov.pi_0)
            cov.C11 += jnp.outer(cov.pi_1, cov.pi_1)
            cov.C01 += jnp.outer(cov.pi_0, cov.pi_1)
            cov.C10 += jnp.outer(cov.pi_1, cov.pi_0)

        argmask = jnp.arange(cov.C00.shape[0])

        if eps_pre is not None:
            argmask_pre = jnp.logical_and(
                jnp.diag(cov.C00) / jnp.max(jnp.diag(cov.C00)) > eps_pre**2,
                jnp.diag(cov.C11) / jnp.max(jnp.diag(cov.C11)) > eps_pre**2,
            )

            if verbose:
                print(f"{jnp.sum(argmask_pre)=} {jnp.sum(~argmask_pre)=} ")

            if jnp.sum(argmask_pre) == 0:
                print(
                    f"WARNING: no modes selectected through argmask pre {jnp.diag(cov.C00)/ jnp.max(jnp.diag(cov.C00))=} {jnp.diag(cov.C11)/ jnp.max(jnp.diag(cov.C11))=}"
                )

            cov.C00 = cov.C00[argmask_pre, :][:, argmask_pre]
            cov.C11 = cov.C11[argmask_pre, :][:, argmask_pre]
            cov.C01 = cov.C01[argmask_pre, :][:, argmask_pre]
            cov.C10 = cov.C10[argmask_pre, :][:, argmask_pre]

            if calc_pi or add_1:
                cov.pi_0 = cov.pi_0[argmask_pre]
                cov.pi_1 = cov.pi_1[argmask_pre]

            argmask = argmask[argmask_pre]

        # start with argmask for auto covariance. Remove all features with variances that are too small, or auto covariance that are too small
        if auto_cov_threshold is not None:
            auto_cov = jnp.einsum(
                "i,i,i->i",
                jnp.diag(cov.C00) ** (-0.5),
                jnp.diag(cov.C01),
                jnp.diag(cov.C11) ** (-0.5),
            )
            argmask_cov = jnp.argsort(auto_cov, descending=True).reshape(-1)

            argmask_cov = argmask_cov[auto_cov[argmask_cov] > auto_cov_threshold]

            if max_features_pre is not None:
                if argmask_cov.shape[0] > max_features_pre:
                    argmask_cov = argmask_cov[:max_features_pre]
                    print(f"reducing argmask_cov to {max_features_pre}")
            else:
                print(f"{argmask_cov.shape=}")

            cov.C00 = cov.C00[argmask_cov, :][:, argmask_cov]
            cov.C11 = cov.C11[argmask_cov, :][:, argmask_cov]
            cov.C01 = cov.C01[argmask_cov, :][:, argmask_cov]
            cov.C10 = cov.C10[argmask_cov, :][:, argmask_cov]

            if calc_pi or add_1:
                cov.pi_0 = cov.pi_0[argmask_cov]
                cov.pi_1 = cov.pi_1[argmask_cov]

            argmask = argmask[argmask_cov]

        print(f"{argmask.shape=}")

        # if add_1:
        if add_1:
            print("adding 1 to T_tilde")

            C10 = jnp.block(
                [
                    [cov.C10, jnp.zeros((cov.C10.shape[1], 1)) if calc_pi else cov.pi_1.reshape((-1, 1))],
                    [jnp.zeros((1, cov.C10.shape[0])) if calc_pi else cov.pi_0.reshape((1, -1)), jnp.array([1])],
                ]
            )

            C00 = jnp.block(
                [
                    [cov.C00, jnp.zeros((cov.C00.shape[1], 1)) if calc_pi else cov.pi_0.reshape((-1, 1))],
                    [jnp.zeros((1, cov.C00.shape[0])) if calc_pi else cov.pi_0.reshape((1, -1)), jnp.array([1])],
                ]
            )

            if not koopman_weight:
                C01 = jnp.block(
                    [
                        [cov.C01, cov.pi_0.reshape((-1, 1)) if not calc_pi else jnp.zeros((cov.C01.shape[1], 1))],
                        [
                            cov.pi_1.reshape((1, -1)) if not calc_pi else jnp.zeros((1, cov.C01.shape[0])),
                            jnp.array([1]),
                        ],
                    ]
                )

                C11 = jnp.block(
                    [
                        [cov.C11, cov.pi_1.reshape((-1, 1)) if not calc_pi else jnp.zeros((cov.C11.shape[1], 1))],
                        [
                            cov.pi_1.reshape((1, -1)) if not calc_pi else jnp.zeros((1, cov.C11.shape[0])),
                            jnp.array([1]),
                        ],
                    ]
                )

            cov.C10 = C10
            cov.C00 = C00

            if not koopman_weight:
                cov.C01 = C01
                cov.C11 = C11

            if not calc_pi:
                cov.pi_0 = None
                cov.pi_1 = None

        if verbose:
            print("diagonalizing C00")

        W0 = cov.whiten(
            "C00",
            epsilon=eps,
            epsilon_pre=eps_pre,
            verbose=verbose,
            use_scipy=use_scipy,
            correlation=correlation_whiten,
        )

        if verbose:
            print(f"{W0.shape=}")

        if verbose:
            print("diagonalizing C11")

        if not koopman_weight:
            if symmetric:
                W1 = W0
                # argmask_1 = argmask_0
                # _trans_g = _trans_f

            else:
                W1 = cov.whiten(
                    "C11",
                    epsilon=eps,
                    epsilon_pre=eps_pre,
                    verbose=verbose,
                    use_scipy=use_scipy,
                    correlation=correlation_whiten,
                )
                if verbose:
                    print(f"{W1.shape=}")

        print("reweighing")

        # if koopman_weight:
        #     KT = W0 @ cov.C10 @ W0.T

        #     from scipy.sparse.linalg import eigs

        #     e, v = eigs(KT.__array__(), k=3)
        #     print(f"koopman reweighing {e=}")

        #     e, v = jnp.array(e), jnp.array(v)

        #     if (p := jnp.sum(jnp.abs(e - 1) < 1e-10)) > 1:
        #         m = jnp.abs(e - 1) < 1e-10
        #         e = e[m]
        #         v = v[:, m]

        #         if p == 2 and add_1:
        #             print("found 2 eigenvalues close to 1, picking the one that does not relate to the added 1")
        #             f0 = v[-1, 0]
        #             f1 = v[-1, 1]

        #             n = f1 - f0  # norm such that eigenvalue sums to 1

        #             f0 /= n
        #             f1 /= n

        #             v = f1 * v[:, [0]] - f0 * v[:, [1]]
        #             e = f1 * e[0] - f0 * e[1]

        #             # print(f"{v=}")

        #         else:
        #             print(f"found {p} eigenvalues close to 1, picking one at random")
        #             print("TODO: implement maximum entropy selection")
        #             e = e[0]
        #             v = v[:, 0]
        #     else:
        #         idx = jnp.argmin(jnp.abs(e - 1))

        #         e = e[idx]
        #         v = v[:, [idx]]

        #     if jnp.abs(e - 1) > 1e-10:
        #         print(f"found eigenvalue {e} with {jnp.abs(e-1)=}, not reweighing")

        #         if only_return_weights:
        #             return w, None

        #     else:

        #         def w_transform(
        #             cv,
        #             nl,
        #             _,
        #             shmap,
        #             shmap_kwargs,
        #             argmask: jnp.Array | None = None,
        #             pi: jnp.Array | None = None,
        #             add_1: bool = False,
        #             q: jnp.Array | None = None,
        #         ):
        #             x = cv.cv

        #             if argmask is not None:
        #                 x = x[argmask]

        #             if pi is not None:
        #                 x = x - pi

        #             if add_1:
        #                 x = jnp.hstack([x, jnp.array([1])])

        #             if q is not None:
        #                 x = x @ q

        #             # print(f"{x.shape=}")

        #             return cv.replace(cv=x, _combine_dims=None)

        #         w_trans = CvTrans.from_cv_function(
        #             w_transform,
        #             static_argnames=["add_1"],
        #             add_1=add_1,
        #             q=W0.T @ v,
        #             pi=cov.pi_0,
        #             argmask=argmask,
        #         )

        #         w_corr_cv, _ = data_loader_output.apply_cv(
        #             f=w_trans,
        #             x=cv_0,
        #             x_t=None,
        #             nl_t=None,
        #             macro_chunk=macro_chunk,
        #             chunk_size=chunk_size,
        #             verbose=verbose,
        #         )

        #         w_corr = []
        #         w_new = []

        #         sign = None

        #         s_corr = 0
        #         s_new = 0

        #         n_neg = 0

        #         for x, wi in zip(w_corr_cv, w):
        #             if sign is None:
        #                 sign = jnp.sign(jnp.sum(jnp.real(x.cv)))

        #             x = jnp.real(x.cv) * sign
        #             x = x.reshape(-1)

        #             m = x <= 0
        #             n_neg += jnp.sum(m)
        #             x = jnp.where(m, 0, x)

        #             s_corr += jnp.sum(x)
        #             w_corr.append(x)

        #             y = wi * x
        #             s_new += jnp.sum(y)
        #             w_new.append(y)

        #         w_corr = [x / s_corr for x in w_corr]
        #         w_new = [x / s_new for x in w_new]

        #         print(f"corrected weights, recomputing model {w_corr[0].shape=} {w_new[0].shape=} ({n_neg=})")

        #         if only_return_weights:
        #             return w_new, w_corr

        #         return KoopmanModel.create(
        #             w=w_new,
        #             # w_t=w_new,
        #             rho=rho,
        #             # rho_t=rho,
        #             cv_0=cv_0,
        #             cv_tau=cv_tau,
        #             nl=nl,
        #             nl_t=nl_t,
        #             add_1=add_1,
        #             eps=eps,
        #             method=method,
        #             symmetric=symmetric,
        #             koopman_weight=False,
        #             out_dim=out_dim,
        #             max_features=max_features,
        #             max_features_pre=max_features_pre,
        #             tau=tau,
        #             macro_chunk=macro_chunk,
        #             chunk_size=chunk_size,
        #             verbose=verbose,
        #             trans=trans,
        #             T_scale=T_scale,
        #             only_diag=only_diag,
        #             calc_pi=calc_pi,
        #             use_scipy=use_scipy,
        #             auto_cov_threshold=auto_cov_threshold,
        #             sparse=sparse,
        #             # n_modes=n_modes,
        #             scaled_tau=scaled_tau,
        #             only_return_weights=only_return_weights,
        #         )

        T_tilde = W1 @ cov.C10 @ W0.T

        if verbose:
            print("koopman': SVD")

        if out_dim is None:
            out_dim = 10

        if out_dim == -1:
            out_dim = T_tilde.shape[0]

        n_modes = out_dim

        k = min(n_modes, T_tilde.shape[0] - 1)
        k = min(n_modes, T_tilde.shape[1] - 1)

        if n_modes + 1 < T_tilde.shape[0] / 5 and sparse:
            from jax.experimental.sparse.linalg import lobpcg_standard
            from jax.random import PRNGKey, uniform

            x0 = uniform(PRNGKey(0), (T_tilde.shape[0], k))

            print(f"using lobpcg with {n_modes} modes ")

            if symmetric and W0.shape[0] == W1.shape[0]:
                s, U, n_iter = lobpcg_standard(
                    T_tilde.T,
                    x0,
                    m=200,
                )

                print(f"{n_iter=} {s=}")

                VT = U.T

            else:
                l, V, n_iter = lobpcg_standard(
                    T_tilde @ T_tilde.T,
                    x0,
                    m=200,
                )

                VT = V.T

                print(n_iter)

                s = l ** (1 / 2)
                s_inv = jnp.where(s > 1e-10, 1 / s, 0)
                U = T_tilde.T @ VT.T @ jnp.diag(s_inv)

        else:
            print(f"using svd with {n_modes} modes")

            if symmetric and W0.shape[0] == W1.shape[0]:
                s, U = jax.numpy.linalg.eigh(
                    T_tilde.T,
                )
                VT = U.T
            else:
                U, s, VT = jax.numpy.linalg.svd(T_tilde.T)

        idx = jnp.argsort(s, descending=True)
        U = U[:, idx]
        s = s[idx]
        VT = VT[idx, :]

        # W0 and W1 are whitening transforms. A whitening transform is still whitening after a rotation
        W0 = U.T @ W0
        W1 = VT @ W1  # V is already transposed

        if out_eps is not None:
            m = jnp.abs(1 - s) < out_eps

            U = U[:, m]
            VT = VT[m, :]
            s = s[m]

            print(f"{jnp.sum(m)=}")

        if out_dim is not None:
            if s.shape[0] < out_dim:
                print(f"found only {s.shape[0]} singular values")

        if verbose:
            print(f"{s[0:min(10, s.shape[0]) ]=}")

        print(f"{add_1=}")

        km = KoopmanModel(
            cov=cov,
            W0=W0,
            W1=W1,
            s=s,
            cv_0=cv_0,
            cv_tau=cv_tau,
            nl=nl,
            nl_t=nl_t,
            w=w,
            # w_t=w_t,
            rho=rho,
            # rho_t=rho_t,
            add_1=add_1,
            max_features=max_features,
            max_features_pre=max_features_pre,
            out_dim=out_dim,
            method=method,
            calc_pi=calc_pi,
            tau=tau,
            trans=trans,
            T_scale=T_scale,
            argmask=argmask,
            only_diag=only_diag,
            eps=eps,
            eps_pre=eps_pre,
            shape=cov.C01.shape[0],
            scaled_tau=scaled_tau,
            correlation_whiten=correlation_whiten,
            # U=U,
            # VT=VT,
        )

        # km = km.weighted_model(
        #     chunk_size=chunk_size,
        #     macro_chunk=macro_chunk,
        #     verbose=verbose,
        # )

        return km

    @staticmethod
    def _add_1(
        cv,
        nl,
        _,
        shmap,
        shmap_kwargs,
    ):
        return cv.replace(cv=jnp.hstack([cv.cv, jnp.array([1])]))

    # def Tk(self, out_dim=None):
    #     # Optimal Data-Driven Estimation of Generalized Markov State Models for Non-Equilibrium Dynamics eq. 30
    #     # T_k = C11^{-1/2} K^T C00^{1/2}
    #     #     = w1.T K.T w0 C00

    #     if out_dim is None:
    #         out_dim = self.s.shape[0]

    #     Tk = self.[:out_dim, :].T @ jnp.diag(self.s[:out_dim])

    #     return Tk

    @property
    def tot_trans(self):
        tr = self.trans

        if self.add_1:
            if tr is not None:
                tr *= CvTrans.from_cv_function(KoopmanModel._add_1)

            else:
                tr = CvTrans.from_cv_function(KoopmanModel._add_1)

        return tr

    def f(
        self,
        out_dim=None,
        remove_constant=True,
        constant_threshold=1e-6,
        skip_first=None,
    ):
        o = self.W0.T
        s = self.s

        if skip_first is None:
            skip_first = self.add_1 or not self.calc_pi

        if skip_first:
            s = s[1:]
            o = o[:, 1:]
            print(f"skipping first mode")

        if remove_constant:
            # o = o[:, 1:]

            # if self.add_1:

            nc = jnp.abs(1 - s) < constant_threshold

            if jnp.sum(nc) > 0:
                print(f"found {jnp.sum(nc)} constant eigenvalues,removing")

            o = o[:, jnp.logical_not(nc)]

        o = o[:, :out_dim]

        tr = CvTrans.from_cv_function(
            data_loader_output._transform,
            static_argnames=["add_1", "add_1_pre"],
            add_1=False,
            add_1_pre=self.add_1,
            q=o,
            pi=self.cov.pi_0,
            argmask=self.argmask,
        )

        if self.trans is not None:
            tr = self.trans * tr

        return tr

    def g(
        self,
        out_dim=None,
        remove_constant=True,
        constant_threshold=1e-6,
        skip_first=None,
    ):
        o = self.W1.T
        s = self.s

        if skip_first is None:
            skip_first = self.add_1 or not self.calc_pi

        if skip_first:
            s = s[1:]
            o = o[:, 1:]
            print(f"skipping first mode")

        # this performs y =  (trans*g_trans)(x) @ Vh[:out_dim,:], but stores smaller matrices

        if remove_constant:
            nc = jnp.abs(1 - s) < constant_threshold

            if jnp.sum(nc) > 0:
                print(f"found {jnp.sum(nc)} constant eigenvalues,removing")

            o = o[:, jnp.logical_not(nc)]

        o = o[:out_dim, :]

        tr = CvTrans.from_cv_function(
            data_loader_output._transform,
            static_argnames=["add_1", "add_1_pre"],
            add_1=False,
            add_1_pre=self.add_1,
            q=o,
            pi=self.cov.pi_1,
            argmask=self.argmask,
        )

        if self.tot_trans is not None:
            tr = self.tot_trans * tr

        return tr

    def koopman_weight(
        self,
        verbose=False,
        chunk_size=None,
        macro_chunk=1000,
        retarget=True,
        epsilon=1e-5,
        max_entropy=True,
    ) -> tuple[list[jax.Array], bool]:
        # Optimal Data-Driven Estimation of Generalized Markov State Models, page 18-19
        # create T_k in the trans basis
        # T_n = C00^{-1} C11 T_k
        # T_K = C11^{-1/2} V S U.T C00^{1/2}
        # T_n mu_corr =  (lambda=1) * mu_corr
        #  C00^{-1/2} C11  C11^ {-1/2} V S UT  C00^{1/2} mu_corr )=  C00^(1/2) mu_corr
        # w0 C11 W1.T V Sigma Ut v = lambda v
        # v= C00^(1/2) v
        # mu_corr = W0^T v

        # out_dim = min(max(int(jnp.sum(jnp.abs(1 - self.s) < epsilon)), 5), self.s.shape[0])

        out_dim = int(jnp.sum(jnp.abs(1 - self.s) < epsilon))
        if out_dim == 0:
            # print("no eigenvalues found close to 1")
            # return self.w, None, False
            out_dim = 1
            print(f"using closest eigenvalue to 1: {self.s=}")

        print("reweighing, new A")

        A = self.W0[:out_dim, :] @ self.cov.C11 @ self.W1.T @ jnp.diag(self.s)[:, :out_dim]

        out_idx = jnp.arange(out_dim)

        l, v = jnp.linalg.eig(A)

        # remove complex eigenvalues, as they cannot be the ground state
        real = jnp.abs(jnp.imag(l)) <= 1e-10
        out_idx = out_idx[real]
        out_idx = out_idx[jnp.argsort(jnp.abs(l[out_idx] - 1))]
        # sort

        print(f"{l[out_idx]=} {out_idx=}")

        mask = jnp.abs(l[out_idx] - 1) < epsilon

        # if jnp.sum(mask) == 0:
        #     print(f"no eigenvalues found close to 1 {l=}")
        #     return self.w, None, False

        # if jnp.sum(mask) <1:
        #     print(f"multiple eigenvalues found close to 1 {l[mask]=}")
        #     return self.w, None, False

        idx = jnp.argmin(jnp.abs(l[out_idx] - 1)).reshape((1,))
        print(f"{idx=} {out_idx[idx]=}")
        l, v = l[out_idx[idx]], v[:, out_idx[idx]]

        v /= jnp.sign(jnp.sum(v))

        # we should be left with one of the options
        idx_v = jnp.argmin(jnp.abs(1 - v))

        # mode = out_idx[idx][idx_v]

        print(f"selecting mode {idx_v=} {v=}")

        if jnp.sum(jnp.abs(1 - v[idx_v]) < epsilon) == 1:
            print(f"found unique eigenmode {l=} {v=}, cleaning close to zero elements")

            v = jnp.round(v)

            print(f"{v=}")

        l, v = jnp.real(l), jnp.real(v)

        # pi = v
        # n_pi = jnp.abs(jnp.sum(pi))
        # pi /= jnp.abs(jnp.sum(pi))

        # v = v[[idx_v],:]

        f_trans_2 = CvTrans.from_cv_function(
            data_loader_output._transform,
            q=self.W0[out_idx[idx], :].T,
            l=None,
            pi=self.cov.pi_0 if self.calc_pi else None,
            argmask=self.argmask,
            add_1_pre=self.add_1,
            static_argnames=["add_1_pre"],
        )

        @partial(CvTrans.from_cv_function, v=v)
        def _get_w(cv: CV, _nl, _, shmap, shmap_kwargs, v: Array):
            x = cv.cv

            return cv.replace(cv=jnp.einsum("i,ij->j", x, v), _combine_dims=None)

        tr = f_trans_2 * _get_w

        w_out_cv, _ = data_loader_output.apply_cv(
            f=tr,
            x=self.cv_0,
            x_t=None,
            nl=self.nl,
            nl_t=None,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
            verbose=verbose,
        )
        w_corr = CV.stack(*w_out_cv).cv

        w_orig = jnp.hstack(self.w)

        def _norm(w):
            w = jnp.where(jnp.sum(w > 0) - jnp.sum(w < 0) > 0, w, -w)  # majaority determines sign
            n_neg = jnp.sum(w < 0)

            w_pos = jnp.where(w >= 0, w, 0)
            w_neg = jnp.where(w < 0, -w, 0)

            n = jnp.sum(w_pos)

            w_pos /= n
            w_neg /= n

            return w_pos, jnp.sum(w_neg), n_neg / w.shape[0]

        w_pos, wf_neg, f_neg = jax.vmap(_norm, in_axes=1)(w_corr)

        print(f"{w_pos=}")

        mode = jnp.argmin(f_neg)

        w_corr = w_pos[mode, :]

        print(f"{f_neg=} {wf_neg=} {mode=} ")

        w_new = w_orig * w_corr
        w_new /= jnp.sum(w_new)

        w_new = data_loader_output._unstack_weights([cvi.shape[0] for cvi in self.cv_0], w_new)
        w_corr = data_loader_output._unstack_weights([cvi.shape[0] for cvi in self.cv_0], w_corr)

        return w_new, w_corr, True

    def weighted_model(
        self,
        chunk_size=None,
        macro_chunk=1000,
        verbose=False,
        **kwargs,
    ) -> KoopmanModel:
        new_w, new_w_corr, b = self.koopman_weight(
            verbose=verbose,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
        )

        if not b:
            return self

        kw = dict(
            w=new_w,
            rho=self.rho,
            cv_0=self.cv_0,
            cv_tau=self.cv_tau,
            nl=self.nl,
            nl_t=self.nl_t,
            add_1=self.add_1,
            eps=self.eps,
            eps_pre=self.eps_pre,
            method=self.method,
            symmetric=False,
            out_dim=self.out_dim,
            tau=self.tau,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
            trans=self.trans,
            T_scale=self.T_scale,
            verbose=verbose,
            calc_pi=self.calc_pi,
            max_features=self.max_features,
            max_features_pre=self.max_features_pre,
            only_diag=self.only_diag,
            correlation_whiten=self.correlation_whiten,
        )

        kw.update(**kwargs)

        return KoopmanModel.create(**kw)

    def timescales(
        self,
        remove_constant=True,
        constant_threshold=1e-6,
        skip_first=None,
    ):
        s = self.s

        if skip_first is None:
            skip_first = self.add_1 or not self.calc_pi

        if skip_first:
            s = s[1:]
            print(f"skipping first mode")

        if remove_constant:
            # s = s[1:]
            nc = jnp.abs(1 - s) < constant_threshold

            if jnp.sum(nc) > 0:
                print(f"found {jnp.sum(nc)} constant eigenvalues,removing from timescales")

            s = s[jnp.logical_not(nc)]

        tau = self.tau
        if tau is None:
            tau = 1
            print("tau not set, assuming 1")

        return -tau / jnp.log(s)


@partial(dataclass, frozen=False, eq=False)
class Covariances:
    C00: jax.Array | None
    C01: jax.Array | None
    C10: jax.Array | None
    C11: jax.Array | None
    pi_0: jax.Array | None
    pi_1: jax.Array | None

    only_diag: bool = False
    trans_f: CvTrans | CvFlow | None = None
    trans_g: CvTrans | CvFlow | None = None
    T_scale: float = 1
    symmetric: bool = False

    @staticmethod
    def create(
        cv_0: list[CV] | list[SystemParams],
        cv_1: list[CV] | list[SystemParams] | None = None,
        nl: list[NeighbourList] | NeighbourList | None = None,
        nl_t: list[NeighbourList] | NeighbourList | None = None,
        w: list[Array] | None = None,
        w_t: list[Array] | None = None,
        calc_pi=True,
        macro_chunk=1000,
        chunk_size=None,
        only_diag=False,
        trans_f: CvTrans | CvFlow | None = None,
        trans_g: CvTrans | CvFlow | None = None,
        T_scale=1,
        symmetric=False,
        calc_C00=True,
        calc_C01=True,
        calc_C10=True,
        calc_C11=True,
        shmap_kwargs=ShmapKwargs.create(),
    ) -> Covariances:
        time_series = cv_1 is not None

        if w is None:
            w = [jnp.ones((cvi.shape[0],)) for cvi in cv_0]

        n = jnp.sum(jnp.array([jnp.sum(wi) for wi in w]))

        w = [wi / n for wi in w]

        if time_series:
            if w_t is None:
                w_t = w

            n_t = jnp.sum(jnp.array([jnp.sum(wi) for wi in w_t]))

            w_t = [wi / n_t for wi in w_t]

        if T_scale != 1:
            raise NotImplementedError("T_scale not implemented")

        # @jax.jit
        def cov_pi(carry, cv_0: CV, cv1: CV | None, w, w_t=None):
            # print(f"{cv_0=} {cv1=}")

            x0 = cv_0.cv
            if cv1 is not None:
                x1 = cv1.cv
            else:
                x1 = None

            (C00, C01, C10, C11, pi0, pi1) = carry

            def c(x, y, w, c_pre):
                if only_diag:
                    einstr = "ni,ni,n->i"
                else:
                    einstr = "ni,nj,n->ij"

                out = jnp.einsum(einstr, x, y, w)
                if c_pre is not None:
                    out += c_pre

                return out

            def p(x, w, p_pre):
                out = jnp.einsum("ni,n->i", x, w)
                if p_pre is not None:
                    out += p_pre
                return out

            if calc_C00:
                C00 = c(x0, x0, w, C00)

            if calc_pi:
                pi0 = p(x0, w, pi0)

            if time_series:
                if calc_C01:
                    C01 = c(x0, x1, w, C01)

                if calc_C10:
                    C10 = c(x1, x0, w, C10)

                if calc_C11:
                    C11 = c(x1, x1, w, C11)

                if calc_pi:
                    pi1 = p(x1, w, pi1)

            return (C00, C01, C10, C11, pi0, pi1)

        if trans_f is not None:

            def f_func(x, nl):
                # print(f"f_func {x=}")

                return trans_f.compute_cv(x, nl, chunk_size=chunk_size, shmap=False)[0]

            f_func = jax.jit(f_func)
            # f_func = padded_shard_map(f_func, kwargs=shmap_kwargs)  # (pmap=True))
        else:

            def f_func(x, nl):
                return x

        if trans_g is not None:

            def g_func(x, nl):
                # print(f"g_func {x=}")
                return trans_g.compute_cv(x, nl, chunk_size=chunk_size, shmap=False)[0]

            g_func = jax.jit(g_func)
            # g_func = padded_shard_map(g_func, kwargs=shmap_kwargs)  # (pmap=True))
        else:

            def g_func(x, nl):
                return x

        out = macro_chunk_map(
            f=f_func,
            ft=g_func,
            op=cv_0[0].stack,
            y=cv_0,
            y_t=cv_1,
            nl=nl,
            nl_t=nl,
            macro_chunk=macro_chunk,
            verbose=True,
            chunk_func=cov_pi,
            chunk_func_init_args=(None, None, None, None, None, None),
            w=w,
            w_t=w_t,
            jit_f=True,
        )

        C00, C01, C10, C11, pi_0, pi_1 = out

        if calc_pi:
            if only_diag:
                if calc_C00:
                    C00 -= pi_0**2

                if time_series:
                    if calc_C11:
                        C11 -= pi_1**2
                    if calc_C01:
                        C01 -= pi_0 * pi_1
                    if calc_C10:
                        C10 -= pi_1 * pi_0

            else:
                if calc_C00:
                    C00 -= jnp.outer(pi_0, pi_0)
                if time_series:
                    if calc_C11:
                        C11 -= jnp.outer(pi_1, pi_1)
                    if calc_C01:
                        C01 -= jnp.outer(pi_0, pi_1)
                    if calc_C10:
                        C10 -= jnp.outer(pi_1, pi_0)

        cov = Covariances(
            C00=C00,
            C01=C01,
            C10=C10,
            C11=C11,
            pi_0=pi_0,
            pi_1=pi_1,
            only_diag=only_diag,
            trans_f=trans_f,
            trans_g=trans_g,
            T_scale=T_scale,
        )

        if symmetric:
            cov = cov.symmetrize()

        return cov

    def whiten(
        self,
        choice,
        epsilon: float = 1e-6,
        epsilon_pre: float = 1e-12,
        out_dim=None,
        max_features=None,
        verbose=False,
        use_scipy=True,
        filter_argmask=True,
        correlation=True,
        cholesky=False,
    ):
        # returns W such that W C W.T = I and hence w.T W = C^-1

        # https://arxiv.org/pdf/1512.00809

        if choice == "C00":
            C = self.C00
        elif choice == "C11":
            C = self.C11
        else:
            raise ValueError(f"choice {choice} not known")

        if self.only_diag:
            raise NotImplementedError("only_diag not implemented")

        V = jnp.diag(C)

        print(f"{correlation=}")

        if correlation:
            V_0 = jnp.max(V)

            m_pre = V / V_0 < epsilon_pre**2

            print(f"removing {jnp.sum(m_pre)} eig. {V_0=}")

            V_sqrt_inv = jnp.where(m_pre, 0, 1 / jnp.sqrt(V))
            P = jnp.einsum("ij,i,j->ij", C, V_sqrt_inv, V_sqrt_inv)

        else:
            P = C

        # W = jnp.linalg.sqrtm(P)

        # G, Theta, _ = jnp.linalg.svd(P)
        theta, G = jnp.linalg.eigh(P)

        # print(f"{theta=} ")
        idx = jnp.argmax(theta)

        mask = theta / theta[idx] > epsilon**2

        print(f"{jnp.sum(mask)=} {theta[mask]=} {theta[~mask]=}")

        theta_inv = jnp.where(mask, 1 / jnp.sqrt(theta), 0)

        W = jnp.einsum(
            "i,ji->ij",
            theta_inv,
            G,
        )

        if correlation:
            W = jnp.einsum(
                "ij,j->ij",
                W,
                V_sqrt_inv,
            )

        W = W[mask, :]

        print(f"{jnp.linalg.norm(W @ C @ W.T - jnp.eye(W.shape[0]))=}")

        return W

    # def shrink(S: Array, n: int, shrinkage="OAS"):
    #     # https://arxiv.org/pdf/1602.08776.pdf appendix b

    #     if shrinkage == "None":
    #         return S
    #     # Todo https://papers.nips.cc/paper_files/paper/2014/file/fa83a11a198d5a7f0bf77a1987bcd006-Paper.pdf
    #     # todo: paper Covariance shrinkage for autocorrelated data
    #     assert shrinkage in ["RBLW", "OAS", "BC"]

    #     p = S.shape[0]
    #     F = jnp.trace(S) / p * jnp.eye(p)

    #     tr_s2 = jnp.trace(S**2)
    #     tr2_s = jnp.trace(S) ** 2

    #     if shrinkage == "RBLW":
    #         rho = ((n - 2) / n * tr_s2 + tr2_s) / ((n + 2) * (tr_s2 - tr2_s / p))
    #     elif shrinkage == "OAS":
    #         # use oracle https://arxiv.org/pdf/0907.4698.pdf, eq 23
    #         rho = ((1 - 2 / p) * tr_s2 + tr2_s) / ((n + 1 - 2 / p) * (tr_s2 - tr2_s / p))

    #     elif shrinkage == "BC":
    #         # https://proceedings.neurips.cc/paper_files/paper/2014/file/fa83a11a198d5a7f0bf77a1987bcd006-Paper.pdf
    #         pass
    #         # shrinkage based on  X
    #         # n = X.shape[0]
    #         # p = X.shape[1]
    #         # b = 20

    #         # u = X - pi_x
    #         # v = Y - pi_y

    #         # S_0 = jnp.einsum("ti,tj->ij", u, u) / (n - 1)
    #         # S_1 = jnp.einsum("ti,tj->ij", u, v) / (n - 1)

    #         # T_0 = jnp.trace(S_0) / p * jnp.eye(p)
    #         # T_1 = jnp.trace(S_1) / p * jnp.eye(p)

    #         # def gamma(s, u, v, S):
    #         #     return (
    #         #         jnp.einsum(
    #         #             "ti,tj,ti,tj,t->ij", u[: n - s, :], w[: n - s], v[: n - s, :], u[s:, :], w[s:], v[s:, :], w
    #         #         )
    #         #         - jnp.sum(w[: n - s]) / jnp.sum(w) * S**2
    #         #     )

    #         # var_BC = gamma(0)
    #         # for i in range(1, b + 1):
    #         #     var_BC += 2 * gamma(i)
    #         # var_BC /= n - 1 - 2 * b + b * (b + 1) / n

    #         # lambda_BC = jnp.einsum("ij,ij", var_BC, var_BC) / jnp.einsum("ij,ij", S_0 - T_0, S_0 - T_0)

    #         # lambda_BC = jnp.clip(lambda_BC, 0, 1)

    #         # print(f"lambda_BC = {lambda_BC}")

    #         #

    #     if rho > 1:
    #         rho = 1

    #     print(f"{rho=}")

    #     def f(C):
    #         return rho * F + (1 - rho) * C

    #     return f

    def symmetrize(self):
        _C00 = self.C00
        _C01 = self.C01
        _C10 = self.C10
        _C11 = self.C11

        pi = None

        if self.pi_0 is not None:
            assert self.pi_1 is not None

            _C00 += jnp.outer(self.pi_0, self.pi_0)
            if _C01 is not None:
                _C01 += jnp.outer(self.pi_0, self.pi_1)

            if _C10 is not None:
                _C10 += jnp.outer(self.pi_1, self.pi_0)

            if _C11 is not None:
                _C11 += jnp.outer(self.pi_1, self.pi_1)

            pi = 0.5 * (self.pi_0 + self.pi_1)

            _C00 -= jnp.outer(pi, pi)
            _C01 -= jnp.outer(pi, pi)
            _C10 -= jnp.outer(pi, pi)
            _C11 -= jnp.outer(pi, pi)

        C00 = (1 / 2) * (_C00 + _C11)

        if self.C01 is not None:
            C01 = (1 / 2) * (_C01 + _C10)
        else:
            C01 = None

        if _C10 is not None:
            C10 = (1 / 2) * (_C10 + _C01)
        else:
            C10 = None

        if _C11 is not None:
            C11 = _C00
        else:
            C10 = None

        return Covariances(
            C00=C00,
            C01=C01,
            C10=C10,
            C11=C11,
            pi_0=pi,
            pi_1=pi,
            only_diag=self.only_diag,
            symmetric=True,
            trans_f=self.trans_f,
            trans_g=self.trans_g,
        )
