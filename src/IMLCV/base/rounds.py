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
import jax.experimental
import jax.experimental.sparse
import jax.experimental.sparse.bcoo
import jax.numpy as jnp
import numpy as np
from equinox import Partial
from IMLCV.base.bias import Bias
from IMLCV.base.bias import CompositeBias
from IMLCV.implementations.bias import RbfBias
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV, CvMetric
from IMLCV.base.CV import CvTrans, CvFlow
from IMLCV.base.CV import NeighbourList, NeighbourListInfo
from IMLCV.base.CV import padded_shard_map
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

from typing import Callable
from jax import vmap
from functools import partial
from IMLCV.implementations.bias import _clip
from IMLCV.base.bias import BiasModify
from IMLCV.base.CV import padded_vmap, macro_chunk_map
from IMLCV.base.bias import NoneBias
from IMLCV.base.CVDiscovery import Transformer
from molmod.units import kjmol
from flax.struct import PyTreeNode
from IMLCV.configs.config_general import Executors

from IMLCV.implementations.CV import identity_trans


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
        T_scale=10,
        macro_chunk=2000,
        macro_chunk_nl=5000,
        only_update_nl=False,
        divide_by_histogram=True,
        n_max=30,
    ) -> data_loader_output:
        # weights = []

        # if T_scale != 1:
        #     raise NotImplementedError("T_scale not implemented")

        if cv_round is None:
            cv_round = self.cv

        sti = self._round_information(c=cv_round).tic

        if new_r_cut == -1:
            new_r_cut = sti.r_cut

        sp: list[SystemParams] = []
        cv: list[CV] = []
        ti: list[TrajectoryInfo] = []
        weights: list[Array] = []
        weights_bincount: list[Array] = []

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

                bias_c.append(traj_info.get_bias())

            if len(sp_c) == 0:
                continue

            if weight_c and out != -1:
                if verbose:
                    print(f"getting weights for cv_round {cvi} {len(sp_c)} trajectories")

                dlo = data_loader_output(
                    sp=sp_c,
                    cv=cv_c,
                    ti=ti_c,
                    sti=sti_c,
                    nl=None,
                    collective_variable=colvar_c,
                    bias=bias_c,
                    # time_series=time_series,
                    ground_bias=ground_bias_c,
                )

                # select points according to free energy divided by histogram count
                w_out = dlo.weights(
                    chunk_size=chunk_size,
                    wham=True,
                    koopman=False,
                    n_max=n_max,
                    wham_sub_grid=3,  # quick settings
                    verbose=verbose,
                    output_bincount=divide_by_histogram,
                )

                if divide_by_histogram:
                    w_c, w_c_bincount = w_out
                else:
                    w_c = w_out

                n = sum([len(wi) for wi in w_c])
                w_c = [w_c[i] * n for i in range(len(w_c))]

            else:
                w_c = [jnp.ones((spi.shape[0])) for spi in sp_c]

                if divide_by_histogram:
                    w_c_bincount = [jnp.ones((spi.shape[0])) for spi in sp_c]

            sp.extend(sp_c)
            cv.extend(cv_c)
            ti.extend(ti_c)
            weights.extend(w_c)

            if divide_by_histogram:
                weights_bincount.extend(w_c_bincount)

            if get_bias_list:
                bias_list.extend(bias_c)

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
            weights_bincount_new: list[Array] = []

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

                if divide_by_histogram:
                    weights_bincount_new.append(weights_bincount[n])

            sp = sp_new
            ti = ti_new
            cv = cv_new
            weights = weights_new
            if divide_by_histogram:
                weights_bincount = weights_bincount_new

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
        out_weights: list[Array] = []

        if time_series:
            out_sp_t = []
            out_cv_t = []
            out_ti_t = []

        if get_bias_list:
            out_biases = []

        for select_weight, spi in zip(weights, sp):
            assert (
                select_weight.shape[0] == spi.shape[0]
            ), f"weights and sp shape are different: {select_weight.shape=} {StopAsyncIteration.shape=}"

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

            for n, select_weight in enumerate(weights):
                reweight = None

                if select_weight is None:
                    probs = None
                else:
                    if lag_n != 0:
                        select_weight = select_weight[:-lag_n]
                    else:
                        select_weight = select_weight

                    reweight = jnp.ones_like(select_weight)

                    if divide_by_histogram:
                        wi_bc = weights_bincount[n]
                        if lag_n != 0:
                            wi_bc = wi_bc[:-lag_n]

                        new_select_weight = select_weight / wi_bc
                        new_reweight = reweight * (select_weight / new_select_weight)

                        reweight = new_reweight
                        select_weight = new_select_weight

                    if T_scale != 1:
                        new_select_weight = select_weight ** (1 / T_scale)
                        new_reweight = reweight * (select_weight / new_select_weight)

                        reweight = new_reweight
                        select_weight = new_select_weight

                    probs = select_weight / jnp.sum(select_weight)

                ni = int(frac * (sp[n].shape[0] - lag_n))

                if ni == 0 and (sp[n].shape[0] - lag_n) > 0:
                    print(f"not enough data points for lag_n {sp[n].shape[0]} < {lag_n}")
                    continue
                    ni = 1

                key, indices = choose(
                    key,
                    probs,
                    out=ni,
                    len=sp[n].shape[0] - lag_n,
                )

                if reweight is None:
                    reweight = jnp.ones_like(weights[n][indices])

                out_sp.append(sp[n][indices])
                out_cv.append(cv[n][indices])
                out_ti.append(ti[n][indices])
                out_weights.append(reweight)

                if time_series:
                    out_sp_t.append(sp[n][indices + lag_n])
                    out_cv_t.append(cv[n][indices + lag_n])
                    out_ti_t.append(ti[n][indices + lag_n])

        else:
            reweight: list[jax.Array] = None

            if weights[0] is None:
                probs = None
            else:
                if lag_n == 0:
                    select_weight = jnp.hstack(weights)
                else:
                    w = []

                    if divide_by_histogram:
                        w_bc = []

                    for n, select_weight in enumerate(weights):
                        if select_weight.shape[0] <= lag_n:
                            print(f"not enough data points for lag_n {select_weight.shape[0]} <= {lag_n}")
                            continue

                        wio = select_weight[:-lag_n]

                        if divide_by_histogram:
                            w_bc.append(weights_bincount[n][:-lag_n])

                        w.append(wio)

                    select_weight = jnp.hstack(w)
                    reweight = jnp.ones_like(select_weight)

                    if divide_by_histogram:
                        wi_bc = jnp.hstack(w_bc)

                        new_select_weight = select_weight / wi_bc
                        new_reweight = reweight * (select_weight / new_select_weight)

                        reweight = new_reweight
                        select_weight = new_select_weight

                    if T_scale != 1:
                        new_select_weight = select_weight ** (1 / T_scale)
                        new_reweight = reweight * (select_weight / new_select_weight)

                        reweight = new_reweight
                        select_weight = new_select_weight

                probs = select_weight / jnp.sum(select_weight)

                print(f"probs = {probs.shape}  {out=} {total=} ")

            key, indices = choose(key, probs, out=int(out), len=total)

            # indices = jnp.sort(indices)

            count = 0

            sp_trimmed = []
            cv_trimmed = []
            ti_trimmed = []

            weights_trimmed = []

            out_biases = []

            if time_series:
                sp_trimmed_t = []
                cv_trimmed_t = []
                ti_trimmed_t = []

            for n, (sp_n, cv_n, ti_n) in enumerate(zip(sp, cv, ti)):
                n_i = sp_n.shape[0] - lag_n

                indices_full = indices[jnp.logical_and(count <= indices, indices < count + n_i)]
                index = indices_full - count

                if len(index) == 0:
                    count += n_i
                    continue

                sp_trimmed.append(sp_n[index])

                cv_trimmed.append(cv_n[index])
                ti_trimmed.append(ti_n[index])

                if reweight is None:
                    weights_trimmed.append(jnp.ones_like(weights[n][index]))
                else:
                    weights_trimmed.append(reweight[indices_full])

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
            out_weights = weights_trimmed

            if time_series:
                out_sp_t = sp_trimmed_t
                out_cv_t = cv_trimmed_t
                out_ti_t = ti_trimmed_t

            if get_bias_list:
                bias_list = out_biases

            # bias = None

        print(f"len(out_sp) = {len(out_sp)} ")

        out_nl = None

        if time_series:
            out_nl_t = None

        # return data
        if time_series:
            tau = None

            arr = []

            # consistency check
            for tii, ti_ti in zip(out_ti, out_ti_t):
                tii: TrajectoryInfo
                ti_ti: TrajectoryInfo

                dt = ti_ti.t - tii.t

                tau = jnp.median(dt) if tau is None else tau

                mask = jnp.allclose(dt, tau)

                if not mask.all():
                    arr.append(jnp.sum(jnp.logical_not(mask)))

            if len(arr) != 0:
                print(
                    f"WARNING:time steps are not equal, {jnp.array(arr)} out of { out   } trajectories have different time steps"
                )

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
                _weights=out_weights,
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
                _weights=out_weights,
            )

        if new_r_cut is not None:
            if verbose:
                print("getting Neighbour List")

            dlo.calc_neighbours(
                r_cut=new_r_cut,
                chunk_size=chunk_size,
                verbose=verbose,
                macro_chunk=macro_chunk_nl,
                only_update=only_update_nl,
            )

        if recalc_cv:
            if verbose:
                print("recalculating CV")
            dlo.recalc(
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
        chunk_size=None,
        T_scale=10,
    ):
        if cv_round is None:
            cv_round = self.cv

        r = self.get_round(c=cv_round)

        common_bias_name = self.full_path(self._name_bias(c=cv_round, r=r))
        common_md_name = self.full_path(self._name_md(c=cv_round, r=r))

        md_engine = MDEngine.load(common_md_name)

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

            except Exception as _:
                print(f"got exception  while collecting md {i}, round {r}, cv {cv_round}, continuing anyway")

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

            if b_name_new.exists():
                print(f"skipping {i=}, new bias found")
                continue

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
                d = future.result()

            except Exception as e:
                print(f"got exception {e} while collecting md {i}, round {round}, cv {cv_round}, continuing anyway")

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
        common_bias_name=None,
        r=None,
    ):
        sps = []
        bs = []
        traj_names = []
        b_names = []
        b_names_new = []
        path_names = []

        sp0_provided = sp0 is not None

        if not sp0_provided:
            dlo_data = rounds.data_loader(
                num=5,
                out=10000,
                split_data=False,
                new_r_cut=None,
                ignore_invalid=ignore_invalid,
                md_trajs=md_trajs,
                cv_round=cv_round,
                min_traj_length=min_traj_length,
                recalc_cv=recalc_cv,
                only_finished=only_finished,
                weight=True,
                T_scale=T_scale,
                time_series=False,
                chunk_size=chunk_size,
            )

            # get the weights of the points

            sp_stack = SystemParams.stack(*dlo_data.sp)
            cv_stack = CV.stack(*dlo_data.cv)
            weights = jnp.hstack(dlo_data.weights())  # retrieve weights
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

            b = CompositeBias.create([Bias.load(common_bias_name), bias])

            b_name = path_name / "bias.json"
            b_name_new = path_name / "bias_new.json"
            b.save(b_name)

            traj_name = path_name / "trajectory_info.h5"

            if not sp0_provided:
                # reweigh data points according to new bias
                probs = weights * jnp.exp(
                    -bias.compute_from_cv(cvs=cv_stack, chunk_size=chunk_size)[0] / (dlo_data.sti.T * boltzmann)
                )
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

            info = NeighbourListInfo.create(
                r_cut=st.r_cut,
                z_array=st.atomic_numbers,
            )
            nl = sp.get_neighbour_list(
                info=info,
            )
            cvs, _ = bias.collective_variable.compute_cv(sp=sp, nl=nl)

        bias.plot(
            name=outputs[0].filepath,
            traj=[cvs],
            offset=True,
        )

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
        samples_per_bin=100,
        min_samples_per_bin=20,
        percentile=1e-1,
        use_executor=True,
        n_max=30,
        vmax=100 * kjmol,
        macro_chunk=10000,
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
            bash_app_python(Rounds._update_CV, executors=Executors.training)(
                execution_folder=self.path(c=cv_round_from), **kw
            ).result()

            return

        Rounds._update_CV(**kw)

    @staticmethod
    def _update_CV(
        rounds: Rounds,
        transformer: Transformer,
        dlo_kwargs={},
        dlo: data_loader_output | None = None,
        chunk_size=None,
        macro_chunk=10000,
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
        samples_per_bin=100,
        min_samples_per_bin=20,
        percentile=1e-1,
        n_max=30,
        vmax=100 * kjmol,
        verbose=True,
    ):
        if dlo is None:
            dlo = rounds.data_loader(
                **dlo_kwargs,
                macro_chunk=macro_chunk,
                verbose=verbose,
                macro_chunk_nl=macro_chunk_nl,
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

        x, _ = dlo.apply_cv_trans(cv_trans=cv_trans, x=dlo.cv, chunk_size=chunk_size, verbose=verbose)

        #
        if plot:
            Transformer.plot_app(
                collective_variables=[dlo.collective_variable, collective_variable],
                cv_data=[dlo.cv, x],
                duplicate_cv_data=True,
                T=dlo.sti.T,
                plot_FES=True,
                weight=dlo.weights(),
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

            new_traj_info.save(round_path / "trajectory_info.h5")
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
    _weights: list[Array] | None = None

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
        metric: CvMetric,
        n_grid=40,
        grid_bounds=None,
        chunk_size=None,
        chunk_size_mid=1,
    ):
        bins, _, cv_mid, _ = metric.grid(n=n_grid, bounds=grid_bounds)

        @partial(CvTrans.from_cv_function, mid=cv_mid)
        def closest_trans(cv: CV, _nl, _, shmap, mid: CV):
            m = jnp.argmin(jnp.sum((mid.cv - cv.cv) ** 2, axis=1), keepdims=True)

            return cv.replace(cv=m)

        def get_histo(data_nums: list[CV], weights: None | list[Array] = None):
            h = jnp.zeros((cv_mid.shape[0],))

            for i in range(len(data_nums)):
                h = h.at[data_nums[i].cv[:, 0]].add(
                    jnp.ones_like(data_nums[i].cv[:, 0]) if weights is None else weights[i]
                )

            return h

        nums = closest_trans.compute_cv_trans(cv_mid, chunk_size=chunk_size)[0].cv

        return cv_mid, nums, bins, closest_trans, get_histo

    @staticmethod
    def _unstack_weights(stack_dims, weights: Array) -> list[Array]:
        return [d.cv.reshape((-1,)) for d in CV(cv=jnp.expand_dims(weights, 1), _stack_dims=stack_dims).unstack()]

    def weights(
        self,
        samples_per_bin=30,
        n_max=50,
        ground_bias=None,
        use_ground_bias=True,
        sign=1,
        T_scale=10,
        chunk_size=None,
        wham=True,
        koopman=True,
        indicator_CV=True,
        wham_sub_grid=3,
        wham_eps=1e-6,
        koopman_eps=1e-10,
        cv_0: list[CV] | None = None,
        cv_t: list[CV] | None = None,
        force_recalc=False,
        macro_chunk=10000,
        verbose=False,
        output_bincount=False,
        max_features_koopman=100,
    ) -> list[jax.Array]:
        if cv_0 is None:
            cv_0 = self.cv

        sd = [a.shape[0] for a in cv_0]
        tot_samples = sum(sd)

        ndim = cv_0[0].shape[1]

        if cv_t is None:
            cv_t = self.cv_t

        # prepare histo

        n = CvMetric.get_n(samples_per_bin=samples_per_bin, samples=tot_samples, n_dims=ndim)

        if n > n_max:
            n = n_max

        if verbose:
            print(f"using {n=}")

        print("getting bounds")
        grid_bounds, _, constants = CvMetric.bounds_from_cv(
            cv_0,
            margin=0.1,
            chunk_size=chunk_size,
            n=10,  # no need to be super accurate
        )

        print("getting histo")
        cv_mid, nums, bins, closest, get_histo = data_loader_output._histogram(
            metric=self.collective_variable.metric,
            n_grid=n,
            grid_bounds=grid_bounds,
            chunk_size=chunk_size,
        )

        grid_nums = None

        if self._weights is not None and not force_recalc:
            print("using precomputed weights")
            w_u = jnp.hstack(self._weights)
            w_u /= jnp.sum(w_u)

        else:
            # TODO:https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00867

            beta = 1 / (self.sti.T * boltzmann)

            if ground_bias is None:
                ground_bias = self.ground_bias

            # get raw rescaling
            w_u = []

            for ti_i in self.ti:
                e = ti_i.e_bias

                if e is None:
                    if verbose:
                        print("WARNING: no bias enerrgy found")
                    e = jnp.zeros((ti_i.sp.shape[0],))

                # energies.append(e)

                e -= jnp.max(e)
                e = jnp.exp(beta * e)

                e /= jnp.mean(e)

                w_u.append(e)

            w_u = jnp.hstack(w_u)

            nan_mask = jnp.isnan(w_u)
            if jnp.any(nan_mask):
                print(f"WARNING: {jnp.sum(nan_mask)}/{w_u.shape[0]} nan values in weights, replacing with zeros")
                w_u = jnp.where(nan_mask, 0, w_u)

            w_u /= jnp.sum(w_u)

            if wham:
                assert self.bias is not None

                if verbose:
                    print("step 1 wham")

                if grid_nums is None:
                    grid_nums, _ = self.apply_cv_trans(
                        closest,
                        cv_0,
                        chunk_size=chunk_size,
                        macro_chunk=macro_chunk,
                    )
                hist = get_histo(grid_nums, data_loader_output._unstack_weights(sd, w_u))

                mask = hist > 1e-16

                dx = []

                sg = []

                for b in bins:
                    d = b[1] - b[0]
                    ls = jnp.linspace(-d / 2, d / 2, num=wham_sub_grid)

                    dx.append(ls[1:] - ls[:-1])

                    sg.append(ls)

                mg = jnp.meshgrid(*sg, indexing="ij")
                dx = jnp.array(dx)

                sub_mg = CV(cv=jnp.reshape(jnp.array(mg), (-1, mg[0].size)).T)

                shape = mg[0].shape

                grid_cv = CV.stack(*[sub_mg + x.cv for x in cv_mid])
                if verbose:
                    print(f"{grid_cv.shape=}")

                    print("step 2: getting biases")

                @vmap
                def _b_ik(sg_b):
                    sg_b = sg_b.reshape(shape)

                    sg_b = jnp.exp(-beta * sg_b)

                    for _ in range(len(shape)):
                        sg_b = jnp.trapezoid(sg_b, axis=0) / sg_b.shape[0]

                    return sg_b

                b_ik = jnp.zeros((len(self.bias), jnp.sum(mask)))

                for i, bi in enumerate(self.bias):
                    if bi is None:
                        o = jnp.ones((jnp.sum(mask),))
                    else:
                        o = bi.compute_from_cv(
                            cvs=grid_cv,
                            # chunk_size=chunk_size,
                        )[0]

                        o = jnp.reshape(o, (cv_mid.shape[0], -1))
                        o = _b_ik(o)[mask]

                    b_ik = b_ik.at[i, :].set(o)

                    print(".", end="")

                f_i = jnp.ones((len(self.bias),))
                N_i = sd

                print("step 3: ground_bis")
                # take intial guess from ground bias
                if ground_bias is not None:
                    a_k = jnp.exp(-beta * self.ground_bias.compute_from_cv(cvs=cv_mid[mask], chunk_size=chunk_size)[0])
                else:
                    a_k = jnp.ones((b_ik.shape[1],))

                if verbose:
                    print(f"{jnp.isnan(a_k).sum()=}")

                a_k /= jnp.sum(a_k)

                from jaxopt import FixedPointIteration

                print("step 4: fixed point")

                def T(a_k, x):
                    b_ik, N_i, f_i, hist_k = x
                    f = jnp.einsum("k,ik->i", a_k, b_ik)
                    f_i = jnp.where(f != 0, 1 / f, jnp.inf)

                    a_k_new = hist_k / jnp.einsum("i,i,ik->k", f_i, N_i, b_ik)
                    a_k_new /= jnp.sum(a_k_new)

                    return a_k_new

                fpi = FixedPointIteration(
                    fixed_point_fun=T,
                    maxiter=1000,
                    tol=wham_eps,
                )
                out = fpi.run(a_k, (b_ik, jnp.array(N_i), f_i, hist[mask]))

                a_k, state = out.params, out.state

                if verbose:
                    print(f"{state=}")

                f_inv = jnp.einsum("k,ik->i", a_k, b_ik)

                w_u = jnp.hstack([wi * fi for wi, fi in zip(data_loader_output._unstack_weights(sd, w_u), f_inv)])

            else:
                if verbose:
                    print("WARNING: not using wham")

                if grid_nums is None:
                    grid_nums, _ = self.apply_cv_trans(closest, cv_0, chunk_size=chunk_size, macro_chunk=macro_chunk)
                hist = get_histo(grid_nums, w_u)

                if use_ground_bias:
                    p_grid = jnp.exp(sign * beta * ground_bias.compute_from_cv(cvs=cv_mid, chunk_size=chunk_size)[0])
                else:
                    p_grid = jnp.ones((cv_mid.cv.shape[0],))

                p_grid = jnp.where(hist <= 1e-16, 0, p_grid / hist)

                w_u = w_u * p_grid[grid_nums]
                w_u /= jnp.sum(w_u)

        if constants and koopman:
            print("not performing koopman weighing because of constants in cv")
            koopman = False

        if koopman and self.time_series:
            # prepare histo

            if verbose:
                print("koopman weights")

            tr = None

            if indicator_CV:
                grid_nums, grid_nums_t = self.apply_cv_trans(
                    closest, cv_0, cv_t, chunk_size=chunk_size, macro_chunk=macro_chunk
                )

                hist = get_histo(grid_nums)
                hist_t = get_histo(grid_nums_t)

                mask = jnp.argwhere(jnp.logical_and(hist > 0, hist_t > 0)).reshape(-1)

                @partial(CvTrans.from_cv_function, mask=mask)
                def get_indicator(cv: CV, nl, _, shmap, mask):
                    out = jnp.zeros((hist.shape[0],))
                    out = out.at[cv.cv].set(1)
                    out = jnp.take(out, mask)

                    return cv.replace(cv=out)

                # if verbose:
                #     print("Calculating indicator CVs")

                # cv_km, cv_km_t = self.apply_cv_trans(
                #     get_indicator,
                #     grid_nums,
                #     grid_nums_t,
                #     chunk_size=chunk_size,
                #     macro_chunk=macro_chunk,
                #     verbose=verbose,
                # )

                cv_km = grid_nums
                cv_km_t = grid_nums_t

                tr = get_indicator

            else:
                cv_km = cv_0
                cv_km_t = cv_t

            if verbose:
                print("constructing koopman model")

            # construct koopman model
            km = self.koopman_model(
                cv_0=cv_km,
                cv_tau=cv_km_t,
                w=data_loader_output._unstack_weights(sd, w_u),
                add_1=True,
                method="tcca",
                symmetric=False,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                out_dim=10,
                verbose=verbose,
                eps=koopman_eps,
                trans=tr,
                max_features=max_features_koopman,
            )
            if verbose:
                print("koopman_weight")
            w_u = jnp.hstack(km.koopman_weight())

        else:
            if verbose:
                print("not using koopman weights")

        if T_scale != 1:
            w_u = w_u ** (1 / T_scale)

        w_u /= jnp.sum(w_u)

        w_u = data_loader_output._unstack_weights(sd, w_u)

        if output_bincount:
            if grid_nums is None:
                grid_nums, _ = self.apply_cv_trans(closest, cv_0, chunk_size=chunk_size, macro_chunk=macro_chunk)

            hist = get_histo(grid_nums, w_u)

            bin_counts = [hist[i.cv].reshape((-1,)) for i in grid_nums]

        if output_bincount:
            return w_u, bin_counts

        return w_u

    @staticmethod
    def _transform(cv, nl, _, shmap, pi, add_1, q):
        cv_v = (cv.cv - pi) @ q

        if add_1:
            cv_v = jnp.hstack([cv_v, jnp.array([1])])

        return cv.replace(cv=cv_v)

    @staticmethod
    def _whiten(
        cv_0: list[CV],
        cv_tau: list[CV] | None = None,
        symmetric=False,
        w=None,
        add_1=False,
        eps=1e-10,
        max_features=2000,
        macro_chunk=10000,
        chunk_size=None,
        canoncial_signs=True,
        verbose=False,
        transform_cv=True,
        sparse=False,
        trans=None,
    ):
        print("whitening: getting covariance")
        cov = Covariances.create(
            cv_0,
            cv_tau,
            w,
            calc_pi=True,
            symmetric=symmetric,
            trans=trans,
        )

        l, q, _ = cov.diagonalize(
            "C00",
            epsilon=eps,
            out_dim=max_features,
        )

        if canoncial_signs:
            signs = jax.vmap(lambda q: jnp.sign(q[jnp.argmax(jnp.abs(q))]), in_axes=1)(q)
            q = jnp.einsum("ij,j->ij", q, signs)

        l_inv_sqrt = 1 / jnp.sqrt(l)

        transform_maf = CvTrans.from_cv_function(
            data_loader_output._transform,
            static_argnames=["add_1"],
            add_1=add_1,
            q=jnp.einsum("ij,j->ij", q, l_inv_sqrt),
            pi=cov.pi_0,
        )

        if verbose:
            print("applying transformation")

        if transform_cv:
            if trans is None:
                t = transform_maf
            else:
                t = trans * transform_maf

            cv_0, cv_tau = data_loader_output._apply(
                x=cv_0,
                x_t=cv_tau,
                nl=None,
                nl_t=None,
                f=lambda x, nl: t.compute_cv_trans(x, nl, chunk_size=chunk_size)[0],
                macro_chunk=macro_chunk,
                verbose=False,
            )

        return cv_0, cv_tau, transform_maf, cov.pi_0, q, l

    def koopman_model(
        self,
        cv_0: list[CV] = None,
        cv_tau: list[CV] = None,
        method="tcca",
        koopman_weight=False,
        symmetric=True,
        w: list[jax.Array] | None = None,
        eps=1e-10,
        max_features=2000,
        out_dim=None,
        add_1=True,
        chunk_size=None,
        macro_chunk=10000,
        verbose=False,
        trans=None,
    ) -> KoopmanModel:
        # TODO: https://www.mdpi.com/2079-3197/6/1/22
        assert method in ["tica", "tcca"]

        if cv_0 is None:
            cv_0 = self.cv

        if cv_tau is None:
            cv_tau = self.cv_t

        if w is not None:
            t = sum([cvi.shape[0] for cvi in cv_0])
            w = [jnp.ones((cvi.shape[0],)) / t for cvi in cv_0]

        return KoopmanModel.create(
            w=w,
            cv_0=cv_0,
            cv_tau=cv_tau,
            add_1=add_1,
            eps=eps,
            method=method,
            symmetric=symmetric,
            out_dim=out_dim,
            koopman_weight=koopman_weight,
            max_features=max_features,
            tau=self.tau,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            verbose=verbose,
            trans=trans,
        )

    def filter_nans(
        self,
        x: list[CV] | None = None,
        x_t: list[CV] | None = None,
        macro_chunk=10000,
    ) -> tuple[data_loader_output, CV, CV]:
        if x is None:
            x = self.cv

        if x_t is None:
            x_t = self.cv_t

        def _check_nan(x: CV, nl, cv, shmap):
            return x.replace(cv=jnp.isnan(x.cv) | jnp.isinf(x.cv))

        check_nan = CvTrans.from_cv_function(_check_nan)

        nan_x, nan_x_t = self.apply_cv_trans(
            check_nan,
            x,
            x_t,
            macro_chunk=macro_chunk,
            pmap=True,
            verbose=True,
        )

        nan = CV.stack(*nan_x).cv

        if nan_x_t is not None:
            nan = jnp.logical_or(nan, CV.stack(*nan_x_t).cv)

        assert not jnp.any(nan), f"found {jnp.sum(nan)}/{len(nan)} nans or infs in the cv data"

        if jnp.any(nan):
            raise

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
        def f(x, nl):
            return cv_trans.compute_cv_trans(
                x,
                nl,
                chunk_size=chunk_size,
                shmap=pmap,
            )[0]

        return self._apply(
            x,
            x_t,
            self.nl,
            self.nl_t,
            f,
            macro_chunk=macro_chunk,
            verbose=verbose,
        )

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
        def f(x, nl):
            return flow.compute_cv_flow(
                x,
                nl,
                chunk_size=chunk_size,
                shmap=pmap,
            )[0]

        return self._apply(
            x,
            x_t,
            self.nl,
            self.nl_t,
            f,
            macro_chunk=macro_chunk,
            verbose=verbose,
        )

    @staticmethod
    def _apply(
        x: CV,
        x_t: CV | None,
        nl: NeighbourList | None,
        nl_t: NeighbourList | None,
        f: Callable,
        macro_chunk=10000,
        verbose=False,
    ):
        if x_t is not None:
            y = [*x, *x_t]
            nl = [*nl, *nl_t] if nl is not None else None

        else:
            y = [*x]
            nl = [*nl] if nl is not None else None

        z = macro_chunk_map(
            f,
            x[0].__class__.stack,
            y,
            nl,
            macro_chunk=macro_chunk,
            verbose=verbose,
        )

        if x_t is not None:
            z, z_t = z[0 : len(x)], z[len(x) :]
        else:
            z_t = None

        for i in range(len(z)):
            assert (
                z[i].shape[0] == x[i].shape[0]
            ), f" shapes do not match {[zi.shape[0] for  zi in z ]} != {[xi.shape[0] for xi in x]}"

            if x_t is not None:
                assert z[i].shape[0] == z_t[i].shape[0]

        return z, z_t

    @staticmethod
    def _get_fes_bias_from_weights(
        T,
        weights: list[jax.Array],
        collective_variable: CollectiveVariable,
        cv: list[CV],
        samples_per_bin=10,
        min_samples_per_bin: int | None = 3,
        n_max=30,
        n_grid=None,
        max_bias=None,
        chunk_size=None,
        macro_chunk=10000,
        max_bias_margin=0.2,
    ) -> RbfBias:
        beta = 1 / (T * boltzmann)

        samples = sum([cvi.shape[0] for cvi in cv])

        if n_grid is None:
            n_grid = CvMetric.get_n(samples_per_bin=samples_per_bin, samples=samples, n_dims=cv[0].shape[1])

        if n_grid > n_max:
            # print(f"reducing n_grid from {n_grid} to {n_max}")
            n_grid = n_max

        if n_grid <= 1:
            raise

        grid_bounds, _, _ = CvMetric.bounds_from_cv(cv, margin=0.1)

        # # do not update periodic bounds
        grid_bounds = jnp.where(
            collective_variable.metric.periodicities,
            collective_variable.metric.bounding_box,
            grid_bounds,
        )

        cv_mid, nums, _, closest, get_histo = data_loader_output._histogram(
            n_grid=n_grid,
            grid_bounds=grid_bounds,
            metric=collective_variable.metric,
        )

        def f(x, nl):
            return closest.compute_cv_trans(x, chunk_size=chunk_size)[0]

        grid_nums, _ = data_loader_output._apply(
            cv,
            None,
            nl=None,
            nl_t=None,
            f=f,
            macro_chunk=macro_chunk,
        )

        p_grid = get_histo(grid_nums, weights)

        mask = p_grid != 0

        p_grid = p_grid[mask]

        fes_grid = -jnp.log(p_grid) / beta
        fes_grid -= jnp.min(fes_grid)

        bias = RbfBias.create(
            cvs=collective_variable,
            cv=cv_mid[mask],
            kernel="thin_plate_spline"
            if jnp.sum(mask) > 1
            else "linear",  # number of points needs to be at leat the degree
            vals=-fes_grid,
        )

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

        prob = Partial(prob, grid_nums_new=grid_nums_new, nums_old=nums_old, p_grid_old=p_grid_old)
        prob = padded_vmap(prob, chunk_size=chunk_size)

        if pmap:
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
            kernel="thin_plate_spline",
        )

        if max_bias is None:
            max_bias = -jnp.min(new_FES_bias_vals)

        return BiasModify.create(
            bias=new_FES_bias,
            fun=_clip,
            kwargs={"a_min": -max_bias, "a_max": 0.0},
        )

    def recalc(self, chunk_size=None, macro_chunk=10000, pmap=True, verbose=False):
        x, x_t = self.apply_cv_flow(
            self.collective_variable.f,
            chunk_size=chunk_size,
            pmap=pmap,
            macro_chunk=macro_chunk,
            verbose=verbose,
        )

        self.cv = x
        self.cv_t = x_t

    def calc_neighbours(self, r_cut, chunk_size=None, macro_chunk=10000, verbose=False, only_update=False):
        if self.time_series:
            y = [*self.sp, *self.sp_t]
        else:
            y = [*self.sp]

        from molmod.units import angstrom

        nl_info = NeighbourListInfo.create(
            r_cut=r_cut,
            r_skin=1 * angstrom,
            z_array=self.sti.atomic_numbers,
        )

        def f(sp: SystemParams, nl=None):
            b, _, _, nl = sp._get_neighbour_list(
                info=nl_info,
                chunk_size=chunk_size,
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
        )

        if self.time_series:
            nl, nl_t = nl[0 : len(self.sp)], nl[len(self.sp) :]
        else:
            nl_t = None

        self.nl = nl
        self.nl_t = nl_t


class KoopmanModel(PyTreeNode):
    pi_0: jax.Array
    pi_1: jax.Array
    q_0: jax.Array
    q_1: jax.Array
    l_0: jax.Array
    l_1: jax.Array

    U: jax.Array
    s: jax.Array
    Vh: jax.Array

    cv_0: list[CV]
    cv_tau: list[CV]

    w: list[jax.Array] | None = None
    eps: float = 1e-10

    add_1: bool = False
    max_features: int = 2000
    out_dim: int | None = None
    method: str = "tcca"

    tau: float | None = None

    trans: CvTrans | None = None

    @staticmethod
    def create(
        w: list[jax.Array] | None,
        cv_0: list[CV],
        cv_tau: list[CV],
        add_1=False,
        eps=1e-10,
        method="tcca",
        symmetric=False,
        out_dim=None,
        koopman_weight=False,
        max_features=None,
        tau=None,
        macro_chunk=10000,
        chunk_size=None,
        verbose=False,
        trans: CvTrans | None = None,
    ):
        if max_features is None:
            max_features = cv_0[0].shape[1]

        if method == "tica":
            (cv0_white, cv_tau_white, transform_maf, pi, q, l) = data_loader_output._whiten(
                cv_0=cv_0,
                cv_tau=cv_tau,
                symmetric=symmetric,
                w=w,
                add_1=add_1,
                eps=eps,
                max_features=max_features,
                verbose=verbose,
                trans=trans,
            )

            # p = jnp.abs(eigval) > 1 + 1e-10
            # in_bounds = jnp.sum(p) == 0
            cov = Covariances.create(
                cv_0=cv0_white,
                cv_1=cv_tau_white,
                w=w,
                calc_pi=False,
                symmetric=symmetric,
            )

            k, u = cov.diagonalize(
                "C01",
                epsilon=eps,
                out_dim=out_dim,
            )

            pi_0, pi_1 = pi, pi
            q_0, q_1 = q, q
            l_0, l_1 = l, l
            U = u
            s = k
            Vh = jnp.linalg.inv(u)

        elif method == "tcca":
            (cv0_white, _, transform_maf_0, pi_0, q_0, l_0) = data_loader_output._whiten(
                cv_0=cv_0,
                w=w,
                eps=eps,
                max_features=max_features,
                add_1=add_1,
                macro_chunk=macro_chunk,
                chunk_size=chunk_size,
                verbose=verbose,
                trans=trans,
            )

            (cv_tau_white, _, transform_maf_1, pi_1, q_1, l_1) = data_loader_output._whiten(
                cv_0=cv_tau,
                w=w,
                eps=eps,
                max_features=max_features,
                add_1=add_1,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                verbose=verbose,
                trans=trans,
            )

            if verbose:
                print("koopman':gettig covariance")

            cov = Covariances.create(
                cv_0=cv0_white,
                cv_1=cv_tau_white,
                w=w,
                symmetric=False,
                calc_pi=False,
            )

            if verbose:
                print("koopman': SVD")

            # if out_dim is None:
            U, s, Vh = jnp.linalg.svd(cov.C01)

            if verbose:
                print(f"{s[0:10]=}")

        else:
            raise ValueError(f"method {method} not known")

        km = KoopmanModel(
            pi_0=pi_0,
            pi_1=pi_1,
            q_0=q_0,
            q_1=q_1,
            l_0=l_0,
            l_1=l_1,
            U=U,
            s=s,
            Vh=Vh,
            cv_0=cv_0,
            cv_tau=cv_tau,
            w=w,
            add_1=add_1,
            max_features=max_features,
            out_dim=out_dim,
            method=method,
            tau=tau,
            trans=trans,
        )

        if koopman_weight:
            km = km.weighted_model(
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                verbose=verbose,
            )

        return km

    @property
    def _q0(self):
        if self.add_1:
            out = jnp.zeros((self.q_0.shape[0] + 1, self.q_0.shape[1] + 1))
            out = out.at[1:, :-1].set(self.q_0)
            out = out.at[0, -1].set(1)

            return out
        return self.q_0

    @property
    def _q1(self):
        if self.add_1:
            out = jnp.zeros((self.q_1.shape[0] + 1, self.q_1.shape[1] + 1))
            out = out.at[1:, :-1].set(self.q_1)
            out = out.at[0, -1].set(1)

            return out
        return self.q_1

    @property
    def _l0(self):
        if self.add_1:
            return jnp.hstack([self.l_0, jnp.array([1])])
        return self.l_0

    @property
    def _l1(self):
        if self.add_1:
            return jnp.hstack([self.l_1, jnp.array([1])])
        return self.l_1

    def C00(self, pow=1):
        return self._q0 @ jnp.diag(self.l0_pow(pow)) @ self._q0.T

    def C11(self, pow=1):
        return self._q1 @ jnp.diag(self.l1_pow(pow)) @ self._q1.T

    def Tk(self, out_dim=None):
        # Optimal Data-Driven Estimation of Generalized Markov State Models for Non-Equilibrium Dynamics eq. 30

        if out_dim is None:
            out_dim = self.U.shape[1]

        return (
            self._q1
            @ jnp.diag(self.l1_pow(-1 / 2))
            @ self.Vh.T[:, :out_dim]
            @ jnp.diag(self.s[:out_dim])
            @ self.U.T[:out_dim, :]
            @ jnp.diag(self.l0_pow(1 / 2))
            @ self._q0.T
        )

    def whiten_f(self):
        return CvTrans.from_cv_function(
            data_loader_output._transform,
            static_argnames=["add_1"],
            add_1=False,
            q=self.q_0 @ jnp.diag(self.l0_pow(-1 / 2)),
            pi=self.pi_0,
        )

    def whiten_g(self):
        return CvTrans.from_cv_function(
            data_loader_output._transform,
            static_argnames=["add_1"],
            add_1=False,
            q=self.q_1 @ jnp.diag(self.l1_pow(-1 / 2)),
            pi=self.pi_1,
        )

    def f(self, out_dim=None):
        if self.add_1:
            o = self.q_0 @ jnp.diag(self.l0_pow(-1 / 2)[:-1]) @ self.U[:-1, 1:]
        else:
            o = self.q_0 @ jnp.diag(self.l0_pow(-1 / 2)) @ self.U

        if out_dim is not None:
            o = o[:, :out_dim]

        tr = CvTrans.from_cv_function(
            data_loader_output._transform,
            static_argnames=["add_1"],
            add_1=False,
            q=o,
            pi=self.pi_0,
        )

        if self.trans is not None:
            tr = self.trans * tr

        return tr

    def g(self, out_dim):
        if self.add_1:
            o = self.q_1 @ jnp.diag(self.l1_pow(-1 / 2)[:-1]) @ self.Vh[:-1, 1 : out_dim + 1]
        else:
            o = self.q_1 @ jnp.diag(self.l1_pow(-1 / 2)) @ self.Vh[:, :out_dim]

        tr = CvTrans.from_cv_function(
            data_loader_output._transform,
            static_argnames=["add_1"],
            add_1=False,
            q=o,
            pi=self.pi_1,
        )

        if self.trans is not None:
            tr = self.trans * tr

        return tr

    def l0_pow(self, pow=1, add_1=None):
        if pow < 0:
            return jnp.where(self._l0 > self.eps, self._l0**pow, 0)

        return self._l0**pow

    def l1_pow(self, pow=1):
        if pow < 0:
            return jnp.where(self._l1 > self.eps, self._l1**pow, 0)

        return self._l1**pow

    def koopman_weight(self, verbose=False) -> list[jax.Array]:
        # Optimal Data-Driven Estimation of Generalized Markov State Models, page 18-19

        n = jnp.sum(jnp.abs(self.s - 1) < 1e-10)

        if n == 0:
            print(
                f"Largest eigenvalue in koopman model is {self.s[0]} instead of 1. Connsider adding constant function"
            )
            n = 1

        if n > 1:
            print(
                f"{n=} eigenvalues with eigenvalue 1 found. This means that there are {n} stationary disjoint distrubitions distributions in the system.  Taking average of eigenvalues"
            )

        if verbose:
            print("getting eig A")
        # TODO : simplify, no calc needed?
        A = self.C00(pow=-1) @ self.C11() @ self.Tk(out_dim=n)
        l, v = jnp.linalg.eig(A)

        del A

        # truncate to only real values
        l = l[:n]
        v = v[:, :n]

        idx = jnp.argmin(jnp.abs(l - 1))

        l0 = l[idx]

        if not jnp.isclose(l0, 1, atol=1e-10):
            print(f"no eigenvalue with value 1: {l0}")

        idx2 = jnp.argwhere(jnp.abs(l - l0) < 1e-10)

        if len(idx2) > 1:
            # it doens't make sense to only take 1, as the order of eigenvlues is arbitrary
            print(
                f"multiple eigenvalues close to {l0=} found: {l[idx2]}. This means that there are {len(idx2) } stationary disjoint distrubitions distributions in the system.  Taking average of eigenvalues"
            )
            v = jnp.real(v[:, idx2].mean(axis=1))
        else:
            v = jnp.real(v[:, idx])

        print("got A ")

        w = []

        def _get_w(cv0_i: CV):
            if self.add_1:
                return jnp.einsum("ni,i->n", cv0_i.cv, v[:-1]) + v[-1]
            return jnp.einsum("ni,i->n", cv0_i.cv, v)

        w = jnp.hstack([_get_w(cv0_i) for cv0_i in self.cv_0])

        mask = w > 0

        n1 = jnp.sum(mask)
        n2 = jnp.sum(jnp.logical_not(mask))

        if n2 > n1:
            w = -w
            mask = jnp.logical_not(mask)
            n2, n1 = n1, n2

        s = jnp.sum(w[mask])

        w /= s

        if not mask.all():
            print(
                f"{ n2 }/{w.shape[0]} negative weights,  mean {  jnp.mean( w[ jnp.logical_not(mask) ])= }, min { jnp.min( w[ jnp.logical_not(mask) ])= }. Setting to zero",
            )

            w = w.at[w < 0].set(0)
        else:
            print("all weights positive")

        return data_loader_output._unstack_weights([cvi.shape[0] for cvi in self.cv_0], w)

    def weighted_model(
        self,
        add_1=False,
        chunk_size=None,
        macro_chunk=10000,
        verbose=False,
    ) -> KoopmanModel:
        return KoopmanModel.create(
            w=self.koopman_weight(
                verbose=verbose,
            ),
            cv_0=self.cv_0,
            cv_tau=self.cv_tau,
            add_1=add_1,
            eps=self.eps,
            method=self.method,
            symmetric=False,
            out_dim=self.out_dim,
            tau=self.tau,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
            trans=self.trans,
        )

    def timescales(self):
        s = self.s
        if self.add_1:
            s = s[1:]

        tau = self.tau
        if tau is None:
            tau = 1
            print("tau not set, assuming 1")

        return -tau / jnp.log(s)


class Covariances(PyTreeNode):
    C00: jax.Array
    C01: jax.Array
    C11: jax.Array
    pi_0: jax.Array
    pi_1: jax.Array

    sparse: bool = False
    only_diag: bool = False
    trans: CvTrans | None = None

    def create(
        cv_0: list[CV],
        cv_1: list[CV] | None = None,
        w: list[Array] | None = None,
        calc_pi=False,
        symmetric=True,
        macro_chunk=10000,
        chunk_size=None,
        only_diag=False,
        sparse=False,
        trans: CvTrans | None = None,
    ) -> Covariances:
        # TODO trans
        n = sum([cvi.shape[0] for cvi in cv_0])

        if trans is not None:
            cv0_shape = jax.eval_shape(lambda x: trans.compute_cv_trans(x)[0].cv, cv_0[0]).shape
        else:
            cv0_shape = cv_0[0].shape

        if cv_1 is not None:
            if trans is not None:
                cv1_shape = jax.eval_shape(lambda x: trans.compute_cv_trans(x)[0].cv, cv_1[0]).shape
            else:
                cv1_shape = cv_1[0].shape

        time_series = cv_1 is not None

        if w is None:
            w = [jnp.ones((cvi.shape[0],)) / n for cvi in cv_0]

        pi_0 = jnp.zeros((cv0_shape[1],))
        if time_series:
            pi_1 = jnp.zeros((cv1_shape[1],))
        else:
            pi_1 = None

        def cov_pi(carry, z_chunk, w):
            if time_series:
                cv0, cv1 = z_chunk.split()
                x0 = cv0.cv
                x1 = cv1.cv
            else:
                x0 = z_chunk.cv

            if only_diag:
                str = "ni,ni,n->i"
            else:
                str = "ni,nj,n->ij"

            C00 = jnp.einsum(str, x0, x0, w)
            pi0 = jnp.einsum("ni,n->i", x0, w)

            if time_series:
                C01 = jnp.einsum(str, x0, x1, w)
                C11 = jnp.einsum(str, x1, x1, w)

                pi1 = jnp.einsum("ni,n->i", x1, w)

            if time_series:
                if carry is not None:
                    C00 += carry[0]
                    C01 += carry[1]
                    C11 += carry[2]
                    pi0 += carry[3]
                    pi1 += carry[4]

                return (C00, C01, C11, pi0, pi1)

            if carry is not None:
                C00 += carry[0]
                pi0 += carry[1]

            return (C00, pi0)

        if cv_1 is not None:
            # trick: combine cv_0 and cv_1. Apply trans to both and then split again

            cv = [CV.combine(x, y) for x, y in zip(cv_0, cv_1)]

            if trans is not None:
                tr = trans + trans

        else:
            cv = cv_0

            if trans is not None:
                tr = trans

        if trans is not None:

            def f(x, _):
                return tr.compute_cv_trans(x)[0]

        else:

            def f(x, _):
                return x

        out = macro_chunk_map(
            f=f,
            op=CV.stack,
            y=cv,
            nl=None,
            macro_chunk=macro_chunk,
            verbose=True,
            chunk_func=cov_pi,
            chunk_func_init_args=None,
            w=w,
        )

        if time_series:
            C00, C01, C11, pi_0, pi_1 = out

        else:
            C00, pi_0 = out
            C01 = None
            C11 = None
            pi_1 = None

        C00 -= jnp.outer(pi_0, pi_0)
        if time_series:
            C11 -= jnp.outer(pi_1, pi_1)
            C01 -= jnp.outer(pi_0, pi_1)

        return Covariances(
            C00=C00,
            C01=C01,
            C11=C11,
            pi_0=pi_0,
            pi_1=pi_1,
            sparse=sparse,
            only_diag=only_diag,
            trans=trans,
        )

    def diagonalize(self, choice, epsilon=1e-10, out_dim=None):
        if choice == "C00":
            C = self.C00
            sym = True
        elif choice == "C11":
            C = self.C11
            sym = True
        elif choice == "C01":
            C = self.C01
            sym = False
        elif choice == "C10":
            C = self.C01.T
            sym = False
        else:
            raise ValueError(f"choice {choice} not known")

        if self.sparse:
            raise NotImplementedError("sparse not implemented")

        if sym:
            k, u = jnp.linalg.eigh(C)
        else:
            k, u = jnp.linalg.eig(C)

            idx = jnp.argsort(jnp.abs(k))
            k = k[idx]
            u = u[:, idx]

        argmask = jnp.argwhere(jnp.abs(k) > epsilon).reshape(-1)

        if len(argmask) == 0:
            raise ValueError("no eigenvalues larger than epsilon")

        if out_dim is not None:
            if out_dim > len(argmask):
                out_dim = len(argmask)

            argmask = argmask[-(out_dim + 1) :]

        k = k[argmask]
        u = u[:, argmask]

        return k, u, argmask
