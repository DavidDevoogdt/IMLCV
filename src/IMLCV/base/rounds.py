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
    SystemParams,
    macro_chunk_map,
    padded_shard_map,
    padded_vmap,
)
from IMLCV.base.CVDiscovery import Transformer
from IMLCV.base.MdEngine import MDEngine, StaticMdInfo, TrajectoryInfo
from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.configs.config_general import Executors
from IMLCV.implementations.bias import RbfBias, _clip


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
        divide_by_histogram=False,
        n_max=30,
        wham=True,
    ) -> data_loader_output:
        if cv_round is None:
            cv_round = self.cv

        if uniform and T_scale != 1:
            print(f"WARNING: uniform and {T_scale=} are not compatible")

        if uniform and divide_by_histogram:
            print("WARNING: uniform and divide_by_histogram are not compatible")

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

        ###################

        if verbose:
            print("obtaining raw data")

        sp: list[SystemParams] = []
        cv: list[CV] = []
        ti: list[TrajectoryInfo] = []
        weights: list[Array] = []
        weights_bincount: list[Array] = []

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
                    bias=bias_c,
                    # time_series=time_series,
                    ground_bias=ground_bias_c,
                )

                # select points according to free energy divided by histogram count
                w_out = dlo.weights(
                    chunk_size=chunk_size,
                    koopman=False,
                    n_max=n_max,
                    # wham_sub_grid=3,  # quick settings
                    verbose=verbose,
                    output_bincount=divide_by_histogram,
                    wham=wham,
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

        for w_i, spi in zip(weights, sp):
            assert (
                w_i.shape[0] == spi.shape[0]
            ), f"weights and sp shape are different: {w_i.shape=} {StopAsyncIteration.shape=}"

        if weights is not None:
            c = 0

            for w in weights:
                if lag_n != 0:
                    if len(w) >= lag_n:
                        n = jnp.sum(jnp.logical_and(w[:-lag_n] != 0, w[lag_n:] != 0))
                        c += n
                else:
                    c += jnp.sum(w != 0)

            total = c

        else:
            total = sum([max(a.shape[0] - lag_n, 0) for a in sp])

        if out == -1:
            out = total

        # if out > total / lag_n:
        #     print(f"not using more data than {total} nonzero data / {lag_n=} ")
        #     out = total / lag_n

        if out > total:
            print(f"not enough data, returning {total} data points instead of {out}")
            out = total

        ###################

        if verbose:
            print(f"total data points {total}, selecting {out}")

        def choose(
            key,
            weight: Array | None,
            out: int,
            # len: int | None,
            histogram_prob: Array | None = None,
            T_scale=1,
        ):
            # if len is None:
            l = weight.shape[0]

            key, key_return = split(key, 2)

            if weight is None:
                weight = jnp.ones(l)
                reweight = jnp.ones_like(weight)

            else:
                reweight = (weight > 0) * 1.0

                if uniform:
                    reweight, weight = weight, reweight

                else:
                    if divide_by_histogram:
                        assert histogram_prob is not None

                        select_weight_new = weight / histogram_prob
                        new_reweight = reweight * weight / select_weight_new

                        weight = select_weight_new
                        reweight = new_reweight

                    if T_scale != 1:
                        select_weight_new = weight ** (1 / T_scale)
                        new_reweight = reweight * weight / select_weight_new

                        weight = select_weight_new
                        reweight = new_reweight

            assert out <= l, f"{out=} {l=}"

            if not uniform and (out, 2 * l):
                print("WARNING: point selection will likely be biased, pass uniform=True to avoid this")

            indices = choice(
                key=key,
                a=l,
                shape=(int(out),),
                p=weight,
                replace=False,
            )

            return key_return, indices, reweight

        def remove_lag(w, lag_n, w_bc=None):
            if lag_n != 0:
                w = w[:-lag_n]
                if w_bc is not None:
                    w_bc = w_bc[:-lag_n]

            return w, w_bc

        out_indices = []
        out_reweights = []
        n_list = []

        if split_data:
            frac = out / total

            for n, w_i in enumerate(weights):
                if w_i is not None:
                    w_i, w_i_bc = remove_lag(
                        w_i,
                        lag_n,
                        weights_bincount[n] if divide_by_histogram else None,
                    )

                ni = int(frac * (sp[n].shape[0] - lag_n))

                key, indices, reweight = choose(
                    key=key,
                    weight=w_i,
                    out=ni,
                    # len=sp[n].shape[0] - lag_n,
                    histogram_prob=w_i_bc,
                    T_scale=T_scale,
                )

                out_indices.append(indices)
                out_reweights.append(reweight)
                n_list.append(n)

        else:
            if weights[0] is None:
                w = None
                w_bc = None
            else:
                w = []

                if divide_by_histogram:
                    w_bc = []
                else:
                    w_bc = None

                for n, w_i in enumerate(weights):
                    w_i, w_i_bc = remove_lag(
                        w_i,
                        lag_n,
                        weights_bincount[n] if divide_by_histogram else None,
                    )

                    w.append(w_i)
                    if divide_by_histogram:
                        w_bc.append(w_i_bc)

                w = jnp.hstack(w)

                if divide_by_histogram and w_bc is not None:
                    w_bc = jnp.hstack(w_bc)

            key, indices, reweight = choose(
                key=key,
                weight=w,
                out=int(out),
                # len=total,
                histogram_prob=w_bc,
                T_scale=T_scale,
            )

            count = 0

            for n, (sp_n) in enumerate(sp):
                n_i = sp_n.shape[0] - lag_n

                indices_full = indices[jnp.logical_and(count <= indices, indices < count + n_i)]
                index = indices_full - count

                if len(index) == 0:
                    count += n_i
                    continue

                out_indices.append(index)
                out_reweights.append(reweight)

                n_list.append(n)

                count += n_i

        ###################
        # storing data    #
        ###################

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
        else:
            out_biases = None

        for n, indices_n, reweights_n in zip(n_list, out_indices, out_reweights):
            out_sp.append(sp[n][indices_n])
            out_cv.append(cv[n][indices_n])
            out_ti.append(ti[n][indices_n])
            out_weights.append(reweights_n[indices_n])

            if time_series:
                out_sp_t.append(sp[n][indices_n + lag_n])
                out_cv_t.append(cv[n][indices_n + lag_n])
                out_ti_t.append(ti[n][indices_n + lag_n])

            if get_bias_list:
                out_biases.append(bias_list[n])

        s = 0
        for ow in out_weights:
            s += jnp.sum(ow)

        out_weights = [ow / s for ow in out_weights]

        print(f"len(out_sp) = {len(out_sp)} ")

        ###################

        if verbose:
            print("Checking data")

        out_nl = None

        if time_series:
            out_nl_t = None

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
        )

        if time_series:
            dlo_kwargs.update(
                sp_t=out_sp_t,
                nl_t=out_nl_t,
                cv_t=out_cv_t,
                ti_t=out_ti_t,
                tau=tau,
            )

        dlo = data_loader_output(
            **dlo_kwargs,
        )

        ###################

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

        ###################

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
        macro_chunk=10000,
        lag_n=20,
        divide_by_histogram=True,
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
                weight=False,
                T_scale=T_scale,
                time_series=False,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                verbose=True,
                # lag_n=lag_n,
            )

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

        bias.plot(
            name=outputs[0].filepath,
            traj=[cvs],
            offset=True,
            margin=0.1,
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

        x, _ = dlo.apply_cv(cv_trans=cv_trans, x=dlo.cv, chunk_size=chunk_size, verbose=verbose)

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
    nl: list[NeighbourList] | NeighbourList | None
    cv: list[CV]
    sti: StaticMdInfo
    ti: list[TrajectoryInfo]
    collective_variable: CollectiveVariable
    sp_t: list[SystemParams] | None = None
    nl_t: list[NeighbourList] | NeighbourList | None = None
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
        return [
            d.cv.reshape((-1,))
            for d in CV(
                cv=jnp.expand_dims(weights, 1),
                _stack_dims=stack_dims,
            ).unstack()
        ]

    def weights(
        self,
        samples_per_bin=50,
        n_max=30,
        n_max_koopman=50,
        ground_bias=None,
        # use_ground_bias=True,
        # sign=1,
        chunk_size=None,
        wham=True,
        koopman=False,
        indicator_CV=True,
        # wham_sub_grid=5,
        wham_eps=1e-10,
        koopman_eps=1e-12,
        koopman_eps_pre=1e-12,
        cv_0: list[CV] | None = None,
        cv_t: list[CV] | None = None,
        force_recalc=False,
        macro_chunk=1000,
        verbose=False,
        output_bincount=False,
        max_features_koopman=10000,
        margin=0.1,
        add_1=True,
        bias_cutoff=8,  # bigger values can lead to nans
        min_f_i=1e-30,
        # cluster_fraction=0.1,
        only_diag=False,
        return_km=False,
        calc_pi=False,
    ) -> list[jax.Array]:
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

        def norm_w(w_stacked):
            return w_stacked / jnp.sum(w_stacked)

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

        if self._weights is not None and not force_recalc:
            print("using precomputed weights")
            w_stacked = jnp.hstack(self._weights)
            w_stacked = check_w(w_stacked)
            w_stacked = norm_w(w_stacked)

        else:
            # TODO:https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00867

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

            grid_nums = None

            beta = 1 / (self.sti.T * boltzmann)

            if ground_bias is None:
                ground_bias = self.ground_bias

            # get raw rescaling
            w_unstacked = []

            nm_tot = []
            n_tot = 0

            offset_i = []

            for ti_i in self.ti:
                e = ti_i.e_bias

                if e is None:
                    if verbose:
                        print("WARNING: no bias enerrgy found")
                    e = jnp.ones((ti_i.sp.shape[0],))

                offset = jnp.median(e)
                offset_i.append(offset)

                u = beta * (e - offset)

                prob = jnp.exp(u)

                # remove everthing that is very improbable
                if bias_cutoff is not None:
                    m = jnp.logical_or(u > bias_cutoff * jnp.log(10) / 2, u < -bias_cutoff * jnp.log(10) / 2)

                    if (n_o := jnp.sum(m)) > 0:
                        nm_tot.append(n_o)
                        n_tot += 1
                        prob = jnp.where(m, 0, prob)

                # keep numbers  small
                # prob /= jnp.sum(prob)

                w_unstacked.append(prob)

            if verbose:
                if n_tot > 0:
                    print(
                        f"WARNING: {n_tot} trajectories have bias energy below {bias_cutoff=}, {sum(nm_tot)=}/{sum(sd)} samples removed (total {sum(nm_tot)/sum(sd)*100:.2f}%)"
                    )
                    print(f"{jnp.array(nm_tot)=}")

            w_stacked = jnp.hstack(w_unstacked)
            w_stacked = check_w(w_stacked)
            w_stacked = norm_w(w_stacked)

            if wham:
                assert self.bias is not None

                if verbose:
                    print("step 1 wham")

                if grid_nums is None:
                    grid_nums, _ = self.apply_cv(
                        closest,
                        cv_0,
                        chunk_size=chunk_size,
                        macro_chunk=macro_chunk,
                    )

                def get_wham(w_unstacked, grid_nums):
                    hist = get_histo(grid_nums, [wi > 0 for wi in w_unstacked])
                    hist_mask = hist > 0

                    b_ik = jnp.zeros((len(w_unstacked), jnp.sum(hist_mask)))

                    for i in range(len(w_unstacked)):
                        # mask part of potential that is not sampled

                        wi_inv = jnp.where(w_unstacked[i] > 0, 1 / w_unstacked[i], 0)

                        hist_i_weights = get_histo([grid_nums[i]], [wi_inv])[hist_mask]
                        hist_i_num = get_histo([grid_nums[i]], [w_unstacked[i] > 0])[hist_mask]

                        b_k = jnp.where(hist_i_num != 0, hist_i_weights / hist_i_num, 0)

                        b_ik = b_ik.at[i, :].set(b_k)

                    print(f"{jnp.max(b_ik)=}")

                    b_ik /= jnp.mean(b_ik)

                    a_k = jnp.ones((b_ik.shape[1],))
                    a_k /= jnp.sum(a_k)

                    N_i = jnp.array([jnp.sum(wi > 0) for wi in w_unstacked])

                    @jax.jit
                    def T(a_k, x):
                        # log_a_k = jnp.log(a_k)

                        b_ik, N_i = x

                        f_i = jnp.einsum("k,ik->i", a_k, b_ik)
                        f_i_safe = jnp.where(f_i < min_f_i, min_f_i, f_i)
                        a_k_new = jnp.einsum("i,ik->k", N_i / f_i_safe, b_ik)
                        a_k_new = a_k_new / jnp.sum(a_k_new)

                        return a_k_new, f_i

                    def norm(a_k, x):
                        a_k_p = T(a_k, x)[0]

                        return 0.5 * jnp.sum((a_k - a_k_p) ** 2) / jnp.shape(a_k)[0]

                    def kl_div(a_k, x):
                        a_k_p, f = T(a_k, x)

                        a_k = jnp.where(a_k >= min_f_i, a_k, min_f_i)
                        a_k_p = jnp.where(a_k_p >= min_f_i, a_k_p, min_f_i)

                        return jnp.sum(a_k * (jnp.log(a_k) - jnp.log(a_k_p)))

                    import jaxopt

                    solver = jaxopt.ProjectedGradient(
                        fun=norm,
                        projection=jaxopt.projection.projection_simplex,  # prob space is simplex
                        maxiter=20000,
                        tol=wham_eps,
                    )

                    out = solver.run(a_k, x=(b_ik, N_i))

                    a_k = out.params

                    # print(f"{a_k=} {out=}")

                    if verbose:
                        n, k = norm(a_k, (b_ik, N_i)), kl_div(a_k, (b_ik, N_i))
                        print(f"wham err={n}, kl divergence={k} {out.state.iter_num=} {out.state.error=} ")

                    _, f = T(a_k, (b_ik, N_i))

                    w_stacked = jnp.hstack([wi * c for wi, c in zip(w_unstacked, f)])

                    print(f"{jnp.isnan(w_stacked).any()}")

                    w_stacked = check_w(w_stacked)
                    w_stacked = norm_w(w_stacked)

                    probs = unstack_w(w_stacked, stack_dims=[wi.shape[0] for wi in w_unstacked])

                    return probs

                w_unstacked = unstack_w(w_stacked)

                w_unstacked = get_wham(
                    w_unstacked,
                    grid_nums,
                )

                w_stacked = jnp.hstack(w_unstacked)

                w_stacked = check_w(w_stacked)
                w_stacked = norm_w(w_stacked)

            else:
                if verbose:
                    print("WARNING: not using wham")

        if koopman and self.time_series:
            n_hist = CvMetric.get_n(
                samples_per_bin=samples_per_bin,
                samples=tot_samples,
                n_dims=ndim,
            )

            if n_hist > n_max:
                n_hist = n_max

            if verbose:
                print(f"using {n_hist=}")

            if verbose:
                print("koopman weights")

            skip = False

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

                    w_pos = [(a > 0) * 1.0 for a in unstack_w(w_stacked)]

                    hist = get_histo(grid_nums, w_pos)
                    hist_t = get_histo(grid_nums_t, w_pos)

                    mask = jnp.argwhere(jnp.logical_and(hist > 0, hist_t > 0)).reshape(-1)

                    @partial(CvTrans.from_cv_function, mask=mask)
                    def get_indicator(cv: CV, nl, _, shmap, mask):
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

            if not skip:
                if verbose:
                    print("constructing koopman model")

                # construct koopman model
                km = self.koopman_model(
                    cv_0=cv_km,
                    cv_tau=cv_km_t,
                    nl=None,
                    w=unstack_w(w_stacked),
                    add_1=True,
                    method="tcca",
                    symmetric=False,
                    chunk_size=chunk_size,
                    macro_chunk=macro_chunk,
                    out_dim=None,
                    verbose=verbose,
                    eps=koopman_eps,
                    eps_pre=koopman_eps_pre,
                    trans=tr,
                    max_features=max_features_koopman,
                    max_features_pre=max_features_koopman,
                    only_diag=only_diag,
                    calc_pi=True,
                )

                if verbose:
                    print("koopman_weight")

                # return km

                w_unstacked, _ = km.koopman_weight(
                    chunk_size=chunk_size,
                    macro_chunk=macro_chunk,
                    verbose=verbose,
                )

                # return w_unstacked, w_unstacked_corr

                w_stacked = jnp.hstack(w_unstacked)
                w_stacked = check_w(w_stacked)
                w_stacked = norm_w(w_stacked)

        else:
            if verbose:
                print("not using koopman weights")

        w_unstacked = unstack_w(w_stacked)

        if output_bincount:
            if grid_nums is None:
                grid_nums, _ = self.apply_cv(
                    closest,
                    cv_0,
                    chunk_size=chunk_size,
                    macro_chunk=macro_chunk,
                )

            hist = get_histo(grid_nums, w_unstacked)

            bin_counts = [hist[i.cv].reshape((-1,)) for i in grid_nums]

            return w_unstacked, bin_counts

        if return_km:
            assert koopman
            return w_unstacked, km

        return w_unstacked

    @staticmethod
    def _transform(
        cv,
        nl,
        _,
        shmap,
        argmask: jnp.Array | None = None,
        pi: jnp.Array | None = None,
        add_1: bool = False,
        q: jnp.Array | None = None,
        l: jnp.Array | None = None,
    ):
        x = cv.cv

        if argmask is not None:
            x = x[argmask]

        if pi is not None:
            x = x - pi

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
        symmetric=False,
        w: list[jax.Array] | None = None,
        eps=1e-6,
        eps_pre=1e-3,
        max_features=500,
        max_features_pre=500,
        out_dim=50,
        add_1=False,
        chunk_size=None,
        macro_chunk=10000,
        verbose=False,
        trans=None,
        T_scale=1,
        only_diag=False,
        calc_pi=False,
    ) -> KoopmanModel:
        # TODO: https://www.mdpi.com/2079-3197/6/1/22
        assert method in ["tica", "tcca"]

        if cv_0 is None:
            cv_0 = self.cv

        if cv_tau is None:
            cv_tau = self.cv_t

        if w is None:
            print("W koopman model not given, will use uniform weights")

            t = sum([cvi.shape[0] for cvi in cv_0])
            w = [jnp.ones((cvi.shape[0],)) / t for cvi in cv_0]

        return KoopmanModel.create(
            w=w,
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

        nan_x, nan_x_t = self.apply_cv(
            check_nan,
            x,
            x_t,
            macro_chunk=macro_chunk,
            shmap=True,
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
        macro_chunk=10000,
        shmap=True,
        verbose=False,
    ) -> tuple[list[CV], list[CV] | None]:
        # if nl is None:
        #     nl = self.nl
        # if nl_t is None:
        #     nl_t = self.nl_t

        if isinstance(f, CvTrans):
            _f = f.compute_cv_trans
        elif isinstance(f, CvFlow):
            _f = f.compute_cv_flow
        else:
            raise ValueError(f"{f=} must be CvTrans or CvFlow")

        def f(x, nl):
            return _f(
                x,
                nl,
                chunk_size=chunk_size,
                shmap=False,
            )[0]

        if shmap:
            # TODO 0909 09:41:06.108200 3141823 collective_ops_utils.h:306] This thread has been waiting for 5000ms for and may be stuck: participant AllReduceParticipantData{rank=27, element_count=1, type=PRED, rendezvous_key=RendezvousKey{run_id=RunId: 5299332, global_devices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], num_local_participants=32, collective_op_kind=cross_module, op_id=3}} waiting for all participants to arrive at rendezvous RendezvousKey{run_id=RunId: 5299332, global_devices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], num_local_participants=32, collective_op_kind=cross_module, op_id=3}
            f = jax.jit(f)
            f = padded_shard_map(f, pmap=True)

        return data_loader_output._apply(
            x=x,
            x_t=x_t,
            nl=nl,
            nl_t=nl_t,
            f=f,
            macro_chunk=macro_chunk,
            verbose=verbose,
            jit_f=not shmap,
        )

    @staticmethod
    def _apply(
        x: CV | SystemParams,
        f: Callable,
        x_t: CV | SystemParams | None = None,
        nl: NeighbourList | NeighbourListUpdate | None = None,
        nl_t: NeighbourList | NeighbourListUpdate | None = None,
        macro_chunk=10000,
        verbose=False,
        jit_f=True,
    ):
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
        )

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
    def _get_fes_bias_from_weights(
        T,
        weights: list[jax.Array],
        collective_variable: CollectiveVariable,
        cv: list[CV],
        samples_per_bin=10,
        min_samples_per_bin: int | None = 3,
        n_max=50,
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

        print(f"getting bounds {n_grid=}")

        if n_grid <= 1:
            raise

        grid_bounds, _, _ = CvMetric.bounds_from_cv(
            cv,
            weights=weights,
            margin=0.2,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
            percentile=1e-6,
        )

        print(f"{grid_bounds=}")

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

        grid_nums, _ = data_loader_output.apply_cv(
            x=cv,
            f=closest,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
        )

        p_grid = get_histo(grid_nums, weights)

        mask = p_grid > 0

        # print(f"{p_grid=}  {p_grid[p_grid<0]} {jnp.sum(p_grid>0)} ")

        p_grid = p_grid[mask]

        fes_grid = -jnp.log(p_grid) / beta
        fes_grid -= jnp.min(fes_grid)

        bias = RbfBias.create(
            cvs=collective_variable,
            cv=cv_mid[mask],
            kernel="linear",
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
        shmap=True,
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

    def recalc(self, chunk_size=None, macro_chunk=10000, shmap=True, verbose=False):
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
        macro_chunk=10000,
        verbose=False,
        only_update=False,
        chunk_size_inner=10,
    ):
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

            def f(nl_update: NeighbourListUpdate | None, sp, *_):
                # print(f"updating neighbour list {nl_update=}")

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
            jit_f=False,
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

    # argmask_0: jax.Array | None
    # argmask_1: jax.Array | None
    argmask: jax.Array | None

    shape: int

    cv_0: list[CV]
    cv_tau: list[CV]
    nl: list[NeighbourList] | NeighbourList | None
    nl_t: list[NeighbourList] | NeighbourList | None

    w: list[jax.Array] | None = None
    eps: float = 1e-10

    only_diag: bool = False
    calc_pi: bool = True

    add_1: bool = False
    max_features: int = 500
    max_features_pre: int = 1000
    out_dim: int | None = None
    method: str = "tcca"

    tau: float | None = None
    T_scale: float = 1.0

    trans: CvTrans | CvFlow | None = None

    @staticmethod
    def create(
        w: list[jax.Array] | None,
        cv_0: list[CV],
        cv_tau: list[CV],
        nl: list[NeighbourList] | NeighbourList | None = None,
        nl_t: list[NeighbourList] | NeighbourList | None = None,
        add_1=True,
        eps=1e-6,
        eps_pre=1e-10,
        method="tcca",
        symmetric=False,
        out_dim=None,  # maximum dimension for koopman model
        koopman_weight=False,
        max_features=500,
        max_features_pre=500,
        tau=None,
        macro_chunk=10000,
        chunk_size=None,
        verbose=False,
        trans: CvTrans | None = None,
        T_scale=1,
        only_diag=False,
        calc_pi=True,
        use_scipy=False,
        auto_cov_threshold=1e-6,
    ):
        # if max_features is None:
        #     max_features = cv_0[0].shape[1]

        # if add_1:
        #     if trans is not None:
        #         trans = trans * CvTrans.from_cv_function(KoopmanModel._add_1)
        #     else:
        #         trans = CvTrans.from_cv_function(KoopmanModel._add_1)

        #  see Optimal Data-Driven Estimation of Generalized Markov State Models
        if verbose:
            print("getting covariances")

        cov = Covariances.create(
            cv_0=cv_0,
            cv_1=cv_tau,
            nl=nl,
            nl_t=nl_t,
            w=w,
            calc_pi=calc_pi,
            only_diag=only_diag,
            symmetric=symmetric,
            T_scale=T_scale,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            trans_f=trans,
            trans_g=trans,
        )
        # correct the covariance matrix for the constant added function

        if add_1:
            if verbose:
                print("adding constant to covariance")

            C00 = jnp.zeros((cov.C00.shape[0] + 1, cov.C00.shape[1] + 1))
            C11 = jnp.zeros((cov.C11.shape[0] + 1, cov.C11.shape[1] + 1))
            C01 = jnp.zeros((cov.C01.shape[0] + 1, cov.C01.shape[1] + 1))

            C00 = C00.at[:-1, :-1].set(cov.C00)
            C11 = C11.at[:-1, :-1].set(cov.C11)
            C01 = C01.at[:-1, :-1].set(cov.C01)

            C00 = C00.at[-1, -1].set(1)
            C00 = C00.at[-1, :-1].set(cov.pi_0)
            C00 = C00.at[:-1, -1].set(cov.pi_0)

            C11 = C11.at[-1, -1].set(1)
            C11 = C11.at[-1, :-1].set(cov.pi_1)
            C11 = C11.at[:-1, -1].set(cov.pi_1)

            C01 = C01.at[-1, -1].set(1)
            C01 = C01.at[-1, :-1].set(cov.pi_1)
            C01 = C01.at[:-1, -1].set(cov.pi_0)

            pi_0 = jnp.zeros((cov.pi_0.shape[0] + 1,))
            pi_1 = jnp.zeros((cov.pi_1.shape[0] + 1,))

            pi_0 = pi_0.at[:-1].set(cov.pi_0)
            pi_1 = pi_1.at[:-1].set(cov.pi_1)

            cov.C00 = C00
            cov.C11 = C11
            cov.C01 = C01

            # if trans is not None:
            #     trans = trans * CvTrans.from_cv_function(KoopmanModel._add_1)
            # else:
            #     trans = CvTrans.from_cv_function(KoopmanModel._add_1)

        # start with argmask for auto covariance. Remove all features with variances that are too small, or auto covariance that are too small
        auto_cov = jnp.einsum("i,i,i->i", jnp.diag(cov.C00) ** (-0.5), jnp.diag(cov.C01), jnp.diag(cov.C11) ** (-0.5))
        argmask = jnp.argsort(auto_cov, descending=True).reshape(-1)

        # print(f"{auto_cov[argmask]=} {cov.C00[argmask, argmask]=} {cov.C11[argmask, argmask]=}")

        # max_std_C00 = jnp.max(jnp.diag(cov.C00))
        # max_std_C11 = jnp.max(jnp.diag(cov.C11))

        # m = argmask[cov.C00[argmask, argmask] > eps_pre * max_std_C00]
        # argmask = argmask[m]
        # m = argmask[cov.C11[argmask, argmask] > eps_pre * max_std_C11]
        # argmask = argmask[m]
        argmask = argmask[auto_cov[argmask] > auto_cov_threshold]

        if max_features_pre is not None:
            if argmask.shape[0] > max_features_pre:
                argmask = argmask[:max_features_pre]
                print(f"reducing argmask to {max_features_pre}")
        else:
            print(f"{argmask.shape=}")

        # print(f"{auto_cov[argmask]=} {cov.C00[argmask, argmask]=} {cov.C11[argmask, argmask]=}")

        cov.C00 = cov.C00[argmask, :][:, argmask]
        cov.C11 = cov.C11[argmask, :][:, argmask]
        cov.C01 = cov.C01[argmask, :][:, argmask]

        if calc_pi:
            cov.pi_0 = cov.pi_0[argmask]
            cov.pi_1 = cov.pi_1[argmask]

        if verbose:
            print("diagonalizing C00")

        W0 = cov.whiten(
            "C00",
            epsilon=eps,
            epsilon_pre=eps_pre,
            verbose=verbose,
            use_scipy=use_scipy,
        )

        if verbose:
            print(f"{W0.shape=}")

        if verbose:
            print("diagonalizing C11")

        if symmetric:
            W1 = W0
            # argmask_1 = argmask_0

        else:
            W1 = cov.whiten(
                "C11",
                epsilon=eps,
                epsilon_pre=eps_pre,
                verbose=verbose,
                use_scipy=use_scipy,
            )
            if verbose:
                print(f"{W1.shape=}")

        cov.C00 = cov.C00
        cov.C11 = cov.C11
        cov.C01 = cov.C01

        if calc_pi:
            cov.pi_0 = cov.pi_0
            cov.pi_1 = cov.pi_1

        Kt = W0 @ cov.C01 @ W1.T

        if verbose:
            print("koopman': SVD")

        # same U,sigma,V as in main text
        U, s, VT = jax.numpy.linalg.svd(Kt, hermitian=symmetric)

        # W0 and W1 are whitening transforms. A whitening transform is still whitening after a rotation
        W0 = U.T @ W0
        W1 = VT @ W1  # V is already transposed

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
            # argmask_0=argmask[argmask_0],
            # argmask_1=argmask[argmask_1],
            only_diag=only_diag,
            eps=eps,
            shape=cov.C01.shape[0],
        )

        if koopman_weight:
            km = km.weighted_model(
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                verbose=verbose,
            )

        return km

    @staticmethod
    def _add_1(
        cv,
        nl,
        _,
        shmap,
    ):
        return cv.replace(cv=jnp.hstack([cv.cv, jnp.array([1])]))

    def Tk(self, out_dim=None):
        # Optimal Data-Driven Estimation of Generalized Markov State Models for Non-Equilibrium Dynamics eq. 30
        # T_k = C11^{-1/2} K^T C00^{1/2}
        #     = w1.T K.T w0 C00

        if out_dim is None:
            out_dim = self.s.shape[0]

        print(f"{self.W0.shape=} {self.W1.shape=} {self.cov.C00.shape=} {self.s.shape=}")

        Tk = self.W1[:out_dim, :].T @ jnp.diag(self.s[:out_dim]) @ self.W0[:out_dim, :] @ self.cov.C00

        return Tk

    @property
    def tot_trans(self):
        tr = self.trans

        if self.add_1 is not None:
            if tr is not None:
                tr *= CvTrans.from_cv_function(KoopmanModel._add_1)

            else:
                tr = CvTrans.from_cv_function(KoopmanModel._add_1)

        return tr

    def f(self, out_dim=None):
        o = self.W0.T

        # # find eigenvalues largest non constant eigenvalues
        # idx = jnp.abs(self.s - 1) < 1e-6

        # if (l := jnp.sum(idx)) != 0:
        #     print(f"found {jnp.sum(idx)} constant eigenvalues, removing {self.s[idx]=}")
        #     o = o[:, jnp.logical_not(idx)]

        # if l==0 and (self.add_1   )

        o = o[:, :out_dim]

        tr = CvTrans.from_cv_function(
            data_loader_output._transform,
            static_argnames=["add_1"],
            add_1=False,
            q=o,
            pi=self.cov.pi_0,
            argmask=self.argmask,
        )

        if self.tot_trans is not None:
            tr = self.tot_trans * tr

        return tr

    def g(self, out_dim=None):
        # this performs y =  (trans*g_trans)(x) @ Vh[:out_dim,:], but stores smaller matrices

        o = self.W1.T

        # find eigenvalues largest non constant eigenvalues
        if self.add_1:
            o = o[1:, :]

        o = o[:out_dim, :]

        tr = CvTrans.from_cv_function(
            data_loader_output._transform,
            static_argnames=["add_1"],
            add_1=False,
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
        epsilon=1e-4,
        max_entropy=True,
    ) -> tuple[list[jax.Array], bool]:
        # Optimal Data-Driven Estimation of Generalized Markov State Models, page 18-19
        # create T_k in the trans basis
        # T_n = C00^{-1} C11 T_k
        # T_K = C11^{-1/2} U S Vh C00^{-1/2} C00
        # T_n mu_corr =  (lambda=1) * mu_corr
        #  C00^{-1/2} C11  C11^ {-1/2} K^T C00^{-1/2}  C00 mu_corr )= (C00^{-1/2} C00 mu_corr)
        # w0 C11 W1.T K^T v = lambda v
        # mu_corr = W0^T v

        out_dim = self.s.shape[0]

        # only use largest eigenvalues of the koopman model
        A = self.W0[:out_dim, :] @ self.cov.C11 @ self.W1[:out_dim, :].T @ jnp.diag(self.s[:out_dim])

        from scipy.sparse.linalg import eigs

        l, v = eigs(
            A=np.array(A),
            sigma=1,
            k=10,
            which="LM",
        )

        print(f"{l=} ")

        l = jnp.array(l)
        v = jnp.array(v)

        mu_corr = self.W0[:out_dim, :].T @ v

        # remove complex eigenvalues, as they cannot be the ground state
        real = jnp.abs(jnp.imag(l)) < self.eps
        l = l[real]
        mu_corr = mu_corr[:, real]

        idx = jnp.argsort(jnp.abs(l - 1), descending=False).reshape((-1,))

        l = jnp.real(l[idx])
        mu_corr = jnp.real(mu_corr[:, idx])

        # mask_pre = jnp.abs(l - 1) < 1e-14

        # if (nl := jnp.sum(mask_pre)) != 0:
        #     print(f"{nl} eigenvalues exact 1, removing")
        #     l = l[jnp.logical_not(mask_pre)]
        #     mu_corr = mu_corr[:, jnp.logical_not(mask_pre)]

        print(f"{l=} ")

        if len(l) == 0:
            print("no eigenvalues found")
            return self.w, False

        l0_eps = jnp.abs(1 - l[0])
        s0_eps = jnp.abs(1 - self.s[0])
        eps_s = jnp.max(jnp.array([l0_eps * 10, s0_eps * 10, self.eps * 10, 1e-2]))

        mask = jnp.abs(l - 1) < eps_s

        # print(f"{mask=} {eps_s=}")

        # include all relevant eigenvalues, or at least 5 if there are more than 5 eigenvalues
        out_dim = max(int(jnp.sum(mask)), min(5, len(l)))

        l = l[:out_dim]
        mu_corr = mu_corr[:, :out_dim]

        ## w_i = xT mu_corr

        f_trans_2 = CvTrans.from_cv_function(
            data_loader_output._transform,
            q=None,
            l=None,
            pi=self.cov.pi_0,
            argmask=self.argmask,
        )

        @partial(CvTrans.from_cv_function, mu_corr=mu_corr)
        def _get_w(cv: CV, _nl, _, shmap, mu_corr: Array):
            x = cv.cv

            return cv.replace(cv=jnp.einsum("i,ij->j", x, mu_corr), _combine_dims=None)

        tr = f_trans_2 * _get_w

        if self.tot_trans is not None:
            tr = self.tot_trans * tr

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
        w_out = CV.stack(*w_out_cv).cv

        w_orig = jnp.hstack(self.w)

        def _norm(w, a=False):
            w = jnp.where(jnp.sum(w) > 0, w, -w)
            n_neg = jnp.sum(w < 0)
            w = w * w_orig

            if a:
                n = jnp.sum(jnp.abs(w))
            else:
                n = jnp.sum(jnp.where(w > 0, w, 0))

            w /= n

            w_pos = jnp.where(w > 0, w, 0)
            w_neg = jnp.where(w < 0, -w, 0)

            return w, w_pos, jnp.sum(w_neg), n_neg / w.shape[0]

        if out_dim > 1:
            print("more than 1 equilibrium distribution, taking maximum entropy dist")

            from jaxopt import ProjectedGradient
            from jaxopt.projection import projection_l1_sphere

            def fun(alpha, w_orig, w_out, l, max_entropy):
                w = jnp.einsum("i,ji->j", alpha, w_out)

                w, w_pos, frac_neg, n_neg = _norm(w)

                if max_entropy:
                    w_pos_safe = jnp.where(w > 0, w, 1)
                    entropy = -jnp.sum(jnp.where(w > 0, w_pos_safe * jnp.log(w_pos_safe), 0))

                    o = -entropy
                else:
                    o = -jnp.sum(w)

                o *= jnp.sum(jnp.abs((1 - jnp.abs(l - 1)) * alpha))

                return o, (w_pos, frac_neg, n_neg)

            fun = Partial(fun, max_entropy=max_entropy)
            fun = jax.jit(fun)

            pg = ProjectedGradient(
                fun=fun,
                projection=projection_l1_sphere,
                tol=1e-10,
                maxiter=3000,
                has_aux=True,
            )

            init = jnp.ones((len(l),))
            init /= jnp.sum(init)

            pg_sol = pg.run(
                init,
                w_out=w_out,
                w_orig=w_orig,
                l=l,
            )

            print(f"{pg_sol=}")

            w_out, frac_neg, n_neg = pg_sol.state.aux

        else:
            w_out = w_out[:, 0]

        _, w_out, frac_neg, n_neg = _norm(w_out)

        print(f"{n_neg=} {frac_neg=} ")

        w_out = data_loader_output._unstack_weights([cvi.shape[0] for cvi in self.cv_0], w_out)

        return w_out, True

    def weighted_model(
        self,
        chunk_size=None,
        macro_chunk=10000,
        verbose=False,
        **kwargs,
    ) -> KoopmanModel:
        new_w, b = self.koopman_weight(
            verbose=verbose,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
        )

        if not b:
            return self

        kw = dict(
            w=new_w,
            cv_0=self.cv_0,
            cv_tau=self.cv_tau,
            nl=self.nl,
            nl_t=self.nl_t,
            add_1=self.add_1,
            eps=self.eps,
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
        )

        kw.update(**kwargs)

        return KoopmanModel.create(**kw)

    def timescales(self):
        s = self.s

        tau = self.tau
        if tau is None:
            tau = 1
            print("tau not set, assuming 1")

        return -tau / jnp.log(s)


@partial(dataclass, frozen=False, eq=False)
class Covariances:
    C00: jax.Array | None
    C01: jax.Array | None
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
        calc_C11=True,
    ) -> Covariances:
        time_series = cv_1 is not None

        if w is None:
            n = sum([cvi.shape[0] for cvi in cv_0])
            w = [jnp.ones((cvi.shape[0],)) / n for cvi in cv_0]

        if T_scale != 1:
            w = [wi ** (1 / T_scale) for wi in w]
            s = jnp.sum(jnp.hstack(w))
            w = [wi / s for wi in w]

        @jax.jit
        def cov_pi(carry, cv_0: CV, cv1: CV | None, w):
            # print(f"{cv_0=} {cv1=}")

            x0 = cv_0.cv
            if cv1 is not None:
                x1 = cv1.cv
            else:
                x1 = None

            (C00, C01, C11, pi0, pi1) = carry

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

                if calc_C11:
                    C11 = c(x1, x1, w, C11)

                if calc_pi:
                    pi1 = p(x1, w, pi1)

            return (C00, C01, C11, pi0, pi1)

        if trans_f is not None:

            def f_func(x, nl):
                # print(f"f_func {x=}")

                return trans_f.compute_cv(x, nl, chunk_size=chunk_size, shmap=False)[0]

            f_func = jax.jit(f_func)
            f_func = padded_shard_map(f_func, pmap=True)
        else:

            def f_func(x, nl):
                return x

        if trans_g is not None:

            def g_func(x, nl):
                # print(f"g_func {x=}")
                return trans_g.compute_cv(x, nl, chunk_size=chunk_size, shmap=False)[0]

            g_func = jax.jit(g_func)
            g_func = padded_shard_map(g_func, pmap=True)
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
            chunk_func_init_args=(None, None, None, None, None),
            w=w,
            jit_f=False,
        )

        C00, C01, C11, pi_0, pi_1 = out

        if calc_pi:
            if only_diag:
                if calc_C00:
                    C00 -= pi_0**2

                if time_series:
                    if calc_C11:
                        C11 -= pi_1**2
                    if calc_C01:
                        C01 -= pi_0 * pi_1

            else:
                if calc_C00:
                    C00 -= jnp.outer(pi_0, pi_0)
                if time_series:
                    if calc_C11:
                        C11 -= jnp.outer(pi_1, pi_1)
                    if calc_C01:
                        C01 -= jnp.outer(pi_0, pi_1)

        cov = Covariances(
            C00=C00,
            C01=C01,
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
        epsilon=1e-12,
        epsilon_pre=1e-12,
        out_dim=None,
        max_features=None,
        verbose=False,
        use_scipy=True,
        filter_argmask=True,
        correlation=True,
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

        # print(f"{V=} {V.shape=}")

        # if epsilon_pre is not None:
        #     argmask = jnp.argsort(V)[::-1].reshape(-1)
        #     argmask = argmask[V[argmask] > V[argmask[0]] * epsilon_pre]

        # print(f"{argmask.shape=} {V[argmask[0]]=}")

        # if max_features is not None:
        #     if len(argmask) > max_features:
        #         print(f"truncating to {max_features=}")
        #         argmask = argmask[:max_features]

        # C = C[argmask, :][:, argmask]
        # V = V[argmask]

        # argmask = None

        if correlation:
            V_sqrt_inv = jnp.where(V < epsilon_pre, 0, 1 / jnp.sqrt(V))
            P = jnp.einsum("ij,i,j->ij", C, V_sqrt_inv, V_sqrt_inv)

        else:
            P = C

        # G, Theta, _ = jnp.linalg.svd(P)
        theta, G = jnp.linalg.eigh(P)
        theta_inv = jnp.where(theta > epsilon, 1 / jnp.sqrt(theta), 0)

        if correlation:
            W = jnp.einsum(
                "i,ji,j->ij",
                theta_inv,
                G,
                V_sqrt_inv,
            )
        else:
            W = jnp.einsum(
                "i,ji->ij",
                theta_inv,
                G,
            )

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
        C00 = self.C00
        C01 = self.C01
        C11 = self.C11

        pi = None

        if self.pi_0 is not None:
            assert self.pi_1 is not None

            C00 += jnp.outer(self.pi_0, self.pi_0)
            if C01 is not None:
                C01 += jnp.outer(self.pi_0, self.pi_1)

            if C11 is not None:
                C11 += jnp.outer(self.pi_1, self.pi_1)

            pi = 0.5 * (self.pi_0 + self.pi_1)

            C00 -= jnp.outer(pi, pi)
            C01 -= jnp.outer(pi, pi)
            C11 -= jnp.outer(pi, pi)

        C00 = (1 / 2) * (self.C00 + self.C11)

        if C01 is not None:
            C01 = (1 / 2) * (self.C01 + self.C01.T)
        if C11 is not None:
            C11 = C00

        return Covariances(
            C00=C00,
            C01=C01,
            C11=C11,
            pi_0=pi,
            pi_1=pi,
            only_diag=self.only_diag,
            symmetric=True,
            trans_f=self.trans_f,
            trans_g=self.trans_g,
        )
