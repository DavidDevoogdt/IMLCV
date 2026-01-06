from __future__ import annotations

import datetime
import os
import shutil
import time
from abc import ABC
from asyncio import Future
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Sequence, TypeVar, cast

import h5py
import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey, choice, split

from IMLCV.base.bias import Bias, BiasModify, CompositeBias, GridBias, RoundBias, StdBias
from IMLCV.base.CV import (
    CV,
    CollectiveVariable,
    CvMetric,
    CvTrans,
    NeighbourList,
    NeighbourListInfo,
    NeighbourListUpdate,
    ShmapKwargs,
    SystemParams,
    macro_chunk_map,
    macro_chunk_map_fun,
    padded_shard_map,
    padded_vmap,
)
from IMLCV.base.CVDiscovery import Transformer
from IMLCV.base.datastructures import MyPyTreeNode, Partial_decorator, jit_decorator, vmap_decorator
from IMLCV.base.MdEngine import EagerTrajectoryInfo, FullTrajectoryInfo, MDEngine, StaticMdInfo, TrajectoryInfo
from IMLCV.base.UnitsConstants import angstrom, atomic_masses, boltzmann, kelvin, kjmol, nanosecond, picosecond
from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.configs.config_general import Executors
from IMLCV.implementations.bias import DTBias, GridMaskBias, RbfBias, _clip
from IMLCV.implementations.CV import _cv_slice, cv_trans_real


@dataclass
class TrajectoryInformation:
    ti: TrajectoryInfo
    cv: int
    round: int
    num: int
    folder: Path
    # name_bias: str | None = None
    # valid: bool = True
    # finished: bool = False

    def get_bias(self) -> Bias | None:
        try:
            return Rounds(self.folder).get_bias(c=self.cv, r=self.round, i=self.num)
        except Exception as e:
            print(f"unable to load bias {e=}")
            return None


@dataclass
class RoundInformation:
    round: int
    # valid: bool
    cv: int
    num: int
    num_vals: Array
    tic: StaticMdInfo
    folder: Path
    # name_bias: str | None = None
    # name_md: str | None = None

    def get_bias(self) -> Bias:
        return Rounds(self.folder).get_bias(c=self.cv, r=self.round)

    def get_engine(self) -> MDEngine:
        return Rounds(self.folder).get_engine(c=self.cv, r=self.round)


@dataclass
class Rounds:
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

    def path(self, c: int | None = None, r: int | None = None, i: int | None = None) -> Path:
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

        shutil.make_archive(str(p), "zip", p)
        shutil.rmtree(p)

    def unzip_cv_round(self, c, r, warn=False):
        p = self.path(c=c, r=r)

        if p.exists():
            if warn:
                print(f"{p} already exists")
            return

        if not p.with_suffix(".zip").exists():
            if warn:
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

            shutil.make_archive(str(p), "zip", p)
            shutil.rmtree(self.path(c=cv, r=r))

    def unzip_cv(self, cv, warn=False):
        print(f"unzipping {cv=}")

        for round_r in self.path(c=cv).glob("round_*.zip"):
            r = round_r.parts[-1][6:-4]

            if r.endswith(".old"):
                continue

            print(f"unzipping {r=}")

            self.unzip_cv_round(cv, r, warn=warn)

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

                shutil.make_archive(str(p), "zip", p)
                shutil.rmtree(self.path(c=cv, r=r))

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

            if repeat is not None:
                atoms = [a.repeat(repeat) for a in atoms]

                print(f"repeating {repeat=} {atoms[0].get_positions().shape=}")

            from ase.io import write

            write(
                filename=self.path(c=c, r=round.round, i=trajejctory.num) / f"trajectory.{ext}",
                images=atoms,
                format=ext,
            )

    def plot(
        self,
        c: int | None = None,
        r: int | None = None,
        i: int | None = None,
        # name: str | Path | None = None,
        plot_fes: bool = True,
        plot_points: bool = True,
        dlo_kwargs={},
    ):
        if c is None:
            c = self.cv

        bias = self.get_bias(c=c, r=r, i=i)

        if plot_points:
            name = self.path(c=c, r=r) / f"bias_plot_{'md_' + str(i) + '_' if i is not None else ''}{r}_points.png"

            if i is None:
                dlo = self.data_loader(
                    num=1,
                    out=-1,
                    cv=c,
                    stop=r,
                    new_r_cut=None,
                    weight=False,
                    split_data=True,
                    **dlo_kwargs,
                )  # type: ignore
                cv_list = dlo.cv
            else:
                cv_list = [self._trajectory_information(c=c, r=r, i=i).ti.CV]

            Transformer.plot_app(
                collective_variables=[bias.collective_variable],
                cv_data=[cv_list],
                name=name,
                T=dlo.sti.T if plot_points else None,
                plot_FES=False,
                duplicate_cv_data=False,
                indicate_plots=None,
                title=f"CV {c} Round {r} {'MD ' + str(i) if i is not None else ''} Points",
            )

        if plot_fes:
            name = self.path(c=c, r=r) / f"bias_plot_{'md_' + str(i) + '_' if i is not None else ''}{r}_fes.png"

            Transformer.plot_app(
                collective_variables=[bias.collective_variable],
                biases=[[bias]],
                duplicate_cv_data=False,
                T=dlo.sti.T if plot_points else None,
                name=name,
                plot_FES=True,
                indicate_plots=None,
                cv_titles=[f"CV {c}"],
            )

    def plot_round(
        self,
        c: int | None = None,
        r: int | None = None,
        name_bias=None,
        name_points=None,
        dlo_kwargs={},
        plot_kwargs=dict(
            indicate_plots=None,
        ),
        plot_points=True,
        plot_fes=True,
        cv_names: list[str] | None = None,
    ):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        if name_bias is None:
            name_bias = self.path(c=c) / f"bias_{r}.png"

        if name_points is None:
            name_points = self.path(c=c) / f"bias_data_{r}.png"

        print(f"{c=} {r=}")

        colvar = self.get_collective_variable(c)
        sti = self.get_static_trajectory_info(c, r)

        if cv_names is None:
            colvar.cvs_name = cv_names

        if plot_fes:
            b = self.get_bias(c=c, r=r)

            Transformer.plot_app(
                collective_variables=[colvar],
                biases=[[b]],
                duplicate_cv_data=False,
                T=sti.T,
                name=name_bias,
                plot_FES=True,
                **plot_kwargs,
            )

        if plot_points:
            print("loading data")

            dlo = self.data_loader(
                num=1,
                out=-1,
                cv=c,
                stop=r,
                new_r_cut=None,
                weight=False,
                split_data=True,
                **dlo_kwargs,
            )  # type: ignore

            print("plotting data")

            Transformer.plot_app(
                collective_variables=[dlo.collective_variable],
                # biases=[[b]],
                cv_data=[[dlo.cv]],
                name=name_points,
                T=dlo.sti.T,
                plot_FES=True,
                duplicate_cv_data=False,
                **plot_kwargs,
            )

    def plot_cv_discovery(
        self,
        start: int,
        end: int,
        additional_collective_variables: list[CollectiveVariable] | None = None,
        additional_collective_variable_names: list[str] | None = None,
        additional_collective_variable_titles: list[str] | None = None,
        plot_biases=True,
        # folder=".",
        ignore_invalid=True,
        only_finished=False,
        get_fes_bias_kwargs={},
        plot_kwargs={},
        duplicate_last_row=False,
    ):
        import jax.numpy as jnp

        from IMLCV.base.CVDiscovery import Transformer
        from IMLCV.base.rounds import DataLoaderOutput

        # f = folder
        # rnds = Rounds.create(folder=f)

        cs = list(range(start, end + 1))

        ncv = len(cs)

        if additional_collective_variables is None:
            ncv_add = 0
            additional_collective_variables = []
            additional_collective_variable_names = []
            additional_collective_variable_titles = []

        else:
            assert additional_collective_variable_names is not None
            assert additional_collective_variable_titles is not None

            ncv_add = len(additional_collective_variables)
            assert len(additional_collective_variable_names) == ncv_add
            assert len(additional_collective_variable_titles) == ncv_add

        bias_matrix = []
        data_matrix = []

        # i: data, j cv

        for i in range(ncv):
            bias_matrix_i = []
            data_matrix_i = []

            for j in range(ncv + ncv_add + 1):
                bias_matrix_i.append(None)
                data_matrix_i.append(None)

            bias_matrix.append(bias_matrix_i)
            data_matrix.append(data_matrix_i)

        collective_variables = []

        for j in range(ncv + ncv_add + 1):
            collective_variables.append(None)

        indicate_plots_arr = [
            [
                "lightgray" if (c < ncv_add) else ("lightblue" if c == (d + ncv_add) else None)
                for c in range(ncv + ncv_add + 1)  # colums
            ]
            for d in range(ncv + ncv_add)  # rows
        ]

        row_colors = jnp.arange(ncv) + ncv_add

        for _i in range(ncv_add):
            collective_variables[_i] = additional_collective_variables[_i]

        T: None | float = None

        for ci, c in enumerate(cs):
            print(f"===={ci=}__{c=}====")

            cv_n = []
            cv_n_p_1 = []
            sp = []

            w = []
            rho = []

            r0 = 0

            is_zipped = self.is_zipped(c=c, r=0)
            if is_zipped:
                print("unzipping")

                self.unzip_cv_round(c=c, r=0)

            _r = self._round_information(c=c, r=r0)

            if T is None:
                T = _r.tic.T
            else:
                assert _r.tic.T == T

            assert _r.valid or ignore_invalid

            rn = _r.num_vals

            rn.sort()

            for i in rn:
                try:
                    _r_i = self._trajectory_information(c=c, r=r0, i=i)
                except Exception as e:
                    print(f"could not load {c=} {r0=} {i=} {e=}, skipping")
                    continue

                if not _r_i.valid and not ignore_invalid:
                    continue

                if (not _r_i.finished) and only_finished:
                    continue
                # no points in collection
                if _r_i.ti.size <= 0:
                    continue

                # if T is None:
                #     T

                cv_n_p_1.append(_r_i.ti.CV)
                cv_n.append(_r_i.ti.CV_orig)
                w.append(_r_i.ti.w)
                rho.append(_r_i.ti.rho)
                sp.append(_r_i.ti.sp)

            print(f"getting colvar {c-1=} {c=}")
            colvar_n = self.get_collective_variable(c - 1)
            colvar_n_p_1 = self.get_collective_variable(c)

            add_cv = []

            for ext_cv in additional_collective_variables:
                cv_ext, _ = DataLoaderOutput.apply_cv(
                    x=sp,
                    f=ext_cv.f,
                )

                add_cv.append(cv_ext)

            collective_variables[ci + ncv_add] = colvar_n
            collective_variables[ci + ncv_add + 1] = colvar_n_p_1

            colvars = [
                *additional_collective_variables,
                colvar_n,
                colvar_n_p_1,
            ]

            cv_data = [
                *add_cv,
                cv_n,
                cv_n_p_1,
            ]

            for _i in range(ncv_add):
                data_matrix[ci][_i] = cv_data[_i]
            for _i in range(2):
                data_matrix[ci][ci + ncv_add + _i] = cv_data[ncv_add + _i]

            if plot_biases:
                biases = []

                for i, (colvar_i, cv_i) in enumerate(zip(colvars, cv_data)):
                    print(f"{i=} {colvar_i.n=} {cv_i[0].shape=}")

                    b, _, _, _ = DataLoaderOutput._get_fes_bias_from_weights(
                        T=T * kelvin,
                        weights=w,
                        rho=rho,
                        collective_variable=colvar_i,
                        cv=cv_i,
                        **get_fes_bias_kwargs,
                    )

                    biases.append(b)

                for _i in range(ncv_add):
                    bias_matrix[ci][_i] = biases[_i]
                for _i in range(2):
                    bias_matrix[ci][ci + ncv_add + _i] = biases[ncv_add + _i]

        assert T is not None

        data_titles = [f"data_{c - 1}" for c in cs]

        if plot_biases:
            print("plotting cv without data")

            Transformer.plot_app(
                collective_variables=collective_variables,
                cv_data=None,
                biases=bias_matrix,
                duplicate_cv_data=False,
                name=self.path() / f"CV_discovery_{start - 1}_{end}.png",
                labels=[
                    *additional_collective_variable_names,
                    *["xyz"] * (ncv + 1),
                ],
                cv_titles=[
                    *additional_collective_variable_titles,
                    *[f"CV{c}" for c in range(start - 1, end + 1)],
                ],
                data_titles=data_titles,
                color_trajectories=False,
                indicate_plots=indicate_plots_arr,
                T=T * kelvin,
                plot_FES=True,
                row_color=row_colors,
                **plot_kwargs,
            )

        print("plotting cv with data")
        if duplicate_last_row:
            data_matrix.append(data_matrix[-1])
            data_titles.append(data_titles[-1])
            row_colors = jnp.hstack([row_colors, jnp.array([row_colors[-1] + 1])])
            indicate_plots_arr.append(
                [
                    "lightgray" if (c < ncv_add) else ("lightblue" if c == ncv + ncv_add else None)
                    for c in range(ncv + ncv_add + 1)
                ]
            )

        # plot with data
        Transformer.plot_app(
            collective_variables=collective_variables,
            cv_data=data_matrix,
            # biases=bias_matrix if plot_biases else None,
            duplicate_cv_data=False,
            name=self.path() / f"CV_discovery_{start - 1}_{end}_data.png",
            labels=[
                *additional_collective_variable_names,
                *["xyz"] * (ncv + 1),
            ],
            cv_titles=[
                *additional_collective_variable_titles,
                *[f"CV{c}" for c in range(start - 1, end + 1)],
            ],
            data_titles=data_titles,
            color_trajectories=False,
            indicate_plots=indicate_plots_arr,
            plot_FES=False,
            T=T * kelvin,
            row_color=row_colors,
            **plot_kwargs,
        )

        if is_zipped:
            print("zipping")
            self.zip_cv_round(c=c, r=0)

    def plot_cv_discovery_v2(
        self,
        start: int,
        end: int,
        extra_collective_variable: CollectiveVariable,
        macro_chunk=100000,
        vmax=30 * kjmol,
        corr=True,
        load_extra_info=True,
        extra_info_title="$\\tau$ (ns)",
        get_fes_bias_kwargs={},
        plot_std=True,
    ):
        import jax.numpy as jnp

        from IMLCV.base.CVDiscovery import Transformer
        from IMLCV.base.rounds import DataLoaderOutput

        dlos = []

        for j in range(start, end + 1):
            dlo = self.data_loader(
                cv=j,
                start=0,
                stop=0,
                out=-1,
                only_finished=False,
                weight=True,
                macro_chunk=macro_chunk,
                verbose=True,
                num=1,
                load_weight=True,
            )

            if load_extra_info:
                timescales = self.extract_timescales(c=j)

            if timescales is None:
                dlo.collective_variable.extra_info = None
            else:
                dlo.collective_variable.extra_info = tuple([f"{float(a):.4f}" for a in timescales])

            dlos.append(dlo)

        if corr:
            f = Transformer.plot_CV_corr
        else:
            f = Transformer.plot_CV

        f(
            collective_variable_projection=extra_collective_variable,
            collective_variables=[a.collective_variable for a in dlos],
            ti=[a.ti for a in dlos],
            # cv_data=[a.cv for a in dlos],
            # sp_data=[a.sp for a in dlos],
            # weights=[[a * b for a, b in zip(a._weights, a._rho)] for a in dlos],
            # std=[a._weights_std for a in dlos],
            # cv_titles=["Round 1", "Round 2"],
            name=self.path() / f"CV_discovery_v2_{start - 1}_{end}_{'std' if plot_std else ''}.png",
            vmax=vmax,
            macro_chunk=macro_chunk,
            extra_info_title=extra_info_title,
            get_fes_bias_kwargs=get_fes_bias_kwargs,
            plot_std=plot_std,
        )

    def extract_timescales(self, c) -> jax.Array | None:
        l = []

        for p in self.path(c=c - 1).glob("_update_CV_*.stdout"):
            try:
                name_no_ext = p.stem  # filename without extension
                idx = int(name_no_ext.rsplit("_", 1)[-1])
                l.append(idx)
            except Exception as e:
                print(f"could not parse index from {p}: {e}")

        timescales: jax.Array | None = None

        if len(l) > 0:
            l.sort()
            l_max = l[-1]

            with open(self.path(c=c - 1) / f"_update_CV_{l_max:0>3}.stdout", "r") as f:
                text = f.read()

                import re

                try:
                    m = re.search(r"timescales:\s*\[(.*?)\]\s*ns", text, flags=re.S)
                    if m:
                        nums = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)", m.group(1))
                        timescales = jnp.array([float(x) for x in nums])

                        print(f"found timescales {timescales} in file")
                    else:
                        print("no timescales found in file")
                except Exception as e:
                    print(f"failed to parse timescales: {e}")

        return timescales

    ######################################
    #             storage                #
    ######################################

    def _c_vals(self) -> list[int]:
        cvs = []

        for cv_c in self.path().glob("cv_*"):
            c = cv_c.parts[-1][3:]

            cvs.append(int(c))

        cvs.sort()

        if len(cvs) == 0:
            cvs.append(-1)

        return cvs

    def _r_vals(self, c: int | None = None):
        if c is None:
            c = self.cv

        rounds: list[int] = []

        for round_r in self.path(c=c).glob("round_*"):
            if round_r.suffix == ".zip":
                continue

            r = int(round_r.parts[-1][6:])

            if not (p := self.path(c=c, r=r) / "static_trajectory_info.h5").exists():
                print(f"could not find {p}")
                continue

            rounds.append(int(r))

        rounds.sort()

        if len(rounds) == 0:
            rounds.append(-1)

        return rounds

    def _i_vals(self, c: int | None = None, r: int | None = None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        i_s: list[int] = []

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

    def _name_bias(self, c, r, i: int | None = None):
        if (p := (self.path(c=c, r=r, i=i) / "bias_new.json")).exists():
            return self.rel_path(p)
        elif (p := (self.path(c=c, r=r, i=i) / "bias_new")).exists():
            return self.rel_path(p)
        elif (p := (self.path(c=c, r=r, i=i) / "bias.json")).exists():
            return self.rel_path(p)
        elif (p := (self.path(c=c, r=r, i=i) / "bias")).exists():
            return self.rel_path(p)

        return None

    def _num_vals(self, c, r: int | None = None):
        if r is not None:
            return len(self._i_vals(c, r=r))

        return len(self._r_vals(c))

    def add_permanent_bias(self, bias: Bias):
        bias.collective_variable.save(self.folder / "permanent_cv.json")
        bias.save(self.folder / "permanent_bias.json")

    def add_cv(self, collective_variable: CollectiveVariable, c: int | None = None):
        if c is None:
            c = self.cv + 1

        directory = self.path(c=c)
        if not os.path.isdir(directory):
            os.mkdir(directory)

        collective_variable.save(self.path(c=c) / "cv.json")

    def get_permanent_bias(self) -> Bias | None:
        if not (p := self.folder / "permanent_bias.json").exists():
            return None

        assert (p_cv := (self.folder / "permanent_cv.json")).exists(), (
            f"cannot load permanent bias for cv {c} if permanent cv does not exist"
        )

        permanent_cv = CollectiveVariable.load(p_cv)

        perm_bias = Bias.load(p, collective_variable=permanent_cv)
        return perm_bias

    def get_bias(self, c: int | None = None, r: int | None = None, i: int | None = None) -> Bias:
        if c is None:
            c = self.cv

        assert (p := self.path(c=c) / "cv.json").exists(), f"cannot find cv at {p}"

        cv = CollectiveVariable.load(p)

        if r is None:
            r = self.get_round(c=c)

        assert (p := self.path(c=c, r=r) / "bias.json").exists(), f"cannot find common bias at {p}"

        bias = Bias.load(p, collective_variable=cv)

        if i is None:
            return bias

        assert (p := self._name_bias(c=c, r=r, i=i)) is not None, f"cannot find individual bias for {c=} {r=} {i=}"

        bias_i = Bias.load(self.folder / p, collective_variable=cv)

        return RoundBias.create(bias_r=bias, bias_i=bias_i)

    def add_round(
        self,
        bias: Bias,
        stic: StaticMdInfo | None = None,
        mde=None,
        c: int | None = None,
        r: int | None = None,
    ):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c) + 1

        if stic is None:
            stic = self.static_trajectory_information(c=c)

        if mde is None:
            mde = self.get_engine(c=c)

        mde.static_trajectory_info = stic

        # bias.collective_variable_path = Path(self.rel_path(self.path(c=c))) / "cv.json"
        # mde.bias = bias

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
        ignore_invalid=True,
        only_finished=False,
        c: int | None = None,
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

            if _r.tic.invalid and not ignore_invalid:
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

                if _r_i.ti.invalid and not ignore_invalid:
                    continue

                if (not _r_i.ti.finished) and only_finished:
                    continue
                # no points in collection
                if _r_i.ti.size <= 0:
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
        num: int = 4,
        out: int = -1,
        split_data: bool = False,
        new_r_cut: float | None = -1.0,
        cv: int | None = None,
        ignore_invalid=True,
        md_trajs: list[int] | None = None,
        start: int | None = None,
        stop: int | None = None,
        time_series: bool = False,
        T_max_over_T=50,
        chunk_size: int | None = None,
        get_colvar: bool = True,
        min_traj_length: int | None = None,
        recalc_cv: bool = False,
        get_bias_list: bool = False,
        num_cv_rounds: int = 1,
        only_finished: bool = True,
        lag_n: int | None = 1,
        lag_tau: float | None = None,
        colvar: CollectiveVariable | None = None,
        verbose: bool = False,
        weight: bool = True,
        macro_chunk: int | None = 2000,
        n_max: int | float = 1e5,
        n_max_lin: int | float = 150,
        calculate_std: bool = True,
        scale_times: bool = False,
        weighing_method: str = "WHAM",
        samples_per_bin: int = 10,
        min_samples_per_bin: int = 3,
        n_hist: int | None = None,
        load_weight: bool = False,
        select_max_bias: float | None = None,
        recalc_bounds=True,
        reweight: bool = True,
        time_correlation_method=None,
        preselect_filter: int = 1,
        n_skip: int = 0,
        load_sp=True,
        equilibration_time: float | None = 0,
    ) -> DataLoaderOutput:
        if cv is None:
            c = self.cv
        else:
            c = cv

        sti = self._round_information(c=c).tic

        if new_r_cut == -1:
            new_r_cut = sti.r_cut

        if not time_series:
            lag_n = 0

        if weight and weighing_method == "WHAM":
            if not get_bias_list:
                get_bias_list = True

        if load_weight:
            get_bias_list = False

        if get_bias_list:
            bias_list: list[Bias] = []

        if lag_tau is not None:
            print(f"using lag_tau instead of lag_n")

        # out = int(out)

        cvrnds = []

        if num_cv_rounds != 1:
            cvrnds = range(max(0, c - num_cv_rounds), c + 1)
            recalc_cv = True

        else:
            cvrnds.append(c)

        try:
            ground_bias = self.get_bias(c=c, r=stop)
        except Exception as e:
            print(f"could not load ground bias {e=}")
            ground_bias = None

        if colvar is not None:
            recalc_cv = True

        if get_colvar or recalc_cv:
            if colvar is None:
                colvar = self.get_collective_variable(c=c)

        if scale_times and not time_series:
            scale_times = False

        ###################

        if verbose:
            print("obtaining raw data")

        # sp: list[SystemParams] = []
        # cv: list[CV] = []
        ti: list[TrajectoryInfo] = []

        weights: list[Array] = []
        weight_wall: list[Array] = []
        p_select: list[Array] = []
        F: list[Array] = []
        density: list[Array] = []
        n_bin: list[Array] = []
        n_eff_bin: list[Array] = []

        bounds: jax.Array | None = None
        nhist: int | None = None

        if calculate_std:
            if not weight:
                calculate_std = False
                print("WARNING: calculate_std is True, but weight is False, setting calculate_std to False")

            # if load_weight:
            #     calculate_std = False
            #     print("WARNING: calculate_std is True, but load_weight is True, setting calculate_std to False")

        if calculate_std and weighing_method != "WHAM":
            calculate_std = False
            print("WARNING: calculate_std is True, but weighing_method is not WHAM, setting calculate_std to False")

        weights_std: list[Array] | None = [] if calculate_std else None

        grid_nums: list[Array] | None = None

        labels: list[Array] = []

        sti_c: StaticMdInfo | None = None

        nl_info: NeighbourListInfo | None = None
        update_info: NeighbourListUpdate | None = None

        frac_full = 1.0

        n_samples_eff_total = 0.0

        if load_weight and lag_n != 0:
            print("WARNING: lag_n is not 0, but load_weight is True. setting lag_n to 0")
            lag_n = 0

        n_tot = 0

        equilibration_n = None

        pb = self.get_permanent_bias()

        print(f"{pb=}")

        for cvi in cvrnds:
            # sti_c: StaticMdInfo | None = None
            # sp_c: list[SystemParams] = []
            # cv_c: list[CV] = []
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

            weight_pb = False

            try:
                colvar_c = self.get_collective_variable(c=cvi)
            except Exception as e:
                print(f"could not load collective variable {e=}")
                colvar_c = None
                weight_c = False

            if load_weight:
                loaded_rho = []
                loaded_w = []
                loaded_sigma = []

            for round_info, traj_info in self.iter(
                start=start,
                stop=stop,
                num=num,
                c=cvi,
                ignore_invalid=ignore_invalid,
                md_trajs=md_trajs,
                only_finished=only_finished,
            ):
                if sti_c is None:
                    sti_c = round_info.tic

                _ti = traj_info.ti

                if equilibration_time is not None:
                    if equilibration_n is None:
                        equilibration_n = int(equilibration_time / (sti_c.timestep * sti_c.save_step))
                        print(f"{equilibration_n=}")

                    if _ti.size < equilibration_n:
                        print("skipping trajectory, not equilibrated")
                        continue

                    _ti = _ti[equilibration_n:]

                if min_traj_length is not None:
                    if traj_info.ti.size < min_traj_length or traj_info.ti.size <= lag_n:
                        # print(f"skipping trajectyory because it's not long enough {traj.ti.size}<{min_traj_length}")
                        continue
                    # else:
                    # print("adding traweights=jectory")

                if n_skip > 0:
                    if _ti.size <= n_skip:
                        continue
                    _ti = _ti[n_skip:]

                ti_c.append(_ti)

                if load_weight:
                    _w_i = _ti.w
                    _rho_i = _ti.rho

                    if _w_i is None or _rho_i is None:
                        raise ValueError(
                            f"weights are not stored for {cvi=} {round_info.round=} {traj_info.num=}. (pass load_weight=False)   "
                        )

                    assert _w_i.shape[0] == _ti.size, f"weights and sp shape are different: {_w_i.shape=} {_ti.size=}"
                    assert _rho_i.shape[0] == _ti.size, (
                        f"weights and sp shape are different: {_rho_i.shape=} {_ti.size=}"
                    )

                    loaded_rho.append(_rho_i)
                    loaded_w.append(_w_i)

                    _sigma_i = _ti.sigma

                    loaded_sigma.append(_sigma_i)

                if new_r_cut is not None:
                    if nl_info is None:
                        nl_info = NeighbourListInfo.create(
                            r_cut=new_r_cut,
                            r_skin=0.0,
                            z_array=sti_c.atomic_numbers,
                        )

                        b, nn, new_nxyz, _ = Partial_decorator(
                            SystemParams._get_neighbour_list,
                            info=nl_info,
                            chunk_size=chunk_size,
                            chunk_size_inner=10,
                            shmap=False,
                            only_update=True,
                            update=update_info,
                        )(
                            self=_ti.sp,
                        )

                        update_info = NeighbourListUpdate.create(
                            num_neighs=int(nn),  # type:ignore
                            nxyz=new_nxyz,
                        )

                        print(f"initializing neighbour list with {nn=} {new_nxyz=}")

                    else:
                        b, nn, new_nxyz, _ = jit_decorator(
                            SystemParams._get_neighbour_list,
                            static_argnames=[
                                # "info",
                                "chunk_size",
                                "chunk_size_inner",
                                "shmap",
                                "only_update",
                                # "update",
                                "verbose",
                                # "shmap_kwargs",
                            ],
                        )(
                            self=_ti.sp,
                            info=nl_info,
                            chunk_size=chunk_size,
                            chunk_size_inner=10,
                            shmap=False,
                            only_update=True,
                            update=update_info,
                        )

                        if new_nxyz is not None:
                            assert update_info is not None
                            assert update_info.nxyz is not None
                            assert update_info.num_neighs is not None

                            if jnp.any(jnp.array(new_nxyz) > jnp.array([2, 2, 2])):
                                print(f"neighbour list is too big {new_nxyz=}")
                                continue

                            if (
                                jnp.any(jnp.array(new_nxyz) > jnp.array(update_info.nxyz))
                                or nn > update_info.num_neighs
                            ):
                                print(f"updating nl with {nn=} {new_nxyz=}")

                                update_info = NeighbourListUpdate.create(
                                    num_neighs=int(nn),  # type:ignore
                                    nxyz=new_nxyz,
                                )

                print(".", end="", flush=True)
                n_tot += 1

                if n_tot % 100 == 0:
                    print("")

                # if cv0 is None:
                #     raise
                #     print(f"recalculating cv for {cvi=} {round_info.round=} {traj_info.num=}")

                #     if colvar is None:
                #         bias = traj_info.get_bias()
                #         assert bias is not None
                #         colvar = bias.collective_variable

                #     assert colvar is not None

                #     nlr = (
                #         _ti.sp.get_neighbour_list(
                #             info=NeighbourListInfo.create(
                #                 r_cut=round_info.tic.r_cut,
                #                 z_array=round_info.tic.atomic_numbers,
                #             ),
                #             chunk_size=chunk_size,
                #         )
                #         if round_info.tic.r_cut is not None
                #         else None
                #     )

                #     cv0, _ = colvar.compute_cv(sp=_ti.sp, nl=nlr)

                # sp_c.append(sp0)
                # cv_c.append(cv0)

                if get_bias_list:
                    bias = traj_info.get_bias()
                    assert bias is not None
                    bias_c.append(bias)

            print("")

            if len(ti_c) == 0:
                continue

            if load_weight:
                w_c = loaded_w
                d_c = loaded_rho
                if calculate_std:
                    weights_std.extend(loaded_sigma)
                ps_c = [jnp.ones((_ti.size)) for _ti in ti_c]
                nb_c = [jnp.ones((_ti.size)) for _ti in ti_c]
                neb_c = [jnp.ones((_ti.size)) for _ti in ti_c]
                F_c = [jnp.zeros((_ti.size)) for _ti in ti_c]  # type: ignore
                grid_nums_c = None

                # if calculate_std:

                labels_c = [0] * len(ti_c)

                # if output_FES_bias:
                #     FES_biases.append(None)

            elif weight_c:
                if verbose:
                    print(f"getting weights for cv_round {cvi} {len(ti_c)} trajectories")

                assert sti_c is not None
                assert colvar_c is not None

                dlo = DataLoaderOutput(
                    # sp=sp_c,
                    # cv=cv_c,
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
                        n_max_lin=n_max_lin,
                        verbose=verbose,
                        samples_per_bin=samples_per_bin,
                        min_samples_per_bin=min_samples_per_bin,
                        macro_chunk=macro_chunk,
                        recalc_bounds=recalc_bounds,
                        compute_std=calculate_std,
                        correlation_method=time_correlation_method,
                        biases=bias_c,
                        n_hist=n_hist,
                    )

                    if bounds is None:
                        bounds = weight_output.bounds
                        nhist = weight_output.n_bins

                        print(f"setting bounds to {bounds=} {nhist=}")

                elif weighing_method == "BC":
                    weight_output = dlo.bincount_weight(
                        chunk_size=chunk_size,
                        n_max=n_max,
                        verbose=verbose,
                        recalc_bounds=recalc_bounds,
                        n_hist=n_hist,
                        n_max_lin=n_max_lin,
                        # return_bias=output_FES_bias,
                    )
                else:
                    raise ValueError(f"{weighing_method=} not supported")

                w_c = weight_output.weight
                ps_c = weight_output.p_select
                F_c = weight_output.F
                d_c = weight_output.density
                nb_c = weight_output.n_bin
                neb_c = weight_output.n_eff_bin if weight_output.n_eff_bin is not None else nb_c

                n_samples_eff_total += weight_output.N_samples_eff

                if calculate_std:
                    weights_std.extend(weight_output.p_std)

                grid_nums_c = weight_output.grid_nums

                labels_c = weight_output.labels

                frac_full = weight_output.frac_full

            else:
                print("setting weights to one!")

                w_c = [jnp.ones((_ti.size)) for _ti in ti_c]
                ps_c = [jnp.ones((_ti.size)) for _ti in ti_c]
                d_c = [jnp.ones((_ti.size)) for _ti in ti_c]
                nb_c = [jnp.ones((_ti.size)) for _ti in ti_c]
                neb_c = [jnp.ones((_ti.size)) for _ti in ti_c]
                F_c = [jnp.zeros((_ti.size)) for _ti in ti_c]

                n_samples_eff_total += sum([_ti.size for _ti in ti_c])
                grid_nums_c = None

                labels_c = [0] * len(ti_c)

            if pb is not None:
                print("applying permanent bias selection")
                beta = 1.0 / (boltzmann * sti_c.T)
                weight_perm = [jnp.exp(-beta * pb.compute_from_system_params(ti_ci.sp)[1].energy) for ti_ci in ti_c]

            else:
                weight_perm = [jnp.ones_like(wi) for wi in w_c]

            # sp.extend(sp_c)
            # cv.extend(cv_c)
            ti.extend(ti_c)

            weight_wall.extend(weight_perm)
            weights.extend(w_c)
            p_select.extend(ps_c)
            F.extend(F_c)
            density.extend(d_c)
            n_bin.extend(nb_c)
            n_eff_bin.extend(neb_c)

            if grid_nums_c is not None:
                if grid_nums is None:
                    print("initializing grid nums")

                    grid_nums = grid_nums_c
                else:
                    print("extending grid nums")

                    grid_nums.extend(grid_nums_c)

                    # print(f"{len(grid_nums)=} {len(sp)=}")

            labels.extend(jnp.array(labels_c))

            if get_bias_list:
                bias_list.extend(bias_c)

        # print(f"{n_bin=} {n_eff_bin=}")

        ###################
        if verbose:
            print("Checking data")

        assert len(ti) != 0, "no data found"

        if load_weight:
            print("loaded weights from disk, not recalculating them")

            dlo = DataLoaderOutput(
                ti=ti,
                sti=sti_c,
                nl=None,
                collective_variable=colvar,  # type: ignore
            )

            return dlo

        ###################

        key = PRNGKey(0)

        if out == -1:
            out = len(ti) - 1

            out_ti = []

            out_biases = bias_list if get_bias_list else None

            for i, a in enumerate(ti):
                a.w = weights[i] * density[i]
                a.rho = p_select[i] / n_bin[i]

                out_ti.append(a)

            out_weights_std = weights_std

            out_labels = labels

        else:
            print(f"{update_info=}")

            if T_max_over_T is not None and ti[0].T is not None:
                # sp_new: list[SystemParams] = []
                # cv_new: list[CV] = []
                ti_new: list[TrajectoryInfo] = []

                w_new: list[Array] = []
                ww_new: list[Array] = []
                ps_new: list[Array] = []
                F_new: list[Array] = []
                d_new: list[Array] = []
                nb_new: list[Array] = []
                neb_new: list[Array] = []
                labels_new: list[Array] = []

                if calculate_std:
                    weights_std_new: list[Array] = []

                if grid_nums is not None:
                    grid_nums_new: list[Array] = []

                if get_bias_list:
                    new_bias_list = []

                for n in range(len(ti)):
                    assert ti[n].T is not None

                    indices = jnp.where(ti[n].T > sti.T * T_max_over_T)[0]  # type: ignore

                    if len(indices) != 0:
                        print(f"temperature threshold surpassed in time_series {n=}, removing the data")
                        continue

                    # sp_new.append(sp[n])

                    # cv_new.append(cv[n])
                    ti_new.append(ti[n])
                    if get_bias_list:
                        new_bias_list.append(bias_list[n])

                    w_new.append(weights[n])
                    ww_new.append(weight_wall[n])
                    ps_new.append(p_select[n])
                    F_new.append(F[n])
                    d_new.append(density[n])
                    nb_new.append(n_bin[n])
                    neb_new.append(n_eff_bin[n])
                    labels_new.append(labels[n])

                    if grid_nums is not None:
                        grid_nums_new.append(grid_nums[n])

                    if calculate_std:
                        weights_std_new.append(weights_std[n])

                # sp = sp_new
                ti = ti_new
                # cv = cv_new

                weights = w_new
                weight_wall = ww_new
                p_select = ps_new
                F = F_new
                density = d_new
                n_bin = nb_new
                n_eff_bin = neb_new

                labels = labels_new

                if grid_nums is not None:
                    grid_nums = grid_nums_new

                if get_bias_list:
                    bias_list = new_bias_list

                if calculate_std:
                    weights_std = weights_std_new

            print(f"len(sp) = {len(ti)}")

            # for j, k in zip(sp, cv):
            #     if j.shape[0] != k.shape[0]:
            #         print(f"shapes do not match {j.shape=} {k.shape=}")

            # for w_i, p_i, spi in zip(weights, p_select, ti.sp):
            #     assert w_i.shape[0] == spi.shape[0], f"weights and sp shape are different: {w_i.shape=} {spi.shape=}"

            #     assert p_i.shape[0] == spi.shape[0], f"p_select and sp shape are different: {p_i.shape=} {spi.shape=}"

            c_list = []
            lag_indices = []
            percentage_list = [] if (weights is not None and scale_times) and (lag_n != 0) else None

            weights_lag = []

            if lag_tau is not None:
                assert sti_c is not None
                dt = sti_c.timestep * sti_c.save_step
                n = int(lag_tau / dt)
                print(f"converting lag_tau {lag_tau} to lag_n {n} with dt={dt}")
                lag_n = n

            if lag_n != 0 and verbose:
                print(f"getting lag indices for {len(ti)} trajectories")

            assert sti_c is not None

            timestep = sti_c.timestep * sti_c.save_step

            @jit_decorator
            @partial(vmap_decorator, in_axes=(0, 0, None, None))
            def get_lag_idx(n, w0, dw: Array, dt):
                tau = lag_n * timestep

                integral = dw / w0 * dt
                integral = jnp.where(jnp.arange(integral.shape[0]) <= n, 0, integral)
                integral = jnp.cumsum(integral)  # sum e^(beta U)

                _index_k = jnp.argmin(jnp.abs((integral[n] + tau) - integral))

                _index_k = jnp.where(_index_k <= n, n, _index_k)

                # first larger index
                index_0 = jnp.where(integral[_index_k] >= integral[n] + tau, _index_k - 1, _index_k)  # type: ignore
                index_1 = jnp.min(jnp.array([index_0 + 1, integral.shape[0] - 1]))  # index after

                indices = jnp.array([index_0, index_1])

                values = integral[indices]

                b = (index_0 <= (integral.shape[0] - 2)) * (jnp.isfinite(values[1]))

                # make sure that percentage is not if int is zero

                percentage = jnp.where(
                    (values[1] - values[0]) < 1e-10,
                    0.0,
                    ((integral[n] + tau) - values[0]) / (values[1] - values[0]),
                )

                return indices[0], indices[1], b, percentage

            for n, ti_i in enumerate(ti):
                if lag_n != 0:
                    if scale_times:
                        scales = weights[n]

                        scales = scales

                        def sinhc(x):
                            return jnp.where(x < 1e-10, 1, jnp.sinh(x) / x)

                        # diffusion = True

                        # if diffusion:
                        #     dw = jnp.exp(-(jnp.log(scales[1:]) - jnp.log(scales[:-1])) / 2) * sinhc(
                        #         (jnp.log(scales[1:]) - jnp.log(scales[:-1])) / 2
                        #     )

                        # else:
                        dw = (scales[1:] + jnp.log(scales[:-1])) / 2

                        assert ti_i.t is not None
                        dt = ti_i.t[1:] - ti_i.t[:-1]

                        from IMLCV.base.UnitsConstants import femtosecond

                        # print(f"{dt[0] / femtosecond}   {lag_n * timestep/ femtosecond=}")
                        # integral = jnp.zeros((scales.shape[0]))
                        # integral = integral.at[1:].set(scales)

                        lag_indices_max, lag_indices_max2, bools, p = get_lag_idx(
                            jnp.arange(scales.shape[0]),
                            scales,
                            dw,
                            dt,
                        )

                        c = jnp.sum(bools)

                        # c = jnp.sum(jnp.logical_and(scales[bools] != 0, scales[lag_indices_max[bools]] != 0))

                        if c == 0:
                            continue

                        assert percentage_list is not None

                        percentage_list.append(p[bools])
                        lag_idx = lag_indices_max[bools]

                        weights_lag.append(jnp.where(bools, scales[lag_indices_max], 0))

                        # ti[n]._t = integral

                    else:
                        c = ti_i.size - lag_n
                        lag_idx = jnp.arange(c) + lag_n

                        weights_lag.append(jnp.where(lag_idx <= ti_i.size, weights[n][lag_idx], 0))

                    c_list.append(c)
                    lag_indices.append(lag_idx)

                else:
                    c = ti_i.size

                    lag_indices.append(jnp.arange(c))
                    c_list.append(c)

                    weights_lag.append(weights[n])

            if scale_times and lag_n != 0:
                # print(f"{c_list=}")

                print(f" {jnp.hstack(c_list)=}")

                ll = []

                assert percentage_list is not None
                for n, (li, pi) in enumerate(zip(lag_indices, percentage_list)):
                    ni = jnp.arange(li.shape[0])
                    dn = li - ni

                    dnp = dn * (1 - pi) + (dn + 1) * pi

                    if jnp.isnan(jnp.mean(dnp)):
                        print(f"{n=} {li=} {pi=}  {ti[n].t=} ")

                    ll.append(jnp.mean(dnp))

                ll = jnp.hstack(ll)

                print(f"average lag index {ll} {jnp.mean(ll)=}")

            total = sum(c_list)

            if out == -1:
                out = total

            if out > n_samples_eff_total:
                print(f"not enough effective datapoints, returning {n_samples_eff_total} data points instead of {out}")
                out = n_samples_eff_total

            ###################

            if verbose:
                print(f"total data points {total}, selecting {out}")

            def choose(
                key,
                weight: list[Array],
                weight_wall: list[Array],
                weight_lag: list[Array],
                ps: list[Array],
                F: list[Array],
                nb: list[Array],
                neb: list[Array],
                density: list[Array],
                grid_nums: list[Array] | None,
                out: int,
            ):
                key, key_return = split(key, 2)

                print("new choice")

                ps_stack = jnp.where(jnp.hstack(nb) != 0, jnp.hstack(ps) / jnp.hstack(nb), 0) * jnp.hstack(weight_wall)
                # ps_stack = jnp.hstack(ps)

                assert jnp.isfinite(ps_stack).all(), "rho_stack contains non-finite values"

                mask = jnp.logical_and(
                    ps_stack > 0.0,
                    jnp.logical_and(
                        jnp.hstack(weight) > 0.0,
                        jnp.hstack(weight_lag) > 0.0,
                    ),
                )

                if select_max_bias is not None:
                    mask = jnp.logical_and(mask, jnp.hstack(F) < select_max_bias)

                    ps_stack = ps_stack

                # print(f"using new p_pair calculation")

                ps_stack = ps_stack[mask]

                nums = jnp.hstack(
                    [jnp.vstack([jnp.full((x.shape[0],), i), jnp.arange(x.shape[0])]) for i, x in enumerate(weight)]
                )

                # n_eff_max = jnp.sum(ps_stack)

                # print(f"effective number of data points {n_eff_max=} out of {total} ")

                # if out >= n_eff_max:
                #     out = int(n_eff_max)
                #     print(f"reducing out to {out=} to match {n_eff_max=}")

                # print(f"{nums.shape=}")

                indices = jnp.argwhere(mask)[
                    choice(
                        key=key,
                        a=ps_stack.shape[0],
                        shape=(int(out),),
                        p=ps_stack,
                        replace=True,
                    )
                ].reshape((-1,))

                _as = jnp.argsort(nums[0, indices]).reshape((-1,))

                indices = indices[_as]

                weight_out = []
                density_out = []

                for Fi, wi, di, nbi in zip(F, weight, density, nb):
                    _vol = di
                    _wi = wi

                    weight_out.append(_wi)
                    density_out.append(_vol)

                return key_return, indices, weight_out, density_out, nums[:, indices], weight

            def remove_lag(w, c):
                w = w[:c]
                return w

            out_indices = []
            out_labels = []

            n_list = []

            # if split_data:
            #     frac = out / total

            #     out_reweights = []
            #     out_dw = []
            #     out_rhos = []

            #     for n, (w_i, wl_i, ps_i, F_i, d_i, nb_i, neb_i, c_i, l_i) in enumerate(
            #         zip(weights, weights_lag, p_select, F, density, n_bin, n_eff_bin, c_list, labels)
            #     ):
            #         # if w_i is not None:
            #         w_i = remove_lag(w_i, c_i)
            #         wl_i = remove_lag(wl_i, c_i)
            #         ps_i = remove_lag(ps_i, c_i)
            #         F_i = remove_lag(F_i, c_i)
            #         d_i = remove_lag(d_i, c_i)
            #         nb_i = remove_lag(nb_i, c_i)
            #         neb_i = remove_lag(neb_i, c_i)

            #         if grid_nums is not None:
            #             gn_i = remove_lag(grid_nums[n], c_i)
            #         else:
            #             gn_i = None

            #         ni = int(frac * c_i)

            #         key, indices, reweight, rerho, _, dw = choose(
            #             key=key,
            #             weight=[w_i],
            #             weight_lag=[wl_i],
            #             ps=[ps_i],
            #             F=[F_i],
            #             nb=[nb_i],
            #             neb=[neb_i],
            #             density=[d_i],
            #             grid_nums=[gn_i] if gn_i is not None else None,
            #             out=ni,
            #         )

            #         out_indices.append(indices)

            #         out_reweights.extend(reweight)
            #         out_rhos.extend(rerho)
            #         out_dw.extend(dw)

            #         n_list.append(n)
            #         out_labels.append(l_i)

            # else:
            _w: list[Array] = []
            _ww: list[Array] = []
            _wl: list[Array] = []
            _ps: list[Array] = []
            _F: list[Array] = []
            _nb: list[Array] = []
            _neb: list[Array] = []
            _d: list[Array] = []

            _grid_nums: list[Array] | None = [] if grid_nums is not None else None

            for n, (w_i, ww_i, wl_i, ps_i, F_i, d_i, nb_i, neb_i, c_i) in enumerate(
                zip(weights, weight_wall, weights_lag, p_select, F, density, n_bin, n_eff_bin, c_list)
            ):
                _w.append(remove_lag(w_i, c_i))
                _ww.append(remove_lag(ww_i, c_i))
                _wl.append(remove_lag(wl_i, c_i))
                _nb.append(remove_lag(nb_i, c_i))
                _d.append(remove_lag(d_i, c_i))
                _ps.append(remove_lag(ps_i, c_i))
                _F.append(remove_lag(F_i, c_i))
                _neb.append(remove_lag(neb_i, c_i))

                if grid_nums is not None:
                    _grid_nums.append(remove_lag(grid_nums[n], c_i))  # type: ignore

            key, indices, out_reweights, out_rhos, nums_full, out_dw = choose(
                key=key,
                weight=_w,
                weight_wall=_ww,
                weight_lag=_wl,
                ps=_ps,
                F=_F,
                nb=_nb,
                neb=_neb,
                density=_d,
                grid_nums=_grid_nums if _grid_nums is not None else None,
                out=int(out),
            )

            print(f"selected {len(indices)} {out=} of {total} data points {len(out_reweights)=} {len(out_rhos)=}")

            count = 0

            for n, n_i in enumerate(c_list):
                indices_full = indices[jnp.logical_and(count <= indices, indices < count + n_i)]
                index = indices_full - count

                # index = nums_full[1, nums_full[0, :] == n]

                if len(index) == 0:
                    count += n_i
                    continue

                out_labels.append(labels[n])
                out_indices.append(index)
                n_list.append(n)

                count += n_i

            print(f"{n_list=}")

            ###################
            # storing data    #
            ###################

            # out_sp: list[SystemParams] = []
            # out_cv: list[CV] = []
            out_ti: list[TrajectoryInfo] = []
            # out_weights: list[Array] = []
            # out_rho: list[Array] = []

            out_weights_std: list[Array] | None = [] if calculate_std else None

            if time_series:
                # out_sp_t: list[SystemParams] = []
                # out_cv_t: list[CV] = []
                # out_ti_t: list[TrajectoryInfo] = []

                # out_weights_t: list[Array] = []
                # out_rho_t: list[Array] = []

                out_dynamic_weights: list[Array] = []

            if get_bias_list:
                out_biases = []
            else:
                out_biases = None

            n_tot = 0

            print("gathering data")

            for n, indices_n in zip(n_list, out_indices):
                print(".", end="", flush=True)
                n_tot += 1

                if n_tot % 100 == 0:
                    print("")
                    n_tot = 0

                ti_n: TrajectoryInfo = ti[n][indices_n]
                ti_n.w = out_reweights[n][indices_n]
                ti_n.rho = out_rhos[n][indices_n]

                if calculate_std:
                    out_weights_std.append(weights_std[n][indices_n])

                if time_series:
                    idx_t = lag_indices[n][indices_n]
                    idx_t_p = idx_t + 1

                    # print(f"{ti_n.shape=} {idx_t.shape=} ")

                    if percentage_list is None:
                        ti_t_n: TrajectoryInfo = ti[n][idx_t]

                        ti_t_n.w = out_reweights[n][idx_t]
                        ti_t_n.rho = out_rhos[n][idx_t]

                        out_dynamic_weights.append(jnp.sqrt(out_dw[n][idx_t] / out_dw[n][indices_n]))

                    else:
                        percentage = percentage_list[n][indices_n]

                        @partial(vmap_decorator, in_axes=(0, 0, 0))
                        def interp(xi, yi, pi):
                            return jax.tree_util.tree_map(
                                lambda xii, yii: (1 - pi) * xii + pi * yii,
                                xi,
                                yi,
                            )

                        # TODO: make interp a CV function and use apply

                        # inerp needs full
                        ti_n_t_ = ti[n][idx_t]
                        ti_n_tp = ti[n][idx_t_p]

                        if isinstance(ti_n_t_, EagerTrajectoryInfo):
                            ti_n_t_ = ti_n_t_.to_full()
                        if isinstance(ti_n_tp, EagerTrajectoryInfo):
                            ti_n_tp = ti_n_tp.to_full()

                        ti_t_n: FullTrajectoryInfo = interp(ti_n_t_, ti_n_tp, percentage)

                        # print(f"{ti_t_n.size=} {ti_n_t_.size=} {ti_n.size=} ")

                        ti_t_n.t = ti_n.t + lag_n * timestep

                        ti_t_n.w = interp(out_reweights[n][idx_t], out_reweights[n][idx_t_p], percentage)
                        ti_t_n.rho = interp(out_rhos[n][idx_t], out_rhos[n][idx_t_p], percentage)

                        # out_ti_t.append(ti_t_n)

                        out_dynamic_weights.append(
                            jnp.sqrt(
                                interp(
                                    out_dw[n][idx_t],
                                    out_dw[n][idx_t_p],
                                    percentage,
                                )
                                / out_dw[n][indices_n]
                            )
                        )

                    ti_n.w_t = ti_t_n.w
                    ti_n.rho_t = ti_t_n.rho
                    ti_n.positions_t = ti_t_n.positions
                    ti_n.cell_t = ti_t_n.cell

                    ti_n.cv_t = ti_t_n.cv

                # do sanity check
                num_expected = indices_n.shape[0]
                assert ti_n.size == num_expected, f"sizes do not match {ti_n.size=} {num_expected=}"
                assert ti_n.w.shape[0] == num_expected, f"sizes do not match {ti_n.w.shape[0]=} {num_expected=}"
                assert ti_n.rho.shape[0] == num_expected, f"sizes do not match {ti_n.rho.shape[0]=} {num_expected=}"
                assert ti_n.positions.shape[0] == num_expected, (
                    f"sizes do not match {ti_n.positions.shape[0]=} {num_expected=}"
                )

                if time_series:
                    assert ti_n.w_t.shape[0] == num_expected, f"sizes do not match {ti_n.w_t.shape[0]=} {num_expected=}"
                    assert ti_n.rho_t.shape[0] == num_expected, (
                        f"sizes do not match {ti_n.rho_t.shape[0]=} {num_expected=}"
                    )
                    assert ti_n.positions_t.shape[0] == num_expected, (
                        f"sizes do not match {ti_n.positions_t.shape[0]=} {num_expected=}"
                    )

                out_ti.append(ti_n)

                if get_bias_list:
                    assert out_biases is not None
                    out_biases.append(bias_list[n])

            # print("")
            # # print(f"c05 {jnp.hstack(out_weights).shape=} {w_full.shape} ")
            # # print(f"c05 {jnp.hstack(out_weights)-w_full=} ")

            # def norm_rho(w, rho):
            #     w_log = [(jnp.log(wi) + jnp.log(rhoi)) for wi, rhoi in zip(w, rho)]

            #     z = jnp.hstack(w_log)
            #     z_max = jnp.max(z)
            #     norm = jnp.log(jnp.sum(jnp.exp(z - z_max))) + z_max

            #     rho = [jnp.exp(jnp.log(rho_i) - norm) for rho_i in rho]

            #     return rho

            # out_rho = norm_rho(out_weights, out_rho)

            # if time_series:
            #     # print(f"normalizing time series rho {out_rho_t=}")

            #     out_rho_t = norm_rho(out_weights_t, out_rho_t)

            # print(f"len(out_sp) = {len(out_sp)} ")

        ###################

        if verbose:
            print("Checking data")

        # assert update_info is not None

        if nl_info is not None:
            assert update_info is not None
            out_nl = NeighbourList(
                info=nl_info,
                update=update_info,
                sp_orig=None,
            )
        else:
            out_nl = None

        if time_series:
            out_nl_t = out_nl

        if lag_tau is not None:
            tau = lag_tau
        else:
            tau = lag_n * sti_c.timestep * sti_c.save_step

        # if time_series:
        #     tau = None

        #     arr = []

        #     for tii, ti_ti in zip(out_ti):
        #         tii: TrajectoryInfo
        #         ti_ti: TrajectoryInfo

        #         assert ti_ti.t is not None
        #         assert tii.t is not None

        #         dt = ti_ti.t - tii.t

        #         tau = jnp.median(dt) if tau is None else tau

        #         mask = jnp.allclose(dt, tau)

        #         if not mask.all():
        #             arr.append(jnp.sum(jnp.logical_not(mask)))

        #     if tau is None:
        #         print("WARNING: tau None")
        #     else:
        #         if len(arr) != 0:
        #             print(
        #                 f"WARNING:time steps are not equal, {jnp.array(arr)} out of {out} trajectories have different time steps"
        #             )

        #         from IMLCV.base.UnitsConstants import femtosecond

        #         print(
        #             f"tau = {tau / femtosecond:.2f} fs, lag_time*timestep*write_step = {lag_n * sti.timestep * sti.save_step / femtosecond:.2f} fs"
        #         )

        ###################

        dlo_kwargs = dict(
            # sp=out_sp,
            nl=out_nl,
            # cv=out_cv,
            ti=out_ti,
            sti=sti,
            collective_variable=colvar,
            time_series=time_series,
            bias=out_biases,
            ground_bias=ground_bias,
            # _weights=out_weights,
            scaled_tau=scale_times,
            # _rho=out_rho,
            labels=out_labels,
            frac_full=frac_full,
            _weights_std=out_weights_std,
            bounds=bounds,
            n_hist=nhist,
        )

        if time_series:
            dlo_kwargs.update(
                nl_t=out_nl_t,  # type:ignore
                tau=tau,  # type:ignore
                dynamic_weights=out_dynamic_weights,
            )

        dlo = DataLoaderOutput(
            **dlo_kwargs,  # type: ignore
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

        # if output_FES_bias:
        #     return dlo, FES_biases

        return dlo

    def iter_ase_atoms(
        self,
        r: int | None = None,
        c: int | None = None,
        num: int = 3,
        r_cut=None,
        minkowski_reduce=True,
        only_finished=False,
        ignore_invalid=True,
    ):
        import ase

        from IMLCV.base.UnitsConstants import angstrom

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

                sp = vmap_decorator(SystemParams.apply_canonicalize, in_axes=(0, None))(sp, op)

            pos_A = sp.coordinates / angstrom
            pbc = sp.cell is not None
            if pbc:
                cell_A = sp.cell / angstrom  # type:ignore

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
        eager: bool = True,
    ) -> TrajectoryInformation:
        if c is None:
            c = self.cv

        if eager:
            # print(f"loading eager trajectory info for {c=} {r=} {i=}")
            ti = EagerTrajectoryInfo.create(self.path(c=c, r=r, i=i) / "trajectory_info.h5")
        else:
            ti = FullTrajectoryInfo.load(self.path(c=c, r=r, i=i) / "trajectory_info.h5")

        return TrajectoryInformation(
            ti=ti,
            cv=c,
            round=r,
            num=i,
            folder=self.folder,
        )

    def static_trajectory_information(self, c: int | None = None, r: int | None = None) -> StaticMdInfo:
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
            cv=int(c),
            folder=self.folder,
            tic=stic,
            num_vals=jnp.array(mdi),
            num=len(mdi),
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

    def get_round(self, c: int | None = None):
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

    def n(self, c: int | None = None, r: int | None = None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)
        return self._round_information(r=r).num

    def invalidate_data(self, c: int | None = None, r: int | None = None, i: int | None = None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        if i is None:
            p = self.path(c=c, r=r, i=i) / "static_trajectory_info.h5"
        else:
            p = self.path(c=c, r=r, i=i) / "trajectory_info.h5"

        if not p.exists():
            print(f"cannot validate data for {c=} {r=} {i=} because traj file does not exist")
            return

        n = "invalid" if i is None else "_invalid"

        try:
            with h5py.File(p, "r+") as hf:
                hf.attrs[n] = True
        except Exception:
            print(f"could not invalidate {c=} {r=} {i=}, writing file")

            with open(self.path(c=c, r=r, i=i) / "invalid", "w"):
                pass

    def validate_data(self, c: int | None = None, r: int | None = None, i: int | None = None):
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
                hf.attrs[n] = False
        except Exception:
            print(f"could not invalidate {c=} {r=} {i=}, writing file")

    def finish_data(self, c: int | None = None, r: int | None = None, i: int | None = None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        if i is None:
            i_list = self._i_vals(c=c, r=r)
        else:
            i_list = [i]

        for i in i_list:
            p = self.path(c=c, r=r, i=i) / "trajectory_info.h5"

            if not p.exists():
                print(f"cannont finish data for {c=} {r=} {i=} because traj file does not exist")

            with h5py.File(p, "r+") as hf:
                hf.attrs["_finished"] = True

    def is_valid(self, c: int | None = None, r: int | None = None, i: int | None = None):
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

    def is_finished(self, c: int | None = None, r: int | None = None, i: int | None = None):
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
                out = hf.attrs["_finished"]
            else:
                return False

        if out:
            return True

        return False

    def frac_finished(self, c: int | None = None, r: int | None = None):
        if c is None:
            c = self.cv
        if r is None:
            r = self.get_round(c=c)

        i_vals = self._i_vals(c=c, r=r)
        n_finished = 0
        n_total = 0
        for i in i_vals:
            if not self.is_valid(c=c, r=r, i=i):
                continue

            n_total += 1

            if self.is_finished(c=c, r=r, i=i):
                n_finished += 1

        return n_finished / n_total if n_total > 0 else 0.0

    def subsample(
        self,
        factor: int | None = None,
        new_save_step: int | None = None,
        c: int | None = None,
        r_range=None,
        i_range=None,
        init_size=None,
    ):
        if c is None:
            c = self.cv

        if factor is None and new_save_step is None:
            raise ValueError("provide subsampling factor or new save step")

        if factor is not None and new_save_step is not None:
            raise ValueError("provide only one of subsampling factor or new save step")

        print(f"subsampling trajectories by factor {factor} for cv round {c}")

        print(f"{r_range=}, {i_range=}")

        if r_range is None:
            _r = self._r_vals(c=c)
        else:
            _r = r_range

        for r in _r:
            if r == 0:
                continue

            if new_save_step is not None:
                stic = self.get_static_trajectory_info(c=c, r=r)
                factor = new_save_step // stic.save_step
                if factor == 0:
                    print(f"new save step {new_save_step} is smaller than current save step {stic.save_step}, skipping")
                    continue

                if factor == 1:
                    print(f"new save step {new_save_step} is equal to current save step {stic.save_step}, skipping")
                    continue

                print(f"{factor=}")

            print(f"subsampling round {r} by factor {factor}")

            if i_range is not None:
                i_vals = i_range
            else:
                i_vals = self._i_vals(c=c, r=r)

            for i in i_vals:
                try:
                    p = self.path(c=c, r=r, i=i) / "trajectory_info.h5"
                    ti = FullTrajectoryInfo.load(p)
                    if init_size is not None:
                        assert ti.size == init_size, f"wrong size {ti.size=}"

                    ti = ti[::factor]
                    ti.save(p)
                    print(".", end="", flush=True)
                except Exception as e:
                    print(f"could not subsample trajectory {c=} {r=} {i=}: {e}")

            print("")

            p = self.path(c=c, r=r) / "static_trajectory_info.h5"

            stic = StaticMdInfo.load(p)
            stic.save_step *= factor
            stic.save(p)

        print("done subsampling")

    def get_collective_variable(
        self,
        c: int | None = None,
    ) -> CollectiveVariable:
        if c is None:
            c = self.cv

        return CollectiveVariable.load(self.path(c=c) / "cv.json")

    def get_static_trajectory_info(
        self,
        c: int | None = None,
        r: int | None = None,
    ):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        return StaticMdInfo.load(self.path(c=c, r=r) / "static_trajectory_info.h5")

    # def get_bias(
    #     self,
    #     c: int | None = None,
    #     r: int | None = None,
    #     i: int | None = None,
    # ) -> Bias:
    #     if c is None:
    #         c = self.cv

    #     if r is None:
    #         r = self.get_round(c=c)

    #     bn = self._name_bias(c=c, r=r, i=i)
    #     assert bn is not None

    #     return Bias.load(self.full_path(bn), root=self.folder)

    def get_engine(self, c: int | None = None, r: int | None = None, i: int | None = None) -> MDEngine:
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        name = self._name_md(c=c, r=r)

        assert name is not None

        return MDEngine.load(
            self.full_path(name),
            bias=self.get_bias(
                c=c,
                r=r,
                i=i,
            ),
            permant_bias=self.get_permanent_bias(),
            filename=None,
        )

    ######################################
    #          MD simulations            #
    ######################################

    def run(self, bias, steps):
        self.run_par([bias], steps)

    def run_par(
        self,
        biases: Sequence[Bias],
        steps,
        plot=True,
        KEY=42,
        sp0: SystemParams | None = None,
        ignore_invalid=True,
        md_trajs: list[int] | None = None,
        cv_round: int | None = None,
        wait_for_plots=False,
        min_traj_length=None,
        recalc_cv=False,
        only_finished=False,
        profile=False,
        chunk_size=None,
        # T_scale=10,
        macro_chunk=2000,
        lag_n=20,
        use_common_bias=True,
        dT=0 * kelvin,
    ):
        if cv_round is None:
            cv_round = self.cv

        r = self.get_round(c=cv_round)

        if use_common_bias:
            common_bias_name = self._name_bias(c=cv_round, r=r)
            assert common_bias_name is not None
            common_bias_name = self.full_path(common_bias_name)
        else:
            common_bias_name = None

        common_md_name = self._name_md(c=cv_round, r=r)
        assert common_md_name is not None
        common_md_name = self.full_path(common_md_name)

        cv_path = self.path(cv_round) / "cv.json"

        pn = self.path(c=cv_round, r=r)

        out = bash_app_python(
            function=Rounds._get_init,
            executors=Executors.training,
            execution_folder=pn,
        )(
            rounds=self,
            KEY=KEY,
            ignore_invalid=ignore_invalid,
            only_finished=only_finished,
            min_traj_length=min_traj_length,
            recalc_cv=recalc_cv,
            # T_scale=T_scale,
            chunk_size=chunk_size,
            md_trajs=md_trajs,
            cv_round=cv_round,
            biases=biases,
            sp0=sp0,
            common_bias_name=common_bias_name,
            r=r,
            macro_chunk=macro_chunk,
            lag_n=lag_n,
            dT=dT,
        ).result()

        # from parsl.dataflow.dflow import AppFuture

        tasks: list[tuple[int, Future]] | None = None
        plot_tasks = []

        from IMLCV.configs.config_general import RESOURCES_DICT

        resources = RESOURCES_DICT[Executors.reference.value]

        print(f"using resources {resources} for md")

        for i, (spi, bi, traj_name, b_name, b_name_new, path_name) in enumerate(zip(*out)):
            future = bash_app_python(
                Rounds.run_md,
                executors=Executors.reference,
                profile=profile,
                execution_folder=path_name,
            )(
                self,
                cv=cv_round,
                r=r,
                i=i,
                inputs=[Path(common_md_name), b_name, Path(self.folder)],
                outputs=[b_name_new, traj_name],
                sp=spi,  # type: ignore
                steps=int(steps),
                parsl_resource_specification=resources,
            )

            if plot:
                plot_file = path_name / "plot.png"

                plot_fut = bash_app_python(
                    Rounds.plot_md_run,
                    execution_folder=path_name,
                )(
                    outputs=[plot_file],
                    rnds=self,
                    fut=future,
                    c=cv_round,
                    r=r,
                    i=i,
                )

                plot_tasks.append(plot_fut)

            if tasks is None:
                tasks = [(i, future)]
            else:
                tasks.append((i, future))

        assert tasks is not None

        # wait for tasks to finish

        finished = jnp.full((len(tasks),), False)
        exception = jnp.full((len(tasks),), False)

        t0 = None
        time_left = 1

        last_report = time.time()

        res_time = 60 * 10  # 10 minutes
        report_time = 60 * 5

        while True:
            if time.time() - last_report > report_time:
                last_report = time.time()
                print(
                    f"{datetime.datetime.now():%H:%M:%S}  finished:{int(jnp.sum(finished))}/{len(tasks)} exceptions: {int(jnp.sum(exception))}"
                )

            if t0 is not None:
                time_left = res_time - (time.time() - t0)

                if time_left < 0:
                    print("time is up")
                    print(f"{datetime.datetime.now():%H:%M:%S}  {jnp.sum(finished)=} {jnp.sum(exception)=}")
                    break

            if jnp.all(jnp.logical_or(finished, exception)):
                print("all tasks finished ")
                print(f"{datetime.datetime.now():%H:%M:%S}  {jnp.sum(finished)=} {jnp.sum(exception)=}")
                break

            if jnp.sum(jnp.logical_or(finished, exception)) > 0.9 * finished.shape[0]:
                print(f"{datetime.datetime.now():%H:%M:%S} most tasks have finished, waiting for max 10 min")
                t0 = time.time()

            for j, (i, future) in enumerate(tasks):
                if t0 is not None:
                    time_left = res_time - (time.time() - t0)

                    if time_left < 0:
                        time_left = 0

                if finished[j] or exception[j]:
                    continue

                try:
                    # wait 1 second for each task
                    future.result(time_left)  # type:ignore

                    finished = finished.at[j].set(True)

                except TimeoutError as _:
                    continue

                except Exception as e:
                    print(f"got exception  while collecting md {i}, round {r}, cv {cv_round}, marking as invalid. {e=}")

                    self.invalidate_data(c=cv_round, r=r, i=i)

                    exception = exception.at[j].set(True)
                    # raise e
                    continue

        unfinished = jnp.argwhere(jnp.logical_not(jnp.logical_or(finished, exception))).reshape((-1))

        if unfinished.shape[0] > 0:
            print(f"canceling {unfinished=} trajectories")

            for j in unfinished:
                i, future = tasks[j]

                future: Future
                try:
                    future.set_exception(TimeoutError)
                except Exception as e:
                    print(f"failed to set exception for {i=},: {e=} ")

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

        common_md_name = self._name_md(c=cv_round, r=round)
        assert common_md_name is not None
        common_md_name = self.full_path(common_md_name)

        # md_engine = MDEngine.load(common_md_name)

        # from parsl.dataflow.dflow import AppFuture

        tasks: list[tuple[int, Future]] | None = None
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

            traj_name = path_name / "trajectory_info.h5"

            future = bash_app_python(
                Rounds.run_md,
                executors=Executors.reference,
                execution_folder=path_name,
            )(
                inputs=[Path(common_md_name), b_name, Path(self.folder)],
                outputs=[b_name_new, traj_name],
                sp=None,  # type: ignore
                steps=int(steps),
            )

            if plot:
                plot_file = path_name / "plot.png"

                plot_fut = bash_app_python(
                    Rounds.plot_md_run,
                    execution_folder=path_name,
                )(
                    rnds=self,
                    i=i,
                    c=cv_round,
                    r=round,
                    inputs=[future.outputs[0]],  # type:ignore
                    outputs=[plot_file],
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
        KEY: jax.Array | int,
        common_bias_name: str | None,
        biases: Sequence[Bias],
        ignore_invalid: bool = False,
        only_finished: bool = True,
        min_traj_length: int | None = None,
        recalc_cv: bool = False,
        # T_scale: float = 10,
        chunk_size: int | None = None,
        md_trajs: list[int] | None = None,
        cv_round: int | None = None,
        sp0: SystemParams | None = None,
        r: int | None = None,
        macro_chunk: int | None = 1000,
        lag_n: int = 20,
        out: int = 20000,
        use_energies: bool = False,
        dT=50 * kelvin,
        # divide_by_histogram=True,
    ):
        sps = []
        bs = []
        traj_names = []
        b_names = []
        b_names_new = []
        path_names = []

        sp0_provided = sp0 is not None

        common_bias = rounds.get_bias(c=cv_round, r=r)
        wall_bias = rounds.get_permanent_bias()

        if wall_bias is not None:
            print("using wall bias to get inital points")

        print(f"{common_bias.collective_variable=}")

        if not sp0_provided:
            dlo_data = rounds.data_loader(
                num=3,
                start=0,
                out=out,
                split_data=False,
                new_r_cut=None,
                ignore_invalid=ignore_invalid,
                md_trajs=md_trajs,
                cv=cv_round,
                min_traj_length=min_traj_length,
                recalc_cv=recalc_cv,
                only_finished=only_finished,
                weight=True,
                weighing_method="BC",
                n_max=1e3,
                # T_scale=T_scale,
                time_series=False,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                verbose=True,
                # lag_n=lag_n,
                recalc_bounds=False,
            )

            T = dlo_data.sti.T

            beta = 1 / (T * boltzmann)

            # get the weights of the points

            sp_stack = SystemParams.stack(*dlo_data.sp)

            if wall_bias is not None:
                print("adding wall bias energies to initial weights")
                _, e_wall = wall_bias.compute_from_system_params(
                    sp=sp_stack,
                    chunk_size=chunk_size,
                )
                e_wall = e_wall.energy
                print(f"{e_wall=}")
            else:
                e_wall = 0.0

            cv_stack = CV.stack(*dlo_data.cv)

            if use_energies:
                try:
                    assert dlo_data.ti is not None, "trajectory information is None"
                    ener_stack = jnp.hstack([x.e_pot for x in dlo_data.ti])  # type: ignore

                    if dlo_data.sti.P is not None:
                        ener_stack += jnp.hstack([x.sp.volume() for x in dlo_data.ti]) * dlo_data.sti.P  # type: ignore

                    print("using energies")

                except Exception as e:
                    print(f"{e=}")
                    ener_stack = None

            # get  weights, and correct for ground state bias.
            # this corrects for the fact that the samples are not uniformly distributed

        else:
            assert sp0.shape[0] == len(biases), (
                f"The number of initials cvs provided {sp0.shape[0]} does not correspond to the number of biases {len(biases)}"
            )

            ener_stack = None

            T = rounds.static_trajectory_information().T

        if isinstance(KEY, int):
            KEY = jax.random.PRNGKey(KEY)

        for i, bias in enumerate(biases):
            path_name = rounds.path(c=cv_round, r=r, i=i)
            if not os.path.exists(path_name):
                os.mkdir(path_name)

            # if common_bias_name is not None:
            #     assert common_bias is not None
            #     b = CompositeBias.create(
            #         [
            #             DTBias.create(
            #                 bias=common_bias,
            #                 dT=dT,
            #                 T=T,
            #             ),
            #             bias,
            #         ]
            #     )
            # else:
            #     b = bias
            b = bias

            # b.collective_variable_path = rounds.rel_path(rounds.path(c=cv_round) / "cv.json")

            b_name = path_name / "bias.json"
            b_name_new = path_name / "bias_new.json"
            b.save(b_name)

            traj_name = path_name / "trajectory_info.h5"

            if not sp0_provided:
                # reweigh data points according to new bias

                ener = bias.compute_from_cv(cvs=cv_stack, chunk_size=chunk_size)[0]

                if use_energies:
                    if ener_stack is not None:
                        # print(f"using energies")
                        ener += ener_stack

                ener += e_wall

                ener -= jnp.min(ener)

                probs = jnp.exp(-ener * beta)
                probs = probs / jnp.sum(probs)

                KEY, k = jax.random.split(KEY, 2)
                index = jax.random.choice(
                    a=probs.shape[0],
                    key=k,
                    p=probs,
                )

                spi = sp_stack[index]

            else:
                assert sp0 is not None
                spi = sp0[i]
                spi = spi.unbatch()

            sps.append(spi)
            bs.append(b)
            traj_names.append(traj_name)
            b_names.append(b_name)
            b_names_new.append(b_name_new)
            path_names.append(path_name)

        return sps, bs, traj_names, b_names, b_names_new, path_names

    # @staticmethod
    def run_md(
        self,
        cv: int,
        r: int,
        i: int,
        steps: int,
        sp: SystemParams | None,
        inputs: list[Path] = [],
        outputs: list[Path] = [],
        parsl_resource_specification={},
    ):
        bias = self.get_bias(c=cv, r=r, i=i)
        assert isinstance(bias, RoundBias)
        permanent_bias = self.get_permanent_bias()

        kwargs = dict(
            trajectory_file=outputs[1],
        )
        if sp is not None:
            kwargs["sp"] = sp  # type: ignore

        md = MDEngine.load(
            inputs[0],
            bias=bias,
            permant_bias=permanent_bias,
            **kwargs,
        )

        md.run(steps)

        # this will save the composite bias
        bias.bias_i.save(outputs[0])

        # delete old bias
        if inputs[1].exists():
            inputs[1].unlink()

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
        assert bias is not None

        cvs = traj.ti.CV

        if cvs is None:
            assert traj is not None
            sp = traj.ti.sp

            info = rnd.tic.neighbour_list_info()
            if info is not None:
                nl = sp.get_neighbour_list(
                    info=rnd.tic.neighbour_list_info(),
                )
            else:
                nl = None

            cvs, _ = bias.collective_variable.compute_cv(sp=sp, nl=nl)

        bias.plot(
            name=outputs[0],
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
        dlo_kwargs: dict | None = None,
        dlo: DataLoaderOutput | None = None,
        chunk_size: int | None = None,
        plot: bool = True,
        new_r_cut: float | None = None,
        new_r_skin: float | None = 2.0 * angstrom,
        save_samples: bool = True,
        save_multiple_cvs: bool = False,
        jac=jax.jacrev,
        cv_round_from: int | None = None,
        cv_round_to: int | None = None,
        test: bool = False,
        max_bias: float | None = None,
        transform_bias: bool = True,
        samples_per_bin: int = 5,
        min_samples_per_bin: int = 1,
        percentile: float = 1e-1,
        use_executor: bool = True,
        n_max: int | float = 1e5,
        vmax: float = 100 * kjmol,
        macro_chunk: int = 1000,
        macro_chunk_nl: int = 5000,
        verbose: bool = False,
        koopman: bool = True,
        output_FES_bias=False,
        equilibration_time=5 * picosecond,
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
            dlo_kwargs["cv"] = cv_round_from

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
            koopman=koopman,
            output_FES_bias=output_FES_bias,
            new_r_skin=new_r_skin,
            equilibration_time=equilibration_time,
        )

        if use_executor:
            bash_app_python(
                Rounds._update_CV,
                executors=Executors.training,
                remove_stderr=False,
                remove_stdout=False,
                execution_folder=self.path(c=cv_round_from),
            )(**kw).result()  # type: ignore

            return

        Rounds._update_CV(**kw)  # type: ignore

    @staticmethod
    def _update_CV(
        rounds: Rounds,
        transformer: Transformer,
        cv_round_from: int,
        cv_round_to: int,
        dlo_kwargs={},
        dlo: DataLoaderOutput | None = None,
        chunk_size=None,
        macro_chunk=1000,
        macro_chunk_nl: int = 5000,
        plot=True,
        new_r_cut=None,
        new_r_skin: float | None = 2.0 * angstrom,
        save_samples=True,
        save_multiple_cvs=False,
        jac=jax.jacrev,
        test=False,
        max_bias=None,
        transform_bias=True,
        samples_per_bin=5,
        min_samples_per_bin=1,
        percentile=1e-1,
        n_max=1e5,
        vmax=100 * kjmol,
        verbose=True,
        koopman=True,
        output_FES_bias=False,
        equilibration_time=5 * picosecond,
    ):
        if dlo is None:
            # plot_folder = rounds.path(c=cv_round_to)
            # cv_titles = [f"{cv_round_from}", f"{cv_round_to}"]

            dlo = rounds.data_loader(
                **dlo_kwargs,
                macro_chunk=macro_chunk,
                verbose=verbose,
                # macro_chunk_nl=macro_chunk_nl,
                min_samples_per_bin=min_samples_per_bin,
                samples_per_bin=samples_per_bin,
                # output_FES_bias=output_FES_bias,
                equilibration_time=equilibration_time,
            )  # type: ignore

            # print(f"{dlo.collective_variable=} {fb[0].collective_variable=}")

            # if fb is not None:
            #     if fb[0] is not None:
            #         Transformer.plot_app(
            #             name=str(plot_folder / "cvdiscovery_pre_bias.png"),
            #             collective_variables=[dlo.collective_variable],
            #             cv_data=None,
            #             biases=[fb[0]],
            #             margin=0.1,
            #             T=dlo.sti.T,
            #             plot_FES=True,
            #             cv_titles=cv_titles,
            #             vmax=vmax,
            #         )

        cvs_new, new_collective_variable, new_bias, w_new = transformer.fit(
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
            koopman=koopman,
        )

        # update state
        rounds.__update_CV(
            new_collective_variable=new_collective_variable,
            new_bias=new_bias,
            cv_round_from=cv_round_from,
            cvs_new=cvs_new,
            dlo=dlo,
            new_r_cut=new_r_cut,
            new_r_skin=new_r_skin,
            save_samples=save_samples,
            save_multiple_cvs=save_multiple_cvs,
            w_new=w_new,
        )

    def transform_CV(
        self,
        cv_trans: CvTrans,
        dlo_kwargs=None,
        dlo: DataLoaderOutput | None = None,
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

        assert dlo is not None

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
        dlo: DataLoaderOutput,
        new_r_cut: float | None = None,
        new_r_skin: float | None = None,
        save_samples=True,
        save_multiple_cvs=False,
        w_new: list[Array] | None = None,
    ):
        if new_r_cut is None:
            new_r_cut = dlo.sti.r_cut

        rounds.add_cv(new_collective_variable, c=cv_round_from + 1)

        stic = rounds.static_trajectory_information(c=cv_round_from)
        stic.r_cut = new_r_cut
        if new_r_skin is not None:
            stic.r_skin = new_r_skin

        rounds.add_round(
            bias=new_bias,
            stic=stic,
            mde=rounds.get_engine(c=cv_round_from),
        )

        # make sure it doesn't take to much space
        rounds.zip_cv_rounds()

        # if save_samples:
        #     first = True

        #     if save_multiple_cvs:
        #         raise NotImplementedError

        #     # if save_multiple_cvs:
        #     #     for dlo_i, cv_new_i in zip(iter(dlo), cvs_new):
        #     #         if not first:
        #     #             rounds.add_cv(new_collective_variable)
        #     #             rounds.add_round(bias=NoneBias.create(new_collective_variable), stic=stic)

        #     #         rounds._copy_from_previous_round(
        #     #             dlo=dlo_i,
        #     #             new_cvs=[cv_new_i],
        #     #             cv_round=cv_round_from,
        #     #         )
        #     #         rounds.add_round(bias=new_bias, stic=stic)

        #     #         first = False

        #     # else:
        rounds._copy_from_previous_round(
            dlo=dlo,
            new_cvs=cvs_new,
            cv_round=cv_round_from,
        )

        rounds.add_round(bias=new_bias, stic=stic)

    def _copy_from_previous_round(
        self,
        dlo: DataLoaderOutput,
        new_cvs: list[CV],
        invalidate: bool = False,
        cv_round: int | None = None,
        w_new: list[Array] | None = None,
        w_t_new: list[Array] | None = None,
    ):
        if cv_round is None:
            cv_round = self.cv - 1

        if w_new is None:
            w_new = dlo._weights

        for i in range(len(dlo.ti)):
            round_path = self.path(c=self.cv, r=0, i=i)
            round_path.mkdir(parents=True, exist_ok=True)

            traj_info = dlo.ti[i]

            assert traj_info.positions is not None
            assert w_new is not None
            assert dlo._rho is not None

            new_traj_info = FullTrajectoryInfo.create(
                positions=traj_info.positions,
                cell=traj_info.cell,
                charges=traj_info.charges,
                e_pot=traj_info.e_pot,
                e_bias=traj_info.e_bias,
                cv=new_cvs[i].cv,
                cv_orig=traj_info.cv,
                w=w_new[i],
                # w_t=w_t_new[i],
                rho=traj_info.rho,
                sigma=dlo._weights_std[i] if dlo._weights_std is not None else None,
                T=traj_info.T,
                P=traj_info.P,
                err=traj_info.err,
                t=traj_info.t,
                size=traj_info.size,
                finished=True,
            )

            new_traj_info.save(round_path / "trajectory_info.h5")

            if invalidate:
                self.invalidate_data(c=self.cv, r=self.round, i=i)


class WeightOutput(MyPyTreeNode):
    weight: list[Array]  # e^( beta U - F_i ), ammount of bias
    p_select: list[Array]  # selection probability, due to sampling
    F: list[Array]  # free energy, if available
    density: list[Array]  # density of bin ik
    n_bin: list[Array]  # number of points in bin_ik
    n_eff_bin: list[Array] | None = None  # effective number of points in bin_ik
    grid_nums: list[Array] | None = None  # grid numbers of each point

    p_std: list[Array] | None = (
        None  # standard deviation of p_select. standerd deviation for bin is obtained by sum_i  p_i**2 p_std_i**2/ (sum_i p_i)**2
    )

    FES_bias: Bias | None = None
    FES_bias_std: Bias | None = None

    labels: list[int] | None = None

    frac_full: float

    bounds: jax.Array | None = None
    n_bins: int | None = None

    N_samples_eff: int | None = None


class DataLoaderOutput(MyPyTreeNode):
    sti: StaticMdInfo
    ti: list[TrajectoryInfo]
    collective_variable: CollectiveVariable
    labels: list[Array] | None = None
    nl: list[NeighbourList] | NeighbourList | None = None
    nl_t: list[NeighbourList] | NeighbourList | None = None
    time_series: bool = False
    tau: float | None = None
    bias: list[Bias] | None = None
    ground_bias: Bias | None = None
    _weights_std: list[Array] | None = None
    dynamic_weights: list[Array] | None = None
    scaled_tau: bool = False

    frac_full: float = 1.0
    bounds: jax.Array | None = None
    n_hist: int | None = None

    @property
    def sp(self) -> list[SystemParams] | None:
        return [ti.sp for ti in self.ti]

    @property
    def sp_t(self) -> list[SystemParams] | None:
        return [ti.sp_t for ti in self.ti] if self.ti[0].sp_t is not None else None

    @property
    def cv(self) -> list[CV] | None:
        return [ti.CV for ti in self.ti] if self.ti[0].CV is not None else None

    @property
    def cv_t(self) -> list[CV] | None:
        return [ti.CV_t for ti in self.ti] if self.ti[0].CV_t is not None else None

    @property
    def _weights(self) -> list[Array] | None:
        return [ti.w for ti in self.ti] if self.ti[0].w is not None else None

    @property
    def _rho(self) -> list[Array] | None:
        return [ti.rho for ti in self.ti] if self.ti[0].rho is not None else None

    @property
    def _rho_t(self) -> list[Array] | None:
        return [ti.rho_t for ti in self.ti] if self.ti[0].rho_t is not None else None

    @property
    def _weights_t(self) -> list[Array] | None:
        return [ti.w_t for ti in self.ti] if self.ti[0].w_t is not None else None

    @staticmethod
    def get_histo(
        data_nums: list[CV],
        weights: None | list[Array] = None,
        log_w: bool = False,
        macro_chunk: int | None = 320,
        verbose: bool = False,
        shape_mask: Array | None = None,
        nn: int = -1,
        f_func: Callable | None = None,
        observable: list[CV] | None = None,
    ) -> jax.Array:
        if f_func is None:

            def _f_func(x, nl):
                return x
        else:
            _f_func = f_func

        if shape_mask is not None:
            nn = int(jnp.sum(shape_mask)) + 1  # +1 for bin -1

        nn_range = jnp.arange(nn)

        if shape_mask is not None:
            nn_range = nn_range.at[-1].set(-1)

        h = jnp.zeros((nn,))

        def _get_histo(
            x: tuple[Array | None, Array, Array | None],
            cv_i: CV,
            obs: CV | None,
            weights: Array | None,
            _weights_t=None,
            _dweights_t=None,
        ) -> tuple[Array | None, Array, Array | None]:
            alpha_factors, h, obs_array = x

            cvi = cv_i.cv[:, 0]

            if not log_w:
                wi = jnp.ones_like(cvi) if weights is None else weights
                h = h.at[cvi].add(wi)

                if obs is not None:
                    # print(f"{obs_array.shape=} {cvi.shape=} {obs.cv.shape=} {wi.shape=}")

                    u = obs.cv * wi[:, None]

                    obs_array = obs_array.at[cvi, :].add(u)  # type: ignore

                return (alpha_factors, h, obs_array)

            if alpha_factors is None:
                alpha_factors = jnp.full_like(h, -jnp.inf)

            # this log of hist. all calculations are done in log space to avoid numerical issues
            w_log = jnp.zeros_like(cvi) if weights is None else weights

            @partial(vmap_decorator, in_axes=(0, None, None, 0))
            def get_max(ref_num, data_num, log_w, alpha):
                val = jnp.where(data_num == ref_num, log_w, -jnp.inf)

                max_new = jnp.max(val)  # type: ignore
                max_tot = jnp.max(jnp.array([alpha, max_new]))

                return max_tot, max_new > alpha, max_tot - alpha

            alpha_factors, alpha_factors_changed, d_alpha = get_max(nn_range, cvi, w_log, alpha_factors)

            # shift all probs by the change in alpha
            h: Array = jnp.where(alpha_factors_changed, h - d_alpha, h)  # type:ignore

            if obs is not None:
                obs_array: jax.Array = jnp.where(alpha_factors_changed[:, None], obs_array - d_alpha, obs_array)  # type:ignore

            m = alpha_factors != -jnp.inf

            p_new = jnp.zeros_like(h)
            p_new = p_new.at[cvi].add(jnp.exp(w_log - alpha_factors[cvi]))

            h: jax.Array = jnp.where(m, jnp.log(jnp.exp(h) + p_new), h)

            if obs is not None:
                obs_array = obs_array.at[cvi].add(obs.cv * jnp.exp(w_log - alpha_factors[cvi]))  # type: ignore

            return (alpha_factors, h, obs_array)

        alpha_factors, h, obs = macro_chunk_map_fun(
            f=_f_func,
            # op=data_nums[0].stack,
            y=data_nums,
            y_t=None if observable is None else observable,
            macro_chunk=macro_chunk,
            verbose=verbose,
            chunk_func=_get_histo,
            chunk_func_init_args=(
                None,
                h,
                jnp.zeros((nn, *observable[0].shape[1:])) if observable is not None else None,
            ),
            w=weights,
            jit_f=True,
        )

        if log_w:
            assert alpha_factors is not None
            h += alpha_factors

            if obs is not None:
                obs += alpha_factors[:, None]

        if shape_mask is not None:
            # print(f"{h[-1]=}")
            h = h[:-1]

            if observable is not None:
                assert obs is not None
                obs = obs[:-1]

        if observable is not None:
            return h, obs

        return h

    @staticmethod
    def _histogram(
        metric: CvMetric,
        n_grid=40,
        grid_bounds=None,
        chunk_size=None,
        chunk_size_mid=1,
    ):
        bins, _, _, cv_mid, bounds = metric.grid(n=n_grid, bounds=grid_bounds)

        # print(f"{bounds=} {grid_bounds=}")

        @partial(CvTrans.from_cv_function, mid=cv_mid, bounds=bounds)
        def closest_trans(cv: CV, _nl, shmap, shmap_kwargs, mid: CV, bounds: jax.Array):
            m = jnp.argmin(jnp.sum((mid.cv - cv.cv) ** 2, axis=1), keepdims=True)
            m = jnp.where(jnp.all(jnp.logical_and(cv.cv > bounds[:, 0], cv.cv < bounds[:, 1])), m, -1)

            return cv.replace(cv=m)

        nums = closest_trans.compute_cv(cv_mid, chunk_size=chunk_size)[0].cv

        return (
            cv_mid,
            nums,
            bins,
            closest_trans,
            Partial_decorator(DataLoaderOutput.get_histo, nn=cv_mid.shape[0]),
        )

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
    def norm_w(w_stacked: jax.Array, ret_n=False) -> jax.Array | tuple[jax.Array, jax.Array]:
        n = jnp.sum(w_stacked)

        if ret_n:
            return w_stacked / n, n

        return w_stacked / n

    @staticmethod
    def check_w(w_stacked: jax.Array) -> jax.Array:  # type:ignore
        if jnp.any(jnp.isnan(w_stacked)):
            print(f"WARNING: w_stacked has {jnp.sum(jnp.isnan(w_stacked))} nan values")
            w_stacked: jax.Array = jnp.where(jnp.isnan(w_stacked), 0, w_stacked)

        if jnp.any(jnp.isinf(w_stacked)):
            print(f"WARNING: w_stacked has {jnp.sum(jnp.isinf(w_stacked))} inf values")
            w_stacked = jnp.where(jnp.isinf(w_stacked), 0, w_stacked)

        if jnp.any(w_stacked < 0):
            print(f"WARNING: w_stacked has {jnp.sum(w_stacked < 0)} neg values")
            w_stacked = jnp.where(w_stacked < 0, 0, w_stacked)

        if jnp.sum(w_stacked) < 1e-16:
            print("WARNING: all w_Stacked values are zero")
            raise

        if len(w_stacked) == 0:
            print("WARNING: len w_Stacked is zero")

        return w_stacked

    def koopman_weight(
        self,
        w: list[Array] | None = None,
        w_t: list[Array] | None = None,
        dw: list[Array] | None = None,
        samples_per_bin: int = 50,
        max_bins: int | float = 1e5,
        out_dim: int = -1,
        chunk_size: int | None = None,
        indicator_CV: bool = True,
        koopman_eps: float = 0,
        koopman_eps_pre: float = 0,
        cv_0: list[CV] | None = None,
        cv_t: list[CV] | None = None,
        macro_chunk: int = 1000,
        verbose: bool = False,
        max_features_koopman: int = 5000,
        margin: float = 0.1,
        add_1=True,
        only_diag: bool = False,
        calc_pi: bool = True,
        sparse: bool = False,
        output_w_corr: bool = False,
        correlation: bool = True,
        return_km: bool = False,
        labels: jax.Array | list[int] | None = None,
        koopman_kwargs: dict = {},
        shrink=False,
        shrinkage_method="bidiag",
    ) -> tuple[list[Array], list[Array] | None, list[Array] | None, list[KoopmanModel] | None]:
        if cv_0 is None:
            cv_0 = self.cv

        if add_1 is None:
            add_1 = not indicator_CV

        sd = [a.shape[0] for a in cv_0]

        ndim = cv_0[0].shape[1]

        if cv_t is None:
            assert self.cv_t is not None
            cv_t = self.cv_t

        def unstack_w(w_stacked, stack_dims=None):
            if stack_dims is None:
                stack_dims = sd
            return self._unstack_weights(stack_dims, w_stacked)

        if w is None:
            assert self._weights is not None
            w = self._weights

        if w_t is None:
            assert self._weights_t is not None
            w_t = self._weights_t

        assert self._weights_t is not None

        if dw is None:
            assert self.dynamic_weights is not None
            dw = self.dynamic_weights

        assert dw is not None

        if verbose:
            print("koopman weights")

        if labels is None:
            labels = jnp.array(self.labels) if self.labels is not None else jnp.ones((len(cv_0),))
        else:
            labels = jnp.array(labels)

        unique_labels = jnp.unique(labels)

        if len(unique_labels) > 1:
            print(f"runninbounds_from_cvg koopman weights for {len(unique_labels)} disconnected regions")

        if indicator_CV:
            print("getting bounds")
            grid_bounds, _, constants = CvMetric.bounds_from_cv(
                cv_0,
                margin=margin,
                # chunk_size=chunk_size,
                # n=20,
            )

            if constants:
                print("not performing koopman weighing because of constants in cv")
                # koopman = False

                out_0 = w
                # out_1 = w_t
                out_2 = None
                out_3 = None
                out_4 = None

                return out_0, out_2, out_3, out_4

            tot_samples = sum(sd)

            n_hist = CvMetric.get_n(
                samples_per_bin=samples_per_bin,
                samples=tot_samples,
                n_dims=ndim,
                max_bins=max_bins,
            )

            if verbose:
                print(f"using {n_hist=}")

            cv_mid, nums, bins, closest, get_histo = DataLoaderOutput._histogram(
                metric=self.collective_variable.metric,
                n_grid=n_hist,
                grid_bounds=grid_bounds,
                chunk_size=chunk_size,
            )

            grid_nums, grid_nums_t = self.apply_cv(closest, cv_0, cv_t, chunk_size=chunk_size, macro_chunk=macro_chunk)
            assert grid_nums_t is not None

        _sd = sd
        _cv_0 = cv_0
        _cv_t = cv_t
        _weights = w
        _weights_t = w_t
        _dynamic_weights = dw

        if indicator_CV:
            _grid_nums = grid_nums
            _grid_nums_t = grid_nums_t

        _rho = self._rho
        _rho_t = self._rho_t
        assert _rho is not None
        assert _rho_t is not None

        w_out: list[jax.Array] = [jnp.array(0.0)] * len(sd)
        w_corr_out: list[jax.Array] = [jnp.array(0.0)] * len(sd)

        w_out_t: list[jax.Array] = [jnp.array(0.0)] * len(sd)
        w_corr_out_t: list[jax.Array] = [jnp.array(0.0)] * len(sd)

        km_out: list[KoopmanModel] = []

        for label in unique_labels:
            print(f"calculating koopman weights for {label=}")

            label_mask = [int(a) for a in jnp.argwhere(labels == label).reshape((-1,))]

            # idx_inv = jnp.arange()

            sd = []
            cv_0 = []
            cv_t = []
            weights = []
            weights_t = []
            weights_dyn = []

            if indicator_CV:
                grid_nums = []
                grid_nums_t = []
            rho = []
            rho_t = []

            for idx in label_mask:
                sd.append(_sd[idx])
                cv_0.append(_cv_0[idx])
                cv_t.append(_cv_t[idx])
                weights.append(_weights[idx])
                weights_t.append(_weights_t[idx])
                weights_dyn.append(_dynamic_weights[idx])

                if indicator_CV:
                    grid_nums.append(_grid_nums[idx])
                    grid_nums_t.append(_grid_nums_t[idx])
                rho.append(_rho[idx])
                rho_t.append(_rho_t[idx])

            if indicator_CV:
                print("getting bounds")

                w_pos = [(a > 0) * 1.0 for a in weights]

                hist = get_histo(grid_nums, w_pos)
                hist_t = get_histo(grid_nums_t, w_pos)

                mask = jnp.argwhere(jnp.logical_and(hist > 0, hist_t > 0)).reshape(-1)

                # print(f"{label=} {mask=} {hist.shape=}")

                @partial(CvTrans.from_cv_function, mask=mask)
                def get_indicator(cv: CV, nl, shmap, shmap_kwargs, mask):
                    out = jnp.zeros((hist.shape[0],))  # type: ignore
                    out = out.at[cv.cv].set(1)
                    out = jnp.take(out, mask)

                    print(f"{out=}")

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

            kpn_kw = dict(
                cv_0=cv_km,
                cv_t=cv_km_t,
                nl=None,
                w=weights,
                w_t=weights_t,
                dynamic_weights=weights_dyn,
                rho=rho,
                rho_t=rho_t,
                add_1=add_1,
                # method="tcca",
                symmetric=False,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                verbose=verbose,
                eps=koopman_eps,
                eps_pre=koopman_eps_pre,
                trans=tr,
                max_features=max_features_koopman,
                max_features_pre=max_features_koopman,
                only_diag=only_diag,
                calc_pi=True,
                scaled_tau=self.scaled_tau,
                sparse=sparse,
                out_dim=out_dim,
                correlation=correlation,
                shrink=shrink,
                shrinkage_method=shrinkage_method,
            )
            kpn_kw |= koopman_kwargs

            km = self.koopman_model(**kpn_kw)  # type: ignore

            try:
                w, w_t, w_corr_t, w_corr_t, b = km.koopman_weight(
                    chunk_size=chunk_size,
                    macro_chunk=macro_chunk,
                    verbose=verbose,
                )
            except Exception as e:
                print(f"koopman reweighing failed for {label=}")
                print(e)

                w_unstacked = weights
                w_unstacked_t = weights_t
                w_corr = [jnp.ones_like(a) for a in weights]
                b = False

            if not b:
                print(f"koopman reweighing failed for {label=}")

            def process(w_corr, w_unstacked=None):
                print(f"{jnp.sum(jnp.hstack(w_corr) < 0)=}  {jnp.sum(~jnp.isfinite(jnp.hstack(w_corr)))=}")

                w_stacked_log = jnp.log(jnp.hstack(w_corr))

                if w_unstacked is not None:
                    w_stacked_log += jnp.log(jnp.hstack(w_unstacked))

                w_stacked_log -= jnp.max(w_stacked_log)
                w_stacked = jnp.exp(w_stacked_log)

                w_stacked = self.check_w(w_stacked)
                w_stacked: jax.Array = self.norm_w(w_stacked)  # type:ignore
                w_unstacked = self._unstack_weights(sd, w_stacked)

                return w_unstacked

            w_unstacked = process(w_corr, weights)
            # print(f"{w_corr_t=} {weights_t=}")
            w_unstacked_t = process(w_corr_t, weights_t)

            if w_corr is not None and output_w_corr:
                w_corr = process(w_corr)

            for n, idx in enumerate(label_mask):
                w_out[idx] = w_unstacked[n]
                w_out_t[idx] = w_unstacked_t[n]
                if w_corr is not None and output_w_corr:
                    w_corr_out[idx] = w_corr[n]

            if return_km:
                km_out.append(None)

        sd = _sd

        out_0 = w_out
        out_1 = w_out_t
        out_2 = None
        out_3 = None

        if output_w_corr:
            out_2 = w_corr_out

        if return_km:
            out_3 = km_out

        return out_0, out_1, out_2, out_3

    def rescale_rho_T(self, factor, rho=None):
        if rho is None:
            rho = self._rho

        if factor == 1:
            return self._rho

        assert factor > 0, "factor must be >0"

        energies = [ti_i.e_pot for ti_i in self.ti]

        beta = 1 / (self.sti.T * boltzmann)

        new_rho = []

        for rho_i, e_i in zip(rho, energies):
            rho_new_i = rho_i * jnp.exp(beta * (1 - 1 / factor) * e_i)
            new_rho.append(rho_new_i)

        return new_rho

    @staticmethod
    def _solve_wham(log_U_ik_nl, H_ik_nl, tau_i_nl, verbose=False):
        # print(f"{H_ik_nl=} {tau_i_nl=} {log_U_ik_nl=}")

        # print(f"{jnp.max(log_U_ik_nl)=} {jnp.min(log_U_ik_nl)=}")

        def log_sum_exp_safe(*x: Array, min_val=None):
            _x: Array = jnp.nansum(jnp.stack(x, axis=0), axis=0)  # type:ignore

            x_max = jnp.nanmax(_x)

            x_max = jnp.where(jnp.isfinite(x_max), x_max, 0.0)

            out = jnp.log(jnp.nansum(jnp.exp(_x - x_max))) + x_max

            if min_val is not None:
                print(f"using {min_val=}")
                out: Array = jnp.where(out < min_val, min_val, out)  # type:ignore

            return out

        def get_N_H(mask_ik: jax.Array, H_ik: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            # print(f"{mask_i.shape=}  {H_ik.shape=} ")

            _H_ik = jnp.where(mask_ik, H_ik, 0)  # type:ignore

            _H_k = jnp.sum(_H_ik, axis=(0))
            _N_i = jnp.sum(_H_ik, axis=(1))

            return _N_i, _H_k, _H_ik

        def get_F_i(F_k: jax.Array, mask_ik: jax.Array, log_x: tuple[Array, Array, Array]):
            log_U_ik, _, _ = log_x

            log_U_ik = jnp.where(mask_ik, log_U_ik, jnp.inf)

            F_i = -vmap_decorator(log_sum_exp_safe, in_axes=(None, 0))(-F_k, -log_U_ik)

            return F_i

        def norm_F_k(F_k: Array):
            # mask_k = jnp.isfinite(F_k)

            F_k = jnp.where(jnp.isfinite(F_k), F_k, jnp.inf)

            F_k_norm = log_sum_exp_safe(-F_k)

            F_k += F_k_norm

            return F_k

        def get_F_k(
            F_i: Array,
            mask_ik: Array,
            log_x: tuple[Array, Array, Array],
        ):
            log_U_ik, H_ik, tau_i_nl = log_x

            # mask_i = jnp.isfinite(F_i)

            N_i, H_k, H_ik = get_N_H(mask_ik, H_ik)

            def _s(*x: Array):
                return jnp.nansum(jnp.array(x))

            # odds of using sample k
            log_ps_ik = vmap_decorator(
                vmap_decorator(
                    _s,
                    in_axes=(None, None, 0, None, 0),
                ),  # k
                in_axes=(0, 0, 0, 0, None),
            )(
                # -jnp.log(H_ik),
                jnp.log(N_i),
                -jnp.log(tau_i_nl),
                -log_U_ik,
                +F_i,
                -vmap_decorator(log_sum_exp_safe, in_axes=(None, None, None, 1))(
                    jnp.log(N_i), -jnp.log(tau_i_nl), F_i, -log_U_ik
                ),
            )

            # log_rho_ik = jnp.where(log_rho_ik < -10 * jnp.log(10), -jnp.inf, log_rho_ik)

            log_w_ik = vmap_decorator(
                vmap_decorator(
                    _s,
                    in_axes=(0, None),
                ),  # k
                in_axes=(0, 0),  # i
            )(
                log_U_ik,
                -F_i,
            )

            log_dens_ik = jax.vmap(jnp.add, in_axes=(0, 0))(
                jnp.log(H_ik),
                -jnp.log(N_i),
            )

            # mask_ik = jnp.logical_and(
            #     jnp.logical_and(mask_ik, jnp.isfinite(log_w_ik)),
            #     jnp.logical_and(jnp.isfinite(log_ps_ik), jnp.isfinite(log_w_ik)),
            # )

            # mask_ik = jnp.logical_and(
            #     mask_ik,
            #     jnp.logical_and(
            #         jnp.isfinite(log_ps_ik),
            #         log_ps_ik > jnp.log(1e-10),
            #     ),
            # )

            # log_ps_ik = jnp.where(mask_ik, log_ps_ik, -jnp.inf)
            # log_w_ik = jnp.where(mask_ik, log_w_ik, -jnp.inf)
            # log_dens_ik = jnp.where(mask_ik, log_dens_ik, -jnp.inf)

            log_ps_ik = jnp.where(jnp.isfinite(log_ps_ik), log_ps_ik, -jnp.inf)
            log_w_ik = jnp.where(jnp.isfinite(log_w_ik), log_w_ik, -jnp.inf)
            log_dens_ik = jnp.where(jnp.isfinite(log_dens_ik), log_dens_ik, -jnp.inf)

            F_k = -vmap_decorator(log_sum_exp_safe, in_axes=(1, 1, 1))(
                log_dens_ik,
                log_w_ik,
                log_ps_ik,
            )  # sum over i

            # F_k = jnp.where(jnp.isfinite(F_k), F_k, -jnp.inf)

            F_k = norm_F_k(F_k)
            return F_k, (mask_ik, log_dens_ik, log_w_ik, log_ps_ik)

        @jit_decorator
        def T(x: tuple[Array, Array], log_x: tuple[Array, Array, Array]):
            F_k, mask_ik = x
            mask_ik = jnp.where(mask_ik == 1.0, True, False)
            # F_k = -jnp.log(a_k)/

            F_i = get_F_i(F_k, mask_ik, log_x)
            F_k, (mask_ik, log_dens_ik, log_w_ik, log_ps_ik) = get_F_k(F_i, mask_ik, log_x)

            return (F_k, jnp.where(mask_ik, 1.0, 0.0)), (F_i, log_w_ik, log_ps_ik, log_dens_ik)

        @jit_decorator
        def norm(x: tuple[Array, Array], log_x: tuple[Array, Array, Array]):
            F_k, mask_ik = x
            (F_k_p, _), _ = T((F_k, mask_ik), log_x)

            a_k = jnp.exp(-F_k)
            a_k_p = jnp.exp(-F_k_p)

            return 0.5 * jnp.sum((a_k - a_k_p) ** 2)

        @jit_decorator
        def kl_div(x: tuple[Array, Array], log_x):
            F_k, mask_ik = x
            (F_k_p, _), _ = T(x, log_x)

            print("inside kl div")

            a_k = jnp.exp(-F_k)

            kl_div_k = a_k * (-F_k + F_k_p)

            return kl_div_k  # , jnp.sum(kl_div_k)

        import jaxopt

        log_x_mask = (log_U_ik_nl, H_ik_nl, tau_i_nl)

        from jaxopt import base
        from jaxopt._src.fixed_point_iteration import FixedPointState

        class FP(jaxopt.FixedPointIteration):
            def update(self, params, state: FixedPointState, *args, **kwargs) -> base.OptStep:
                next_params, aux = self._fun(params, *args, **kwargs)

                F_k, _ = params
                F_k_p, _ = next_params

                a_k = jnp.exp(-F_k)
                error = jnp.sum(a_k * (-F_k + F_k_p))

                next_state = FixedPointState(
                    iter_num=state.iter_num + 1,
                    error=error,  # type:ignore
                    aux=aux,
                    num_fun_eval=state.num_fun_eval + 1,
                )

                if self.verbose:
                    self.log_info(next_state, error_name="Distance btw Iterates")
                return base.OptStep(params=next_params, state=next_state)

        solver = FP(
            fixed_point_fun=T,
            # history_size=5,
            tol=1e-14,
            implicit_diff=True,
            has_aux=True,
            maxiter=10000,
        )

        # solver.optimality_fun = optimality_fun  # type:ignore

        # T_next = T(
        #     (
        #         jnp.full((log_U_ik_nl.shape[1],), jnp.log(1.0 / log_U_ik_nl.shape[1])),
        #         jnp.full(H_ik_nl.shape, 1.0),
        #     ),
        #     log_x_mask,
        # )

        # print(f"{T_next=}")

        out = solver.run(
            (
                jnp.full((log_U_ik_nl.shape[1],), 0.0),
                jnp.full(H_ik_nl.shape, 1.0),
            ),
            log_x=log_x_mask,
        )

        # out.state = cast(FixedPointState, out.state)

        # print(f"{out.state=}")

        F_k_nl, mask_ik_float = out.params
        mask_ik_nl = jnp.where(mask_ik_float == 1.0, True, False)

        assert out.state.aux is not None
        F_i_nl, log_w_ik_nl, log_ps_ik_nl, log_dens_ik_nl = out.state.aux

        # print(f"{jnp.exp(log_w_ik_nl)=}")

        if jnp.isnan(out.state.error):
            print(f"error is nan {jnp.sum(mask_ik_nl)=} {out.state.iter_num=}")
            raise

        # print(f"wham done! {out.state.error=} ")

        if verbose:
            n, k = norm((F_k_nl, mask_ik_float), log_x_mask), kl_div((F_k_nl, mask_ik_float), log_x_mask)
            print(f"wham err={n}, kl divergence={jnp.sum(k)} {out.state.iter_num=} {out.state.error=} ")

        return log_w_ik_nl, log_ps_ik_nl, F_i_nl, log_dens_ik_nl, mask_ik_nl, F_k_nl

    def wham_weight(
        self,
        samples_per_bin: int = 10,
        min_samples_per_bin: int = 1,
        n_max: float | int = 1e5,
        n_max_lin: int | None = None,
        chunk_size: int | None = None,
        biases: list[Bias] = None,
        cv_0: list[CV] | None = None,
        cv_t: list[CV] | None = None,
        macro_chunk: int | None = 1000,
        verbose: bool = False,
        margin: float = 0.0,
        compute_std: bool = False,
        sparse_inverse: bool = True,
        recalc_bounds=True,
        correlation_method: str | None = None,
        blav_pair=False,
        n_subgrids: int = 5,
        n_hist: int | None = None,
        get_e_wall=True,
    ) -> WeightOutput:
        if cv_0 is None:
            cv_0 = self.cv

        if cv_t is None:
            cv_t = self.cv_t

        # TODO:https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00867

        # get raw rescaling
        u_unstacked = []
        e_unstacked = []

        beta = 1 / (self.sti.T * boltzmann)

        for ti_i in self.ti:
            e = ti_i.e_bias

            if e is None:
                if verbose:
                    print("WARNING: no bias enerrgy found")
                e = jnp.zeros((ti_i.shape,))

            e -= jnp.min(e)

            u = beta * e

            u_unstacked.append(u)

            e_unstacked.append(ti_i.e_pot * beta)

        (
            frac_full,
            labels,
            x_labels,
            num_labels,
            grid_nums_mask,
            get_histo,
            bins,
            cv_mid,
            hist_mask,
            _,
            bounds,
            tau_i,
        ) = self.get_bincount(
            cv_0,
            n_max=n_max,
            n_max_lin=n_max_lin,
            n_hist=n_hist,
            margin=margin,
            chunk_size=chunk_size,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            macro_chunk=macro_chunk,
            verbose=verbose,
            recalc_bounds=recalc_bounds,
            correlation_method=correlation_method,
        )

        len_i = len(u_unstacked)
        len_k = jnp.sum(hist_mask)
        log_U_ik = jnp.full((len_i, len_k), jnp.inf)
        H_ik = jnp.zeros((len_i, len_k))

        label_i = []

        dx = jnp.array(
            jnp.meshgrid(
                *[jnp.linspace(-(a[0] - a[1]) / 2, (a[0] - a[1]) / 2, n_subgrids) for a in bins], indexing="ij"
            )
        )
        dx = CV(cv=dx.reshape((dx.shape[0], -1)).T)
        n_elem = dx.cv.shape[0]

        print(f"{dx.shape=}")

        @partial(vmap_decorator, in_axes=(0, None), out_axes=0)
        def get_b(center: CV, bias_i: Bias):
            biases, _ = bias_i.compute_from_cv(dx.replace(cv=dx.cv + center.cv))

            u = -beta * biases
            m = jnp.max(u)

            return jnp.log(jnp.sum(jnp.exp(u - m))) + m - jnp.log(n_elem)

        for i in range(len_i):
            if verbose:
                print(".", end="", flush=True)
                if (i + 1) % 100 == 0:
                    print("")

            t_log_U_ik = -get_b(cv_mid[hist_mask], biases[i])

            # print(f"{_log_U_ik=}")

            # # # # log sum exp(-u_i)
            # log_ue: jax.Array = get_histo(
            #     [grid_nums_mask[i]],
            #     [u_unstacked[i] + e_unstacked[i]],
            #     log_w=True,
            #     shape_mask=hist_mask,
            #     macro_chunk=macro_chunk,
            # )  # type:ignore

            # log_e: jax.Array = get_histo(
            #     [grid_nums_mask[i]],
            #     [e_unstacked[i]],
            #     log_w=True,
            #     shape_mask=hist_mask,
            #     macro_chunk=macro_chunk,
            # )  # type:ignore

            # # # mean of e^(-beta U)
            # _log_U_ik: jax.Array = -jnp.where(
            #     jnp.isneginf(_log_H_ik),
            #     -jnp.inf,
            #     log_e - log_ue,
            # )  # type:ignore
            # print(f"{_log_U_ik_2-_log_U_ik=}")

            t_log_H_ik: jax.Array = get_histo(
                [grid_nums_mask[i]],
                None,
                log_w=True,
                shape_mask=hist_mask,
                macro_chunk=macro_chunk,
            )  # type:ignore

            H_ik = H_ik.at[i, :].set(jnp.exp(t_log_H_ik))
            log_U_ik = log_U_ik.at[i, :].set(t_log_U_ik)

            # assign label to trajectory
            labels_i = jnp.sum(x_labels[:, grid_nums_mask[i].cv.reshape(-1)], axis=1)
            label_i.append(jnp.argmax(labels_i))

        # print(f"{jnp.sum(H_ik,axis=0 )=}")
        # print(f"{jnp.sum(H_ik,axis=1 )=}")

        # _log_U_ik = log_U_ik
        # _H_ik = H_ik

        w_ik = jnp.full((log_U_ik.shape[0], log_U_ik.shape[1]), 0.0)
        ps_ik = jnp.full((log_U_ik.shape[0], log_U_ik.shape[1]), 0.0)
        dens_ik = jnp.full((log_U_ik.shape[0], log_U_ik.shape[1]), 0.0)
        mask_ik = jnp.full((log_U_ik.shape[0], log_U_ik.shape[1]), False)

        F_k = jnp.full((log_U_ik.shape[1],), jnp.inf)
        F_i = jnp.full((log_U_ik.shape[0],), jnp.inf)

        tau_i = tau_i

        label_i = jnp.array(label_i).reshape((-1))

        print(f"{jnp.max(log_U_ik)=}")

        for nl in range(0, num_labels):
            mk = labels == nl
            nk = int(jnp.sum(mk))
            arg_mk = jnp.argwhere(mk).reshape((-1,))

            mi = label_i == nl
            ni = jnp.sum(mi)
            arg_mi = jnp.argwhere(mi).reshape((-1,))

            if nk == 0 or ni == 0:
                print(f"skipping label {nl} with {nk} bins and {ni} trajectories")
                continue

            print(f"running wham with {nk} bins and {ni} trajectories")

            # # this is just a meshgrid

            @partial(vmap_decorator, in_axes=(0, None))
            @partial(vmap_decorator, in_axes=(None, 0))
            def _get_arg_ik(i, k):
                return jnp.array([i, k])

            arg_ik = _get_arg_ik(arg_mi, arg_mk)
            # print(f"{arg_ik.shape=}")
            # print(f"{arg_ik=}")
            # print(f"{ arg_mk=}")
            # print(f"{ arg_mi=}")

            H_ik_nl = H_ik[arg_ik[:, :, 0], arg_ik[:, :, 1]]
            log_U_ik_nl = log_U_ik[arg_ik[:, :, 0], arg_ik[:, :, 1]]

            log_w_ik_nl, log_ps_ik_nl, F_i_nl, log_dens_ik_nl, mask_ik_nl, F_k_new = self._solve_wham(
                log_U_ik_nl,
                H_ik_nl,
                tau_i[mi],
                verbose=verbose,
            )

            # print(f"{jnp.sum(jnp.exp(log_ps_ik_nl),axis=0)=}")
            # print(f"{jnp.sum(jnp.exp(log_ps_ik_nl),axis=1)=}")
            # print(f"{jnp.sum(jnp.exp(-F_k_new),axis=0)=}")

            H_ik_nl = jnp.where(mask_ik_nl, H_ik_nl, 0.0)

            mask_ik = mask_ik.at[arg_ik[:, :, 0], arg_ik[:, :, 1]].set(mask_ik_nl)
            w_ik = w_ik.at[arg_ik[:, :, 0], arg_ik[:, :, 1]].set(jnp.exp(log_w_ik_nl))
            ps_ik = ps_ik.at[arg_ik[:, :, 0], arg_ik[:, :, 1]].set(jnp.exp(log_ps_ik_nl))
            dens_ik = dens_ik.at[arg_ik[:, :, 0], arg_ik[:, :, 1]].set(jnp.exp(log_dens_ik_nl))
            H_ik = H_ik.at[arg_ik[:, :, 0], arg_ik[:, :, 1]].set(H_ik_nl)

            F_k = F_k.at[arg_mk].set(F_k_new)
            F_i = F_i.at[arg_mi].set(F_i_nl)

        F_k /= beta
        F_i /= beta
        log_U_ik /= beta

        print(f"{jnp.max(F_k)/kjmol=} {jnp.min(F_k)/kjmol=}  {F_k/kjmol=}")

        H_ik_tau = H_ik / tau_i.reshape((-1, 1))

        N_i_tau = jnp.sum(H_ik_tau, axis=1)

        mask_i = jnp.any(mask_ik, axis=1)
        mask_k = jnp.any(mask_ik, axis=0)

        if not jnp.all(mask_i):
            print(f"WARNING: some trajectories have no valid bins {jnp.argwhere(jnp.logical_not(mask_i))=}")

        if not jnp.all(mask_k):
            print(f"WARNING: some bins have no valid samples {jnp.argwhere(jnp.logical_not(mask_k))=}")

        # print(f"valid i: {jnp.any(mask_ik, axis=1)=} valid_k: {jnp.any(mask_ik, axis=0)=}")

        if compute_std:
            H_ik_mask = H_ik[mask_i, :][:, mask_k]
            log_U_ik_mask = log_U_ik[mask_i, :][:, mask_k]
            F_i_mask = F_i[mask_i]
            F_k_mask = F_k[mask_k]
            N_i_tau_mask = N_i_tau[mask_i]
            H_ik_tau_mask = H_ik_tau[mask_i, :][:, mask_k]

            log_N_i_tau_mask = jnp.log(N_i_tau_mask)
            log_N_tot_tau_mask = jnp.log(jnp.sum(H_ik_tau_mask))

            nk = F_k_mask.shape[0]
            ni = log_N_i_tau_mask.shape[0]

            # see thermolib derivation

            print(f"test: {nk=} {ni=}")

            def _s(*x):
                return jnp.sum(jnp.stack(x, axis=0), axis=0)

            A = [
                [
                    jnp.diag(
                        vmap_decorator(
                            lambda log_b_i, F_k_mask: jnp.sum(
                                jnp.exp(-F_k_mask * beta + log_N_i_tau_mask + F_i_mask * beta - log_b_i * beta)
                            ),
                            in_axes=(1, 0),
                        )(log_U_ik_mask, F_k_mask)
                    ),
                    -jnp.exp(
                        vmap_decorator(
                            vmap_decorator(_s, in_axes=(0, None, None, 0)),  # k
                            in_axes=(None, 0, 0, 0),  # i
                        )(-F_k_mask * beta, log_N_i_tau_mask, F_i_mask * beta, -log_U_ik_mask * beta)
                    ).T,
                    -jnp.exp(
                        vmap_decorator(
                            vmap_decorator(_s, in_axes=(0, None, None, 0)),  # k
                            in_axes=(None, 0, 0, 0),  # i
                        )(-F_k_mask * beta, log_N_i_tau_mask, F_i_mask * beta, -log_U_ik_mask * beta)
                    ).T,
                    -jnp.exp(-F_k_mask * beta + log_N_tot_tau_mask).reshape((nk, 1)),
                ],
                [
                    -jnp.exp(
                        vmap_decorator(
                            vmap_decorator(
                                _s,
                                in_axes=(None, 0, 0, 0),
                            ),  # i
                            in_axes=(0, None, None, 1),  # k
                        )(-F_k_mask * beta, log_N_i_tau_mask, F_i_mask * beta, -log_U_ik_mask * beta)
                    ).T,
                    jnp.diag(jnp.exp(log_N_i_tau_mask)),
                    jnp.diag(jnp.exp(log_N_i_tau_mask)),
                    jnp.zeros((ni, 1)),
                ],
                [
                    -jnp.exp(
                        vmap_decorator(
                            vmap_decorator(
                                _s,
                                in_axes=(None, 0, 0, 0),
                            ),  # i
                            in_axes=(0, None, None, 1),  # k
                        )(-F_k_mask * beta, log_N_i_tau_mask, F_i_mask * beta, -log_U_ik_mask * beta)
                    ).T,
                    jnp.diag(jnp.exp(log_N_i_tau_mask)),
                    jnp.zeros((ni, ni)),
                    jnp.zeros((ni, 1)),
                ],
                [
                    -jnp.exp(-F_k_mask * beta + log_N_tot_tau_mask).reshape((1, nk)),
                    jnp.zeros((1, ni)),
                    jnp.zeros((1, ni)),
                    jnp.zeros((1, 1)),
                ],
            ]

            F = jnp.block(A)

            # if sparse_inverse:
            #     from scipy.sparse import csc_matrix

            #     F_sparse = csc_matrix((F.__array__()).__array__())

            #     from scipy.sparse.linalg import splu

            #     B = splu(
            #         F_sparse,
            #         permc_spec="NATURAL",
            #     )

            #     print(f"{B.shape=} {B.nnz=}")

            #     Bx = B.solve(
            #         jnp.eye(F.shape[0]).__array__(),
            #     )  # only interested in the first nk + ni columns

            #     cov_sigma = jnp.diag(jnp.array(Bx))

            # else:
            eigval, V = jnp.linalg.eigh(F)

            # print(f"{eigval=}")

            l_inv: Array = jnp.where(jnp.abs(eigval) == 0, jnp.inf, 1 / eigval)  # type:ignore

            cov_sigma = V @ jnp.diag(l_inv) @ V.T

            print(f"{cov_sigma.shape=}")

            sigma_Fk_mask = jnp.diag(cov_sigma)[:nk]
            sigma_Fi_mask = jnp.diag(cov_sigma)[nk : nk + ni]

            sigma_Fk_mask: Array = jnp.where(sigma_Fk_mask < 0, 0, jnp.sqrt(sigma_Fk_mask)) / beta  # type:ignore
            sigma_Fi_mask = jnp.where(sigma_Fi_mask < 0, 0, jnp.sqrt(sigma_Fi_mask)) / beta

            sigma_Fk = jnp.full((len_k,), jnp.inf)
            sigma_Fk = sigma_Fk.at[mask_k].set(sigma_Fk_mask)

            sigma_Fi = jnp.full((len_i,), jnp.inf)
            sigma_Fi = sigma_Fi.at[mask_i].set(sigma_Fi_mask)

            # use sigma_Fi to esimate error on single weight

            # fk_fun = lambda x: get_F_k(beta * x, mask_ik, (beta * log_U_ik, H_ik))[0] / beta

            # jac_F_k = jax.jacrev(fk_fun, argnums=(0))(
            #     F_i,
            # )

            # print(f"{jac_F_k.shape=} {ni=} {nk=}")

            # sigma_k = jnp.einsum(
            #     "ki,ij,lj->kl", jac_F_k, cov_sigma[nk : nk + ni, :][:, nk : nk + ni] / beta**2, jac_F_k
            # )
            # sigma_k = jnp.sqrt(jnp.diag(sigma_k))
            # print(f"{sigma_k / sigma_Fk}")

            print(f"{sigma_Fk/kjmol=}  {sigma_Fi/kjmol=}  ")

        s = 0.0

        s_ps_k = jnp.zeros((len_k + 1,))

        w_out = []
        ps_out = []
        dens_out = []
        F_k_out = []
        n_bin_out = []
        n_eff_out = []
        grid_nums_out = []
        sigma_out = []

        # E_k = jnp.zeros((len_k,))

        dw_mins = []
        dw_maxs = []

        w_tot_sq = jnp.zeros((len_k + 1,))
        w_tot_k = jnp.zeros((len_k + 1,))

        H_ik_req = jnp.zeros((len_i, len_k + 1))

        k_count = jnp.array((len_k,))

        for i in range(len_i):
            k_arr = grid_nums_mask[i].cv.reshape(-1)

            H_ik_req = H_ik_req.at[i, k_arr].add(1)

            mask = jnp.logical_or(jnp.logical_not(mask_ik[i, k_arr]), k_arr == -1)

            # print(f"{i=} {k_arr=} ")
            # print(f"{jnp.sum(mask)=}")

            grid_nums_out.append(k_arr)

            # get difference between binned and actual bias

            # dw = jnp.where(mask, 0.0, jnp.exp(u_unstacked[i] - log_U_ik[i, k_arr] * beta))
            # dw_inv = jnp.where(dw > 0, 1 / dw, 0)
            # print(f"{1/dw=}")

            # dw_mins.append(jnp.min(dw))
            # dw_maxs.append(jnp.max(dw))

            w = jnp.where(mask, 0.0, w_ik[i, k_arr])  # * dw
            ps = jnp.where(mask, 0.0, ps_ik[i, k_arr])
            dens = jnp.where(mask, 0.0, dens_ik[i, k_arr])  # * dw_inv
            n_bin = jnp.where(mask, 0.0, H_ik[i, k_arr])
            n_eff = jnp.where(mask, 0.0, H_ik[i, k_arr] / tau_i[i])

            # https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.3c01423?ref=article_openPDF

            w_out.append(w)
            ps_out.append(ps)
            dens_out.append(dens)
            n_bin_out.append(n_bin)
            n_eff_out.append(n_eff)
            F_k_out.append(jnp.where(mask, jnp.inf, F_k[k_arr]))

            # if compute_std:
            #     sigma_out.append(
            #         jnp.where(mask, jnp.inf, sigma_Fk[k_arr] * jnp.einsum("i,ij->j", 1 / tau_i, H_ik[:, k_arr]) ** 0.5)
            #     )

            s += jnp.sum(jnp.where(mask, 0.0, (w * dens) * ps / n_bin))

            w_tot_k = w_tot_k.at[k_arr].add(w * dens / n_bin * ps)
            w_tot_sq = w_tot_sq.at[k_arr].add((w * dens / n_bin * ps) ** 2)

            s_ps_k = s_ps_k.at[k_arr].add(ps / n_bin)

        # print(f"{ jnp.linalg.norm(H_ik-H_ik_req[:,:-1])=}")

        # print(f"{jnp.sum(H_ik_req,axis=0)=}")
        # print(f"{jnp.sum(H_ik,axis=1)=}")

        # print(
        #     f"{jnp.mean(jnp.array(dw_mins))=}  {jnp.min(jnp.array(dw_mins)  )= }  {jnp.mean(jnp.array(dw_maxs))=} {jnp.max(jnp.array(dw_maxs)  )= }"
        # )

        print(f"{s_ps_k=} ")
        print(f"{s=}")

        # if compute_std:

        # new value of sigma
        sigma_ind = jnp.full((len_k + 1,), jnp.inf)
        sigma_ind = sigma_ind.at[:-1].set(sigma_Fk)
        sigma_ind = sigma_ind / jnp.exp(jnp.log(w_tot_sq) / 2 - jnp.log(w_tot_k))

        if compute_std:
            sigma_out = []
            simga_sq_k_test = jnp.zeros((len_k + 1,))

            for i in range(len_i):
                k_arr = grid_nums_mask[i].cv.reshape(-1)

                # w = w_out[i] * dens_out[i]

                sigma = sigma_ind[k_arr]

                sigma_out.append(sigma)
                simga_sq_k_test = simga_sq_k_test.at[k_arr].add(
                    (w_out[i] * dens_out[i] * sigma * ps_out[i] / n_bin_out[i]) ** 2
                )

            sigma_k_test = jnp.sqrt(simga_sq_k_test / w_tot_k**2)

            print(f"{( jnp.linalg.norm(sigma_k_test[:-1] -sigma_Fk)/kjmol)=} {sigma_k_test/kjmol=}  ")

        mask_k = jnp.any(mask_ik, axis=0)

        # print(f"{E_k/kjmol=}")

        output_weight_kwargs = {
            "weight": w_out,
            "p_select": ps_out,
            "n_bin": n_bin_out,
            "density": dens_out,
            "F": F_k_out,
            "grid_nums": grid_nums_out,
            "labels": label_i,
            "frac_full": frac_full,
            "p_std": sigma_out if compute_std else None,
            "bounds": bounds,
            "n_bins": len(bins[0]),
            "n_eff_bin": n_eff_out,
            "N_samples_eff": jnp.sum(N_i_tau),
        }

        # if return_bias or output_free_energy:
        #     mask_k = jnp.any(mask_ik, axis=0)

        #     fes_grid = F_k[mask_k]

        #     cv_grid = cv_mid[hist_mask][mask_k]

        #     if (nn := jnp.sum(jnp.isnan(fes_grid))) != 0:
        #         print(f" found {nn=} nans in  ")

        #         raise

        #     if (nn := jnp.sum(jnp.isinf(fes_grid))) != 0:
        #         print(f" found {nn=} infs in  ")

        #         raise

        #     if smooth_bias:
        #         if (nn := jnp.sum(jnp.isnan(sigma_Fk))) != 0:
        #             print(f" found {nn=} nans in sigma_ak")

        #             raise

        #         if (nn := jnp.sum(jnp.isinf(sigma_Fk))) != 0:
        #             print(f" found {nn=} infs in sigma_ak")

        #             raise

        #     fes_grid -= jnp.min(fes_grid)

        #     # print(f"{fes_grid/kjmol=}")

        #     # mask = fes_grid > bias_cutoff

        #     # if jnp.sum(mask) > 0:
        #     #     print(f"found {jnp.sum(mask)} bins with bias above cutoff")
        #     #     cv_grid = cv_grid[mask]
        #     #     fes_grid = fes_grid[mask]

        #     print("computing rbf, including smoothing")

        #     range_frac = jnp.array([b[1] - b[0] for b in bins]) / (
        #         self.collective_variable.metric.bounding_box[:, 1] - self.collective_variable.metric.bounding_box[:, 0]
        #     )
        #     epsilon = 1 / (0.815 * range_frac)

        #     bias = RbfBias.create(
        #         cvs=self.collective_variable,
        #         cv=cv_grid,
        #         kernel="thin_plate_spline",
        #         vals=-fes_grid,
        #         epsilon=epsilon,
        #         # smoothing=(sigma_Fk / sigma_ref) ** 2 * 1e-4 if smooth_bias else None,
        #     )

        #     print(f"{compute_std=}")

        #     if compute_std:
        #         print("log_exp_slice")

        #         range_frac = jnp.array([b[1] - b[0] for b in bins]) / (
        #             self.collective_variable.metric.bounding_box[:, 1]
        #             - self.collective_variable.metric.bounding_box[:, 0]
        #         )
        #         epsilon = 1 / (0.815 * range_frac)

        #         sigma_Fk = cast(jax.Array, sigma_Fk)

        #         sigma_bias = RbfBias.create(
        #             cvs=self.collective_variable,
        #             cv=cv_mid[hist_mask],
        #             kernel="thin_plate_spline",
        #             vals=sigma_Fk,
        #             slice_exponent=2,
        #             log_exp_slice=False,
        #             slice_mean=True,
        #             epsilon=epsilon,
        #         )

        #         from IMLCV.base.bias import StdBias

        #         bias = StdBias.create(bias, sigma_bias)

        #     output_weight_kwargs["FES_bias"] = bias

        return WeightOutput(**output_weight_kwargs)

    def bincount_weight(
        self,
        samples_per_bin: int = 10,
        n_max: float | int = 1e5,
        n_max_lin: float | int = 150,
        n_hist: int | None = None,
        chunk_size: int | None = None,
        cv_0: list[CV] | None = None,
        cv_t: list[CV] | None = None,
        macro_chunk: int | None = 1000,
        verbose: bool = False,
        margin: float = 0.1,
        min_samples: int = 3,
        recalc_bounds=True,
        compute_labels=False,
    ) -> WeightOutput:
        if cv_0 is None:
            cv_0 = self.cv

        if cv_t is None:
            cv_t = self.cv_t

        # TODO:https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00867

        beta = 1 / (self.sti.T * boltzmann)

        # len = [c.shape[0] for c in cv_0]
        len_i = len(cv_0)

        # get raw rescaling
        # u_unstacked = []
        # e_unstacked = []
        # beta = 1 / (self.sti.T * boltzmann)

        # # print(f"capped")

        # high_b = []

        # for ti_i in self.ti:
        #     e = ti_i.e_bias

        #     if e is None:
        #         if verbose:
        #             print("WARNING: no bias enerrgy found")
        #         e = jnp.zeros((ti_i.sp.shape[0],))

        #     e -= jnp.min(e)

        #     # if (p := jnp.sum(e > bias_cutoff)) > 0:
        #     #     high_b.append(p)
        #     #     e = e.at[e > bias_cutoff].set(bias_cutoff)

        #     u = beta * e

        #     u_unstacked.append(u)

        #     e_unstacked.append(ti_i.e_pot)

        # print(f"capped")

        frac_full, labels, x_labels, num_labels, grid_nums_mask, get_histo, bins, cv_mid, hist_mask, n_hist, _, corr = (
            self.get_bincount(
                cv_0,
                n_max=n_max,
                n_hist=n_hist,
                n_max_lin=n_max_lin,
                margin=margin,
                chunk_size=chunk_size,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples,
                macro_chunk=macro_chunk,
                verbose=verbose,
                recalc_bounds=recalc_bounds,
                compute_labels=compute_labels,
            )
        )

        w_out = []
        ps_out = []
        dens_out = []
        F_k_out = []
        n_bin_out = []

        H_ik = hist_mask

        print(f"{H_ik.shape=}")

        F_k = jnp.zeros_like(n_hist / beta)

        # E_k = jnp.zeros((len_k,))

        # mask_k = jnp.any(mask_ik, axis=0)

        label_i = []

        for i in range(len_i):
            gi = grid_nums_mask[i].cv.reshape(-1)

            if compute_labels:
                labels_i = jnp.sum(x_labels[:, grid_nums_mask[i].cv.reshape(-1)], axis=1)
                label_i.append(jnp.argmax(labels_i))
            else:
                label_i.append(0)

            n_b = n_hist[gi]

            w = jnp.ones_like(n_b)
            ps = jnp.ones_like(n_b)
            dens = n_b
            n_bin = n_b

            w_out.append(w)
            ps_out.append(ps)
            dens_out.append(dens)
            n_bin_out.append(n_bin)
            F_k_out.append(F_k[gi])

            # E_k = E_k.at[gi].add(e_unstacked[i][gi] * ps / n_bin)

        output_weight_kwargs = {
            "weight": w_out,
            "p_select": ps_out,
            "n_bin": n_bin_out,
            "density": dens_out,
            "F": F_k_out,
            # "grid_nums": grid_nums_mask,
            "labels": label_i,
            "frac_full": frac_full,
            "N_samples_eff": jnp.sum(jnp.array([a.shape[0] / b for a, b in zip(cv_0, corr)])),
        }

        return WeightOutput(**output_weight_kwargs)

    @staticmethod
    def _correlation_time(
        cv_0: list[CV],
        verbose: bool = True,
        method: str = None,
        const_acf: float = 5.0,
        periodicities: jax.Array | None = None,
    ):
        # https://juser.fz-juelich.de/record/152532/files/FZJ-2014-02136.pdf
        # block-averaging estimate of integrated correlation times for diagnostics

        print(f"starting block-averaging estimation of integrated correlation times")

        tau_list: list[float] = []
        TE_list: list[float] = []

        if periodicities is not None:
            print("applying periodicity to cv for correlation time estimation")
            cv_0, _ = DataLoaderOutput.apply_cv(
                CvTrans.from_cv_function(PeriodicKoopmanModel._exp_periodic, periodicities=periodicities),
                cv_0,
            )

        @partial(jax.jit, static_argnames=["norm"])
        def _acf(cv: CV, norm=True):
            x = cv.cv

            # https://emcee.readthedocs.io/en/stable/tutorials/autocorr/

            @partial(vmap_decorator, in_axes=1)
            def f(x):
                x = x - jnp.mean(x, axis=0)

                corr = jax.scipy.signal.correlate(x, x, mode="full")
                corr = corr[-x.shape[0] :]
                corr /= jnp.arange(x.shape[0], 0, -1)

                if norm:
                    corr = jnp.where(corr[0] == 0, corr, corr / corr[0])

                # integrated autocorrelation time (use lags 1..M)

                def hilbert(x, N=None, axis=-1):
                    N = x.shape[axis]

                    Xf = jax.numpy.fft.fft(x, N, axis=axis)
                    h = jnp.zeros(N, dtype=Xf.dtype)
                    if N % 2 == 0:
                        h = h.at[0].set(1)
                        h = h.at[N // 2].set(1)
                        h = h.at[1 : N // 2].set(2)

                    else:
                        h = h.at[0].set(1)
                        h = h.at[1 : (N + 1) // 2].set(2)

                    if x.ndim > 1:
                        ind = [jnp.newaxis] * x.ndim
                        ind[axis] = slice(None)
                        h = h[tuple(ind)]
                    x = jax.numpy.fft.ifft(Xf * h, axis=axis)
                    return x

                # Compute the analytic signal using the Hilbert transform
                envelope = jnp.abs(hilbert(corr))

                print(f"envelope shape: {envelope.shape}")

                envelope = envelope[: envelope.shape[0] // 2]

                # Define the loss function
                def loss_fn(tau, y):
                    x = jnp.arange(len(y))

                    y_pred = jnp.exp(-x / tau)

                    res = jnp.sum((y - y_pred) ** 2)

                    # print(f"{res=}")

                    return res

                # Initial guess for parameters

                # Optimize the parameters
                from jaxopt import GradientDescent

                solver = GradientDescent(fun=loss_fn)

                x0 = jnp.array([1.0])

                result = solver.run(x0, envelope)

                print(f"Fitted parameters: {result.params}")

                tau = result.params

                # return tau
                return (1 + jnp.exp(-1 / tau)) / (1 - jnp.exp(-1 / tau))

            time_scales = f(x)

            return jnp.max(time_scales)

        out = []

        @jax.jit
        def _blav(cv: CV):
            # data = cv_trans.compute_cv(sp)[0].cv.reshape((sp.shape[0], -1))
            data = cv.cv.reshape((cv.shape[0], -1))

            errors = []
            block_sizes = []

            for bs in range(1, data.shape[0] // 10 + 1):
                n_blocks = data.shape[0] // bs
                block_means = jnp.mean(data[: n_blocks * bs, :].reshape((n_blocks, bs, -1)), axis=1)
                block_var_naive = jnp.var(block_means, axis=0, ddof=1) / n_blocks

                errors.append(jnp.sqrt(block_var_naive))
                block_sizes.append(bs)

            errors = jnp.array(errors)

            # try:
            from jaxopt import GaussNewton, ScipyMinimize

            def fit_func(x, B, err):
                t_int, TE = x

                TE = jnp.exp(TE)
                t_int = jnp.exp(t_int) + 1

                return TE * jnp.sqrt(B / (B + t_int - 1)) - err

            @partial(jax.vmap, in_axes=(None, 1), out_axes=0)
            def solve(block_sizes, errors):
                solver = GaussNewton(residual_fun=fit_func, tol=1e-6)
                out = solver.run((0.0, errors[0]), block_sizes, errors)

                log_t_int_est, log_TE_est = out.params

                t_int_est = jnp.exp(log_t_int_est) + 1
                TE_est = jnp.exp(log_TE_est)

                return t_int_est, TE_est

            t_int, TE = solve(jnp.array(block_sizes), errors)

            t_int = jnp.where(jnp.isfinite(t_int), t_int, jnp.inf)

            t_int = jnp.where(t_int > data.shape[0] / 2, data.shape[0] / 2, t_int)

            t_int = jnp.where(t_int < 1.0, 1.0, t_int)

            return t_int[i]

        from IMLCV.base.UnitsConstants import angstrom

        for i, cv_i in enumerate(cv_0):
            if verbose:
                print(".", end="", flush=True)
                if (i + 1) % 100 == 0:
                    print("")

            cv_data = cv_i

            if method == None:
                t_int_i = _blav(cv_data)
            elif method == "acf":
                t_int_i = _acf(cv_data)
            else:
                raise ValueError(f"mehod {method=} unknown, choose from 'acf','blav'")

            tau_list.append(t_int_i)

        tau_i = jnp.array(tau_list)
        # TE_i = jnp.array(TE_list)

        print(f"Estimated integrated correlation times per trajectory: {tau_i} {tau_i.shape} ")

        return tau_i

    def get_bincount(
        self,
        cv_0: list[CV] | None = None,
        n_max=1e5,
        n_max_lin: int | None = 150,
        margin=0.0,
        chunk_size=None,
        samples_per_bin=20,
        min_samples_per_bin=1,
        macro_chunk: int | None = 1000,
        verbose=False,
        compute_labels=True,
        recalc_bounds=True,
        n_hist: int | None = None,
        bounds: jax.Array | None = None,
        correlation_method: str | None = None,
    ):
        if cv_0 is None:
            cv_0 = self.cv

        if n_hist is None:
            n_hist = self.n_hist

        if bounds is None:
            bounds = self.bounds

        return DataLoaderOutput._get_bincount(
            cv_0=cv_0,
            n_max=n_max,
            margin=margin,
            chunk_size=chunk_size,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            macro_chunk=macro_chunk,
            verbose=verbose,
            metric=self.collective_variable.metric,
            compute_labels=compute_labels,
            recalc_bounds=recalc_bounds,
            n_hist=n_hist,
            bounds=bounds,
            correlation_method=correlation_method,
            n_max_lin=n_max_lin,
        )

    @staticmethod
    def _get_bincount(
        cv_0: list[CV],
        metric: CvMetric,
        n_max=1e5,
        n_hist: int | None = None,
        n_max_lin: int | None = None,
        bounds: jax.Array | None = None,
        margin=0.0,
        chunk_size=None,
        samples_per_bin=20,
        min_samples_per_bin=1,
        macro_chunk: int | None = 1000,
        verbose=False,
        compute_labels=True,
        frac_full: float | None = 1.0,
        recalc_bounds=True,
        correlation_method: str | None = None,
    ):
        # helper method

        ndim = cv_0[0].shape[1]

        print(f"{bounds=}{metric=} {recalc_bounds=}")

        if bounds is not None:
            print(f"using provided {bounds=}")
            grid_bounds = bounds

        else:
            print("getting bounds")

            if recalc_bounds:
                bounds, _, constants = CvMetric.bounds_from_cv(
                    cv_0,
                    margin=0,
                    # chunk_size=chunk_size,
                    macro_chunk=macro_chunk,
                    n=30,
                    verbose=verbose,
                )

            else:
                bounds = metric.bounding_box

            bounds_margin = (bounds[:, 1] - bounds[:, 0]) * margin
            # bounds_margin = jnp.where(   self.periodicities, )
            bounds = bounds.at[:, 0].set(bounds[:, 0] - bounds_margin)
            bounds = bounds.at[:, 1].set(bounds[:, 1] + bounds_margin)

            grid_bounds = jax.vmap(lambda x, y: jnp.where(metric.extensible, y, x), in_axes=(1, 1), out_axes=(1))(
                metric.bounding_box,
                bounds,
            )

        print(f"{grid_bounds=}")

        if correlation_method is not None:
            print("correlating data to get effective sample size")
            tau_i = DataLoaderOutput._correlation_time(
                cv_0,
                verbose=verbose,
                method=correlation_method,
                periodicities=metric.periodicities,
            )

            sd = [int(jnp.ceil(a.shape[0] / t)) for a, t in zip(cv_0, tau_i)]
            tot_samples = sum(sd)

        else:
            sd = [a.shape[0] for a in cv_0]
            tot_samples = sum(sd)

        # prepare histo

        def get_hist(n_hist):
            if verbose:
                print(f"using {n_hist=}")

            if n_hist <= 3:
                print(f"WARNING: {n_hist=}, adjusting to 3")
                n_hist = 3

            print("getting histo")
            cv_mid, nums, bins, closest, get_histo = DataLoaderOutput._histogram(
                metric=metric,
                n_grid=n_hist,
                grid_bounds=grid_bounds,
                chunk_size=chunk_size,
            )

            grid_nums, _ = DataLoaderOutput.apply_cv(
                closest,
                cv_0,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                verbose=verbose,
            )

            if verbose:
                print("getting histo")

            log_hist = get_histo(
                grid_nums,
                [jnp.full(c.shape[0], -jnp.log(t)) for t, c in zip(tau_i, cv_0)] if correlation_method else None,
                log_w=True,
                macro_chunk=macro_chunk,
                verbose=verbose,
            )

            print(f"{min_samples_per_bin=}")

            # print(f"{jnp.exp(hist)=}")
            hist_mask = log_hist >= jnp.log(min_samples_per_bin)  # at least 1 sample

            print(f"{jnp.sum(hist_mask)=}")

            return cv_mid, nums, bins, closest, get_histo, grid_nums, jnp.exp(log_hist[hist_mask]), hist_mask

        if n_hist is None:
            if frac_full is None:
                print(f"getting frac full")

                if n_max > 1e3:
                    n_hist = CvMetric.get_n(
                        samples_per_bin=samples_per_bin,
                        samples=tot_samples,
                        n_dims=ndim,
                        max_bins=jnp.min(jnp.array([1e3, n_max])),  # 10 000 test grid points
                    )

                    # pre test to get empty fraction
                    cv_mid, nums, bins, closest, get_histo, grid_nums, _n_hist_mask, hist_mask = get_hist(n_hist)

                    frac_full = jnp.sum(hist_mask) / hist_mask.shape[0]

                    print(f"{frac_full=}")
                else:
                    frac_full = 1.0

            n_hist = CvMetric.get_n(
                samples_per_bin=samples_per_bin,
                samples=tot_samples,
                n_dims=ndim,
                max_bins=n_max / frac_full,  # compensate for empty spaces
            )

            print(f"{n_hist=}")
        else:
            print(f"using provided {n_hist=}")

        if n_max_lin is not None:
            if n_hist > n_max_lin:
                print(f"capping n_hist from {n_hist} to {n_max_lin}")
                n_hist = n_max_lin

        cv_mid, nums, bins, closest, get_histo, grid_nums, _n_hist_mask, hist_mask = get_hist(n_hist)

        frac_full = jnp.sum(hist_mask) / hist_mask.shape[0]

        idx_inv = jnp.full(hist_mask.shape[0], -1)
        idx_inv = idx_inv.at[hist_mask].set(jnp.arange(jnp.sum(hist_mask)))

        # print(f"{jnp.argwhere(hist_mask)=}")

        grid_nums_mask = [g.replace(cv=idx_inv[g.cv]) for g in grid_nums]

        # print(f"{grid_nums=}")
        # print(f"{grid_nums_mask=}")

        n_hist_mask = jnp.sum(hist_mask)
        # print(f"{_n_hist_mask=}")

        if compute_labels:
            x = jnp.full((0, n_hist_mask), False)

            for fni, gn in enumerate(grid_nums_mask):
                b = jnp.full(n_hist_mask + 1, False)
                b = b.at[gn.cv.reshape(-1)].set(True)
                b = b[:-1]

                in_rows, new_rows = vmap_decorator(
                    lambda u, v: (jnp.logical_and(u, v).any(), jnp.logical_or(u, v)), in_axes=(None, 0)
                )(b, x)

                if in_rows.any():
                    n = jnp.sum(in_rows)

                    rr = jnp.argwhere(in_rows).reshape(-1)

                    if n == 1:  # add visited parts to row
                        x = x.at[rr[0], :].set(new_rows[rr[0], :])

                    else:  # merge connected parts
                        rows = jnp.vstack([x[rr, :], b])
                        new_row = jnp.any(rows, axis=0)

                        x = x.at[rr[0], :].set(new_row)

                        x = jnp.delete(x, rr[1:], 0)

                else:  # create row
                    x = jnp.vstack([x, b])

            # print(f"{x=}")

            labels = vmap_decorator(lambda x: jnp.argwhere(x, size=1).reshape(()), in_axes=1)(x)

            num_labels = jnp.unique(labels).shape[0]

            if num_labels > 1:
                print(f"found {num_labels} different regions {labels=}")

            # print(f"{labels.shape=}")

            x_labels = jnp.hstack([x, jnp.full((x.shape[0], 1), False)])
        else:
            labels, x_labels, num_labels = None, None, None

        return (
            frac_full,
            labels,
            x_labels,
            num_labels,
            grid_nums_mask,
            get_histo,
            bins,
            cv_mid,
            hist_mask,
            _n_hist_mask,
            grid_bounds,
            tau_i if correlation_method else jnp.ones((len(cv_0),)),
        )

    @staticmethod
    def _transform(
        cv,
        nl,
        shmap,
        shmap_kwargs,
        argmask: jax.Array | None = None,
        pi: jax.Array | None = None,
        add_1: bool = False,
        add_1_pre: bool = False,
        q: jax.Array | None = None,
        l: jax.Array | None = None,
    ) -> jax.Array:
        x = cv.cv

        # print(f"inside {x.shape=} {q=} {argmask=} ")

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
        cv_0: list[CV] | list[SystemParams] | None = None,
        cv_t: list[CV] | list[SystemParams] | None = None,
        nl: list[NeighbourList] | NeighbourList | None = None,
        nl_t: list[NeighbourList] | NeighbourList | None = None,
        # method="tcca",
        only_return_weights=False,
        symmetric=False,
        rho: list[jax.Array] | None = None,
        w: list[jax.Array] | None = None,
        rho_t: list[jax.Array] | None = None,
        w_t: list[jax.Array] | None = None,
        dynamic_weights: list[jax.Array] | None = None,
        eps: float = 0.0,
        eps_pre: float = 0.0,
        eps_shrink: float = 1e-3,
        max_features=5000,
        max_features_pre=5000,
        out_dim=-1,
        add_1=True,
        chunk_size=None,
        macro_chunk=1000,
        verbose=False,
        trans=None,
        T_scale: float = 1.0,
        only_diag=False,
        calc_pi=False,
        scaled_tau=None,
        sparse=True,
        correlation=True,
        auto_cov_threshold: float | None = None,
        shrink=True,
        shrinkage_method="bidiag",
        generator=False,
        use_w=True,
        periodicities: jax.Array | None = None,
        iters_nonlin: int = 10000,
        epochs: int = 2000,
        batch_size: int = 10000,
        init_learnable_params=True,
        entropy_reg: float = 0.0,
    ) -> "KoopmanModel":
        # TODO: https://www.mdpi.com/2079-3197/6/1/22

        # assert method in ["tica", "tcca"]

        if scaled_tau is None:
            scaled_tau = self.scaled_tau

        if cv_0 is None:
            cv_0 = self.cv

        if nl is None:
            nl = self.nl

        if rho is None:
            if self._rho is not None:
                rho = self._rho
            else:
                print("W koopman model not given, will use uniform weights")

                t = sum([cvi.shape[0] for cvi in cv_0])
                rho = [jnp.ones((cvi.shape[0],)) / t for cvi in cv_0]

        if T_scale != 1.0:
            print(f"rescaling rho for T_scale {T_scale=}")

            rho = self.rescale_rho_T(T_scale, rho=rho)

        if w is None:
            if self._weights is not None:
                w = self._weights
            else:
                print("W koopman model not given, will use uniform weights")

                t = sum([cvi.shape[0] for cvi in cv_0])
                w = [jnp.ones((cvi.shape[0],)) / t for cvi in cv_0]

        if not generator:
            if cv_t is None:
                cv_t = self.cv_t

            if nl_t is None:
                nl_t = self.nl_t

            assert cv_t is not None

            if rho_t is None:
                if self._rho_t is not None:
                    rho_t = self._rho_t
                else:
                    print("W_t koopman model not given, will use w")

                    rho_t = rho

            if T_scale != 1.0:
                rho_t = self.rescale_rho_T(T_scale, rho=rho_t)

            if w_t is None:
                if self._weights_t is not None:
                    w_t = self._weights_t
                else:
                    print("W_t koopman model not given, will use w")

                    w_t = w

        if dynamic_weights is None:
            # print("dynamic weights not given, will use uniform weights")

            if self.dynamic_weights is not None:
                dynamic_weights = self.dynamic_weights
            else:
                dynamic_weights = [jnp.ones((cvi.shape[0],)) for cvi in cv_0]

            # print(f"{dynamic_weights=} ")

        return KoopmanModel.create(
            w=w,
            w_t=w_t,
            dynamic_weights=dynamic_weights,
            rho=rho,
            rho_t=rho_t,
            cv_0=cv_0,
            cv_t=cv_t,
            nl=nl,
            nl_t=nl_t,
            add_1=add_1,
            eps=eps,
            eps_pre=eps_pre,
            # method=method,
            symmetric=symmetric,
            out_dim=out_dim,
            eps_shrink=eps_shrink,
            # koopman_weight=koopman_weight,
            max_features=max_features,
            max_features_pre=max_features_pre,
            tau=self.tau,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            verbose=verbose,
            trans=trans,
            # T_scale=T_scale,
            only_diag=only_diag,
            calc_pi=calc_pi,
            scaled_tau=scaled_tau,
            sparse=sparse,
            only_return_weights=only_return_weights,
            correlation_whiten=correlation,
            auto_cov_threshold=auto_cov_threshold,
            shrink=shrink,
            shrinkage_method=shrinkage_method,
            generator=generator,
            use_w=use_w,
            periodicities=periodicities,
            iters_nonlin=iters_nonlin,
            epochs=epochs,
            batch_size=batch_size,
            init_learnable_params=init_learnable_params,
            entropy_reg=entropy_reg,
        )

    def filter_nans(
        self,
        x: list[CV] | None = None,
        x_t: list[CV] | None = None,
        macro_chunk=1000,
    ):
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
        f: CvTrans,
        x: list[CV] | list[SystemParams],
        x_t: list[CV] | list[SystemParams] | None = None,
        nl: list[NeighbourList] | NeighbourList | None = None,
        nl_t: list[NeighbourList] | NeighbourList | None = None,
        chunk_size: int | None = None,
        macro_chunk: int | None = 1000,
        shmap: bool = True,
        shmap_kwargs=ShmapKwargs.create(),
        verbose: bool = False,
        print_every: int = 10,
        jit_f: bool = True,
    ) -> tuple[list[CV], list[CV] | None]:
        _f = f.compute_cv

        if x_t is not None:
            assert isinstance(x_t[0], x[0].__class__), "x_t must be of the same type as x"

        def __f(x: X, nl: NeighbourList | None) -> CV:
            return _f(
                x,
                nl,
                chunk_size=chunk_size,
                shmap=False,
            )[0]

        # if shmap:
        #     __f = padded_shard_map(__f, kwargs=shmap_kwargs)
        #     jit_f = False

        out = DataLoaderOutput._apply(
            x=x,
            x_t=x_t,
            nl=nl,
            nl_t=nl_t,
            f=__f,
            macro_chunk=macro_chunk,
            verbose=verbose,
            jit_f=jit_f,
            print_every=print_every,
        )

        # if shmap:
        #     print(f"putting out ")
        #     jax.debug.inspect_array_sharding(out, callback=print)
        #     out = jax.device_put(out, jax.devices()[0])

        #     jax.debug.inspect_array_sharding(out, callback=print)

        return out

    @staticmethod
    def _apply(
        x: list[X],
        f: Callable[[X, NeighbourList | None], X2],
        x_t: list[X] | None = None,
        nl: list[NeighbourList] | NeighbourList | None = None,
        nl_t: list[NeighbourList] | NeighbourList | None = None,
        macro_chunk: int | None = 1000,
        verbose: bool = False,
        jit_f: bool = True,
        print_every: int = 10,
    ) -> tuple[list[X2], list[X2] | None]:
        if verbose:
            print("inside _apply")

        out = macro_chunk_map(
            f=f,
            # op=x[0].__class__.stack,
            y=x,
            y_t=x_t,
            nl=nl,
            nl_t=nl_t,
            macro_chunk=macro_chunk,
            verbose=verbose,
            jit_f=jit_f,
            print_every=print_every,
        )

        if verbose:
            print("outside _apply")
        # jax.debug.inspect_array_sharding(out[0], callback=print)

        if x_t is not None:
            z, z_t = out

        else:
            z = out
            z_t = None

        z = cast(list[X2], z)
        if z_t is not None:
            z_t = cast(list[X2], z_t)

        for i in range(len(z)):
            assert z[i].shape[0] == x[i].shape[0], (
                f" shapes do not match {[zi.shape[0] for zi in z]} != {[xi.shape[0] for xi in x]}"
            )

            if z_t is not None:
                assert z[i].shape[0] == z_t[i].shape[0]

        return z, z_t

    @staticmethod
    def _apply_bias(
        x: list[CV],
        bias: Bias,
        chunk_size=None,
        macro_chunk=1000,
        verbose=False,
        shmap=True,
        shmap_kwargs=ShmapKwargs.create(),
        jit_f=True,
    ) -> list[jax.Array]:
        def f(x: CV, _):
            b = bias.compute_from_cv(
                x,
                chunk_size=chunk_size,
                shmap=False,
            )[0]

            return x.replace(cv=b.reshape((-1, 1)))

        # if shmap:
        #     f = padded_shard_map(f, kwargs=shmap_kwargs)  # (pmap=True))

        out, _ = DataLoaderOutput._apply(
            x=x,
            f=f,
            macro_chunk=macro_chunk,
            verbose=verbose,
            jit_f=jit_f,
        )

        # if shmap:
        #     out = jax.device_put(out, jax.devices()[0])

        return [a.cv[:, 0] for a in out]

    def apply_bias(self, bias: Bias, chunk_size=None, macro_chunk=1000, verbose=False, shmap=False) -> list[Array]:
        return DataLoaderOutput._apply_bias(
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
        sigma=0.01,
        key=jax.random.PRNGKey(0),
    ) -> tuple[CV, SystemParams, NeighbourList | NeighbourListInfo | None, TrajectoryInfo]:
        import jax.numpy as jnp

        from IMLCV.implementations.bias import HarmonicBias

        # print(f"{self.collective_variable=}")

        mu = self.collective_variable.metric.bounding_box[:, 1] - self.collective_variable.metric.bounding_box[:, 0]
        k = self.sti.T * boltzmann / (mu**2 * 2 * sigma**2)

        center_bias = HarmonicBias.create(cvs=self.collective_variable, k=k, q0=point)

        center = self.apply_bias(
            bias=center_bias,
        )

        p = jnp.exp(-jnp.hstack(center) / (self.sti.T * boltzmann))
        p /= jnp.sum(p)

        n = []

        for i, x in enumerate(self.cv):
            n.append(jnp.stack([jnp.full(x.shape[0], i), jnp.arange(x.shape[0])], axis=1))

        n = jnp.vstack(n)

        idx = jax.random.choice(key, a=n, p=p, shape=(), replace=True)

        return (
            self.cv[idx[0]][idx[1]],
            self.sp[idx[0]][idx[1]],
            self.nl[idx[0]][idx[1]] if self.nl is not None else None,
            self.ti[idx[0]],
        )

    def get_fes_bias_from_weights(
        self,
        cv: list[CV] | None = None,
        weights: list[Array] | None = None,
        rho: list[Array] | None = None,
        weights_std: list[Array] | None = None,
        samples_per_bin=100,
        min_samples_per_bin: int | None = 5,
        n_max=1e5,
        n_max_lin=1000,
        max_bias=None,
        chunk_size=None,
        macro_chunk=1000,
        max_bias_margin=0.2,
        rbf_bias=True,
        kernel="thin_plate_spline",
        collective_variable: CollectiveVariable | None = None,
        set_outer_border=True,
        rbf_degree: int | None = None,
        smoothing: float = -1,
        frac_full=1.0,
        recalc_bounds=True,
        output_density_bias=False,
        observable: list[CV] | None = None,
        overlay_mask=False,
        std_bias=True,
        margin=0.3,
        n_hist: int | None = None,
        bounds: jax.Array | None = None,
    ):
        if cv is None:
            cv = self.cv

        if weights is None:
            weights = self._weights

        assert weights is not None

        if rho is None:
            rho = self._rho

        assert rho is not None

        if weights_std is None:
            weights_std = self._weights_std

        if collective_variable is None:
            collective_variable = self.collective_variable

        if frac_full is None:
            frac_full = self.frac_full

        if n_hist is None:
            if self.n_hist is not None:
                print("using stored n_hist")
            n_hist = self.n_hist

        if bounds is None:
            if self.bounds is not None:
                print("using stored bounds")

            bounds = self.bounds

        return DataLoaderOutput._get_fes_bias_from_weights(
            T=self.sti.T,
            weights=weights,
            rho=rho,
            weights_std=weights_std,
            collective_variable=collective_variable,
            cv=cv,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            n_max_lin=n_max_lin,
            n_max=n_max,
            max_bias=max_bias,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            max_bias_margin=max_bias_margin,
            rbf_bias=rbf_bias,
            kernel=kernel,
            set_outer_border=set_outer_border,
            rbf_degree=rbf_degree,
            smoothing=smoothing,
            frac_full=frac_full,
            recalc_bounds=recalc_bounds,
            output_density_bias=output_density_bias,
            observable=observable,
            overlay_mask=overlay_mask,
            std_bias=std_bias,
            margin=margin,
            bounds=bounds,
            n_hist=n_hist,
        )

    @staticmethod
    def _get_fes_bias_from_weights(
        T,
        weights: list[jax.Array],
        rho: list[jax.Array],
        collective_variable: CollectiveVariable,
        cv: list[CV],
        samples_per_bin: int = 1000,
        min_samples_per_bin: int = 5,
        n_max: int | float = 1e5,
        n_max_lin: int | float = 100,
        max_bias: float | None = None,
        chunk_size: int | None = None,
        macro_chunk: int | None = 1000,
        max_bias_margin: float = 0.2,
        rbf_bias: bool = True,
        kernel: str = "thin_plate_spline",
        set_outer_border: bool = True,
        rbf_degree: int | None = None,
        smoothing: float | None = -1,
        frac_full=1.0,
        compute_frac_full=False,
        grid_bias_order: int = 0,
        verbose=False,
        recalc_bounds=True,
        margin=0.3,
        output_density_bias=False,
        observable: list[CV] | None = None,
        overlay_mask=False,
        std_bias=True,
        weights_std: list[jax.Array] | None = None,
        n_hist: int | None = None,
        bounds: jax.Array | None = None,
    ) -> tuple[Bias, StdBias | None, Bias | None, dict]:
        beta = 1 / (T * boltzmann)

        _, _, _, _, grid_nums_mask, get_histo, bins, cv_mid, hist_mask, n_hist_mask, bounds, _ = (
            DataLoaderOutput._get_bincount(
                cv,
                metric=collective_variable.metric,
                n_hist=n_hist,
                n_max_lin=n_max_lin,
                bounds=bounds,
                n_max=n_max,
                margin=margin,
                chunk_size=chunk_size,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
                macro_chunk=macro_chunk,
                verbose=verbose,
                compute_labels=False,
                frac_full=frac_full,
                recalc_bounds=recalc_bounds,
            )
        )

        def safe_log_sum(*x):
            p_out = None

            for xi in x:
                p = jnp.where(xi > 0, jnp.log(xi), jnp.full_like(xi, -jnp.inf))

                if p_out is None:
                    p_out = p
                else:
                    p_out += p

            return p_out

        w_log = [safe_log_sum(wi, rhoi) for wi, rhoi in zip(weights, rho)]

        log_w_rho_grid_mask = get_histo(
            grid_nums_mask,
            w_log,
            macro_chunk=macro_chunk,
            log_w=True,
            shape_mask=hist_mask,
        )

        # mask = hist_mask

        # mask = jnp.logical_and(
        #     hist_mask,
        #     jnp.isfinite(log_w_rho_grid_mask),
        # )

        fes_grid = jnp.full(hist_mask.shape, jnp.inf)
        fes_grid = fes_grid.at[hist_mask].set(-log_w_rho_grid_mask / beta)
        fes_grid -= jnp.nanmin(fes_grid)

        mask_tot = jnp.isfinite(fes_grid)

        # if set_outer_border and rbf_bias:

        # add host point at every corner

        #     print(f"getting borders {collective_variable.metric.periodicities=} ")

        #     border_points = vmap_decorator(
        #         lambda x: jnp.sum(
        #             vmap_decorator(
        #                 lambda p, s, e, per: jnp.logical_and(
        #                     jnp.logical_or(p == s, p == e),
        #                     jnp.logical_not(per),
        #                 )
        #             )(
        #                 x,
        #                 cv_mid.cv[0, :],
        #                 cv_mid.cv[-1, :],
        #                 jnp.logical_not(collective_variable.metric.extensible),
        #             )
        #         )
        #         >= 1
        #     )(cv_mid.cv)

        #     print(f"{jnp.sum(border_points)=} {max_bias=}")

        #     fes_grid = fes_grid.at[border_points].set(jnp.max(fes_grid[mask]))

        #     # select data
        #     mask_tot = jnp.logical_or(mask, border_points)
        # else:
        # mask_tot = mask

        if std_bias and weights_std is None:
            print("no weights std given, cannot compute std bias")
            std_bias = False

        if std_bias:
            log_w_sigma_sq = get_histo(
                grid_nums_mask,
                [safe_log_sum(wi, wi, wstdi, wstdi, rhoi, rhoi) for wi, rhoi, wstdi in zip(weights, rho, weights_std)],
                macro_chunk=macro_chunk,
                log_w=True,
                shape_mask=hist_mask,
            )

            # print(f"{log_w_sigma_sq=}")

            log_w = get_histo(
                grid_nums_mask,
                [safe_log_sum(wi, rhoi) for wi, rhoi in zip(weights, rho)],
                macro_chunk=macro_chunk,
                log_w=True,
                shape_mask=hist_mask,
            )

            log_w_sigma_grid = jnp.full(hist_mask.shape, -jnp.inf)
            log_w_sigma_grid = log_w_sigma_grid.at[hist_mask].set(log_w_sigma_sq) / 2
            log_w_grid = log_w_grid = jnp.full(hist_mask.shape, -jnp.inf)
            log_w_grid = log_w_grid.at[hist_mask].set(log_w)

            # print(f"{log_w_sigma_grid=}")
            # print(f"{log_w_grid=}")

            std_grid = jnp.exp(log_w_sigma_grid - log_w_grid)

            # print(f"{std_grid/kjmol=}")

        bounds_gb = jnp.array([[(a[0] + a[1]) / 2, (a[-1] + a[-2]) / 2] for a in bins])
        shape = tuple([len(a) - 1 for a in bins])

        print(f"{fes_grid.shape=} {shape=} ")

        bounds_adjusted = GridBias.adjust_bounds(bounds=bounds_gb, n=shape[0])

        if rbf_bias:
            fes_grid_selection = fes_grid[mask_tot]
            cv_selection = cv_mid[mask_tot]

            range_frac = jnp.array([b[1] - b[0] for b in bins]) / (
                collective_variable.metric.bounding_box[:, 1] - collective_variable.metric.bounding_box[:, 0]
            )
            epsilon = 1 / (0.815 * range_frac * jnp.sqrt(collective_variable.metric.ndim))

            bias = RbfBias.create(
                cvs=collective_variable,
                cv=cv_selection,
                kernel=kernel,
                vals=-fes_grid_selection,
                epsilon=epsilon,
                degree=rbf_degree,
                smoothing=smoothing,
                sigma=std_grid[mask_tot] if std_bias else None,
            )

            if overlay_mask:
                grid_mask_bias = GridMaskBias(
                    collective_variable=collective_variable,
                    n=shape[0],
                    bounds=bounds_adjusted,
                    vals=fes_grid.reshape(shape) * 0.0 + 1.0,
                    order=grid_bias_order,
                )

                bias = CompositeBias.create(biases=[bias, grid_mask_bias], fun=jnp.prod)

        else:
            bias = GridBias(
                collective_variable=collective_variable,
                n=shape[0],
                bounds=bounds_adjusted,
                vals=-fes_grid.reshape(shape),
                order=grid_bias_order,
            )

        if std_bias:
            std_bias = StdBias.create(
                bias=GridBias(
                    collective_variable=collective_variable,
                    n=shape[0],
                    bounds=bounds_adjusted,
                    vals=log_w_grid.reshape(shape) / beta,
                    order=grid_bias_order,
                ),
                log_exp_sigma=GridBias(
                    collective_variable=collective_variable,
                    n=shape[0],
                    bounds=bounds_adjusted,
                    vals=log_w_sigma_grid.reshape(shape),
                    order=grid_bias_order,
                ),
            )
        else:
            std_bias = None

        if output_density_bias:
            shape = tuple([len(a) - 1 for a in bins])
            dv = jnp.prod(jnp.array([(a[1] - a[0]) for a in bins]))
            # v = jnp.prod(jnp.array(n_hist_mask.shape)) * dv

            dens = n_hist_mask / jnp.sum(n_hist_mask * dv)

            dens_grid = jnp.full(hist_mask.shape, jnp.inf)
            dens_grid = dens_grid.at[hist_mask].set(-jnp.log10(dens))

            dens_bias = GridBias(
                collective_variable=collective_variable,
                n=shape[0] + 1,
                bounds=jnp.array([[(a[0] + a[1]) / 2, (a[-1] + a[-2]) / 2] for a in bins]),
                vals=-dens_grid.reshape(shape),
                order=grid_bias_order,
            )
        else:
            dens_bias = None

        if observable is not None:
            print(f"reweighting observable")

            shape = tuple([len(a) - 1 for a in bins])

            # print(f"{observable=} {grid_nums_mask=}")
            x = get_histo(
                grid_nums_mask,
                [jnp.exp(a) for a in w_log],
                observable=observable,
                log_w=False,
                macro_chunk=macro_chunk,
                verbose=verbose,
                shape_mask=hist_mask,
            )

            _, obs = x

            obs = jax.vmap(lambda x: x / jnp.exp(log_w_rho_grid_mask), in_axes=1, out_axes=1)(obs)

            print(f"{hist_mask.shape=} {obs.shape=}")

            obs_grid = jnp.full((hist_mask.shape[0], *obs.shape[1:]), jnp.nan)
            obs_grid = obs_grid.at[hist_mask, :].set(obs)

            obs = dict(
                n=shape[0] + 1,
                bounds=jnp.array([[(a[0] + a[1]) / 2, (a[-1] + a[-2]) / 2] for a in bins]),
                vals=obs_grid.reshape((*shape, -1)),
                order=grid_bias_order,
            )

        else:
            obs = None

        return bias, std_bias, dens_bias, obs

    def get_transformed_fes(
        self,
        new_cv: list[CV],
        new_colvar: CollectiveVariable,
        samples_per_bin=5,
        min_samples_per_bin: int = 1,
        chunk_size=1,
        smoothing=1,
        max_bias=None,
        shmap=False,
        n_grid_old=50,
        n_grid_new=30,
    ) -> RbfBias:
        old_cv = self.cv
        old_cv_stack = CV.stack(*self.cv)
        new_cv_stack = CV.stack(*new_cv)

        # get bins for new CV
        grid_bounds_new, _, _ = CvMetric.bounds_from_cv(new_cv, margin=0.1)
        cv_mid_new, nums_new, _, closest_new, get_histo_new = DataLoaderOutput._histogram(
            n_grid=n_grid_new,
            grid_bounds=grid_bounds_new,
            metric=new_colvar.metric,
        )
        grid_nums_new = closest_new.compute_cv(new_cv_stack, chunk_size=chunk_size)[0].cv

        # get bins for old CV
        grid_bounds_old, _, _ = CvMetric.bounds_from_cv(old_cv, margin=0.1)
        cv_mid_old, nums_old, _, closest_old, get_histo_old = DataLoaderOutput._histogram(
            n_grid=n_grid_old,
            grid_bounds=grid_bounds_old,
            metric=self.collective_variable.metric,
        )
        grid_nums_old = closest_old.compute_cv(old_cv_stack, chunk_size=chunk_size)[0].cv

        assert self.ground_bias is not None

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

        @partial(vmap_decorator, in_axes=(None, 0))
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

        prob = Partial_decorator(prob, grid_nums_new=grid_nums_new, nums_old=nums_old, p_grid_old=p_grid_old)
        prob = padded_vmap(prob, chunk_size=chunk_size)

        if shmap:
            prob = padded_shard_map(prob)

        num_grid_new, p_grid_new = prob(nums_new)  # type:ignore

        # mask = num_grid_new >= min_samples_per_bin

        # print(f"{jnp.sum(mask)}/{mask.shape[0]} bins with samples")

        # p_grid_new = p_grid_new  # [mask]

        fes_grid = -jnp.log(p_grid_new) / beta
        fes_grid -= jnp.min(fes_grid)

        raise NotImplementedError("determine eps")

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
        raise NotImplementedError("determine eps")

        if T is None:
            T = self.sti.T

        _, _, cv_grid, _, _ = self.collective_variable.metric.grid(n=n_grid)
        new_cv_grid, _, log_det = trans.compute_cv(cv_grid, log_Jf=True)

        FES_bias_vals, _ = self.ground_bias.compute_from_cv(cv_grid)

        new_FES_bias_vals = FES_bias_vals + T * boltzmann * log_det
        new_FES_bias_vals -= jnp.max(new_FES_bias_vals)

        # weight = jnp.exp(new_FES_bias_vals / (T * boltzmann))
        # weight /= jnp.sum(weight)

        bounds, _, _ = self.collective_variable.metric.bounds_from_cv(
            new_cv_grid,
            weights=self._weight,
            rho=self._rho,
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

    def recalc(
        self, chunk_size: int | None = None, macro_chunk: int | None = 1000, shmap: bool = False, verbose: bool = False
    ):
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
            assert self.sp_t is not None
            y = [*self.sp, *self.sp_t]
        else:
            y = [*self.sp]

        from IMLCV.base.UnitsConstants import angstrom

        print(f"nl info ")

        nl_info = NeighbourListInfo.create(
            r_cut=r_cut,
            r_skin=0 * angstrom,
            z_array=self.sti.atomic_numbers,
        )

        # @partial(jit_decorator, static_argnames=["update"])
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

            def f0(nl_update: NeighbourListUpdate | None, sp: SystemParams, *_):
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
                        num_neighs=int(nn),
                        stack_dims=None,
                    )

                b, new_nn, new_xyz, _ = _f(sp, nl_update)

                if not jnp.all(b):
                    n_xyz, nn = nl_update.nxyz, nl_update.num_neighs

                    assert n_xyz is not None

                    nl_update = NeighbourListUpdate.create(
                        nxyz=tuple([max(a, int(b)) for a, b in zip(n_xyz, new_xyz)]),
                        num_neighs=max(nn, int(new_nn)),
                        stack_dims=None,
                    )

                    print(f"updating neighbour list {nl_update=}")

                return nl_update

            def f_inner(x: SystemParams, nl: NeighbourList | None):
                return x

            # print(f"test")

            nl_update = macro_chunk_map_fun(
                f=f_inner,
                y=y,
                nl=None,
                macro_chunk=macro_chunk,
                verbose=verbose,
                jit_f=False,
                chunk_func=f0,
                w=None,
            )

            assert nl_update is not None

            self.nl = NeighbourList(
                info=nl_info,
                update=nl_update,
                sp_orig=None,
            )

            self.nl_t = self.nl if self.time_series else None

            return

        def f(sp: SystemParams, _: NeighbourList | None) -> NeighbourList:
            b, _, _, xnl = sp._get_neighbour_list(
                info=nl_info,
                chunk_size=chunk_size,
                chunk_size_inner=chunk_size_inner,
                shmap=False,
                only_update=only_update,
            )  # type:ignore

            assert jnp.all(b)
            assert xnl is not None

            return xnl

        nl = macro_chunk_map(
            f=f,
            # op=SystemParams.stack,
            y=y,
            nl=None,
            macro_chunk=macro_chunk,
            verbose=verbose,
            jit_f=True,
        )  # type:ignore

        if self.time_series:
            nl, nl_t = nl[0 : len(self.sp)], nl[len(self.sp) :]
        else:
            nl_t = None

        nl = cast(list[NeighbourList], nl)
        if nl_t is not None:
            nl_t = cast(list[NeighbourList], nl_t)

        self.nl = nl
        self.nl_t = nl_t


class KoopmanModel(MyPyTreeNode):
    s: jax.Array

    cov: Covariances
    W0: jax.Array
    W1: jax.Array

    argmask: jax.Array | None
    argmask_s: jax.Array | None

    shape: int

    cv_0: list[CV] | list[SystemParams]
    cv_t: list[CV] | list[SystemParams]
    nl: list[NeighbourList] | NeighbourList | None
    nl_t: list[NeighbourList] | NeighbourList | None

    w: list[jax.Array] | None = None
    rho: list[jax.Array] | None = None

    w_t: list[jax.Array] | None = None
    rho_t: list[jax.Array] | None = None

    periodicities: jax.Array | bool | None = None

    use_w: bool = True

    dynamic_weights: list[jax.Array] | None = None

    shrink: bool = False
    shrinkage_method: str = "bidiag"

    eps: float = 1e-10
    eps_pre: float | None = None

    only_diag: bool = False
    calc_pi: bool = True

    scaled_tau: bool = False

    add_1: bool = True
    max_features: int = 5000
    max_features_pre: int = 5000
    out_dim: int | None = None
    # method: str = "tcca"
    correlation_whiten: bool = True
    verbose: bool = True

    tau: float | None = None
    # T_scale: float = 1.0
    eps_shrink: float = 1e-3

    generator: bool = False

    trans: CvTrans | CvTrans | None = None

    constant_threshold: float = 1e-14
    entropy_reg: float = 0.0

    @staticmethod
    def create(
        w: list[jax.Array] | None,
        rho: list[jax.Array] | None,
        w_t: list[jax.Array] | None,
        rho_t: list[jax.Array] | None,
        cv_0: list[CV] | list[SystemParams],
        cv_t: list[CV] | list[SystemParams] | None,
        dynamic_weights: list[jax.Array] | None = None,
        nl: list[NeighbourList] | NeighbourList | None = None,
        nl_t: list[NeighbourList] | NeighbourList | None = None,
        add_1=True,
        eps: float = 1e-14,
        eps_pre: float = 0,
        # method="tcca",
        symmetric=False,
        out_dim=-1,  # maximum dimension for koopman model
        max_features=5000,
        max_features_pre=5000,
        tau=None,
        macro_chunk=1000,
        chunk_size=None,
        verbose=True,
        trans: CvTrans | None = None,
        # T_scale: float = 1.0,
        only_diag=False,
        calc_pi=False,
        use_scipy=False,
        auto_cov_threshold=None,
        sparse=True,
        # n_modes=10,
        scaled_tau=False,
        only_return_weights=False,
        correlation_whiten=False,
        out_eps=None,
        constant_threshold: float = 1e-14,
        shrink=True,
        shrinkage_method="bidiag",
        eps_shrink=1e-3,
        glasso=True,
        generator: bool = False,
        use_w=True,
        periodicities: jax.Array | None = None,
        exp_period=True,
        vamp_r=2,
        iters_nonlin=int(1e4),
        print_nonlin_every=10,
        epochs=2000,
        batch_size=10000,
        init_learnable_params=True,
        min_std=1e-2,  # prevents collapse into singularities
        entropy_reg=0.0,
    ):
        #  see Optimal Data-Driven Estimation of Generalized Markov State Models
        if verbose:
            print(f"getting covariancesm {generator=}")

        print(f"{out_dim=} {eps=} {eps_pre=}")
        print(f"{calc_pi=} {add_1=}   ")

        print(f"new tot w")

        def tot_w(w, rho):
            # wi = exp(-beta U_bias)
            # reweighing:  exp(-beta' U_bias)

            w_log = [jnp.log(wi) + jnp.log(rhoi) for wi, rhoi in zip(w, rho)]
            z = jnp.hstack(w_log)

            z_max = jnp.max(z)
            norm = jnp.log(jnp.sum(jnp.exp(z - z_max))) + z_max

            w_tot = [jnp.exp(w_log_i - norm) for w_log_i in w_log]

            s = 0
            for wi in w_tot:
                s += jnp.sum(wi)

            print(f"{s=}")

            return w_tot

        w_tot = tot_w(w, rho)

        if not generator:
            assert cv_t is not None
            assert w_t is not None
            assert rho_t is not None

            w_tot_t = tot_w(w_t, rho_t)
        else:
            w_tot_t = None

        # if periodicities is not None:
        #     print(f"using periodicities in cv transform {periodicities=}")

        #     idx_periodic = jnp.argwhere(periodicities).flatten()
        #     idx_nonperiodic = jnp.argwhere(jnp.logical_not(periodicities)).flatten()

        #     trans_periodic = CvTrans.from_cv_function(
        #         _cv_slice,
        #         indices=idx_periodic,
        #     )

        #     trans_nonperiodic = CvTrans.from_cv_function(
        #         _cv_slice,
        #         indices=idx_nonperiodic,
        #     )

        #     if exp_period:
        #         if trans is not None:
        #             cv_0, cv_t = DataLoaderOutput.apply_cv(
        #                 trans,
        #                 cv_0,
        #                 cv_t,
        #                 nl,
        #                 nl_t,
        #                 chunk_size=chunk_size,
        #                 shmap=False,
        #                 macro_chunk=macro_chunk,
        #                 verbose=verbose,
        #             )

        #         cv_0_per, cv_t_per = DataLoaderOutput.apply_cv(
        #             trans_periodic,
        #             cv_0,
        #             cv_t,
        #             nl,
        #             nl_t,
        #             chunk_size=chunk_size,
        #             shmap=False,
        #             macro_chunk=macro_chunk,
        #             verbose=verbose,
        #         )

        #         cv_0_nonper, cv_t_nonper = DataLoaderOutput.apply_cv(
        #             trans_nonperiodic,
        #             cv_0,
        #             cv_t,
        #             nl,
        #             nl_t,
        #             chunk_size=chunk_size,
        #             shmap=False,
        #             macro_chunk=macro_chunk,
        #             verbose=verbose,
        #         )

        #         print(f"###########3 building nonper model ")

        #         km_nonperiodic = KoopmanModel.create(
        #             w=w,
        #             rho=rho,
        #             w_t=w_t,
        #             rho_t=rho_t,
        #             cv_0=cv_0_nonper,
        #             cv_t=cv_t_nonper,
        #             dynamic_weights=dynamic_weights,
        #             nl=nl,
        #             nl_t=nl_t,
        #             add_1=add_1,
        #             eps=eps,
        #             eps_pre=eps_pre,
        #             # method=method,
        #             symmetric=symmetric,
        #             out_dim=out_dim,
        #             max_features=max_features,
        #             max_features_pre=max_features_pre,
        #             tau=tau,
        #             macro_chunk=macro_chunk,
        #             chunk_size=chunk_size,
        #             verbose=verbose,
        #             trans=None,
        #             # T_scale=T_scale,
        #             only_diag=only_diag,
        #             calc_pi=calc_pi,
        #             use_scipy=use_scipy,
        #             auto_cov_threshold=auto_cov_threshold,
        #             sparse=sparse,
        #             scaled_tau=scaled_tau,
        #             correlation_whiten=correlation_whiten,
        #             constant_threshold=constant_threshold,
        #             shrink=shrink,
        #             shrinkage_method=shrinkage_method,
        #             eps_shrink=eps_shrink,
        #             glasso=glasso,
        #             generator=generator,
        #             use_w=use_w,
        #             periodicities=None,
        #         )

        #         print(f"###########3 building per model ")

        #         def _periodic(cv: CV, _nl, shmap, shmap_kwargs, periodicities=periodicities):
        #             return cv.replace(cv=jnp.exp(1.0j * cv.cv))

        #         _trans = CvTrans.from_cv_function(_periodic, periodicities=periodicities)

        #         # cov = Covariances.create(
        #         #     cv_0=cv_0_per,  # type: ignore
        #         #     nl=nl,
        #         #     w=w_tot if use_w else None,
        #         #     calc_pi=True,
        #         #     only_diag=only_diag,
        #         #     symmetric=symmetric,
        #         #     chunk_size=chunk_size,
        #         #     macro_chunk=macro_chunk,
        #         #     trans_f=CvTrans.from_cv_function(PeriodicKoopmanModel._exp_periodic),
        #         #     verbose=verbose,
        #         #     shrink=False,
        #         #     shrinkage_method=shrinkage_method,
        #         #     pi_argmask=jnp.array([-1]) if add_1 else None,
        #         #     eps_shrink=eps_shrink,
        #         #     generator=generator,
        #         # )

        #         # # print(f"{jnp.diag(cov.rho_00)=}")

        #         # print(f"{jnp.angle(cov.pi_0)=}")

        #         # _trans_periodic = CvTrans.from_cv_function(
        #         #     PeriodicKoopmanModel._center_periodic,
        #         #     periodicities=periodicities,
        #         #     angles=jnp.angle(cov.pi_0),
        #         # )

        #         km_periodic = KoopmanModel.create(
        #             w=w,
        #             rho=rho,
        #             w_t=w_t,
        #             rho_t=rho_t,
        #             cv_0=cv_0_per,
        #             cv_t=cv_t_per,
        #             dynamic_weights=dynamic_weights,
        #             nl=nl,
        #             nl_t=nl_t,
        #             add_1=add_1,
        #             eps=eps,
        #             eps_pre=eps_pre,
        #             # method=method,
        #             symmetric=symmetric,
        #             out_dim=out_dim,
        #             max_features=max_features,
        #             max_features_pre=max_features_pre,
        #             tau=tau,
        #             macro_chunk=macro_chunk,
        #             chunk_size=chunk_size,
        #             verbose=verbose,
        #             trans=CvTrans.from_cv_function(PeriodicKoopmanModel._exp_periodic),
        #             only_diag=True,
        #             calc_pi=False,
        #             use_scipy=use_scipy,
        #             auto_cov_threshold=auto_cov_threshold,
        #             sparse=sparse,
        #             scaled_tau=scaled_tau,
        #             correlation_whiten=correlation_whiten,
        #             constant_threshold=constant_threshold,
        #             shrink=shrink,
        #             shrinkage_method=shrinkage_method,
        #             eps_shrink=eps_shrink,
        #             glasso=glasso,
        #             generator=generator,
        #             use_w=use_w,
        #             periodicities=None,
        #         )

        #         print(f"##### done")

        #         return PeriodicKoopmanModel(
        #             km_periodic=km_periodic,
        #             km_nonperiodic=km_nonperiodic,
        #             periodicities=periodicities,
        #             trans=trans,
        #             trans_periodic=trans_periodic,
        #             trans_nonperiodic=trans_nonperiodic,
        #         )

        if periodicities is not None:
            _tr_per = CvTrans.from_cv_function(PeriodicKoopmanModel._exp_periodic, periodicities=periodicities)

            if trans is not None:
                trans *= _tr_per
            else:
                trans = _tr_per

        variational = False
        if trans is not None:
            if trans.num_learnable_params > 0:
                variational = True

        # if variational and add_1:
        #     add_1 = False

        if add_1:
            print("adding 1 to  basis set")

            from IMLCV.implementations.CV import append_trans

            _add_1 = append_trans(v=jnp.array([1]))

            if trans is None:
                trans = _add_1
            else:
                trans *= _add_1

        # print(f" {calc_pi=}")

        # print(f"{dynamic_weights=}")

        if variational:
            # batch_size = 1e4
            # epochs = 50

            assert trans is not None
            print(f"using variational approach {trans.num_learnable_params=} ")

            trans.learnable_params_shape

            if init_learnable_params:
                print("initializing learnable params")
                learable_params = trans.init_learnable_params(jax.random.PRNGKey(42))
            # trans = trans.apply_learnable_params(learable_params)
            else:
                print("using default learnable params")
                learable_params = trans.get_learnable_params()

            import optax

            eta = 0.1

            assert not generator

            tau = tau if tau is not None else 1.0

            @jax.jit
            def objective_fn(params, x_0, x_t, w):
                if generator:
                    learnable_params = params
                else:
                    learnable_params = params

                _trans = trans.apply_learnable_params(learnable_params)

                print(f"{x_0=} {x_t=} ")

                y_0, _ = _trans.compute_cv(x_0, nl)
                y_t, _ = _trans.compute_cv(x_t, nl_t) if x_t is not None else (None, None)

                cov = Covariances.create(
                    cv_0=[y_0],
                    cv_1=[y_t] if y_t is not None else None,
                    nl=nl,
                    nl_t=nl_t,
                    w=[w],
                    w_t=[w] if x_t is not None else None,
                    trans_f=None,
                    trans_g=None,
                    shrink=False,
                    calc_pi=True,
                    get_diff=False,
                    pi_argmask=jnp.array([-1]) if add_1 else None,
                    symmetric=symmetric,
                    generator=False,
                    chunk_size=None,
                    macro_chunk=None,
                )

                W0 = cov.whiten_rho("rho_00", apply_mask=False, epsilon=1e-12, cholesky=False)

                if symmetric:
                    W1 = W0
                else:
                    W1 = cov.whiten_rho("rho_11", apply_mask=False, epsilon=1e-12, cholesky=False)

                # trim W to out_dim
                if out_dim is not None and out_dim > 0:
                    W0 = W0[: out_dim + 1 :]
                    W1 = W1[: out_dim + 1 :]

                # if generator:
                #     K = W0 @ cov.rho_gen @ W1.T
                # else:
                K = W0 @ cov.rho_01 @ W1.T

                loss = -jnp.sum(K**2)  # vamp-2 score

                if entropy_reg > 0.0:
                    print("computing entropy regularization")

                    # z_0 = y_0.cv @ cov.sigma_0_inv @ W0.T

                    print(f"{y_0.cv.shape=} {cov.pi_0.shape=} {cov.sigma_0_inv.shape=} {W0.T.shape=}")

                    z_0 = jnp.einsum("ni,i,ik->nk", y_0.cv - cov.pi_0, cov.sigma_0_inv, W0.T)

                    jax.debug.print("z mean {} std {}", jnp.mean(z_0, axis=0), jnp.std(z_0, axis=0))

                    if add_1:
                        z_0 = z_0[:, :-1]  # remove constant basis

                    # silverman rule
                    c_inv = (4 / (z_0.shape[0] * (z_0.shape[1] + 2))) ** (-2 / (z_0.shape[1] + 4))

                    # print(f"{H_inv.shape=}")

                    def soft_binning_entropy_stable(z_0, w, grid_size=20, range_limit=4.0):
                        n_frames, d = z_0.shape

                        # 1. Setup Grid
                        grid_axis = jnp.linspace(-range_limit, range_limit, grid_size)
                        grid_coords = jnp.meshgrid(*(grid_axis for _ in range(d)))
                        grid_points = jnp.stack([g.ravel() for g in grid_coords], axis=-1)  # (G^d, d)

                        # 2. Kernel parameters
                        # Whitening is done, so h^2 = 1 / c_inv
                        h_sq = 1.0 / c_inv

                        # 3. Compute Log-Memberships
                        # Instead of exp(-0.5 * dist^2 / h_sq), we stay in log space
                        @jax.vmap
                        def compute_log_unnormalized_density(grid_p):
                            print(f"{z_0.shape=} {grid_p.shape=} ")

                            # Distance from one grid point to all CV points
                            sq_dist = jnp.sum((z_0 - grid_p) ** 2, axis=-1)
                            log_kernels = -0.5 * sq_dist / h_sq

                            # We need to account for weights: log(sum(w * exp(log_kernels)))
                            # This is exactly what jax.nn.logsumexp supports with the 'b' (weights) argument
                            return jax.nn.logsumexp(log_kernels, b=w)

                        # log_p_bins: (grid_size**d,)
                        log_p_bins = compute_log_unnormalized_density(grid_points)

                        # 4. Normalize the Log-Distribution
                        # We want p_i = exp(log_p_i) / sum(exp(log_p_j))
                        # So log(p_i) = log_p_i - logsumexp(log_p_all)
                        log_p_normalized = log_p_bins - jax.nn.logsumexp(log_p_bins)

                        # 5. Shannon Entropy: H = -sum(p * log(p))
                        # Since we have log(p), this is -sum(exp(log_p) * log_p)
                        p_normalized = jnp.exp(log_p_normalized)
                        entropy = -jnp.sum(p_normalized * log_p_normalized)

                        return -entropy  # Return negative entropy to minimize

                    # Integration into your loss
                    loss_entropy = soft_binning_entropy_stable(z_0, w)

                    jax.debug.print("Entropy loss: {le}", le=loss_entropy)
                    jax.debug.print("Current total loss before entropy: {l}", l=loss)

                    loss += entropy_reg * loss_entropy

                    # # @jax.vmap
                    # def _entropy(z):
                    #     p, wp = z

                    #     @jax.vmap
                    #     def _inner(q, wq):
                    #         s = q - p

                    #         print(f"{s=}")
                    #         print(f"{H_inv=} {s.shape=}{wq=}{wp=} ")

                    #         kernel = jnp.exp(-0.5 * s @ H_inv @ s)
                    #         return wq * kernel

                    #     return -wp * jnp.log(jnp.sum(_inner(y_0.cv, w) + 1e-10))

                    # entropy = jnp.sum(jax.lax.map(_entropy, (y_0.cv, w)))

                    # loss -= entropy_reg * entropy

                # # dK = W0 @ cov.rho_gen @ W1.T / (300 * kelvin * boltzmann)
                # # dl, _ = jnp.linalg.eigh(dK)

                # if symmetric:
                #     l, u = jnp.linalg.eigh(K)

                # else:
                #     l, u = jnp.linalg.eig(K)

                #     # if not generator:
                # l = -tau / jnp.log(l) / nanosecond

                # remove the constant eigenvalue
                # l = l[:-1]
                # # l = l[1:]
                # # dl = dl[1:]
                # # dl = dl[::-1]

                # # ts = -tau / jnp.log(l) / nanosecond
                # # else:
                # #     l = l
                # # jax.debug.print("timescales: {l}", l=ts)
                # # jax.debug.print("dK matrix: {dl}", dl=dl)
                # # jax.debug.print("dK/ts  {}", dl / ts)
                # # jax.debug.print("improved eigenvalues: {l}", l=l)

                # print(f"{l=}")

                # # miximize timescales
                # loss = -jnp.sum(l)

                # loss = -jnp.sum(K**2)

                return loss, (None)

            optimizer = optax.adamw(learning_rate=1e-4, weight_decay=1e-5)

            cv_out_shape, _ = jax.eval_shape(trans.compute_cv, cv_0[0])
            # lambdas = jnp.zeros((cv_out_shape.cv.shape[1],))

            # # Initialize the optimizer state
            opt_state = optimizer.init(learable_params)

            @jax.jit
            def _body_fun(opt_state, params, x0, xt, w):
                (loss_val, _), grads = jax.value_and_grad(objective_fn, has_aux=True)(params, x0, xt, w)
                print(f"loss_val {loss_val} {grads=}")

                updates, opt_state = optimizer.update(grads, opt_state, params)

                print(f"{updates=}")
                params = optax.apply_updates(params, updates)

                return opt_state, params, loss_val, _

            x_0_stack = CV.stack(*cv_0) if isinstance(cv_0[0], CV) else SystemParams.stack(*cv_0)
            x_t_stack = (
                (CV.stack(*cv_t) if isinstance(cv_0[0], CV) else SystemParams.stack(*cv_t))
                if cv_t is not None
                else None
            )
            w_tot_stack = jnp.hstack(w_tot)

            for epoch in range(epochs):
                opt_state, learable_params, loss_train, _ = _body_fun(
                    opt_state, learable_params, x_0_stack, x_t_stack, w_tot_stack
                )

                # # compute validation loss
                # loss_val, _ = objective_fn(learable_params, x_0_val, x_t_val, w_val)

                print(f"{epoch=}  {loss_train=}   ")

            # create random K-folds splits and pick one validation fold

            # num_folds = 5  # change as needed
            # N = x_0_stack.shape[0]

            # if batch_size > N / 5:
            #     print(f"adjusting batch size from {batch_size} to {N // 5} ")
            #     batch_size = N // 5

            # # deterministic key for fold generation (can be exposed/seeded)
            # key = jax.random.PRNGKey(42)
            # key_1, key = jax.random.split(key, 2)
            # perm = jax.random.permutation(key_1, N)

            # # compute fold sizes and split indices
            # num_folds = N // batch_size
            # # num_folds = 1

            # base = N // num_folds
            # extras = N % num_folds
            # sizes = [base + (1 if i < extras else 0) for i in range(num_folds)]
            # boundaries = jnp.cumsum(jnp.array(sizes))[:-1]
            # folds_idx = list(jnp.split(perm, boundaries))

            # print(f" {N=} {base=} ")

            # # select a random validation fold
            # key_1, key = jax.random.split(key, 2)
            # val_fold = int(jax.random.randint(key_1, (), 0, num_folds))

            # val_idx = folds_idx[val_fold]
            # train_idx = jnp.concatenate([folds_idx[i] for i in range(num_folds) if i != val_fold], axis=0)

            # # gather train/validation stacks for cv_0 and cv_t (works for structured CV dataclasses)
            # x_0_val = x_0_stack[val_idx]  # validation set (rename if you prefer train/val swapped)
            # x_0_train = x_0_stack[train_idx]
            # w_val = w_tot_stack[val_idx]
            # w_train = w_tot_stack[train_idx]

            # x_t_train = x_t_stack[train_idx] if x_t_stack is not None else None
            # x_t_val = x_t_stack[val_idx] if x_t_stack is not None else None

            # for epoch in range(epochs):
            #     # permute training data each epoch
            #     key_1, key = jax.random.split(key, 2)
            #     perm = jax.random.permutation(key_1, x_0_train.shape[0])
            #     _x_0_train = x_0_train[perm]
            #     _w_train = w_train[perm]
            #     if x_t_train is not None:
            #         _x_t_train = x_t_train[perm]

            #     for i in range(num_folds - 1):
            #         x_i = _x_0_train[i * base : (i + 1) * base]
            #         x_t_i = _x_t_train[i * base : (i + 1) * base] if x_t_train is not None else None
            #         w_i = _w_train[i * base : (i + 1) * base]

            #         opt_state, learable_params, loss_train, _ = _body_fun(opt_state, learable_params, x_i, x_t_i, w_i)

            #     # opt_state, learable_params, loss_train, _ = _body_fun(
            #     #     opt_state, learable_params, x_0_train, x_t_train, w_train
            #     # )

            #     # compute validation loss
            #     loss_val, _ = objective_fn(learable_params, x_0_val, x_t_val, w_val)

            #     print(f"{epoch=}  {loss_train=} {loss_val=}  ")

            trans = trans.apply_learnable_params(learable_params)
            print(f"optimized trans done ")

            # cov = Covariances.create(
            #     cv_0=cv_0,  # type: ignore
            #     cv_1=cv_t,  # type: ignore
            #     nl=nl,
            #     nl_t=nl_t,
            #     w=w_tot if use_w else None,
            #     w_t=w_tot_t if use_w else None,
            #     dynamic_weights=dynamic_weights,
            #     calc_pi=calc_pi,
            #     only_diag=only_diag,
            #     symmetric=symmetric,
            #     chunk_size=chunk_size,
            #     macro_chunk=macro_chunk,
            #     trans_f=trans,
            #     trans_g=trans,
            #     verbose=verbose,
            #     shrink=False,
            #     shrinkage_method=shrinkage_method,
            #     pi_argmask=jnp.array([-1]) if add_1 else None,
            #     eps_shrink=eps_shrink,
            #     generator=generator,
            # )

            # return KoopmanModel(
            #     cov=cov,
            #     W0=U,
            #     W1=U,
            #     s=jnp.exp(-l_train),
            #     cv_0=cv_0,
            #     cv_t=cv_t,
            #     nl=nl,
            #     nl_t=nl_t,
            #     w=w,
            #     w_t=w_t,
            #     rho=rho,
            #     rho_t=rho_t,
            #     dynamic_weights=dynamic_weights,
            #     add_1=add_1,
            #     max_features=max_features,
            #     max_features_pre=max_features_pre,
            #     out_dim=out_dim,
            #     # method=method,
            #     calc_pi=calc_pi,
            #     tau=tau,
            #     trans=trans,
            #     argmask=None,
            #     only_diag=only_diag,
            #     eps=eps,
            #     eps_pre=eps_pre,
            #     shape=cov.rho_00.shape[0],
            #     scaled_tau=scaled_tau,
            #     correlation_whiten=correlation_whiten,
            #     constant_threshold=constant_threshold,
            #     verbose=verbose,
            #     shrink=shrink,
            #     shrinkage_method=shrinkage_method,
            #     eps_shrink=eps_shrink,
            #     # use_w=use_w,
            #     generator=generator,
            #     # periodicities=periodicities,
            #     argmask_s=None,
            # )

        cov = Covariances.create(
            cv_0=cv_0,  # type: ignore
            cv_1=cv_t,  # type: ignore
            nl=nl,
            nl_t=nl_t,
            w=w_tot if use_w else None,
            w_t=w_tot_t if use_w else None,
            dynamic_weights=dynamic_weights,
            calc_pi=calc_pi,
            only_diag=only_diag,
            symmetric=symmetric,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            trans_f=trans,
            trans_g=trans,
            verbose=verbose,
            shrink=False,
            shrinkage_method=shrinkage_method,
            pi_argmask=jnp.array([-1]) if add_1 else None,
            eps_shrink=eps_shrink,
            generator=generator,
        )

        # print(f"{cov.pi_0=}")

        # print(f"{cov=}")

        argmask = cov.mask(eps_pre, max_features_pre, auto_cov_threshold)

        if not generator:
            # first mask before shrinking
            if shrink:
                cov = cov.shrink()

        x = cov.decompose(
            out_dim,
            eps=eps,
            out_eps=out_eps,
            glasso_whiten=glasso,
            verbose=verbose,
            # generator=generator,
        )

        if only_diag:
            argmask_s, s = x
            W0 = None
            W1 = None

            argmask = argmask
        else:
            W0, W1, s = x

        if out_dim is not None:
            if s.shape[0] < out_dim:
                print(f"found only {s.shape[0]} singular values")

        if verbose:
            print(f"{s[0:min(10, s.shape[0]) ]=}")

        print(f"{add_1=} new weights")  # if self.periodicities is not None:
        #     print("applying periodicities to f")

        #     _real = CvTrans.from_cv_function(lambda cv, nl, shmap, shmap_kwargs: cv.replace(cv=jnp.abs(cv.cv)))

        #     tr *= _real

        km = KoopmanModel(
            cov=cov,
            W0=W0,
            W1=W1,
            s=s,
            cv_0=cv_0,
            cv_t=cv_t,
            nl=nl,
            nl_t=nl_t,
            w=w,
            w_t=w_t,
            rho=rho,
            rho_t=rho_t,
            dynamic_weights=dynamic_weights,
            add_1=add_1,
            max_features=max_features,
            max_features_pre=max_features_pre,
            out_dim=out_dim,
            # method=method,
            calc_pi=calc_pi,
            tau=tau,
            trans=trans,
            argmask=argmask,
            only_diag=only_diag,
            eps=eps,
            eps_pre=eps_pre,
            shape=cov.rho_00.shape[0],
            scaled_tau=scaled_tau,
            correlation_whiten=correlation_whiten,
            constant_threshold=constant_threshold,
            verbose=verbose,
            shrink=shrink,
            shrinkage_method=shrinkage_method,
            eps_shrink=eps_shrink,
            # use_w=use_w,
            generator=generator,
            # periodicities=periodicities,
            argmask_s=argmask_s if only_diag else None,
            entropy_reg=entropy_reg,
        )

        return km

    @staticmethod
    def _add_1(
        cv,
        nl,
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

        # if self.add_1:
        #     if tr is not None:
        #         tr *= CvTrans.from_cv_function(KoopmanModel._add_1)

        #     else:
        #         tr = CvTrans.from_cv_function(KoopmanModel._add_1)

        return tr

    def f_stiefel(
        self,
        out_dim=None,
        n_skip=None,
        remove_constant=True,
        mask: jax.Array | None = None,
        optimizer_kwargs: dict = {},
    ):
        W0, _ = self.cov.decompose_pymanopt(out_dim=out_dim, optimizer_kwargs=optimizer_kwargs)

        # sigma_1_inv = self.cov.sigma_1_inv
        # assert sigma_1_inv is not None
        o = W0.T  # jnp.diag(sigma_1_inv) @ self.W1.T
        # s = self.s

        # n_skip = self.get_n_skip(
        #     n_skip=n_skip,
        #     remove_constant=remove_constant,
        # )

        # # s = s[n_skip:]
        # o = o[:, n_skip:]

        if mask is not None:
            # s = s[mask]
            o = o[:, mask]

        o = o[:, :out_dim]

        tr = CvTrans.from_cv_function(
            DataLoaderOutput._transform,
            static_argnames=["add_1", "add_1_pre"],
            add_1=False,
            add_1_pre=False,
            q=o,
            pi=self.cov.pi_1,
            argmask=self.argmask,
        )

        if self.tot_trans is not None:
            tr = self.tot_trans * tr

        return tr

    def f(
        self,
        out_dim=None,
        remove_constant=True,
        n_skip: int | None = None,
        mask: jax.Array | None = None,
        inv_exp=True,
    ):
        if self.only_diag:
            argmask_s = self.argmask_s

        else:
            o = self.W0.T  # jnp.diag(self.cov.sigma_0_inv) @ self.W0.T

        s = self.s

        n_skip = self.get_n_skip(
            n_skip=n_skip,
            remove_constant=remove_constant,
        )

        s = s[n_skip:]

        if self.only_diag:
            argmask_s = argmask_s[n_skip:]
        else:
            o = o[:, n_skip:]

        if mask is not None:
            s = s[mask]
            if self.only_diag:
                argmask_s = argmask_s[mask]
            else:
                o = o[:, mask]

        if self.only_diag:
            argmask_s = argmask_s[:out_dim]
        else:
            o = o[:, :out_dim]

        tr = CvTrans.from_cv_function(
            DataLoaderOutput._transform,
            static_argnames=["add_1", "add_1_pre"],
            add_1=False,
            add_1_pre=False,
            q=o if not self.only_diag else None,
            pi=self.cov.pi_0[argmask_s] if (self.only_diag and self.cov.pi_0 is not None) else self.cov.pi_0,
            argmask=self.argmask if not self.only_diag else self.argmask[argmask_s],
        )

        if self.trans is not None:
            tr = self.trans * tr

        if self.cov.C00.dtype == jnp.complex_ is not None and inv_exp:
            print("applying periodicities to f")

            _real = CvTrans.from_cv_function(PeriodicKoopmanModel._inv_exp_periodic)

            tr *= _real

        return tr, None

    def g(
        self,
        out_dim=None,
        n_skip=None,
        remove_constant=True,
        mask: jax.Array | None = None,
    ):
        sigma_1_inv = self.cov.sigma_1_inv
        assert sigma_1_inv is not None
        o = self.W1.T  # jnp.diag(sigma_1_inv) @ self.W1.T
        s = self.s

        n_skip = self.get_n_skip(
            n_skip=n_skip,
            remove_constant=remove_constant,
        )

        s = s[n_skip:]
        o = o[:, n_skip:]

        if mask is not None:
            s = s[mask]
            o = o[:, mask]

        o = o[:, :out_dim]

        tr = CvTrans.from_cv_function(
            DataLoaderOutput._transform,
            static_argnames=["add_1", "add_1_pre"],
            add_1=False,
            add_1_pre=False,
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
        epsilon=1e-6,
        max_entropy=True,
        out_dim=None,
    ) -> tuple[list[Array], list[Array], list[Array] | None, list[Array] | None, bool]:
        # Optimal Data-Driven Estimation of Generalized Markov State Models, page 18-19
        # create T_k in the trans basis
        # C00^{-1} C11 T_k
        # W0 C00 W0.T = I
        # W1 C11 W1.T = I
        # W0 C01 W1.T = sigma
        # _______________
        # W1 C11 W0.T sigma x = x   <- eigensolver
        # mu_ref = W1.T@x

        # assert not self.shrink, "cannot compute equilibrium density from shrunken covariances"

        if out_dim is None:
            out_dim = int(jnp.sum(jnp.abs(1 - self.s) < epsilon))

        if out_dim == 0:
            # print("no eigenvalues found close to 1")
            # return self.w, None, False
            out_dim = 1

            i = int(jnp.min(jnp.array([5, self.s.shape[0]])))

            print(f"using closest eigenvalue to 1: {self.s[:i]=}")

        print("reweighing,  A")

        # A = self.W1 @ self.cov.rho_11 @ jnp.diag(self.cov.sigma_1 * self.cov.sigma_0_inv) @ self.W0.T @ jnp.diag(self.s)
        A = self.W1 @ self.cov.C11 @ self.W0.T @ jnp.diag(self.s)

        lv, v = jnp.linalg.eig(A)

        print(f"eigenvalues {lv=}")

        # remove complex eigenvalues, as they cannot be tshe ground state
        real = jnp.abs(jnp.imag(lv)) <= 1e-3
        out_idx = jnp.argwhere(real).reshape((-1))  # out_idx[real]
        out_idx = out_idx[jnp.argsort(jnp.abs(lv[out_idx] - 1))]
        # sort

        n = jnp.min(jnp.array([out_idx.shape[0], 10]))
        print(f"{lv[out_idx[:n]]=} ")

        lv, v = lv[out_idx], v[:, out_idx]

        lv, v = jnp.real(lv), jnp.real(v)

        lv, v = lv[(0,)], v[:, (0,)]

        sigma_1_inv = self.cov.sigma_1_inv

        assert sigma_1_inv is not None

        mu_ref = self.W1.T @ v  # jnp.diag(sigma_1_inv) @ self.W1.T @ v

        f_trans_2 = CvTrans.from_cv_function(
            DataLoaderOutput._transform,
            q=None,
            l=None,
            pi=self.cov.pi_0 if self.calc_pi else None,
            argmask=self.argmask,
            add_1_pre=False,
            static_argnames=["add_1_pre"],
        )

        @partial(CvTrans.from_cv_function, v=mu_ref)
        def _get_w(cv: CV, _nl, shmap, shmap_kwargs, v: Array):
            x = cv.cv

            return cv.replace(cv=jnp.einsum("i,ij->j", x, v), _combine_dims=None)

        tr = f_trans_2 * _get_w

        if self.trans is not None:
            tr = self.trans * tr

        print(f"{self.cv_0[0].shape}")

        w_out_cv, w_out_cv_t = DataLoaderOutput.apply_cv(
            f=tr,
            x=self.cv_0,  # type:ignore
            x_t=self.cv_t,  # type:ignore
            nl=self.nl,
            # nl_t=self.nl_t,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
            verbose=verbose,
        )
        assert w_out_cv_t is not None

        w_corr = CV.stack(*w_out_cv).cv
        w_corr_t = CV.stack(*w_out_cv_t).cv

        def _norm(w):
            w = jnp.where(jnp.sum(w > 0) - jnp.sum(w < 0) > 0, w, -w)  # majaority determines sign
            n_neg = jnp.sum(w < 0)

            w_pos: jax.Array = jnp.where(w >= 0, w, 0)  # type: ignore
            w_neg: jax.Array = jnp.where(w < 0, -w, 0)  # type: ignore

            n = jnp.sum(w_pos)

            w_pos /= n
            w_neg /= n

            return w_pos, jnp.sum(w_neg), n_neg / w.shape[0]

        w_pos, wf_neg, f_neg = vmap_decorator(_norm, in_axes=1)(w_corr)
        w_pos_t, wf_neg_t, f_neg_t = vmap_decorator(_norm, in_axes=1)(w_corr_t)

        x = jnp.logical_and(wf_neg == 0, f_neg == 0)

        nm = jnp.sum(x)

        if nm == 0:
            print(f"didn't find modes with positive weights, aborting {wf_neg=} {f_neg=}")
            assert self.w is not None
            assert self.w_t is not None

            return self.w, self.w_t, None, None, False

        if nm > 1:
            print(f"found multiple modes with positive weights, merging {wf_neg=} {f_neg=}")
        else:
            print(f"succes")

        def norm_w_corr(w: jax.Array):
            log_w = jnp.log(w)
            log_w -= jnp.max(log_w)
            log_w_norm = jnp.log(jnp.sum(jnp.exp(log_w)))

            w_norm = jnp.exp(log_w - log_w_norm) * w.shape[0]
            # w_norm = DataLoaderOutput._unstack_weights([cvi.shape[0] for cvi in self.cv_0], w_norm)

            return w_norm

        print(f"{w_pos.shape=}")

        w_corr = norm_w_corr(jnp.sum(w_pos[x, :], axis=0))
        w_corr_t = norm_w_corr(jnp.sum(w_pos_t[x, :], axis=0))

        print(f"  {jnp.std(w_corr)=}  ")

        def get_new_w(w_orig: list[Array], w_corr: Array):
            log_w_orig = jnp.log(jnp.hstack(w_orig))

            w_new_log = jnp.log(w_corr) + log_w_orig

            mm = jnp.max(w_new_log)

            w_new_log -= mm

            w_new_log_norm = jnp.log(jnp.sum(jnp.exp(w_new_log)))

            w_new = jnp.exp(w_new_log - w_new_log_norm)

            w_new = jnp.real(w_new)

            w_new = DataLoaderOutput._unstack_weights([cvi.shape[0] for cvi in self.cv_0], w_new)

            return w_new

        assert self.w is not None
        assert self.w_t is not None

        w_new = get_new_w(self.w, w_corr)
        w_new_t = get_new_w(self.w_t, w_corr_t)

        # print(f"{  jnp.isnan(jnp.hstack(w_new)).any()=}")

        w_corr = DataLoaderOutput._unstack_weights([cvi.shape[0] for cvi in self.cv_0], w_corr)
        w_corr_t = DataLoaderOutput._unstack_weights([cvi.shape[0] for cvi in self.cv_t], w_corr_t)

        return w_new, w_new_t, w_corr, w_corr_t, True

    def weighted_model(
        self,
        chunk_size=None,
        macro_chunk=1000,
        # verbose=False,
        out_dim=None,
        **kwargs,
    ) -> KoopmanModel:
        if self.only_diag:
            print("cannot reweight model with only diagonal covariances")
            return self

        new_w, new_w_t, _, _, b = self.koopman_weight(
            verbose=self.verbose,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            out_dim=out_dim,
        )

        # if not b:
        #     return self

        kw = dict(
            w=new_w,
            rho=self.rho,
            w_t=new_w_t,
            rho_t=self.rho_t,
            dynamic_weights=self.dynamic_weights,
            cv_0=self.cv_0,
            cv_t=self.cv_t,
            nl=self.nl,
            nl_t=self.nl_t,
            add_1=self.add_1,
            eps=self.eps,
            eps_pre=self.eps_pre,
            # method=self.method,
            symmetric=False,
            out_dim=self.out_dim,
            tau=self.tau,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
            trans=self.trans,
            # T_scale=self.T_scale,
            verbose=self.verbose,
            calc_pi=self.calc_pi,
            max_features=self.max_features,
            max_features_pre=self.max_features_pre,
            only_diag=self.only_diag,
            correlation_whiten=self.correlation_whiten,
            constant_threshold=self.constant_threshold,
            shrink=self.shrink,
            shrinkage_method=self.shrinkage_method,
            eps_shrink=self.eps_shrink,
            use_w=self.use_w,
            generator=self.generator,
            entropy_reg=self.entropy_reg,
            # periodicities=self.periodicities,
            # argmask_s=self.argmask_s,
        )

        kw.update(**kwargs)

        return KoopmanModel.create(**kw)  # type:ignore

    def get_n_skip(
        self,
        n_skip: int | None = None,
        remove_constant=True,
    ):
        s = self.s

        if n_skip is None:
            if self.add_1 or not self.calc_pi:
                n_skip = 1
            else:
                n_skip = 0

        s = s[n_skip:]

        if remove_constant:
            nc = jnp.abs(1 - s) < self.constant_threshold

            if jnp.sum(nc) > 0:
                print(f"found {jnp.sum(nc)} constant eigenvalues,removing from timescales")

            s = s[jnp.logical_not(nc)]

            n_skip += int(jnp.sum(nc))

        print(f"got {n_skip=}")

        return n_skip

    def timescales(
        self,
        remove_constant=True,
        n_skip=None,
        mask: jax.Array | None = None,
    ):
        s = self.s

        n_skip = self.get_n_skip(
            n_skip=n_skip,
            remove_constant=remove_constant,
        )

        s = s[n_skip:]

        if mask is not None:
            s = s[mask]

        tau = self.tau
        if tau is None:
            tau = 1
            print("tau not set, assuming 1")

        return -tau / jnp.log(s)


class PeriodicKoopmanModel(MyPyTreeNode):
    km_periodic: KoopmanModel
    km_nonperiodic: KoopmanModel
    periodicities: jax.Array
    trans_periodic: CvTrans
    trans_nonperiodic: CvTrans
    trans: CvTrans | None = None

    @staticmethod
    def _reorder(cv: CV, _nl, shmap, shmap_kwargs, idx):
        return cv.replace(cv=cv.cv.at[idx].set(cv.cv))

    @staticmethod
    def _center_periodic(cv: CV, _nl, shmap, shmap_kwargs, periodicities, angles):
        return cv.replace(
            cv=jnp.mod(cv.cv - angles + jnp.pi, 2 * jnp.pi) - jnp.pi,  # center around angles
        )

    @staticmethod
    def _exp_periodic(cv: CV, _nl, shmap, shmap_kwargs, periodicities=None):
        if periodicities is None:
            return cv.replace(cv=jnp.exp(1.0j * cv.cv))

        return cv.replace(
            cv=jnp.where(
                periodicities,
                jnp.exp(1.0j * cv.cv),
                cv.cv,
            )
        )

    @staticmethod
    def _inv_exp_periodic(cv: CV, _nl, shmap, shmap_kwargs, periodicities=None):
        return cv.replace(cv=jnp.real(cv.cv))

        # if periodicities is None:
        #     return cv.replace(cv=jnp.angle(cv.cv))

        # return cv.replace(
        #     cv=jnp.where(
        #         periodicities,
        #         jnp.real(cv.cv),
        #         cv.cv,
        #     )
        # )

    def _get_s_ext(
        self,
        remove_constant=True,
        n_skip: int | None = None,
    ):
        s_ext_per = jnp.vstack([self.km_periodic.s, jnp.full_like(self.km_periodic.s, 0)])

        skip_per = self.km_periodic.get_n_skip(
            n_skip=n_skip,
            remove_constant=remove_constant,
        )

        s_ext_nonper = jnp.vstack([self.km_nonperiodic.s, jnp.full_like(self.km_nonperiodic.s, 1)])

        n_skip_nonper = self.km_nonperiodic.get_n_skip(
            n_skip=n_skip,
            remove_constant=remove_constant,
        )

        s_ext = jnp.hstack([s_ext_per[:, skip_per:], s_ext_nonper[:, n_skip_nonper:]])

        s_ext = jnp.concatenate(  # type: ignore
            [
                s_ext,
                jnp.arange(s_ext.shape[1]).reshape((1, -1)),
            ],
            axis=0,
        )

        # print(f"{s_ext=}")

        s_ext = s_ext[:, jnp.argsort(s_ext[0, :], descending=True)]

        return s_ext

    def f(
        self,
        out_dim=None,
        remove_constant=True,
        n_skip: int | None = None,
        mask: jax.Array | None = None,
        inv_exp=True,
    ):
        s_ext = self._get_s_ext(
            remove_constant=remove_constant,
            n_skip=n_skip,
        )

        # print(f"{s_ext=}")

        o = s_ext[:, :out_dim]

        n_per = jnp.sum(o[1, :] == 0)
        n_nonper = jnp.sum(o[1, :] == 1)

        # print(f"{n_per=} {n_nonper=}")

        tr_per, _ = self.km_periodic.f(
            out_dim=n_per,
            remove_constant=remove_constant,
            n_skip=n_skip,
            mask=mask,
            inv_exp=inv_exp,
        )

        tr_nonper, _ = self.km_nonperiodic.f(
            out_dim=n_nonper,
            remove_constant=remove_constant,
            n_skip=n_skip,
            mask=mask,
            inv_exp=inv_exp,
        )

        idx = jnp.hstack(
            [
                jnp.argwhere(o[1, :] == 0, size=int(n_per), fill_value=-1).reshape(-1),
                jnp.argwhere(o[1, :] == 1, size=int(n_nonper), fill_value=-1).reshape(-1),
            ]
        )

        print(f"{idx=} {n_per=} {n_nonper=} ")

        out_tr = (self.trans_periodic * tr_per + self.trans_nonperiodic * tr_nonper) * CvTrans.from_cv_function(
            PeriodicKoopmanModel._reorder,
            idx=idx,
        )

        if self.trans is not None:
            out_tr = self.trans * out_tr

        return out_tr, o[1, :] == 0

    def timescales(
        self,
        remove_constant=True,
        n_skip: int | None = None,
    ):
        s_ext = self._get_s_ext(
            remove_constant=remove_constant,
            n_skip=n_skip,
        )

        return -self.km_periodic.tau / jnp.log(s_ext[0, :])

    def weighted_model(
        self,
        chunk_size=None,
        macro_chunk=1000,
        out_dim=None,
        **kwargs,
    ) -> PeriodicKoopmanModel:
        km_per = self.km_periodic.weighted_model(
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            out_dim=out_dim,
            **kwargs,
        )

        # km_nonper = self.km_nonperiodic.weighted_model(
        #     chunk_size=chunk_size,
        #     macro_chunk=macro_chunk,
        #     out_dim=out_dim,
        #     **kwargs,
        # )

        return PeriodicKoopmanModel(
            km_periodic=km_per,
            km_nonperiodic=self.km_nonperiodic,
            periodicities=self.periodicities,
            trans_periodic=self.trans_periodic,
            trans_nonperiodic=self.trans_nonperiodic,
            trans=self.trans,
        )


X = TypeVar("X", "CV", "SystemParams", "NeighbourList")
X2 = TypeVar("X2", "CV", "SystemParams", "NeighbourList")


class Covariances(MyPyTreeNode):
    rho_00: jax.Array
    rho_01: jax.Array | None
    rho_10: jax.Array | None
    rho_11: jax.Array | None

    rho_gen: jax.Array | None

    pi_s_0: jax.Array | None
    pi_s_1: jax.Array | None
    sigma_0: jax.Array
    sigma_1: jax.Array | None

    d_rho_00: jax.Array | None

    bc: list[jax.Array] | None = None  # used in BC estimator
    shrinkage_method: str = "bidiag"

    time_series: bool = True

    # n: int | None = None

    W_0: jax.Array | None = None
    W_1: jax.Array | None = None

    only_diag: bool = False
    trans_f: CvTrans | CvTrans | None = None
    trans_g: CvTrans | CvTrans | None = None
    # T_scale: float = 1.0
    symmetric: bool = False
    pi_argmask: Array | None = None

    w2: float
    w3: float

    @staticmethod
    def create(
        cv_0: list[X],
        cv_1: list[X] | None = None,
        nl: list[NeighbourList] | NeighbourList | None = None,
        nl_t: list[NeighbourList] | NeighbourList | None = None,
        w: list[Array] | None = None,
        w_t: list[Array] | None = None,
        dynamic_weights: list[Array] | None = None,
        calc_pi=True,
        macro_chunk=1000,
        chunk_size=None,
        only_diag=False,
        trans_f: CvTrans | CvTrans | None = None,
        trans_g: CvTrans | CvTrans | None = None,
        # T_scale=1,
        symmetric=False,
        calc_C00=True,
        calc_C01=True,
        calc_C10=True,
        calc_C11=True,
        shmap_kwargs=ShmapKwargs.create(),
        verbose=True,
        shrink=True,
        # BC=False,
        shrinkage_method="bidiag",
        pi_argmask: Array | None = None,
        get_diff=False,
        bessel_correct=True,
        generator=False,
        eps_shrink=1e-3,
    ) -> Covariances:
        time_series = cv_1 is not None

        # assert not only_diag

        # if generator:
        #     generator = True
        #     print("generator not implemented, setting to False")

        if not time_series:
            if get_diff:
                print("cannot get differences without time series, setting to False")
                get_diff = False

            # assert trans_f is not None
            # assert trans_g is not None

        # print(f"{nl=}")

        if w is None:
            w = [jnp.ones((cvi.shape[0],)) for cvi in cv_0]

        n = jnp.sum(jnp.array([jnp.sum(wi) for wi in w]))

        w = [wi / n for wi in w]
        if time_series:
            if w_t is None:
                w_t = [jnp.ones((cvi.shape[0],)) for cvi in cv_1]

            n = jnp.sum(jnp.array([jnp.sum(wi) for wi in w_t]))
            w_t = [wi / n for wi in w_t]

        if dynamic_weights is None:
            dynamic_weights = [jnp.ones((cvi.shape[0],)) for cvi in cv_0]

        # print(f"{dynamic_weights=}")

        print(f"shrinking {shrinkage_method =} ")

        BC = shrinkage_method == "BC"

        print(f"{nl=}")

        def cov_pi(
            carry: tuple[
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                list[jax.Array] | None,
                jax.Array | None,
            ],
            cv_0: CV,
            cv_1: CV | None,
            w: jax.Array | None,
            w_t: jax.Array | None = None,
            diff_w: jax.Array | None = None,
        ):
            assert w is not None

            w_t = w  # _t / (diff_w**2)

            # print(f"{nl=}")

            # print(f"{w=} {w_t=} {diff_w=}")

            (
                rho_00_prev,
                rho_01_prev,
                rho_10_prev,
                rho_11_prev,
                rho_gen_prev,
                pi_0_prev,
                pi_1_prev,
                sigma_0_prev,
                sigma_1_prev,
                w_prev,
                w_prev_t,
                diff_w_prev_0,
                diff_w_prev_1,
                bc,
                d_rho_prev,
            ) = carry

            if (trans_f is not None) and generator and (trans_g is not None):
                assert not only_diag, "only_diag not implemented for generator with transformations"

                def phi(x):
                    # print(f"phi: {x=} ")
                    cv, _ = trans_f.compute_cv(x, nl, shmap=False)
                    return cv

                # sp shape (macro_chunk, n_atoms, 3)

                shape = cv_0.shape[1:]
                n_shape = len(shape)

                # print(f"{shape=} {n_shape=}")

                # print(f"{shape=} {n_shape=}")

                # linearize instead of jvp, to avoid creating full jacobian
                # works well for repeated function application
                cv_0_trans, f_jvp = jax.linearize(phi, cv_0)

                if sigma_0_prev is not None:
                    sigma_0_inv = jnp.where(sigma_0_prev == 0, 1, 1 / sigma_0_prev)

                def get(L, ni):
                    print(f"outside {ni=}")

                    @partial(jax.vmap, in_axes=0)
                    def _get(ni):
                        print(f"inside {ni=}")

                        u = jnp.zeros_like(cv_0.coordinates)

                        u = u.at[:, *ni].set(1.0)

                        x1 = cv_0.replace(coordinates=u)

                        df = f_jvp(x1)

                        # print(f"{df.shape=} {sigma_0_prev=}")

                        if sigma_0_prev is not None:
                            df = df * sigma_0_inv

                        return jnp.einsum("nl,nk,n->lk", df.cv, df.cv, w)

                    dL = jnp.sum(_get(ni), axis=0)

                    print(f"{dL=}")

                    # need to divide by mass
                    dL /= atomic_masses[jnp.array(nl.info.z_array)[ni[0, 0]]]

                    return L + dL, None

                ni = jnp.stack(jnp.meshgrid(*[jnp.arange(n) for n in shape], indexing="ij"), axis=2)

                # jax.debug.print("ni before scan {}", ni)

                print(f"{ni=}")

                print(f"{ni.shape=}")

                dL, _ = jax.lax.scan(
                    get,
                    jnp.zeros((cv_0_trans.shape[1], cv_0_trans.shape[1])),
                    ni,
                )

                print(f"{dL=}")

                cv_0 = cv_0_trans

            assert not cv_0.atomic

            x_0 = cv_0.cv

            if cv_1 is not None:
                x_1 = cv_1.cv
            else:
                x_1 = None

            if sigma_1_prev is not None:
                assert x_1 is not None
                sigma_1_inv = jnp.where(sigma_1_prev == 0, 1, 1 / sigma_1_prev)
                x_1 = jnp.einsum("ni,i->ni", x_1, sigma_1_inv)
            else:
                sigma_1_inv = None

            if sigma_0_prev is not None:
                sigma_0_inv = jnp.where(sigma_0_prev == 0, 1, 1 / sigma_0_prev)

                x_0 = jnp.einsum("ni,i->ni", x_0, sigma_0_inv)

            if get_diff:
                if (sigma_0_prev is not None) and (sigma_1_prev is not None):
                    dx = (x_0 * sigma_0_prev - x_1 * sigma_1_prev) * jnp.sqrt(sigma_0_inv * sigma_1_inv)

                else:
                    dx = x_0 - x_1

            def get_pi(_x, _w, _pi_prev, _w_prev, calc_pi):
                _dw = jnp.sum(_w)

                if _w_prev is None:
                    _w_tot = _dw
                else:
                    _w_tot = _w_prev + _dw

                if not calc_pi:
                    return None, None, _w_tot, _dw

                _pi_new_dw = jnp.einsum("ni,n->i", _x, _w)

                if _pi_prev is None:
                    _pi_tot = _pi_new_dw / _dw

                    _dpi = None
                else:
                    _pi_tot = (_pi_prev * _w_prev + _pi_new_dw) / (_w_tot)

                    _dpi = _pi_tot - _pi_prev

                if pi_argmask is not None:
                    # print(f"{pi_argmask=}")

                    _pi_tot = _pi_tot.at[pi_argmask].set(0.0)
                    if _dpi is not None:
                        _dpi = _dpi.at[pi_argmask].set(0.0)

                    # print(f"{_pi_tot}")

                return _pi_tot, _dpi, _w_tot, _dw

            pi_0_new, dpi_0, w_tot, dw = get_pi(x_0, w, pi_0_prev, w_prev, calc_pi)

            # jax.debug.print("dw_o {}", dw_0)

            if time_series:
                pi_1_new, dpi_1, w_tot_t, dw_t = get_pi(x_1, w_t, pi_1_prev, w_prev_t, calc_pi)

                # diff_w = jnp.sqrt(w * w_t)  # * w_dw

                w0_dyn = w  # / diff_w
                w1_dyn = w_t  # / diff_w

                _, _, diff_tot_0, diff_dw_0 = get_pi(None, w0_dyn, None, diff_w_prev_0, False)
                _, _, diff_tot_1, diff_dw_1 = get_pi(None, w1_dyn, None, diff_w_prev_1, False)

            def c(_x, _y, _w, _c_prev, _pi_x, _pi_y, _d_pi_x, _d_pi_y, _w_prev, _dw, dx=False):
                if _d_pi_x is not None and not dx:
                    assert _d_pi_y is not None

                    if only_diag:
                        _c_prev += jnp.conj(_d_pi_x) * _d_pi_y
                    else:
                        _c_prev += jnp.outer(jnp.conj(_d_pi_x), _d_pi_y)

                if calc_pi and not dx:
                    assert _pi_x is not None
                    assert _pi_y is not None

                    u = _x - _pi_x
                    v = _y - _pi_y
                else:
                    u = _x
                    v = _y

                if only_diag:
                    _c_new_dw = jnp.einsum("ni,ni,n->i", jnp.conj(u), v, _w)  # in case of complex cv
                else:
                    _c_new_dw = jnp.einsum("ni,nj,n->ij", jnp.conj(u), v, _w)  # in case of complex cv

                if _c_prev is None:
                    return _c_new_dw / _dw

                return (_w_prev * _c_prev + _c_new_dw) / (_w_prev + _dw)

            if calc_C00:
                rho_00 = c(x_0, x_0, w, rho_00_prev, pi_0_new, pi_0_new, dpi_0, dpi_0, w_prev, dw)

            rho_01, rho_10, rho_11 = None, None, None

            if time_series:
                assert w is not None

                if calc_C01:
                    rho_01 = c(
                        x_0, x_1, w0_dyn, rho_01_prev, pi_0_new, pi_1_new, dpi_0, dpi_1, diff_w_prev_0, diff_dw_0
                    )

                if calc_C10:
                    # rho_10 = c(x_1, x_0, w, rho_10_prev, pi_1_new, pi_0_new, dpi_1, dpi_0, w_prev, dw)
                    rho_10 = c(
                        x_1, x_0, w1_dyn, rho_10_prev, pi_1_new, pi_0_new, dpi_1, dpi_0, diff_w_prev_1, diff_dw_1
                    )

                if calc_C11:
                    rho_11 = c(x_1, x_1, w_t, rho_11_prev, pi_1_new, pi_1_new, dpi_1, dpi_1, w_prev_t, dw_t)

            if generator:
                if rho_gen_prev is not None:
                    rho_gen = (rho_gen_prev * w_prev + dL) / (w_prev + dw)
                else:
                    rho_gen = dL / dw
            else:
                rho_gen = None

            if get_diff:
                # d_rho = c(dx, dx, w, d_rho_prev, None, None, None, None, w_prev, dw, dx=True)
                d_rho = c(dx, dx, w, d_rho_prev, None, None, None, None, w_prev, dw, dx=True)
            else:
                d_rho = None

            # convert everything to new sigma

            diag_rho_00 = jnp.diag(rho_00) if not only_diag else rho_00

            d_sigma_0 = jnp.sqrt(jnp.where(diag_rho_00 <= 0, 0.0, diag_rho_00))
            f0 = jnp.where(d_sigma_0 == 0, 1.0, 1 / d_sigma_0)

            pi_0_new = jnp.einsum("i,i->i", pi_0_new, f0) if calc_pi else None

            def get_rho(rho, f_left, f_right):
                if only_diag:
                    return jnp.einsum("i,i,i->i", rho, jnp.conj(f_left), f_right)
                return jnp.einsum("ij,i,j->ij", rho, jnp.conj(f_left), f_right)

            rho_00 = get_rho(rho_00, f0, f0)

            if sigma_0_prev is None:
                sigma_0 = d_sigma_0
            else:
                sigma_0 = jnp.where(sigma_0_prev == 0, d_sigma_0, sigma_0_prev * d_sigma_0)

            if generator:
                rho_gen = get_rho(rho_gen, f0, f0)

            if time_series:
                diag_rho_11 = jnp.diag(rho_11) if not only_diag else rho_11

                d_sigma_1 = jnp.sqrt(jnp.where(diag_rho_11 <= 0, 0.0, diag_rho_11))  # type: ignore
                f1 = jnp.where(d_sigma_1 <= 0, 1.0, 1 / d_sigma_1)

                pi_1_new = jnp.einsum("i,i->i", pi_1_new, f1) if calc_pi else None

                if rho_01 is not None:
                    rho_01 = get_rho(rho_01, f0, f1)
                if rho_10 is not None:
                    rho_10 = get_rho(rho_10, f1, f0)

                if rho_11 is not None:
                    rho_11 = get_rho(rho_11, f1, f1)

                if sigma_1_prev is None:
                    sigma_1 = d_sigma_1
                else:
                    sigma_1 = jnp.where(sigma_1_prev == 0, d_sigma_1, sigma_1_prev * d_sigma_1)

            else:
                sigma_1 = None
                rho_01 = None
                rho_10 = None
                rho_11 = None

            if get_diff:
                d_rho = get_rho(d_rho, f0, f0)

            if BC:
                assert not only_diag, "BC not implemented for only_diag"

                def _bc(_x, _y, _w, _bc_prev, _w_prev, _dw):
                    _bc_new_dw = jnp.einsum("ni,nj,ni,nj,n->ij", _x, _x, _y, _y, _w)

                    if _bc_prev is None:
                        return _bc_new_dw / _dw

                    return (_w_prev * _bc_prev + _bc_new_dw) / (_w_prev + _dw)

                assert calc_pi is False, "bias correction with pi not implemented"

                bc_00 = _bc(x_0, x_0, w, bc[0] if bc is not None else None, w_prev, dw)
                bc_01 = _bc(x_0, x_1, w, bc[1] if bc is not None else None, w_prev, dw)

                bc_00 = jnp.einsum("ij,i,j->ij", bc_00, f0**2, f0**2)
                bc_01 = jnp.einsum("ij,i,j->ij", bc_01, f0**2, f1**2)

                bc = [bc_00, bc_01]

            return (
                rho_00,
                rho_01,
                rho_10,
                rho_11,
                rho_gen,
                pi_0_new,
                pi_1_new if time_series else None,
                sigma_0,
                sigma_1 if time_series else None,
                w_tot,
                w_tot_t if time_series else None,
                # None,
                diff_tot_0 if time_series else None,
                diff_tot_1 if time_series else None,
                bc,
                d_rho,
            )

        if trans_f is not None:

            def f_func(x: X, nl: NeighbourList | None) -> CV:
                # print(f"{x=} {nl=} {generator=}")

                if not generator:
                    x, _ = trans_f.compute_cv(x, nl, chunk_size=chunk_size, shmap=False)
                # else:
                #     print(f"{x=}")
                return x

        else:
            assert isinstance(cv_0[0], CV)

            def f_func(x: X, nl: NeighbourList | None) -> CV:
                assert isinstance(x, CV)
                return x

        if trans_g is not None:

            def g_func(x: X, nl: NeighbourList | None) -> CV:
                x, _ = trans_g.compute_cv(x, nl, chunk_size=chunk_size, shmap=False)

                return x

        else:

            def g_func(x: X, nl: NeighbourList | None) -> CV:
                assert isinstance(x, CV)
                return x

        g_func = cast(Callable[[CV | SystemParams, NeighbourList | None], CV], g_func)

        chunk_func_init_args = cast(
            tuple[
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                list[jax.Array] | None,
                jax.Array | None,
            ],
            (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
        )

        # print(f"{cv_1=}")

        # with jax.debug_nans():
        out = macro_chunk_map_fun(
            # f=lambda x, nl: x,
            # ft=lambda x, nl: x,
            f=f_func,
            ft=g_func,
            y=cv_0,
            y_t=cv_1,
            nl=nl,
            nl_t=nl,
            macro_chunk=macro_chunk,
            verbose=verbose,
            chunk_func=cov_pi,
            chunk_func_init_args=chunk_func_init_args,
            w=w,
            w_t=w_t,
            d_w=dynamic_weights,
            jit_f=True,
        )

        (
            rho_00,
            rho_01,
            rho_10,
            rho_11,
            rho_gen,
            pi_s_0,
            pi_s_1,
            sigma_0,
            sigma_1,
            _w,
            _wt,
            _dw_0,
            _dw_1,
            bc,
            d_rho,
        ) = out

        print(f"{_w=} {_wt=} {_dw_0=} {_dw_1=}")

        print(f"{rho_gen=}")

        # # divide time lagged covariances by total weight
        # if time_series:
        #     print(f"correcting")

        #     assert _dw_0 is not None
        #     rho_01 /= _dw_0
        #     rho_10 /= _dw_1

        if BC:
            # print(f"{bc=}")

            bc_00 = bc[0] - rho_00**2
            bc_01 = bc[1] - rho_00**2

            # print(f"{bc_00=} {rho_00=}")

            bc = [bc_00, bc_01]

        # print(f"test {rho_00=}")
        assert rho_00 is not None
        assert sigma_0 is not None

        # bessel correction: https://univ-lyon1.hal.science/hal-04693522/file/shrinkage_eusipco_2024_reviewed.pdf

        w_stack = jnp.hstack(w)

        w2 = jnp.sum(w_stack**2) / (jnp.sum(w_stack) ** 2)
        w3 = jnp.sum(w_stack**3) / (jnp.sum(w_stack) ** 3)

        print(f"{w2=} {w3=}")

        # if bessel_correct:
        eps = w2

        print(f"bessel correction {eps=} {1/eps=}  {w_stack.shape[0]=} ")

        # C -> (1-eps) C
        # sigma -> sqrt(1-eps) sigma
        # rho: stays the same
        # pi: stays the same

        gamma = 1 / jnp.sqrt(1 - eps)

        if sigma_0 is not None:
            sigma_0 *= gamma

        if sigma_1 is not None:
            sigma_1 *= gamma

        if pi_s_0 is not None:
            pi_s_0 /= gamma

        if pi_s_1 is not None:
            pi_s_1 /= gamma

        # rho_00 = jnp.real(rho_00)

        cov = Covariances(
            rho_00=rho_00,
            rho_01=rho_01,
            rho_10=rho_10,
            rho_11=rho_11,
            rho_gen=rho_gen,
            pi_s_0=pi_s_0,
            pi_s_1=pi_s_1,
            only_diag=only_diag,
            trans_f=trans_f,
            trans_g=trans_g,
            # T_scale=T_scale,
            sigma_0=sigma_0,
            sigma_1=sigma_1,
            # n=sum([a.shape[0] for a in cv_0]),
            bc=bc,
            shrinkage_method=shrinkage_method,
            d_rho_00=d_rho,
            w2=w2,
            w3=w3,
            time_series=time_series,
            symmetric=not time_series,
        )

        # print(f"pre {jnp.isnan(cov.C00).any()}")

        # print(f"shrinking {cov.C00 is None =} {cov.C11 is None=}")

        # print(f"debug nans v2")
        # with jax.debug_nans():
        if symmetric:
            cov = cov.symmetrize()

        # print(f"shrinking {cov.C00 is None =} {cov.C11 is None=}")

        # print(f"sym {jnp.isnan(cov.C00).any()}")0

        if shrink:
            cov = cov.shrink(eps_shrink=eps_shrink)

        # print(f"shrunk {jnp.isnan(cov.C00).any()}")

        return cov

    @property
    def generator(self):
        return self.rho_gen is not None

    @property
    def pi_0(self):
        if self.pi_s_0 is None:
            return None
        return jnp.einsum("i,i->i", self.pi_s_0, self.sigma_0)

    @property
    def pi_1(self):
        if self.pi_s_1 is None:
            return None
        return jnp.einsum("i,i->i", self.pi_s_1, self.sigma_1)

    @property
    def C00(self):
        if self.only_diag:
            return jnp.einsum("i,i->i", self.rho_00, self.sigma_0**2)

        return jnp.einsum("ij,i,j->ij", self.rho_00, self.sigma_0, self.sigma_0)

    @property
    def C_gen(self):
        if self.only_diag:
            return jnp.einsum("i,i->i", self.rho_gen, self.sigma_0**2)

        return jnp.einsum("ij,i,j->ij", self.rho_gen, self.sigma_0, self.sigma_0)

    @property
    def d_C00(self):
        if self.only_diag:
            return jnp.einsum("i,i->i", self.d_rho_00, self.sigma_0**2)

        return jnp.einsum("ij,i,j->ij", self.d_rho_00, self.sigma_0, self.sigma_0)

    @property
    def C01(self):
        if self.rho_01 is None:
            return None
        assert self.sigma_1 is not None

        if self.only_diag:
            return jnp.einsum("i,i->i", self.rho_01, self.sigma_0 * self.sigma_1)

        return jnp.einsum("ij,i,j->ij", self.rho_01, self.sigma_0, self.sigma_1)

    @property
    def C10(self):
        if self.rho_10 is None:
            return None
        assert self.sigma_1 is not None

        if self.only_diag:
            return jnp.einsum("i,i->i", self.rho_10, self.sigma_1 * self.sigma_0)

        return jnp.einsum("ij,i,j->ij", self.rho_10, self.sigma_1, self.sigma_0)

    @property
    def C11(self):
        if self.rho_11 is None:
            return None
        assert self.sigma_1 is not None

        if self.only_diag:
            return jnp.einsum("i,i->i", self.rho_11, self.sigma_1**2)

        return jnp.einsum("ij,i,j->ij", self.rho_11, self.sigma_1, self.sigma_1)

    @property
    def sigma_0_inv(self):
        return jnp.where(self.sigma_0 == 0, 0, 1 / self.sigma_0)

    @property
    def sigma_1_inv(self):
        if self.sigma_1 is None:
            return None
        return jnp.where(self.sigma_1 == 0, 0, 1 / self.sigma_1)

    # def glasso_whiten(self, choice: str):
    #     from gglasso.problem import glasso_problem
    #     import numpy as np

    #     if choice == "rho_00":
    #         rho = self.rho_00
    #         sigma_inv = self.sigma_0_inv
    #         C = self.C00
    #     elif choice == "rho_11":
    #         rho = self.rho_11
    #         sigma_inv = self.sigma_1_inv
    #         C = self.C11
    #     else:
    #         raise ValueError(f"choice {choice} not known")

    #     assert rho is not None

    #     print(f"glasso on {choice} {rho.shape=}")

    #     P = glasso_problem(
    #         S=rho.__array__(),
    #         N=self.n_eff,
    #         reg_params={"lambda1": 1e-3},
    #         latent=False,
    #         do_scaling=False,
    #     )

    #     P.solve()

    #     # lambda1_range = np.logspace(0, -3, 10)
    #     # modelselect_params = {"lambda1_range": lambda1_range}

    #     # P.model_selection(modelselect_params=modelselect_params, method="eBIC", gamma=0.1)

    #     sol = P.solution.precision_
    #     sol_l, sol_U = jnp.linalg.eigh(sol)
    #     inv_cov_glasso = sol_U @ jnp.diag(sol_l ** (0.5)) @ sol_U.T

    #     W = jnp.einsum(
    #         "ij,j->ij",
    #         inv_cov_glasso,
    #         sigma_inv,
    #     )

    #     print(f"{jnp.linalg.norm(W @ C @ W.T - jnp.eye(W.shape[0]))=}")

    #     return W

    def whiten_rho(
        self,
        choice: str,
        epsilon: float = 1e-4,
        max_features: int | None = None,
        verbose=False,
        cholesky=False,
        apply_mask: bool = True,
    ) -> Array:
        # returns W such that W C W.T = I and hence w.T W = C^-1

        # https://arxiv.org/pdf/1512.00809

        if choice == "rho_00":
            C = self.C00
            sigma = self.sigma_0
        elif choice == "rho_11":
            C = self.C11
            sigma = self.sigma_1
        else:
            raise ValueError(f"choice {choice} not known")

        assert C is not None

        if self.only_diag:
            return 1 / jnp.sqrt(C) * sigma

        C = C

        print(f"{jnp.linalg.norm(C - jnp.conj(C.T))=}")

        print(f"{epsilon=}")

        if cholesky:
            print(f"cholesky")

            import scipy

            # this is pivoted cholesky

            if C.dtype == jnp.complex_:
                cho = scipy.linalg.lapack.zpstrf
                print("complex cholesky")
            else:
                cho = scipy.linalg.lapack.dpstrf

            X, P, r, info = cho(C, tol=epsilon**2, lower=True)
            X = jnp.array(X)

            # print(f"{X=}")

            pi = jnp.eye(P.shape[0])[:, P - 1][:, :r]
            X = X.at[jnp.triu_indices(X.shape[0], 1)].set(0)  # set upper half to zero
            X = X[:r, :][:, :r]

            err = jnp.linalg.norm(pi.T @ C @ pi - X @ jnp.conj(X).T)

            print(f" rank reduced chol {err=} rank {r} ")

            # P_out = P[:r]

            X_inv = jax.scipy.linalg.solve_triangular(
                X,
                jnp.eye(*X.shape),  # type: ignore
                lower=True,
            )

            W = X_inv @ pi.T

        else:
            # else:
            theta, G = jnp.linalg.eigh(C)

            idx = jnp.argmax(theta)
            mask = jnp.abs(theta / theta[idx]) > epsilon**2

            if verbose:
                print(f"{jnp.sum(mask)=} ")

            theta_safe = jnp.where(mask, theta, 1)

            theta_inv = jnp.where(mask, 1 / jnp.sqrt(theta_safe), 0)

            W = jnp.einsum(
                "i,ji->ij",
                # V_inv,
                theta_inv,
                G,
            )
            if apply_mask:
                W = W[mask, :]

        if apply_mask:
            if max_features is not None:
                if W.shape[0] > max_features:
                    print(f"whiten: reducing dim to {max_features=}")
                    W = W[:max_features, :]

        print(f"{jnp.linalg.norm(W @ C @ jnp.conj(W).T - jnp.eye(W.shape[0]))=}")

        # if return_P:
        #     return W, P_out

        W = W @ jnp.diag(sigma)

        return W

    def mask(
        self,
        eps_pre: float | None,
        max_features: int | None = 2000,
        auto_cov_threshold: float | None = None,
    ):
        argmask = jnp.arange(self.rho_00.shape[0])
        if eps_pre is not None:
            if self.sigma_1 is not None:
                b = jnp.logical_and(self.sigma_0 > eps_pre, self.sigma_1 > eps_pre)
            else:
                b = self.sigma_0 > eps_pre
            argmask = argmask[b]

        # print(f"{argmask.shape=}")

        if self.rho_01 is not None:
            d = jnp.diag(self.rho_01) if not self.only_diag else self.rho_01
        else:
            assert self.rho_gen is not None
            d = jnp.exp(-jnp.diag(self.rho_gen)) if not self.only_diag else jnp.exp(-self.rho_gen)

        argsort = jnp.argsort(d[argmask], descending=True)
        argmask = argmask[argsort]

        # print(f"{argmask.shape=}")

        if auto_cov_threshold is not None:
            b = d[argmask] > auto_cov_threshold
            argmask = argmask[b]

        # print(f"{argmask.shape=}")

        if max_features is not None:
            if argmask.shape[0] > max_features:
                argmask = argmask[:max_features]

        def mask_rho(rho: jax.Array | None) -> jax.Array | None:
            if rho is None:
                return None
            if self.only_diag:
                return rho[argmask]
            return rho[argmask, :][:, argmask]

        # self.rho_00 = self.rho_00[argmask, :][:, argmask]
        self.rho_00 = mask_rho(self.rho_00)

        if self.rho_11 is not None:
            # self.rho_11 = self.rho_11[argmask, :][:, argmask]
            self.rho_11 = mask_rho(self.rho_11)
        if self.rho_01 is not None:
            # self.rho_01 = self.rho_01[argmask, :][:, argmask]
            self.rho_01 = mask_rho(self.rho_01)
        if self.rho_10 is not None:
            # self.rho_10 = self.rho_10[argmask, :][:, argmask]
            self.rho_10 = mask_rho(self.rho_10)

        if self.pi_s_0 is not None:
            self.pi_s_0 = self.pi_s_0[argmask]
        if self.pi_s_1 is not None:
            self.pi_s_1 = self.pi_s_1[argmask]

        self.sigma_0 = self.sigma_0[argmask]
        if self.sigma_1 is not None:
            self.sigma_1 = self.sigma_1[argmask]

        if self.d_rho_00 is not None:
            # self.d_rho_00 = self.d_rho_00[argmask, :][:, argmask]
            self.d_rho_00 = mask_rho(self.d_rho_00)

        if self.rho_gen is not None:
            # self.rho_gen = self.rho_gen[argmask, :][:, argmask]
            self.rho_gen = mask_rho(self.rho_gen)

        return argmask

    def decompose(
        self,
        out_dim: int | None = None,
        sparse=True,
        eps: float = 1e-2,
        out_eps: float | None = None,
        glasso_whiten=False,
        verbose=True,
        apply_mask: bool = True,
        # generator=True,
        # shrink=False,
    ):
        W_0 = self.whiten_rho("rho_00", epsilon=eps, apply_mask=apply_mask)

        print(f"{W_0.shape=}")

        if not self.symmetric:
            W_1 = self.whiten_rho("rho_11", epsilon=eps, apply_mask=apply_mask)
        else:
            W_1 = W_0

        if self.only_diag:
            if self.generator:
                T_tilde = jnp.einsum("i,i,i->i", self.rho_gen, W_0, jnp.conj(W_0))
            else:
                assert self.rho_01 is not None
                T_tilde = jnp.einsum("i,i,i->i", self.rho_01, W_1, jnp.conj(W_0))

        else:
            if self.generator:
                # T_tilde = W_1 @ (self.C10 - self.C00) @ W_0.T
                # T_tilde = (W_0 @ (self.C01 - self.C00) @ W_1.T).T

                T_tilde = (W_0 @ (self.rho_gen) @ jnp.conj(W_0).T).T

            else:
                assert self.rho_01 is not None
                T_tilde = (W_0 @ self.rho_01 @ jnp.conj(W_1).T).T

        print(f"{T_tilde.shape=}")

        if out_dim is None:
            out_dim = 10

        if out_dim == -1:
            out_dim = T_tilde.shape[0]

        if out_dim < 20:
            out_dim = 20

        n_modes = out_dim

        k = min(n_modes, T_tilde.shape[0] - 1)
        if not self.only_diag:
            k = min(n_modes, T_tilde.shape[1] - 1)

        if self.only_diag:
            s = T_tilde
            idx = jnp.argsort(s, descending=True)[:n_modes]

            # W_0 = jnp.diag(W_0)[idx, :]
            # W_1 = jnp.diag(W_1)[idx, :]
            # s = s[idx]

            return idx, s

        if n_modes + 1 < T_tilde.shape[0] / 5 and sparse:
            from jax.experimental.sparse.linalg import lobpcg_standard
            from jax.random import PRNGKey, uniform

            x0 = uniform(PRNGKey(0), (T_tilde.shape[0], k))

            print(f"using lobpcg with {n_modes} modes ")

            if self.symmetric:
                # matrix should be psd

                s, U, n_iter = lobpcg_standard(
                    T_tilde.T,
                    x0,
                    m=200,
                )

                print(f"{n_iter=} {s=}")

                # VT = U.T
                V = U

            else:
                l, V, n_iter = lobpcg_standard(
                    T_tilde @ T_tilde.T,
                    x0,
                    m=200,
                )

                # VT = V.T

                print(n_iter)

                s = l ** (1 / 2)
                s_inv: jax.Array = jnp.where(s > 1e-12, 1 / s, 0)  # type: ignore
                U = T_tilde.T @ jnp.conj(VT).T @ jnp.diag(s_inv)

        else:
            if self.symmetric and W_0.shape[0] == W_1.shape[0]:
                print("using eigh")
                s, U = jax.numpy.linalg.eigh(
                    T_tilde.T,
                )
                # VT = U.T
                V = U
            else:
                print("using svd")
                U, s, VT = jax.numpy.linalg.svd(T_tilde.T)
                V = jnp.conj(VT).T

        if self.generator:
            print(f"generator, shiften spectrum {s=}")
            s = jnp.exp(-s)

        idx = jnp.argsort(s, descending=True)
        U = U[:, idx]
        s = s[idx]
        V = V[:, idx]

        W_0 = U.T @ W_0
        W_1 = V.T @ W_1

        if out_eps is not None:
            m = jnp.abs(1 - s) < out_eps

            U = U[:, m]
            VT = VT[m, :]
            s = s[m]

            print(f"{jnp.sum(m)=}")

        W_0 = jnp.einsum("ij,j->ij", W_0, self.sigma_0_inv)
        W_1 = jnp.einsum("ij,j->ij", W_1, self.sigma_1_inv if self.sigma_1_inv is not None else self.sigma_0_inv)

        return W_0, W_1, s

    @property
    def p(self):
        return self.rho_00.shape[0]

    @property
    def n_eff(self):
        return 1.0 / self.w2

    def shrink(self, shrinkage=None, eps_shrink: float = 1e-3):
        if self.only_diag:
            print("skip shrinkage for only_diag")
            return self

        if shrinkage is None:
            shrinkage = self.shrinkage_method
        # https://arxiv.org/pdf/1602.08776.pdf appendix b

        print(f"shrinking with {shrinkage}")

        n = self.n_eff
        assert n is not None, "C00 not provided"

        if shrinkage == "glasso":
            # https://arxiv.org/pdf/2403.02979v1#page=4.31

            from IMLCV.tools.pcglasso import pcglasso

            S = jnp.block([[self.C00, self.C01], [self.C10, self.C11]])

            d_opt, R, W = pcglasso(
                S,
                lmbda=1e-1,
                tol_newton=1e-5,
                tol_glasso=1e-5,
                tol_tot=1e-5,
                max_iter=100,
                max_iter_glasso=100,
                alpha=0.0,
                verbose=3,
                par=False,
            )

            print(f"{d_opt=} {R=} {W=}")

            raise

        # Todo https://papers.nips.cc/paper_files/paper/2014/file/11459f04a46a9e348cdeee6986fcf5f2-Paper.pdf
        # todo: paper Covariance shrinkage for autocorrelated data
        # assert shrinkage in ["RBLW", "OAS", "bidiag", "BC"]

        if self.time_series:
            p = self.C00.shape[0]

            C = jnp.block([[self.C00, self.C01], [self.C10, self.C11]])

            if shrinkage == "bidiag":
                T = jnp.block(
                    [
                        [jnp.diag(jnp.diag(self.C00)), jnp.diag(jnp.diag(self.C01))],
                        [jnp.diag(jnp.diag(self.C10)), jnp.diag(jnp.diag(self.C11))],
                    ]
                )

            elif shrinkage == "diag":
                T = jnp.block(
                    [
                        [jnp.diag(jnp.diag(self.C00)), self.C01 * 0],
                        [self.C10 * 0, jnp.diag(jnp.diag(self.C11))],
                    ]
                )
            elif shrinkage == "constant":
                T = jnp.block(
                    [
                        [jnp.mean(jnp.diag(self.C00)) * jnp.eye(self.C00.shape[0]), self.C01 * 0],
                        [self.C10 * 0, jnp.mean(jnp.diag(self.C11)) * jnp.eye(self.C11.shape[0])],
                    ]
                )
            elif shrinkage == "biconstant":
                T = jnp.block(
                    [
                        [
                            jnp.mean(jnp.diag(self.C00)) * jnp.eye(self.C00.shape[0]),
                            jnp.mean(jnp.diag(self.C01)) * jnp.eye(self.C01.shape[0]),
                        ],
                        [
                            jnp.mean(jnp.diag(self.C10)) * jnp.eye(self.C10.shape[0]),
                            jnp.mean(jnp.diag(self.C11)) * jnp.eye(self.C11.shape[0]),
                        ],
                    ]
                )
            else:
                raise NotImplementedError

        else:
            assert self.C00 is not None

            C = self.C00

            p = C.shape[0]
            if shrinkage == "bidiag":
                T = jnp.diag(jnp.diag(C))
            elif shrinkage == "diag":
                T = jnp.diag(jnp.diag(C))
            elif shrinkage == "constant":
                T = jnp.mean(jnp.diag(C)) * jnp.eye(C.shape[0])
            else:
                raise NotImplementedError

        assert self.w2 is not None
        assert self.w3 is not None

        gamma = 1.0 / (1.0 - self.w2)
        nu = 1 - self.w2 - 2 * self.w3
        eta = self.w2 + self.w2**2 - 2 * self.w3
        eps = self.w2

        C = C / gamma  # with bessel correction, C = gamma S

        # Shrinkage MMSE estimators of covariances beyond the zero-mean and stationary variance assumptions

        def get_C(lambd):
            return (1 - lambd) * C + lambd * T

        def get_lambd(C):
            tr_CS = jnp.trace(C @ C.T) - jnp.sum(jnp.diag(C) * jnp.diag(C))
            tr_CS_diag = jnp.trace(C) * jnp.trace(C) - jnp.sum(jnp.diag(C) * jnp.diag(C))

            return ((gamma * nu + eps - 1) * tr_CS + gamma * eta * tr_CS_diag) / (
                (gamma * nu) * tr_CS + gamma * eta * tr_CS_diag
            )

        lambd = 0

        for i in range(10):
            C_est = get_C(lambd)

            lambd_new = get_lambd(C_est)

            if jnp.abs(lambd - lambd_new) < 1e-6:
                break

            lambd = lambd_new

            print(f"{i}: {lambd=}")

        print(f"{i}: {lambd=}, finished")

        if lambd > 1:
            lambd = 1

        # figure out new C matrix

        def get_rho(i, j):
            X = C_est
            if i == 0:
                X = X[:p, :]
                sigma_0_inv = self.sigma_0_inv
            else:
                X = X[p:, :]
                sigma_0_inv = self.sigma_1_inv

            if j == 0:
                X = X[:, :p]
                sigma_1_inv = self.sigma_0_inv
            else:
                X = X[:, p:]
                sigma_1_inv = self.sigma_1_inv

            # print(f"{X.shape=} {sigma_0_inv.shape=}")

            return jnp.einsum("i,ij,j->ij", sigma_0_inv, X, sigma_1_inv)

        if not self.time_series:
            return Covariances(
                rho_00=get_rho(0, 0),
                pi_s_0=self.pi_s_0,
                pi_s_1=None,
                sigma_0=self.sigma_0,
                sigma_1=None,
                only_diag=self.only_diag,
                rho_01=None,
                rho_11=None,
                rho_10=None,
                rho_gen=self.rho_gen,
                trans_f=self.trans_f,
                trans_g=self.trans_g,
                symmetric=self.symmetric,
                shrinkage_method=shrinkage,
                d_rho_00=self.d_rho_00,
                w2=self.w2,
                w3=self.w3,
            )

        return Covariances(
            rho_00=get_rho(0, 0),
            rho_01=get_rho(0, 1),
            rho_11=get_rho(1, 1),
            pi_s_0=self.pi_s_0,
            pi_s_1=self.pi_s_1,
            rho_10=get_rho(1, 0),
            rho_gen=self.rho_gen,
            sigma_0=self.sigma_0,
            sigma_1=self.sigma_1,
            only_diag=self.only_diag,
            trans_f=self.trans_f,
            trans_g=self.trans_g,
            symmetric=self.symmetric,
            # n=self.n,
            shrinkage_method=shrinkage,
            d_rho_00=self.d_rho_00,
            w2=self.w2,
            w3=self.w3,
        )

    def symmetrize(self):
        if not self.time_series:
            return self

        _rho_00 = self.rho_00
        _rho_01 = self.rho_01
        _rho_10 = self.rho_10
        _rho_11 = self.rho_11
        _pi_s_0 = self.pi_s_0
        _pi_s_1 = self.pi_s_1

        _d_rho_00 = self.d_rho_00

        # make sigma same

        _sigma = jnp.sqrt(self.sigma_0 * self.sigma_1)

        d_sigma_0 = jnp.where(_sigma != 0, self.sigma_0 / _sigma, 0)
        d_sigma_1 = jnp.where(_sigma != 0, self.sigma_1 / _sigma, 0)

        # print(f"{d_sigma_0=}")

        def get_rho(rho, f1, f2) -> jax.Array:
            if self.only_diag:
                return jnp.einsum("i,i->i", rho, jnp.conj(f1) * f2)
            return jnp.einsum("ij,i,j->ij", rho, jnp.conj(f1), f2)

        # _rho_00 = jnp.einsum("ij,i,j->ij", _rho_00, d_sigma_0, d_sigma_0)
        # _rho_01 = jnp.einsum("ij,i,j->ij", _rho_01, d_sigma_0, d_sigma_1)
        # _rho_10 = jnp.einsum("ij,i,j->ij", _rho_10, d_sigma_1, d_sigma_0)
        # _rho_11 = jnp.einsum("ij,i,j->ij", _rho_11, d_sigma_1, d_sigma_1)

        _rho_00 = get_rho(_rho_00, d_sigma_0, d_sigma_0)
        _rho_01 = get_rho(_rho_01, d_sigma_0, d_sigma_1)
        _rho_10 = get_rho(_rho_10, d_sigma_1, d_sigma_0)
        _rho_11 = get_rho(_rho_11, d_sigma_1, d_sigma_1)

        if self.d_rho_00 is not None:
            # _d_rho_00 = jnp.einsum("ij,i,j->ij", _d_rho_00, d_sigma_0, d_sigma_0)
            _d_rho_00 = get_rho(_d_rho_00, d_sigma_0, d_sigma_0)

        # make pi same

        calc_pi = self.pi_s_0 is not None

        def get_outer(pi1, pi2):
            if self.only_diag:
                return jnp.einsum("i,i->i", pi1, pi2)
            return jnp.einsum("i,j->ij", pi1, pi2)

        if calc_pi:
            assert self.pi_s_1 is not None
            _pi_s_0 = jnp.einsum("i,i->i", _pi_s_0, d_sigma_0)
            _pi_s_1 = jnp.einsum("i,i->i", _pi_s_1, d_sigma_1)

            _pi_s = 0.5 * (_pi_s_0 + _pi_s_1)

            # _rho_00 = _rho_00 + jnp.outer(_pi_s_0, _pi_s_0) - jnp.outer(_pi_s, _pi_s)
            # _rho_11 = _rho_11 + jnp.outer(_pi_s_1, _pi_s_1) - jnp.outer(_pi_s, _pi_s)
            # _rho_01 = _rho_01 + jnp.outer(_pi_s_0, _pi_s_1) - jnp.outer(_pi_s, _pi_s)
            # _rho_10 = _rho_10 + jnp.outer(_pi_s_1, _pi_s_0) - jnp.outer(_pi_s, _pi_s)

            _rho_00 = _rho_00 + get_outer(_pi_s_0, _pi_s_0) - get_outer(_pi_s, _pi_s)
            _rho_11 = _rho_11 + get_outer(_pi_s_1, _pi_s_1) - get_outer(_pi_s, _pi_s)
            _rho_01 = _rho_01 + get_outer(_pi_s_0, _pi_s_1) - get_outer(_pi_s, _pi_s)
            _rho_10 = _rho_10 + get_outer(_pi_s_1, _pi_s_0) - get_outer(_pi_s, _pi_s)

        sym_rho_00 = (1 / 2) * (_rho_00 + _rho_11)
        sym_rho_01 = (1 / 2) * (_rho_01 + _rho_10)

        d_sigma = jnp.sqrt(jnp.diag(sym_rho_00)) if not self.only_diag else jnp.sqrt(sym_rho_00)
        d_sigma_inv = jnp.where(d_sigma != 0, 1 / d_sigma, 0)

        # sym_rho_00 = jnp.einsum("ij,i,j->ij", sym_rho_00, d_sigma_inv, d_sigma_inv)
        # sym_rho_01 = jnp.einsum("ij,i,j->ij", sym_rho_01, d_sigma_inv, d_sigma_inv)
        sym_rho_00 = get_rho(sym_rho_00, d_sigma_inv, d_sigma_inv)
        sym_rho_01 = get_rho(sym_rho_01, d_sigma_inv, d_sigma_inv)

        if calc_pi:
            _pi_s = jnp.einsum("i,i->i", _pi_s, d_sigma)
        else:
            _pi_s = None

        _sigma *= d_sigma

        return Covariances(
            rho_00=sym_rho_00,
            rho_11=sym_rho_00,
            rho_01=sym_rho_01,
            rho_10=sym_rho_01,
            rho_gen=self.rho_gen,
            pi_s_0=_pi_s,
            pi_s_1=_pi_s,
            sigma_0=_sigma,
            sigma_1=_sigma,
            only_diag=self.only_diag,
            symmetric=True,
            trans_f=self.trans_f,
            trans_g=self.trans_g,
            # n=self.n,
            shrinkage_method=self.shrinkage_method,
            bc=self.bc,
            d_rho_00=_d_rho_00,
            w2=self.w2,
            w3=self.w3,
        )
