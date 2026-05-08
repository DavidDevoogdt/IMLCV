from __future__ import annotations

import datetime
import os
import shutil
import time
from asyncio import Future
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Sequence

import h5py
import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey, choice, split

from IMLCV.base.bias import Bias, RoundBias
from IMLCV.base.CV import (
    CollectiveVariable,
    CvTrans,
)
from IMLCV.base.CVDiscovery import Transformer
from IMLCV.base.dataobjects import (
    CV,
    EagerTrajectoryInfo,
    FullTrajectoryInfo,
    NeighbourList,
    NeighbourListInfo,
    NeighbourListUpdate,
    StaticMdInfo,
    SystemParams,
    TrajectoryInfo,
)
from IMLCV.base.dataset import DataLoaderOutput
from IMLCV.base.datastructures import Partial_decorator, jit_decorator, vmap_decorator
from IMLCV.base.MdEngine import MDEngine
from IMLCV.base.plot import plot_app, plot_CV, plot_CV_corr
from IMLCV.base.UnitsConstants import (
    angstrom,
    boltzmann,
    kelvin,
    kjmol,
    picosecond,
)
from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.configs.config_general import Executors


@dataclass
class TrajectoryInformation:
    ti: TrajectoryInfo
    cv: int
    round: int
    num: int
    folder: Path

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
        if "equilibration_time" not in dlo_kwargs:
            dlo_kwargs["equilibration_time"] = None

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

            plot_app(
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

            plot_app(
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
        if "equilibration_time" not in dlo_kwargs:
            dlo_kwargs["equilibration_time"] = None

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

            plot_app(
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

            plot_app(
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
        # additional_collective_variable_names: list[str] | None = None,
        # additional_collective_variable_titles: list[str] | None = None,
        plot_biases=True,
        # folder=".",
        ignore_invalid=True,
        only_finished=False,
        get_fes_bias_kwargs={},
        plot_kwargs={},
        duplicate_last_row=False,
    ):
        import jax.numpy as jnp

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
            additional_collective_variable_names = [colvar.cvs_name for colvar in additional_collective_variables]
            additional_collective_variable_titles = [colvar.name for colvar in additional_collective_variables]

            # assert additional_collective_variable_names is not None
            # assert additional_collective_variable_titles is not None

            ncv_add = len(additional_collective_variables)
            # assert len(additional_collective_variable_names) == ncv_add
            # assert len(additional_collective_variable_titles) == ncv_add
        print(f"{ncv=} {ncv_add=}")

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

            assert not _r.tic.invalid or ignore_invalid

            rn = _r.num_vals

            rn.sort()

            for i in rn:
                try:
                    _r_i = self._trajectory_information(c=c, r=r0, i=i)
                except Exception as e:
                    print(f"could not load {c=} {r0=} {i=} {e=}, skipping")
                    continue

                if _r_i.ti.invalid and not ignore_invalid:
                    continue

                if (not _r_i.ti.finished) and only_finished:
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

            plot_app(
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
        plot_app(
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
        vmax_std=5 * kjmol,
        corr=True,
        load_extra_info=True,
        extra_info_title="$\\tau$ (ns)",
        get_fes_bias_kwargs={},
        plot_std=True,
        plot_dens=True,
    ):
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
                equilibration_time=None,
            )

            if load_extra_info:
                timescales = self.extract_timescales(c=j)

            if timescales is None:
                dlo.collective_variable.extra_info = None
            else:
                dlo.collective_variable.extra_info = tuple([f"{float(a):.4f}" for a in timescales])

            dlos.append(dlo)

        if corr:
            f = plot_CV_corr
        else:
            f = plot_CV

        f(
            collective_variable_projection=extra_collective_variable,
            collective_variables=[a.collective_variable for a in dlos],
            ti=[a.ti for a in dlos],
            name=self.path() / f"CV_discovery_v2_{start - 1}_{end}.png",
            vmax=vmax,
            vmax_std=vmax_std,
            macro_chunk=macro_chunk,
            extra_info_title=extra_info_title,
            get_fes_bias_kwargs=get_fes_bias_kwargs,
            plot_std=plot_std,
            plot_dens=plot_dens,
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
            "cannot load permanent bias because permanent cv does not exist"
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

        if i is not None:
            assert (p := self._name_bias(c=c, r=r, i=i)) is not None, f"cannot find individual bias for {c=} {r=} {i=}"

            bias_i = Bias.load(self.folder / p, collective_variable=cv)

        else:
            bias_i = None

        sti = self.get_static_trajectory_info(c=c, r=r)

        return RoundBias.create(bias_r=bias, bias_i=bias_i, bias_scale=sti.bias_scale)

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
        n_skip: int | None = None,
        load_sp=True,
        equilibration_time: float | None = -1,
        use_energies_bincounts: bool = False,
    ) -> DataLoaderOutput:
        if cv is None:
            c = self.cv
        else:
            c = cv

        print(f"{n_max=}")
        # if scale_times:
        #     print("WARNING: scale_times is True, but this is not implemented yet, setting scale_times to False")
        #     scale_times = False

        print(f"{time_correlation_method=}")

        if equilibration_time is not None and n_skip is not None:
            print("WARNING: both equilibration_time and n_skip are not None, setting n_skip to None")
            n_skip = None

        if equilibration_time:
            if equilibration_time == -1:
                print("WARNING: equilibration_time is -1, computing from data")
            else:
                print(f"{equilibration_time/picosecond=}")

        if n_skip is not None:
            print(f"{n_skip=}")

        sti = self._round_information(c=c).tic

        if lag_tau is not None:
            if lag_n is not None:
                print(f"WARNING: both lag_tau and lag_n are not None, using lag_tau")

            dt = sti.timestep * sti.save_step
            n = int(lag_tau / dt)
            print(f"converting lag_tau {lag_tau / picosecond:.3f} ps to lag_n {n} with dt={dt / picosecond} ps")
            lag_n = n

        if new_r_cut == -1:
            new_r_cut = sti.r_cut

        if new_r_cut is not None:
            print(f"{new_r_cut/angstrom=}")

        if not time_series:
            lag_n = 0

        if weight and weighing_method == "WHAM":
            if not get_bias_list:
                get_bias_list = True

        if load_weight:
            get_bias_list = False

        if get_bias_list:
            bias_list: list[Bias] = []

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
        nbins: int | None = None

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

            def _p(symb, n_tot):
                print(symb, end="", flush=True)
                n_tot += 1

                if n_tot % 100 == 0:
                    print("")

                return n_tot

            if equilibration_time == -1:
                eq_arr = []

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

                if min_traj_length is not None:
                    if traj_info.ti.size < min_traj_length:
                        # print(f"skipping trajectyory because it's not long enough ")
                        n_tot = _p("x", n_tot)

                    # else:
                    # print("adding traweights=jectory")

                if equilibration_time is not None:
                    if equilibration_time == -1:
                        if time_correlation_method is None:
                            cutoff = 0

                        else:
                            [cutoff], _ = DataLoaderOutput._correlation_time(
                                cv_0=[_ti.CV],
                                find_cutoff=True,
                                method=time_correlation_method,
                            )

                        eq_arr.append(cutoff)

                        if _ti.size <= cutoff:
                            n_tot = _p("x", n_tot)

                            continue

                        _ti = _ti[cutoff:]

                    else:
                        equilibration_n = int(equilibration_time / (sti_c.timestep * sti_c.save_step))
                        # print(f"{equilibration_n=}")

                        if _ti.size <= equilibration_n:
                            n_tot = _p("x", n_tot)

                            continue

                        _ti = _ti[equilibration_n:]

                elif n_skip is not None:
                    if _ti.size <= n_skip:
                        n_tot = _p("x", n_tot)

                        continue
                    _ti = _ti[n_skip:]

                if lag_n is not None and lag_n > 0:
                    if _ti.size <= lag_n:
                        n_tot = _p("x", n_tot)

                        continue
                    # else:
                    # print("adding traweights=jectory")

                if get_bias_list:
                    bias = traj_info.get_bias()
                    if bias is None:
                        # print(f"could not load bias for {cvi=} {round_info.round=} {traj_info.num=}, skipping bias")
                        n_tot = _p("x", n_tot)

                        continue
                    bias_c.append(bias)

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
                        # print(f"got neighbour list with {nn=} {new_nxyz=}")

                        if not b:
                            print(f"nn needs update {nn=}")

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
                        else:
                            if nn > update_info.num_neighs:
                                print(f"updating nl with {nn=} {new_nxyz=}")
                                update_info = NeighbourListUpdate.create(
                                    num_neighs=int(nn),  # type:ignore
                                    nxyz=update_info.nxyz,
                                )

                n_tot = _p(".", n_tot)

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

            print("")

            if len(ti_c) == 0:
                continue

            if equilibration_time == -1:
                print(f"{jnp.array(eq_arr)=}   {jnp.array(eq_arr)*(sti_c.timestep * sti_c.save_step)/picosecond=}")

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

                nhist = 1
                nbins = 1

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

                # dlo._correlation_time()

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
                        nbins = weight_output.n_bins
                        nhist = weight_output.n_hist

                        print(f"setting bounds to {bounds=} {nhist=}")

                elif weighing_method == "BC":
                    weight_output = dlo.bincount_weight(
                        chunk_size=chunk_size,
                        n_max=n_max,
                        verbose=verbose,
                        recalc_bounds=recalc_bounds,
                        n_hist=n_hist,
                        n_max_lin=n_max_lin,
                        use_energies=use_energies_bincounts,
                        # return_bias=output_FES_bias,
                    )

                    if bounds is None:
                        bounds = weight_output.bounds
                        nbins = weight_output.n_bins
                        nhist = weight_output.n_hist

                else:
                    raise ValueError(f"{weighing_method=} not supported")

                w_c = weight_output.weight
                ps_c = weight_output.p_select
                F_c = weight_output.F
                d_c = weight_output.density
                nb_c = weight_output.n_bin
                neb_c = weight_output.n_eff_bin if weight_output.n_eff_bin is not None else nb_c

                # n_samples_eff_total += weight_output.N_samples_eff

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

                # n_samples_eff_total += sum([_ti.size for _ti in ti_c])
                grid_nums_c = None

                labels_c = [0] * len(ti_c)

                nhist = 1
                nbins = 1

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

            c_list = []
            lag_indices = []
            percentage_list = [] if (weights is not None and scale_times) and (lag_n != 0) else None

            weights_lag = []
            # log_weights_girsanov = []

            if lag_n != 0 and verbose:
                print(f"getting lag indices for {len(ti)} trajectories")

            assert sti_c is not None

            timestep = sti_c.timestep * sti_c.save_step

            @jit_decorator
            @partial(vmap_decorator, in_axes=(0, 0, None, None, None))
            def get_lag_idx(n, u0, sum_u, diff_u: Array, dt):
                log_tau = jnp.log(lag_n * timestep)

                def log_sinhc(x):
                    x = jnp.abs(x)
                    # logsumexp trick

                    x_safe = jnp.where(x < 1e-5, 1e-5, x)

                    t1 = jnp.log(1 - jnp.exp(-2 * x_safe))
                    t2 = x_safe
                    t3 = -jnp.log(x_safe)

                    return jnp.where(x < 1e-5, 0, t1 + t2 + t3)

                log_dt = jnp.log(dt) - sum_u / 2 + u0 + log_sinhc(diff_u / 2)

                log_dt = jnp.where(jnp.arange(log_dt.shape[0]) <= n, -jnp.inf, log_dt)
                log_t = jax.lax.cumlogsumexp(log_dt)

                # jax.debug.print("log_dt: {log_dt}", log_dt=log_dt)

                _index_k = jnp.argmin(jnp.abs(log_tau - log_t))
                _index_k = jnp.where(_index_k <= n, n, _index_k)

                # first larger index
                index_0 = jnp.where(log_t[_index_k] >= log_tau, _index_k - 1, _index_k)  # type: ignore
                index_1 = jnp.min(jnp.array([index_0 + 1, log_t.shape[0] - 1]))  # index after

                indices = jnp.array([index_0, index_1])

                log_values = log_t[indices]

                values = jnp.exp(log_values)

                b = (index_0 <= (log_t.shape[0] - 2)) * (jnp.isfinite(log_values[1]))
                b = b * (values[1] > values[0])

                # make sure that percentage is not if int is zero

                percentage = jnp.where(
                    jnp.abs(log_values[1] - log_values[0]) < 1e-5,
                    0.0,
                    (jnp.exp(log_tau) - values[0]) / (values[1] - values[0]),
                )

                return indices[0], indices[1], b, percentage

            for n, ti_i in enumerate(ti):
                if lag_n != 0:
                    idx = jnp.arange(c)
                    if scale_times:
                        scales = weights[n]

                        # w = e^(U)
                        _u = jnp.log(scales)

                        _u = jnp.where(jnp.isfinite(_u), _u, 1e10)

                        _su = _u[1:] + _u[:-1]
                        _du = _u[1:] - _u[:-1]

                        assert ti_i.t is not None
                        dt = ti_i.t[1:] - ti_i.t[:-1]

                        dt = jnp.where(dt > 0, dt, timestep)  # replace non-finite dt with a small value

                        # print(f"{jnp.isnan(_u).any()=}, {jnp.isinf(_u).any()=}, {_u.shape=}")

                        with jax.debug_nans():
                            lag_indices_max, lag_indices_max2, bools, p = get_lag_idx(
                                jnp.arange(scales.shape[0]), _u, _su, _du, dt
                            )

                        c = jnp.sum(bools)

                        if c == 0:
                            print(f"WARNING: no valid lag indices found for trajectory {n=},  skipping lagged data")

                            continue

                        assert percentage_list is not None

                        percentage_list.append(p[bools])
                        lag_idx = lag_indices_max[bools]

                        weights_lag.append(jnp.where(bools, scales[lag_indices_max], 0))

                        # ti[n]._t = integral

                    else:
                        c = ti_i.size - lag_n
                        if c <= 0:
                            print(
                                f"WARNING: lag_n {lag_n} for {n=} is larger than trajectory length {ti_i.size},  appending zeros"
                            )

                            weights_lag.append(jnp.zeros_like(weights[n]))
                            # log_weights_girsanov.append(jnp.zeros_like(weights[n]))
                            lag_idx = jnp.arange(ti_i.size)  # append all indices, but they will be masked out later

                        else:
                            i
                            lag_idx = idx + lag_n
                            weights_lag.append(weights[n][lag_idx])

                    # if ti_i.A is not None:
                    #     log_weights_girsanov.append(ti_i.A[lag_idx] - ti_i.A[idx])
                    # else:
                    #     log_weights_girsanov.append(jnp.zeros_like(weights[n][lag_idx]))

                    c_list.append(c)
                    lag_indices.append(lag_idx)

                else:
                    c = ti_i.size

                    lag_indices.append(jnp.arange(c))
                    c_list.append(c)

                    weights_lag.append(weights[n])
                    # log_weights_girsanov.append(jnp.zeros_like(weights[n]))

            # print(f"{log_weights_girsanov=}")

            # norm = jnp.max(jnp.hstack(log_weights_girsanov))
            # log_weights_girsanov = [wg - norm for wg in log_weights_girsanov]

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

            # if out > n_samples_eff_total:
            #     print(f"not enough effective datapoints, returning {n_samples_eff_total} data points instead of {out}")
            #     out = n_samples_eff_total

            if out > total:
                print(f"not enough total datapoints, returning {total} data points instead of {out}")
                out = total

            n_out_per_bin = out / nbins

            print(f"{n_out_per_bin=} {nhist=} {nbins=}")

            ###################

            if verbose:
                print(f"total data points {total}, selecting {out}")

            def choose(
                key,
                weight: list[Array],
                weight_wall: list[Array],
                weight_lag: list[Array],
                # log_weight_girsanov: list[Array],
                ps: list[Array],
                F: list[Array],
                nb: list[Array],
                neb: list[Array],
                density: list[Array],
                grid_nums: list[Array] | None,
                out: int,
                sigma: list[Array] | None = None,
            ):
                try:
                    key, key_return = split(key, 2)

                    print(f"new choice {out=}")

                    nbstack = jnp.hstack(nb)
                    psstack = jnp.hstack(ps)
                    # log_wgstack = jnp.hstack(log_weight_girsanov)

                    log_nb = jnp.log(nbstack)
                    log_ps = jnp.log(psstack)
                    log_ww = jnp.hstack(weight_wall)

                    log_ps_eff = log_ps + log_ww - log_nb  # + log_wgstack
                    log_ps_eff = jnp.where(jnp.isfinite(log_ps_eff), log_ps_eff, -jnp.inf)
                    log_ps_eff = jnp.where(jnp.isnan(log_ps_eff), -jnp.inf, log_ps_eff)

                    log_ps_eff -= jnp.nanmax(log_ps_eff)
                    ps_stack = jnp.exp(log_ps_eff)

                    # print(f"{ps_stack=}")

                    # assert jnp.isfinite(ps_stack).all(), "rho_stack contains non-finite values"

                    mask = ps_stack > 0.0

                    # print(f"{ps_stack=} {jnp.hstack(weight)=} {jnp.hstack(weight_lag)=}")

                    # print(f"number of valid data points after masking {jnp.sum(mask)=} out of {total=}")

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
                    out_simga = []

                    beta = 1.0 / (boltzmann * sti_c.T)

                    # average number of points per bin

                    print(f"{len(F)=} {len(weight)=} {len(density)=} {len(nb)=}")

                    for i, (
                        Fi,
                        wi,
                        di,
                        nbi,
                        # wgi,
                    ) in enumerate(
                        zip(
                            F,
                            weight,
                            density,
                            nb,
                            # log_weight_girsanov,
                        )
                    ):
                        do = jnp.where(
                            jnp.isfinite(Fi) & jnp.isfinite(wi) & jnp.isfinite(di) & (nbi != 0),
                            jnp.exp(beta * Fi + jnp.log(wi) + jnp.log(di)),
                            0.0,
                        )

                        density_out.append(do)
                        weight_out.append(jnp.exp(-beta * Fi))

                        if calculate_std:
                            sigma_new = sigma[i] * jnp.sqrt(ps[i] / nbi) * jnp.sqrt(n_out_per_bin)

                            out_simga.append(sigma_new)  # type: ignore

                    return key_return, indices, weight_out, density_out, nums[:, indices], weight, out_simga
                except Exception as e:
                    for x in [weight, weight_wall, weight_lag, ps, F, nb, neb, density]:
                        print(f"{[xi.shape for xi in x]=}")

                    raise e

            def remove_lag(w, c):
                w = w[:c]
                return w

            out_indices = []
            out_labels = []

            n_list = []

            _w: list[Array] = []
            _ww: list[Array] = []
            _wl: list[Array] = []
            _ps: list[Array] = []
            _F: list[Array] = []
            _nb: list[Array] = []
            _neb: list[Array] = []
            _d: list[Array] = []
            _w_std: list[Array] | None = [] if calculate_std else None
            # _wg: list[Array] = []

            _grid_nums: list[Array] | None = [] if grid_nums is not None else None

            for n, (
                w_i,
                ww_i,
                wl_i,
                ps_i,
                F_i,
                d_i,
                nb_i,
                neb_i,
                c_i,
                # wg_i,
            ) in enumerate(
                zip(
                    weights,
                    weight_wall,
                    weights_lag,
                    p_select,
                    F,
                    density,
                    n_bin,
                    n_eff_bin,
                    c_list,
                    # log_weights_girsanov,
                )
            ):
                _w.append(remove_lag(w_i, c_i))
                _ww.append(remove_lag(ww_i, c_i))
                _wl.append(remove_lag(wl_i, c_i))
                _nb.append(remove_lag(nb_i, c_i))
                _d.append(remove_lag(d_i, c_i))
                _ps.append(remove_lag(ps_i, c_i))
                _F.append(remove_lag(F_i, c_i))
                _neb.append(remove_lag(neb_i, c_i))
                # _wg.append(remove_lag(wg_i, c_i))

                if calculate_std:
                    assert weights_std is not None
                    _w_std.append(remove_lag(weights_std[n], c_i))  # type: ignore

                if grid_nums is not None:
                    _grid_nums.append(remove_lag(grid_nums[n], c_i))  # type: ignore

            key, indices, out_reweights, out_rhos, nums_full, out_dw, out_sigma = choose(
                key=key,
                weight=_w,
                weight_wall=_ww,
                weight_lag=_wl,
                # log_weight_girsanov=_wg,
                ps=_ps,
                F=_F,
                nb=_nb,
                neb=_neb,
                density=_d,
                grid_nums=_grid_nums if _grid_nums is not None else None,
                out=int(out),
                sigma=_w_std,
            )

            print(f"selected {len(indices)} {out=} of {total} data points {len(out_reweights)=} {len(out_rhos)=}")

            count = 0

            for n, n_i in enumerate(c_list):
                indices_full = indices[jnp.logical_and(count <= indices, indices < count + n_i)]
                index = indices_full - count

                # index = nums_full[1, nums_full[0, :] == n]

                if len(index) == 0:
                    count += n_i
                    # print(f"no data points selected for trajectory {n=}, skipping")
                    continue

                out_labels.append(labels[n])
                out_indices.append(index)
                n_list.append(n)

                count += n_i

            print(f"{n_list=}")

            ###################
            # storing data    #
            ###################

            out_ti: list[TrajectoryInfo] = []

            # out_weights_std: list[Array] | None = [] if calculate_std else None

            # if time_series:
            #     out_dynamic_weights: list[Array] = []

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
                    ti_n.sigma = out_sigma[n][indices_n]

                if time_series:
                    idx_t = lag_indices[n][indices_n]
                    idx_t_p = idx_t + 1

                    # print(f"{ti_n.shape=} {idx_t.shape=} ")

                    if percentage_list is None:
                        ti_t_n: TrajectoryInfo = ti[n][idx_t]

                        ti_t_n.w = out_reweights[n][idx_t]
                        ti_t_n.rho = out_rhos[n][idx_t]

                        ti_t_n.w_dyn = jnp.exp(ti_t_n.A - ti_n.A)

                        # out_dynamic_weights.append(jnp.sqrt(out_dw[n][idx_t] / out_dw[n][indices_n]))

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

                        ti_t_n.w_dyn = jnp.exp(interp(ti_t_n.A[idx_t], ti_t_n.A[idx_t_p], percentage) - ti_n.A)

                    ti_n.w_dyn = ti_t_n.w_dyn
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

            # def norm_rho(w: list[tuple[Array]]):
            #     w_log = [jnp.sum(jnp.array([jnp.log(x) for x in w_i])) for w_i in w]

            #     z = jnp.hstack(w_log)
            #     z_max = jnp.max(z)
            #     norm = jnp.log(jnp.sum(jnp.exp(z - z_max))) + z_max

            #     print(f"normalizing rho with norm {norm=}")

            #     rho = [jnp.exp(jnp.log(t[-1]) - norm) for t in w]

            #     return rho

            # if out_ti[0].w_dyn is not None:
            #     out = norm_rho([(out_ti_i.w, out_ti_i.rho, out_ti_i.w_dyn) for out_ti_i in out_ti])
            #     for i in range(len(out_ti)):
            #         out_ti[i].w_dyn = out[i]

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

            print(f"{out_nl.info=} {out_nl.update=}")
        else:
            out_nl = None

        if time_series:
            out_nl_t = out_nl

            if lag_tau is not None:
                tau = lag_tau
            else:
                tau = lag_n * timestep

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
            # _weights_std=out_weights_std,
            bounds=bounds,
            n_hist=nhist,
        )

        if time_series:
            dlo_kwargs.update(
                nl_t=out_nl_t,  # type:ignore
                tau=tau,  # type:ignore
                # dynamic_weights=out_dynamic_weights,
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

    def data_loader_cv(self, c: int):
        pass

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
            i_list = self._i_vals(c=c, r=r)
        else:
            i_list = [i]

        for i in i_list:
            p = self.path(c=c, r=r, i=i) / "trajectory_info.h5"

            if not p.exists():
                print(f"cannot invalidate data for {c=} {r=} {i=} because traj file does not exist")
                return

            n = "_invalid"

            try:
                with h5py.File(p, "r+") as hf:
                    hf.attrs[n] = False

                print(f"validated {c=} {r=} {i=}")
            except Exception:
                print(f"could not validate {c=} {r=} {i=}, writing file")

    def validate_round(self, c: int | None = None, r: int | None = None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        p = self.path(c=c, r=r) / "static_trajectory_info.h5"

        if not p.exists():
            print(f"cannot validate data for {c=} {r=}  because traj file does not exist")
            return

        n = "invalid"

        try:
            with h5py.File(p, "r+") as hf:
                hf.attrs[n] = False
        except Exception:
            print(f"could not validate {c=} {r=}, writing file")

    def finish_data(self, c: int | None = None, r: int | None = None, i: int | None = None):
        if c is None:
            c = self.cv

        if r is None:
            r = self.get_round(c=c)

        if i is None:
            i_list = self._i_vals(c=c, r=r)
        else:
            i_list = [i]

        print(f"finishing data for {c=} {r=} {i_list=}")

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
        # use_common_bias=True,
        dT=0 * kelvin,
        add_umbrella=True,
    ):
        if cv_round is None:
            cv_round = self.cv

        r = self.get_round(c=cv_round)

        # if use_common_bias:
        #     common_bias_name = self._name_bias(c=cv_round, r=r)
        #     assert common_bias_name is not None
        #     common_bias_name = self.full_path(common_bias_name)
        # else:
        #     common_bias_name = None

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
            # common_bias_name=common_bias_name,
            r=r,
            macro_chunk=macro_chunk,
            lag_n=lag_n,
            dT=dT,
            use_energies=True,
            add_umbrella=add_umbrella,
        ).result()

        # from parsl.dataflow.dflow import AppFuture

        tasks: list[tuple[int, Future]] | None = None
        plot_tasks = []

        from IMLCV.configs.config_general import RESOURCES_DICT

        resources = RESOURCES_DICT[Executors.reference.value]

        print(f"using resources {resources} for md")

        for i, (spi, traj_name, b_name, b_name_new, path_name) in enumerate(zip(*out)):
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
        # common_bias_name: str | None,
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
        use_energies: bool = True,
        dT=50 * kelvin,
        f_scale=0.8,
        add_umbrella=True,
        # divide_by_histogram=True,
    ):
        sps = []
        # bs = []
        traj_names = []
        b_names = []
        b_names_new = []
        path_names = []

        sp0_provided = sp0 is not None

        common_bias = rounds.get_bias(c=cv_round, r=r)
        wall_bias = rounds.get_permanent_bias()

        if wall_bias is not None:
            print("using wall bias to get inital points")

        # print(f"{common_bias.collective_variable=}")

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
                weight=False,
                # weighing_method="BC",
                use_energies_bincounts=use_energies,  # select according to e^{-beta E_pot} and uniform accross CV space
                n_max=1e4,
                n_max_lin=50,
                # T_scale=T_scale,
                time_series=False,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                verbose=True,
                # lag_n=lag_n,
                recalc_bounds=False,
                equilibration_time=None,
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

            # w = jnp.hstack(dlo_data._weights) * jnp.exp(-e_wall * beta) * jnp.hstack(dlo_data._rho)

            # if use_energies:
            #     # ener_stack = (
            #     #     f_scale
            #     #     * common_bias.compute_from_cv(
            #     #         cvs=cv_stack,
            #     #         chunk_size=chunk_size,
            #     #     )[0]
            #     # )  # -F

            #     ener_stack = jnp.hstack([x.e_pot for x in dlo_data.ti])  # type: ignore

            #     if dlo_data.sti.P is not None:
            #         ener_stack += jnp.hstack([x.sp.volume() for x in dlo_data.ti]) * dlo_data.sti.P  # type: ignore

            #         print("using energies")

            # get  weights, and correct for ground state bias.
            # this corrects for the fact that the samples are not uniformly distributed

        else:
            assert sp0.shape[0] == len(biases), (
                f"The number of initials cvs provided {sp0.shape[0]} does not correspond to the number of biases {len(biases)}"
            )

            # ener_stack = None

            T = rounds.static_trajectory_information().T

        if isinstance(KEY, int):
            KEY = jax.random.PRNGKey(KEY)

        for i, bias in enumerate(biases):
            path_name = rounds.path(c=cv_round, r=r, i=i)
            if not os.path.exists(path_name):
                os.mkdir(path_name)

            if add_umbrella:
                b = bias
            else:
                from IMLCV.base.bias import NoneBias

                b = NoneBias(collective_variable=bias.collective_variable)

            b_name = path_name / "bias.json"
            b_name_new = path_name / "bias_new.json"
            b.save(b_name)

            traj_name = path_name / "trajectory_info.h5"

            if not sp0_provided:
                # reweigh data points according to new bias

                ener = bias.compute_from_cv(cvs=cv_stack, chunk_size=chunk_size)[0]

                # if use_energies:
                #     if ener_stack is not None:
                #         # print(f"using energies")
                #         ener += ener_stack

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
            # bs.append(b)
            traj_names.append(traj_name)
            b_names.append(b_name)
            b_names_new.append(b_name_new)
            path_names.append(path_name)

        return sps, traj_names, b_names, b_names_new, path_names

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
        equilibration_time=-1,
        n_max_lin: int = 100,
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

        if "vmax" in dlo_kwargs:
            vmax = dlo_kwargs.pop("vmax")

        if "n_max" in dlo_kwargs:
            n_max = dlo_kwargs.pop("n_max")

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
            n_max_lin=n_max_lin,
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
        equilibration_time=-1,
        n_max_lin: int = 100,
    ):
        if dlo is None:
            # plot_folder = rounds.path(c=cv_round_to)
            # cv_titles = [f"{cv_round_from}", f"{cv_round_to}"]

            dlo = rounds.data_loader(
                **dlo_kwargs,
                macro_chunk=macro_chunk,
                verbose=verbose,
                n_max=n_max,
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
            n_max_lin=n_max_lin,
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
            plot_app(
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
                sigma=traj_info.sigma,
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
