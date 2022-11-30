from __future__ import annotations

import os
import shutil
from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import ase
import h5py
import jax
import jax.numpy as jnp
from filelock import FileLock
from molmod.constants import boltzmann
from parsl.data_provider.files import File

from IMLCV import KEY
from IMLCV.base.bias import Bias, CompositeBias
from IMLCV.base.CV import SystemParams
from IMLCV.base.MdEngine import MDEngine, StaticTrajectoryInfo, TrajectoryInfo
from IMLCV.external.parsl_conf.bash_app_python import bash_app_python


@dataclass
class TrajectoryInformation:
    ti: TrajectoryInfo
    valid: bool
    round: int
    num: int
    folder: Path
    name_bias: str | None = None

    def get_bias(self) -> Bias:
        assert self.name_bias is not None
        return Bias.load(self.folder / self.name_bias)


@dataclass
class RoundInformation:
    round: int
    valid: bool
    num: int
    tic: StaticTrajectoryInfo
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
    def __init__(self, folder: str | Path = "output", copy=True) -> None:
        """
        this class saves all relevant info in a hdf5 container. It is build as follows:
        root
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
        """

        folder = Path(folder)

        if folder.exists():
            if copy:
                folder = Rounds._get_copy(folder=Path(folder))
        else:
            folder.mkdir(parents=True)

        self.folder = folder
        self.lock = FileLock(self.h5filelock_name)

        # load h5file or create from files if corrupt/non existent
        try:
            with self.lock:
                self.h5file = h5py.File(self.h5file_name, mode="r+")
        except:

            self._make_file()
            self.recover()

    ######################################
    #             IO                     #
    ######################################

    @staticmethod
    def _get_copy(folder: Path) -> Path:

        i = 0
        while True:
            p = folder.parent / (f"{folder.name}_{i:0>3}")
            if p.exists():
                i += 1
            else:
                shutil.copytree(folder, p)
                break

        return p

    def _make_file(self):
        if not Path(self.h5file_name).exists():
            # create the file
            with self.lock:
                with h5py.File(self.h5file_name, mode="w-", swmr=True):
                    pass

        self.h5file = h5py.File(self.h5file_name, mode="r+")

    @property
    def h5file_name(self):
        return self.full_path(self.path() / "results.h5")

    @property
    def h5filelock_name(self):
        return self.full_path(self.path() / "results.h5.lock")

    def full_path(self, name) -> str:
        return str(self.folder / Path(name))

    def rel_path(self, name):
        return str(Path(name).relative_to(self.folder))

    def path(self, r=None, i=None) -> Path:
        p = Path(self.folder)
        if r is not None:
            p /= f"round_{r}"
            if i is not None:
                p /= f"md_{i}"

        return p

    @staticmethod
    def load(folder: str | Path, copy=False):

        return Rounds(folder=folder, copy=copy)

    def write_xyz(self, r: int | None = None, num: int = 1, repeat=None):

        from ase.io.extxyz import write_extxyz

        for i, (atoms, round, trajejctory) in enumerate(
            self.iter_ase_atoms(r=r, num=num)
        ):
            with open(
                self.path(r=round.round, i=trajejctory.num) / "trajectory.xyz",
                mode="w",
            ) as f:
                if repeat is not None:
                    atoms = [a.repeat(repeat) for a in atoms]

                write_extxyz(f, atoms)

    ######################################
    #             storage                #
    ######################################

    def recover(self):

        for round_r in self.path().glob("round_*"):
            r = round_r.parts[-1][6:]

            f = self.h5file

            if r not in f.keys():

                attr = {}
                sti = StaticTrajectoryInfo.load(
                    filename=self.path(r=r) / "static_trajectory_info.h5"
                )

                if (p := (self.path(r) / "bias")).exists():
                    attr["name_bias"] = self.rel_path(p)

                if (p := (self.path(r) / "engine")).exists():
                    attr["name_bias"] = self.rel_path(p)

                self.add_round(attr=attr, stic=sti, r=r)

            for md_i in round_r.glob("md_*"):

                i = md_i.parts[-1][3:]

                if i in f[f"{r}"].keys():
                    continue

                tin = md_i / "trajectory_info.h5"

                if not (tin).exists():
                    if not (md_i / "bash_app_trajectory_info.h5").exists():
                        continue
                    else:
                        tin = md_i / "bash_app_trajectory_info.h5"

                traj = TrajectoryInfo.load(tin)
                self.add_md(i=i, r=r, d=traj, attrs=None)

    def add_md(self, i, d: TrajectoryInfo, attrs=None, bias: str | None = None, r=None):
        if r is None:
            r = self.round

        with self.lock:

            f = self.h5file
            if f"{i}" not in f[f"{r}"]:
                f.create_group(f"{r}/{i}")

            d.save(filename=self.path(r=r, i=i) / "trajectory_info.h5")

            if attrs is None:
                attrs = {}

            if bias is not None:
                attrs["name_bias"] = bias

            if attrs is not None:
                for key, val in attrs.items():
                    if val is not None:
                        f[f"{r}/{i}"].attrs[key] = val

            f[f"{r}/{i}"].attrs["valid"] = True
            f[f"{r}"].attrs["num"] += 1

            f.flush()

    def add_round(self, stic: StaticTrajectoryInfo, r=None, attr=None):

        if r is None:
            r = self.round + 1

        if attr is None:
            attr = {}

        dir = self.path(r=r)
        if not dir.exists():
            dir.mkdir(parents=True)

        with self.lock:
            f = self.h5file
            f.create_group(f"{r}")

            for key in attr:
                if attr[key] is not None:
                    f[f"{r}"].attrs[key] = attr[key]

            stic.save(self.path(r=r) / "static_trajectory_info.h5")

            f[f"{r}"].attrs["num"] = 0
            f[f"{r}"].attrs["valid"] = True

            f.flush()

    def add_round_from_md(self, md: MDEngine):

        r = self.round + 1

        directory = self.path(r=r)
        if not os.path.isdir(directory):
            os.mkdir(directory)

        name_md = directory / "engine"
        name_bias = directory / "bias"
        md.save(name_md)
        md.bias.save(name_bias)
        md.bias.collective_variable

        attr = {}

        attr["name_md"] = self.rel_path(name_md)
        attr["name_bias"] = self.rel_path(name_bias)

        self.add_round(attr=attr, stic=md.static_trajectory_info, r=r)

    ######################################
    #             retreiveal             #
    ######################################

    def iter(
        self, start=None, stop=None, num=3
    ) -> Iterable[tuple[RoundInformation, TrajectoryInformation]]:
        if stop is None:
            stop = self.round

        if start is None:
            start = 0

        for r0 in range(max(stop - (num - 1), start), stop + 1):

            _r = self.round_information(r=r0)

            if not _r.valid:
                continue

            for i in range(_r.num):
                _r_i = self.get_trajectory_information(r=r0, i=i)

                if not _r_i.valid:
                    continue

                yield _r, _r_i

    def iter_ase_atoms(self, r: int | None = None, num: int = 3):

        from molmod import angstrom

        for round, trajejctory in self.iter(stop=r, num=num):

            traj = trajejctory.ti

            pos_A = traj.positions / angstrom
            pbc = traj.cell is not None
            if pbc:
                cell_A = traj.cell / angstrom
                # vol_A3 = traj.volume / angstrom**3
                # vtens_eV = traj.vtens / electronvolt
                # stresses_eVA3 = vtens_eV / vol_A3

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
                # atoms.info["stress"] = stresses_eVA3
            else:
                atoms = [
                    ase.Atoms(
                        numbers=round.tic.atomic_numbers,
                        masses=round.tic.masses,
                        positions=positions,
                    )
                    for positions in pos_A
                ]

            # if traj.gpos is not None:
            #     atoms.arrays["forces"] = -traj.gpos * angstrom / electronvolt
            # if traj.e_pot is not None:
            #     atoms.info["energy"] = traj.e_pot / electronvolt

            yield atoms, round, trajejctory

    def get_trajectory_information(self, r: int, i: int) -> TrajectoryInformation:

        with self.lock:
            f = self.h5file
            d = f[f"{r}/{i}"]

            ti = TrajectoryInfo.load(self.path(r=r, i=i) / "trajectory_info.h5")
            r_attr = {key: d.attrs[key] for key in d.attrs}

            f.flush()

        return TrajectoryInformation(
            ti=ti, **r_attr, round=r, num=i, folder=self.folder
        )

    def round_information(self, r: int) -> RoundInformation:

        with self.lock:
            f = self.h5file

            folder = self.path(r=r)

            stic = StaticTrajectoryInfo.load(folder / "static_trajectory_info.h5")

            d = f[f"{r}"].attrs
            r_attr = {key: d[key] for key in d}

        return RoundInformation(
            round=int(r),
            folder=folder,
            tic=stic,
            **r_attr,
        )

    ######################################
    #           Properties               #
    ######################################
    def _set_attr(self, name, value, r=None, i=None):
        with self.lock:
            f = self.h5file
            if r is not None:
                f2 = f[f"{r}"]
            else:
                f2 = f

            if i is not None:
                assert r is not None, "also provide round"
                f2 = f[f"/{i}"]

            f2.attrs[name] = value

            f2.flush()

    def _get_attr(self, name, r=None, i=None):

        with self.lock:
            f = self.h5file
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
        return self.round_information(r=self.round).tic.T

    @property
    def P(self):
        return self.round_information(r=self.round).tic.P

    @property
    def round(self):
        with self.lock:
            f = self.h5file
            return len(f.keys()) - 1

    def n(self, r=None):
        if r is None:
            r = self.round
        return self.round_information(r=self.round).num

    def invalidate_data(self, r=None, i=None):
        if r is None:
            r = self.round

        self._set_attr(name="valid", value=False, r=r, i=i)

    def _validate(self, md: MDEngine):

        pass

    def get_bias(self, r=None, i=None) -> Bias:
        if r is None:
            r = self.round

        bn = self._get_attr("name_bias", r=r, i=i)
        return Bias.load(self.full_path(bn))

    def get_engine(self, r=None) -> MDEngine:
        if r is None:
            r = self.round

        name = self._get_attr("name_bias", r=r)
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
    ):
        with self.lock:
            f = self.h5file
            common_bias_name = self.full_path(f[f"{self.round}"].attrs["name_bias"])
            common_md_name = self.full_path(f[f"{self.round}"].attrs["name_md"])
        from parsl.dataflow.dflow import AppFuture

        tasks: list[tuple[int, AppFuture]] | None = None
        plot_tasks = []
        md_engine = MDEngine.load(common_md_name)

        sp: SystemParams | None = None

        for _, t in self.iter(num=2):
            sp0 = t.ti.sp
            if sp is None:
                sp = sp0
            else:
                sp += sp0
        if sp is None:  # no provious round available
            sp = md_engine.sp
        sp = sp.batch()

        for i, bias in enumerate(biases):

            temp_name = self.path(r=self.round, i=i)
            if not os.path.exists(temp_name):
                os.mkdir(temp_name)

            # construct bias
            if bias is None:
                b = Bias.load(common_bias_name)
            else:
                b = CompositeBias([Bias.load(common_bias_name), bias])

            b_name = self.full_path(temp_name / "bias")
            b_name_new = self.full_path(temp_name / "bias_new")
            b.save(b_name)

            traj_name = self.full_path(temp_name / "trajectory_info.h5")

            @bash_app_python
            def run(
                steps: int,
                sp: SystemParams | None,
                inputs=[],
                outputs=[],
            ):

                bias = Bias.load(inputs[1].filepath)

                print("loading umbrella with sp = {sp}")

                kwargs = dict(
                    bias=bias,
                    trajectory_file=outputs[1].filepath,
                )

                if sp is not None:
                    kwargs["sp"] = sp

                md = MDEngine.load(inputs[0].filepath, **kwargs)
                md.run(steps)
                bias.save(outputs[0].filepath)
                d = md.get_trajectory()
                return d

            @bash_app_python
            def plot_app(
                st: StaticTrajectoryInfo, traj: TrajectoryInfo, inputs=[], outputs=[]
            ):

                bias = Bias.load(inputs[0].filepath)
                sp = traj.sp
                if st.equilibration is not None:
                    if traj.t is not None:
                        sp = sp[traj.t > st.equilibration]

                cvs, _ = bias.collective_variable.compute_cv(sp=sp)
                bias.plot(name=outputs[0].filepath, traj=[cvs])

            if bias is not None and sp.shape[0] > 1:
                bs = bias.compute_from_system_params(sp).energy
                probs = jnp.exp(-bs / (md_engine.static_trajectory_info.T * boltzmann))
                probs = probs / jnp.linalg.norm(probs)
            else:
                probs = None

            index = jax.random.choice(
                a=sp.shape[0],
                key=KEY,
                p=probs,
            )

            future = run(
                sp=sp[index],
                inputs=[File(common_md_name), File(b_name)],
                outputs=[File(b_name_new), File(traj_name)],
                steps=int(steps),
                stdout=self.full_path(temp_name / "md.stdout"),
                stderr=self.full_path(temp_name / "md.stderr"),
            )

            if plot:

                plot_file = self.full_path(temp_name / "plot.pdf")

                plot_fut = plot_app(
                    traj=future,
                    st=md_engine.static_trajectory_info,
                    inputs=[future.outputs[0]],
                    outputs=[File(plot_file)],
                    stdout=self.full_path(temp_name / "plot.stdout"),
                    stderr=self.full_path(temp_name / "plot.stderr"),
                )

                plot_tasks.append(plot_fut)

            if tasks is None:
                tasks = [(i, future)]
            else:
                tasks.append((i, future))

        assert tasks is not None

        # wait for tasks to finish
        for i, future in tasks:
            d = future.result()
            self.add_md(d=d, bias=future.outputs[0].filename, i=i)

        # wait for plots to finish
        if plot:
            for future in plot_tasks:
                d = future.result()
