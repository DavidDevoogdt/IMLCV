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
import numpy as np
from filelock import FileLock
from jax import Array
from molmod.constants import boltzmann
from parsl.data_provider.files import File

from configs.bash_app_python import bash_app_python
from IMLCV.base.bias import Bias, CompositeBias
from IMLCV.base.CV import SystemParams
from IMLCV.base.MdEngine import MDEngine, StaticTrajectoryInfo, TrajectoryInfo

# todo: invaildate with files instead of db tha gets deleted


@dataclass
class TrajectoryInformation:
    ti: TrajectoryInfo
    round: int
    num: int
    folder: Path
    name_bias: str | None = None
    valid: bool = True

    def get_bias(self) -> Bias:
        assert self.name_bias is not None
        return Bias.load(self.folder / self.name_bias)


@dataclass
class RoundInformation:
    round: int
    valid: bool
    num: int
    num_vals: Array
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
    def __init__(
        self, folder: str | Path = "output", copy=True, new_folder=True
    ) -> None:
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

        # create the file
        # with self.lock:
        #     with h5py.File(self.h5file_name, mode="w-"):
        #         pass

        self.h5file = h5py.File(self.h5file_name, mode="w")

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
                    attr["name_md"] = self.rel_path(p)

                if (p := (self.path(r) / "invalid")).exists():
                    attr["valid"] = False

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

                bias = None
                if (p := (md_i / "new_bias")).exists():
                    bias = self.rel_path(p)
                elif (p := (md_i / "bias")).exists():
                    bias = self.rel_path(p)

                attr = None

                if (p := (md_i / "invalid")).exists():
                    attr["valid"] = False

                traj = TrajectoryInfo.load(tin)
                self.add_md(i=i, r=r, d=traj, bias=bias, attrs=attr)

    def add_md(self, i, d: TrajectoryInfo, attrs=None, bias: str | None = None, r=None):
        if r is None:
            r = self.round

        with self.lock:
            f = self.h5file
            if f"{i}" not in f[f"{r}"]:
                f.create_group(f"{r}/{i}")

            # check if in recover mode
            if not (p := self.path(r=r, i=i) / "trajectory_info.h5").exists():
                d.save(filename=p)

            if attrs is None:
                attrs = {}

            if "valid" in attrs:
                attrs["valid"] = True

            if bias is not None:
                attrs["name_bias"] = bias

            if attrs is not None:
                for key, val in attrs.items():
                    if val is not None:
                        f[f"{r}/{i}"].attrs[key] = val

            f[f"{r}"].attrs["num"] += 1
            f[f"{r}"].attrs["num_vals"] = np.append(f[f"{r}"].attrs["num_vals"], int(i))

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
            f[f"{r}"].attrs["num_vals"] = np.array([], dtype=np.int32)
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
        self,
        start=None,
        stop=None,
        num=3,
        ignore_invalid=False,
    ) -> Iterable[tuple[RoundInformation, TrajectoryInformation]]:
        if stop is None:
            stop = self.round

        if start is None:
            start = 0

        for r0 in range(max(stop - (num - 1), start), stop + 1):
            _r = self.round_information(r=r0)

            if not _r.valid and not ignore_invalid:
                continue

            for i in _r.num_vals:
                _r_i = self.get_trajectory_information(r=r0, i=i)

                if not _r_i.valid and not ignore_invalid:
                    continue

                # no points in collection
                if _r_i.ti._size <= 0:
                    continue

                yield _r, _r_i

    def iter_ase_atoms(self, r: int | None = None, num: int = 3):
        from molmod import angstrom

        for round, trajejctory in self.iter(stop=r, num=num):
            traj = trajejctory.ti

            pos_A = traj._positions / angstrom
            pbc = traj._cell is not None
            if pbc:
                cell_A = traj._cell / angstrom
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

    def round_information(self, r: int | None = None) -> RoundInformation:
        if r is None:
            r = self.round

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

            f.flush()

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

        if not (p := self.path(r=r, i=i) / "invalid").exists():
            with open(p, "w+"):
                pass

        self._set_attr(name="valid", value=False, r=r, i=i)

    def get_bias(self, r=None, i=None) -> Bias:
        if r is None:
            r = self.round

        bn = self._get_attr("name_bias", r=r, i=i)
        return Bias.load(self.full_path(bn))

    def get_engine(self, r=None) -> MDEngine:
        if r is None:
            r = self.round

        name = self._get_attr("name_md", r=r)
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
    ):
        if isinstance(KEY, int):
            KEY = jax.random.PRNGKey(KEY)

        with self.lock:
            f = self.h5file
            common_bias_name = self.full_path(f[f"{self.round}"].attrs["name_bias"])
            common_md_name = self.full_path(f[f"{self.round}"].attrs["name_md"])
        from parsl.dataflow.dflow import AppFuture

        tasks: list[tuple[int, AppFuture]] | None = None
        plot_tasks = []
        md_engine = MDEngine.load(common_md_name)

        r_cut = md_engine.static_trajectory_info.r_cut
        z_array = md_engine.static_trajectory_info.atomic_numbers

        if sp0 is None:
            tis: TrajectoryInfo | None = None

            # also get smaples form init round
            for ri, ti in self.iter(start=0, num=2, ignore_invalid=True):
                # sp0 = ti.ti.sp
                if tis is None:
                    tis = ti.ti
                else:
                    assert tis is not None
                    tis += ti.ti

            if tis is None:
                tis = TrajectoryInfo(
                    _positions=md_engine.sp.coordinates, _cell=md_engine.sp.cell
                )
        else:
            assert sp0.shape[0] == len(
                biases
            ), f"The number of initials cvs provided {sp.shape[0]} does not correspond to the number of biases {len(biases)}"

        for i, bias in enumerate(biases):
            path_name = self.path(r=self.round, i=i)
            if not os.path.exists(path_name):
                os.mkdir(path_name)

            # construct bias
            if bias is None:
                b = Bias.load(common_bias_name)
            else:
                b = CompositeBias([Bias.load(common_bias_name), bias])

            b_name = self.full_path(path_name / "bias")
            b_name_new = self.full_path(path_name / "bias_new")
            b.save(b_name)

            traj_name = self.full_path(path_name / "trajectory_info.h5")

            @bash_app_python(executors=["reference"])
            def run(
                steps: int,
                sp: SystemParams | None,
                inputs=[],
                outputs=[],
            ):
                bias = Bias.load(inputs[1].filepath)

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

            @bash_app_python(executors=["default"])
            def plot_app(
                st: StaticTrajectoryInfo, traj: TrajectoryInfo, inputs=[], outputs=[]
            ):
                bias = Bias.load(inputs[0].filepath)

                if st.equilibration is not None:
                    if traj._t is not None:
                        traj = traj[traj._t > st.equilibration]

                cvs = traj.CV
                if cvs is None:
                    sp = traj.sp
                    nl = sp.get_neighbour_list(
                        r_cut=st.r_cut, z_array=st.atomic_numbers
                    )
                    cvs, _ = bias.collective_variable.compute_cv(sp=sp, nl=nl)

                bias.plot(name=outputs[0].filepath, traj=[cvs])

            if sp0 is None:

                if tis.shape > 1:  # type: ignore
                    assert tis is not None

                    # compensate for bias of current simulation
                    if tis.cv is not None:
                        bs, _ = bias.compute_from_cv(cvs=tis.CV)
                    else:
                        sp = tis.sp

                        nl = sp.get_neighbour_list(r_cut=r_cut, z_array=z_array)
                        bs = bias.compute_from_system_params(sp=sp, nl=nl).energy

                    # compensate for bias of previous
                    bs += tis.e_pot

                    probs = jnp.exp(
                        -bs / (md_engine.static_trajectory_info.T * boltzmann)
                    )
                    probs = probs / jnp.sum(probs)

                    KEY, k = jax.random.split(KEY, 2)

                    index = jax.random.choice(
                        a=tis.shape,  # type: ignore
                        key=k,
                        p=probs,
                    )

                    spi = tis[index].sp

                else:
                    spi = tis[0].sp
            else:
                spi = sp0[i]

            spi = spi.unbatch()
            print(f"settin {spi=}")

            future = run(
                sp=spi,  # type: ignore
                inputs=[File(common_md_name), File(b_name)],
                outputs=[File(b_name_new), File(traj_name)],
                steps=int(steps),
                execution_folder=path_name,
                stdout="md.stdout",
                stderr="md.stderr",
            )

            if plot:
                plot_file = self.full_path(path_name / "plot.pdf")

                plot_fut = plot_app(
                    traj=future,
                    st=md_engine.static_trajectory_info,
                    inputs=[future.outputs[0]],
                    outputs=[File(plot_file)],
                    execution_folder=path_name,
                    stdout="plot.stdout",
                    stderr="plot.stderr",
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
