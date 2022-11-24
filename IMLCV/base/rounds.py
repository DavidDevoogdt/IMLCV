from __future__ import annotations

import logging
import os
import shutil
from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import ase
import dill
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from filelock import FileLock
from molmod.constants import boltzmann
from parsl.data_provider.files import File

from IMLCV import KEY
from IMLCV.base.bias import Bias, CompositeBias, NoneBias
from IMLCV.base.CV import SystemParams
from IMLCV.base.MdEngine import MDEngine, StaticTrajectoryInfo, TrajectoryInfo
from IMLCV.external.parsl_conf.bash_app_python import bash_app_python

logging.getLogger("filelock").setLevel(logging.DEBUG)


class Rounds(ABC):

    # ENGINE_KEYS = ["T", "P", "timecon_thermo", "timecon_baro"]

    def __init__(self, folder: str | Path = "output") -> None:
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

        self.round = -1
        self.i = 0

        if not os.path.isdir(folder):
            os.makedirs(folder)

        self.folder = Path(folder).resolve()

        # if not Path(self.h5file_name).exists():

        self.lock = FileLock(self.h5filelock_name)

        self._make_file()

    def _make_file(self):
        # create the file
        with self.lock:
            with h5py.File(self.h5file_name, mode="w-", swmr=True):
                pass

        self.h5file = h5py.File(self.h5file_name, mode="r+")
        # self.lock = Lock()

    # def __del__(self):
    #     self.h5file.close()

    @property
    def h5file_name(self):
        return self.full_path(self.path() / "results.h5")

    @property
    def h5filelock_name(self):
        return self.full_path(self.path() / "results.h5.lock")

    # def __del__(self):
    #     self.h5file.close()

    def save(self):
        with open(self.full_path("rounds"), "wb") as f:

            d = self.__dict__.copy()

            del d["h5file"]

            dill.dump(d, f)

    @staticmethod
    def load(folder: str | Path, sti: StaticTrajectoryInfo | None = None):
        with open(f"{folder}/rounds", "rb") as f:
            self = object.__new__(RoundsMd)
            self.__dict__.update(dill.load(f))
            self.folder = Path(folder).resolve()

        try:
            with self.lock:
                self.h5file = h5py.File(self.h5file_name, mode="r+")
        except:
            if Path(self.h5file_name).exists():
                shutil.move(self.h5file_name, f"{self.h5file_name}_bak")
            self._make_file()
            self.recover()

        return self

    def add(self, i, d: TrajectoryInfo, attrs=None, r=None):
        if r is None:
            r = self.round

        with self.lock:
            print("enetered add")
            print(f"{i} got lock file {self.h5file_name}")
            f = self.h5file
            print(f"{i} got hdf5 file")

            if f"{i}" not in f[f"{r}"]:
                f.create_group(f"{r}/{i}")
            if "trajectory_info" not in f[f"{r}/{i}"]:
                f[f"{r}/{i}"].create_group("trajectory_info")
                d._save(hf=f[f"{r}/{i}/trajectory_info"])

            if attrs is not None:
                for key, val in attrs.items():
                    if val is not None:
                        f[f"{r}/{i}"].attrs[key] = val

            f[f"{r}/{i}"].attrs["valid"] = True
            f[f"{r}"].attrs["num"] += 1

            f.flush()

    def new_round(self, attr, stic: StaticTrajectoryInfo, r=None):

        self.round += 1
        self.i = 0

        if r is None:
            r = self.round

        dir = self.path(round=r)
        if not dir.exists():
            dir.mkdir(parents=True)

        with self.lock:
            f = self.h5file
            f.create_group(f"{r}")

            for key in attr:
                if attr[key] is not None:
                    f[f"{r}"].attrs[key] = attr[key]

            f[f"{r}"].create_group("static_trajectory_info")
            stic._save(hf=f[f"{r}/static_trajectory_info"])

            f[f"{r}"].attrs["num"] = 0
            f[f"{r}"].attrs["valid"] = True

            f.flush()

        self.save()

    def iter(self, start=None, stop=None, num=3) -> Iterable[tuple[Round, Trajectory]]:
        if stop is None:
            stop = self.round

        if start is None:
            start = 0

        for r0 in range(max(stop - (num - 1), start), stop + 1):

            _r = self._get_r(r=r0)

            if not _r.valid:
                continue

            for i in range(_r.num):
                _r_i = self._get_r_i(r=r0, i=i)

                if not _r_i.valid:
                    continue

                yield _r, _r_i

    @dataclass
    class Trajectory:
        ti: TrajectoryInfo
        valid: bool
        round: int
        num: int
        folder: Path
        name_bias: str | None = None

        def get_bias(self) -> Bias:
            assert self.name_bias is not None
            return Bias.load(self.folder / self.name_bias)

    def _get_r_i(self, r: int, i: int) -> Trajectory:

        with self.lock:
            f = self.h5file
            d = f[f"{r}/{i}"]

            ti = TrajectoryInfo._load(hf=d["trajectory_info"])
            r_attr = {key: d.attrs[key] for key in d.attrs}

            f.flush()

        return Rounds.Trajectory(ti=ti, **r_attr, round=r, num=i, folder=self.folder)

    @dataclass
    class Round:
        round: int
        # names: list[str]
        valid: bool
        num: int
        tic: StaticTrajectoryInfo
        name_bias: str | None = None
        name_md: str | None = None

    def _get_r(self, r: int) -> Round:

        with self.lock:
            f = self.h5file

            stic = StaticTrajectoryInfo._load(hf=f[f"{r}/static_trajectory_info"])

            d = f[f"{r}"].attrs
            r_attr = {key: d[key] for key in d}

        return Rounds.Round(
            round=int(r),
            # names=names,
            tic=stic,
            **r_attr,
        )

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

    @property
    def T(self):
        return self._get_r(r=self.round).tic.T

    @property
    def P(self):
        return self._get_r(r=self.round).tic.P

    def n(self, r=None):
        if r is None:
            r = self.round
        return self._get_r(r=self.round).num

    def invalidate_data(self, r=None, i=None):
        if r is None:
            r = self.round

        self._set_attr(name="valid", value=False, r=r, i=i)

    def full_path(self, name) -> str:
        return str(self.folder / Path(name))

    def rel_path(self, name):
        return str(Path(name).relative_to(self.folder))

    def path(self, round=None, i=None) -> Path:
        p = Path(self.folder)
        if round is not None:
            p /= f"round_{round}"
            if i is not None:
                p /= f"md_{i}"

        return p


class RoundsCV(Rounds):
    """class for unbiased rounds."""


class RoundsMd(Rounds):
    """helper class to save/load all data in a consistent way.

    Gets passed between modules
    """

    def __init__(self, folder="output") -> None:
        super().__init__(folder=folder)

    @staticmethod
    def load(folder) -> RoundsMd:
        return Rounds.load(folder=folder)

    def _add(
        self,
        traj: TrajectoryInfo,
        md: MDEngine,
        bias: str,
        i: int,
        r: int | None = None,
    ):
        """adds all the saveble info of the md simulation."""

        if i is None:
            i = self.i
            self.i += 1

        self._validate(md)

        attr = {}
        attr["name_bias"] = bias

        super().add(d=traj, attrs=attr, i=i, r=r)

    def _validate(self, md: MDEngine):

        pass

    def iter_atoms(
        self, r: int | None = None, num: int = 3
    ) -> tuple[ase.Atoms, Rounds.Round, Rounds.Trajectory]:

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

    def write_xyz(self, r: int | None = None, num: int = 1, repeat=None):

        from ase.io.extxyz import write_extxyz

        for i, (atoms, round, trajejctory) in enumerate(self.iter_atoms(r=r, num=num)):
            with open(
                self.path(round=round.round, i=trajejctory.num) / "trajectory.xyz",
                mode="w",
            ) as f:
                if repeat is not None:
                    atoms = [a.repeat(repeat) for a in atoms]

                write_extxyz(f, atoms)

    def new_round(self, md: MDEngine):

        r = self.round + 1

        directory = self.path(round=r)
        if not os.path.isdir(directory):
            os.mkdir(directory)

        name_md = directory / "engine"
        name_bias = directory / "bias"
        md.save(name_md)
        md.bias.save(name_bias)

        attr = {}

        attr["name_md"] = self.rel_path(name_md)
        attr["name_bias"] = self.rel_path(name_bias)

        super().new_round(attr=attr, stic=md.static_trajectory_info)

    def get_bias(self, r=None, i=None) -> Bias:
        if r is None:
            r = self.round

        with self.lock:
            f = self.h5file
            if i is None:
                bn = f[f"{r}"].attrs["name_bias"]
            else:
                bn = f[f"{r}"][i].attrs["name_bias"]
        return Bias.load(self.full_path(bn))

    def get_engine(self, r=None) -> MDEngine:
        if r is None:
            r = self.round
        with self.lock:

            f = self.h5file
            name = f[f"{r}"].attrs["name_md"]

        return MDEngine.load(self.full_path(name), filename=None)

    def recover(self):
        rounds = -1
        self.round = -1

        for round_r in self.path().glob("round_*"):
            rounds += 1
            r = round_r.parts[-1][6:]

            f = self.h5file

            if r not in f.keys():
                assert (round_r / "bias").exists()
                assert (round_r / "engine").exists()

                attr = {}

                attr["name_md"] = self.rel_path(round_r / "engine")
                attr["name_bias"] = self.rel_path(round_r / "bias")

                sti = MDEngine.load(
                    self.full_path(round_r / "engine")
                ).static_trajectory_info

                directory = self.path(round=r)

                super().new_round(attr=attr, stic=sti, r=r)

            rr = self._get_r(r=r)
            md = MDEngine.load(file=self.full_path(rr.name_md))

            for md_i in round_r.glob("md_*"):

                i = md_i.parts[-1][3:]

                if i in f[f"{r}"].keys():
                    continue

                if not (md_i / "bias").exists():
                    continue

                tin = md_i / "trajectory_info.h5"

                if not (tin).exists():
                    if not (md_i / "bash_app_trajectory_info.h5").exists():
                        continue
                    else:
                        tin = md_i / "bash_app_trajectory_info.h5"

                traj = TrajectoryInfo.load(tin)

                self._add(bias=self.rel_path(md_i / "bias"), traj=traj, md=md, i=i, r=r)

        self.save()

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

            temp_name = self.path(round=self.round, i=i)
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
            self._add(
                traj=d,
                md=md_engine,
                bias=future.outputs[0].filename,
                i=i,
            )

        if plot:
            for future in plot_tasks:
                d = future.result()

        self.i += len(tasks)

    def grid_points(self, cvs, map=True, r=None):
        """finds systemparams that with cv that are close to given CV"""

    def unbias_rounds(self, steps=1e5, num=1e7, calc=False) -> RoundsCV:
        raise NotImplementedError
        md = self.get_engine()
        if self.n() > 1 or isinstance(self.get_bias(), NoneBias) or calc:
            from IMLCV.base.Observable import Observable

            obs = Observable(self)
            fesBias = obs.fes_bias(plot=True)

            md = md.new_bias(fesBias, filename=None, write_step=5)
            self.new_round(md)

        if self.n() == 0:
            self.run(None, steps)

        # construct rounds object
        r = self._get_r_i(self.round, 0)
        props = self._get_r(self.round)
        beta = 1 / (props["T"] * boltzmann)
        bias = Bias.load(r["attr"]["name_bias"])

        p = r["positions"][:]
        c = r.get("cell")
        t_orig = r["t"][:]

        def _interp(x_new, x, y):
            if y is not None:
                return jnp.apply_along_axis(
                    lambda yy: jnp.interp(x_new, x, yy), arr=y, axis=0
                )
            return None

        def bt(x, x_orig):
            pt = partial(_interp, x=x_orig, y=p)
            ct = partial(_interp, x=x_orig, y=c)
            return bias.compute_coor(coordinates=pt(x), cell=ct(x))[0]

        bt = jnp.vectorize(bt, excluded=[1])

        def integr(fx, x):
            return np.array([0, *np.cumsum((fx[1:] + fx[:-1]) * (x[1:] - x[:-1]) / 2)])

        t = t_orig[:]
        eb = np.exp(-beta * bt(t, t_orig))
        tau = integr(1 / eb, t)
        tau_new = np.linspace(start=0.0, stop=tau.max(), num=int(num))

        p_new = _interp(tau_new, tau, r["positions"])
        c_new = _interp(tau_new, tau, r.get("cell"))

        roundscv = RoundsCV(f"{self.folder}_unbiased")
        roundscv.new_round(props)
        roundscv.add(
            0,
            TrajectoryInfo(
                e_pot=None,
                positions=p_new,
                gpos=None,
                cell=c_new,
                t=tau_new,
            ),
        )

        return roundscv
