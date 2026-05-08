"""MD engine class peforms MD simulations in a given NVT/NPT ensemble."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
from pathlib import Path
from time import time

import cloudpickle
import jax
import jsonpickle

# from ase.data import atomic_masses
from typing_extensions import Self

from IMLCV import unpickler
from IMLCV.base.bias import Bias, Energy, EnergyResult
from IMLCV.base.dataobjects import CV, FullTrajectoryInfo, NeighbourList, ShmapKwargs, StaticMdInfo, SystemParams
from IMLCV.base.datastructures import MyPyTreeNode, field
from IMLCV.base.UnitsConstants import angstrom, bar, kjmol

######################################
#             MDEngine               #
######################################


class MDEngine(MyPyTreeNode, ABC):
    """Base class for MD engine."""

    bias: Bias
    permanent_bias: Bias | None = field(default=None)
    energy: Energy
    sp: SystemParams
    static_trajectory_info: StaticMdInfo
    trajectory_info: FullTrajectoryInfo | None = field(default=None)
    trajectory_file: Path | None = field(pytree_node=False, default=None)
    time0: float = field(default_factory=time)

    step: int = 1

    nl: NeighbourList | None = None
    # r_skin = 1.0 * angstrom

    @classmethod
    def create(
        cls,
        bias: Bias,
        energy: Energy,
        static_trajectory_info: StaticMdInfo,
        trajectory_info=None,
        trajectory_file=None,
        sp: SystemParams | None = None,
        **kwargs,
    ) -> Self:
        cont = False

        create_kwargs = {}

        if trajectory_file is not None:
            trajectory_file = Path(trajectory_file)
            # continue with existing file if it exists
            if Path(trajectory_file).exists():
                trajectory_info = FullTrajectoryInfo.load(trajectory_file)
                cont = True

        if not cont or trajectory_info is None:
            create_kwargs["step"] = 1
            if sp is not None:
                energy.sp = sp

        else:
            create_kwargs["step"] = trajectory_info.size
            energy.sp = trajectory_info.sp[-1]

        # self.update_nl()

        kwargs.update(create_kwargs)

        out = cls(
            bias=bias,
            energy=energy,
            static_trajectory_info=static_trajectory_info,
            trajectory_info=trajectory_info,
            trajectory_file=trajectory_file,
            **kwargs,
        )

        out.update_nl()

        return out

    def update_nl(self):
        if self.static_trajectory_info.r_cut is None:
            return None

        if self.nl is None:
            info = self.static_trajectory_info.neighbour_list_info()
            assert info is not None

            self.nl = self.sp.get_neighbour_list(info)  # jitted update
            return

        self.nl = self.nl.slow_update_nl(self.sp)

    def save(self, file):
        filename = Path(file)

        b = self.bias
        pb = self.permanent_bias
        self.bias = None
        self.permanent_bias = None

        if filename.suffix == ".json":
            with open(filename, "w") as f:
                f.writelines(jsonpickle.encode(self, indent=1, use_base85=True))  # type:ignore
        else:
            with open(filename, "wb") as f:
                cloudpickle.dump(self, f)

        self.bias = b
        self.permanent_bias = pb

    @staticmethod
    def load(file, bias: Bias, permant_bias: Bias | None = None, **kwargs) -> MDEngine:
        filename = Path(file)

        if filename.suffix == ".json":
            with open(filename) as f:
                self = jsonpickle.decode(f.read(), context=unpickler)
        else:
            with open(filename, "rb") as f:
                self = cloudpickle.load(f)

        assert isinstance(self, MDEngine), f"{self=}"

        self.bias = bias
        self.permanent_bias = permant_bias

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

            if key == "trajectory_file":
                continue

        assert isinstance(self, MDEngine), f"{self=}"

        if (key := "trajectory_file") in kwargs.keys():
            print(f"loading ti  {self.trajectory_file} ")
            trajectory_file = kwargs[key]
            if Path(trajectory_file).exists():
                print("updating sp from trajectory file")

                self.trajectory_info = FullTrajectoryInfo.load(self.trajectory_file)
                self.step = self.trajectory_info.size * self.static_trajectory_info.save_step
                self.sp = self.trajectory_info.sp[-1]

                print(f"loaded ti  {self.step=} ")

        # else:
        #     print(f"{self.sp=}")

        self.time0 = time()

        # self.update_nl()

        return self

    def run(self, steps):
        """run the integrator for a given number of steps.

        Args:
            steps: number of MD steps
        """

        self.update_nl()

        if self.step != 1:
            steps = steps - self.step
            print(f"previous run had {self.step} steps, running for additional {int(steps)} steps!")
        else:
            print(f"running for {int(steps)} steps!")

        if steps <= 0:
            print("No steps to run, exiting")
            return

        try:
            for _ in range(int(steps)):
                self._step()

            if self.trajectory_info is not None:
                self.trajectory_info._finished = True

        except Exception as err:
            if self.step == 1:
                raise err
            print("The calculator finished early with error , marking as invalid")
            if self.trajectory_info is not None:
                self.trajectory_info._invalid = True

            raise err

        if self.trajectory_info is not None:
            self.trajectory_info = self.trajectory_info._shrink_capacity()
            if self.trajectory_file is not None:
                self.trajectory_info.save(self.trajectory_file)

    @abstractmethod
    def _step(self):
        raise NotImplementedError

    def get_trajectory(self) -> FullTrajectoryInfo:
        assert self.trajectory_info is not None

        return self.trajectory_info._shrink_capacity()

    def save_step(
        self,
        sp: SystemParams,
        T: jax.Array | None = None,
        P: jax.Array | None = None,
        t: jax.Array | None = None,
        err: jax.Array | None = None,
        cv: jax.Array | None = None,
        e_bias: jax.Array | None = None,
        e_pot: jax.Array | None = None,
        A: jax.Array | None = None,
        gpos_mag=0.0,
        gpos_bias_mag=0.0,
        canonicalize=False,
    ):
        screen_log = self.step % self.static_trajectory_info.screen_log == 0
        save_step = self.step % self.static_trajectory_info.save_step == 0
        write_step = self.step % self.static_trajectory_info.write_step == 0

        if screen_log or save_step or write_step:
            # print(f"{self.step=}  {screen_log=} or {save_step=} or {write_step=}")

            ti = FullTrajectoryInfo.create(
                positions=sp.coordinates,
                cell=sp.cell,
                e_pot=e_pot,
                e_bias=e_bias,
                cv=cv,
                T=T,
                P=P,
                t=t,
                err=err,
                A=A,
            )

            if self.step == 1:
                str = f"{'step': ^10s}"
                str += f"|{'cons err': ^10s}"
                str += f"|{'e_pot[Kj/mol]': ^15s}"
                str += f"|{'e_bias[Kj/mol]': ^15s}"
                if ti._P is not None:
                    str += f"|{'P[bar]': ^10s}"
                str += f"|{'T[K]': ^10s}|{'walltime[s]': ^11s}"

                ss = "|\u2207\u2093U\u1d47|[Kj/\u212b]"
                str += f"|{ss: ^13s}"

                ss = "|\u2207\u2093U\u1d47|[Kj/\u212b]"
                str += f"|{ss: ^13s}"

                str += f"|{'A': ^10s}"

                str += f"|{' CV': ^10s}"
                print(str, sep="")
                print(f"{'=' * len(str)}")

            if self.step % self.static_trajectory_info.screen_log == 0:
                str = f"{self.step: >10d}"
                assert ti._err is not None
                assert ti._T is not None
                assert ti._e_pot is not None
                assert ti._e_bias is not None

                str += f"|{ti._err[0]: >10.4f}"
                str += f"|{ti._e_pot[0] / kjmol: >15.8f}"
                str += f"|{ti._e_bias[0] / kjmol: >15.8f}"
                if ti._P is not None:
                    str += f" {ti._P[0] / bar: >10.8f}"
                str += f" {ti._T[0]: >10.2f} {time() - self.time0: >11.2f}"
                str += f"|{(gpos_mag / kjmol * angstrom): >13.2f}"
                str += f"|{(gpos_bias_mag / kjmol * angstrom): >13.2f}"

                str += f"|{ti._A[0]: >10.5f}"

                if ti._cv is not None:
                    str += f"| {ti._cv[0, :]}"
                print(str)

            if self.step % self.static_trajectory_info.save_step == 0:
                # write step to trajectory
                if self.trajectory_info is None:
                    self.trajectory_info = ti
                else:
                    self.trajectory_info += ti

            if self.step % self.static_trajectory_info.write_step == 0:
                if self.trajectory_file is not None:
                    self.trajectory_info.save(self.trajectory_file)  # type: ignore

            assert self.bias is not None

        self.bias = self.bias.update_bias(self)

        self.step += 1

    def get_energy(
        self,
        sp: SystemParams,
        nl: NeighbourList | None = None,
        gpos: bool = False,
        vtens: bool = False,
        manual_vir=None,
        eps=1e-5,
    ) -> EnergyResult:
        return self.energy.compute_from_system_params(
            gpos=gpos,
            vir=vtens,
            sp=sp,
            nl=nl,
            manual_vir=manual_vir,
            eps=eps,
        )

    def get_bias(
        self,
        sp: SystemParams,
        nl: NeighbourList | None = None,
        gpos: bool = False,
        vtens: bool = False,
        shmap: bool = False,
        use_jac=False,
        push_jac=False,
        rel=False,
        shmap_kwargs=ShmapKwargs.create(),
    ) -> tuple[CV, EnergyResult]:
        assert self.bias is not None

        cv, ener = self.bias.compute_from_system_params(
            sp=sp,
            nl=nl,
            gpos=gpos,
            vir=vtens,
            shmap=shmap,
            use_jac=use_jac,
            push_jac=push_jac,
            rel=rel,
            shmap_kwargs=shmap_kwargs,
        )

        if self.permanent_bias is not None:
            cv2, ener2 = self.permanent_bias.compute_from_system_params(
                sp=sp,
                nl=nl,
                gpos=gpos,
                vir=vtens,
                shmap=shmap,
                use_jac=use_jac,
                push_jac=push_jac,
                rel=rel,
                shmap_kwargs=shmap_kwargs,
            )

            ener = ener + ener2

        return cv, ener

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        removed = []
        try:
            f_names = [f.name for f in fields(self.__class__)]

            for k in statedict.keys():
                if k not in f_names:
                    removed.append(k)

            for k in removed:
                del statedict[k]

            self.__class__.__init__(self, **statedict)

            if self.trajectory_file is not None:
                if Path(self.trajectory_file).exists():
                    assert self.trajectory_info is not None

                    self.step = self.trajectory_info.size * self.static_trajectory_info.save_step
                    self.sp = self.trajectory_info.sp[-1]

            self.time0 = time()

        except Exception as e:
            print(
                f"tried to initialize {self.__class__} with from {statedict=} {f'{removed=}' if len(removed) == 0 else ''} but got exception",
            )
            raise e

        # self.update_nl()
