from pathlib import Path
from typing import Iterator

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hsluv import hsluv_to_rgb
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvFlow
from IMLCV.base.CV import CvMetric
from IMLCV.base.CV import CvTrans
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CV import SystemParams
from IMLCV.base.MdEngine import StaticMdInfo
from IMLCV.base.rounds import Rounds
from IMLCV.implementations.CV import scale_cv_trans
from jax import Array
from jax import jacrev
from jax import vmap
from jax.random import choice
from jax.random import PRNGKey
from jax.random import split
from matplotlib import gridspec
from matplotlib.figure import Figure


class Transformer:
    def __init__(
        self,
        outdim,
        periodicity=None,
        bounding_box=None,
        descriptor=CvFlow,
    ) -> None:
        self.outdim = outdim

        if periodicity is None:
            periodicity = [False for _ in range(self.outdim)]

        self.periodicity = periodicity

        self.descriptor = descriptor

    def pre_fit(
        self,
        z: list[SystemParams],
        nl: list[NeighbourList] | None,
        chunk_size=None,
        scale=True,
    ) -> tuple[list[CV], CvFlow]:
        f = self.descriptor

        x: list[CV] = []

        for i, zi in enumerate(z):
            nli = nl[i] if nl is not None else None
            x.append(f.compute_cv_flow(zi, nli, chunk_size=chunk_size))

        if scale:
            g = scale_cv_trans(CV.stack(*x), lower=0, upper=1)
            x = [g.compute_cv_trans(xi)[0] for xi in x]

            f = f * g

        return x, f

    def fit(
        self,
        sp_list: list[SystemParams],
        nl_list: list[NeighbourList] | None,
        chunk_size=None,
        prescale=True,
        postscale=True,
        jac=jacrev,
        **fit_kwargs,
    ) -> tuple[CV, CollectiveVariable]:
        print("starting pre_fit")

        x, f = self.pre_fit(
            sp_list,
            nl_list,
            scale=prescale,
            chunk_size=chunk_size,
        )

        print("starting fit")
        y, g = self._fit(
            x,
            nl_list,
            chunk_size=chunk_size,
            **fit_kwargs,
        )

        print("starting post_fit")
        z, h = self.post_fit(y, scale=postscale)

        cv = CollectiveVariable(
            f=f * g * h,
            metric=CvMetric(
                periodicities=self.periodicity,
                bounding_box=jnp.vstack(
                    [jnp.min(z.cv, axis=0), jnp.max(z.cv, axis=0)],
                ).T,
            ),
            jac=jac,
        )

        return z, cv

    def _fit(
        self,
        x: list[CV],
        nl: list[NeighbourList] | None,
        chunk_size=None,
        **kwargs,
    ) -> tuple[CV, CvTrans]:
        raise NotImplementedError

    def post_fit(self, y: list[CV], scale) -> tuple[CV, CvTrans]:
        y = CV.stack(*y)
        if not scale:
            return y, CvTrans.from_cv_function(lambda x, _: x)
        h = scale_cv_trans(y)
        return h.compute_cv_trans(y)[0], h


class CVDiscovery:
    """convert set of coordinates to good collective variables."""

    def __init__(self, transformer: Transformer) -> None:
        # self.rounds = rounds
        self.transformer = transformer

    @staticmethod
    def data_loader(
        rounds: Rounds,
        num=4,
        out=-1,
        split_data=False,
        new_r_cut=None,
        cv_round=None,
        filter_bias=False,
        filter_energy=False,
    ) -> tuple[list[SystemParams], list[NeighbourList] | None, CollectiveVariable, StaticMdInfo, list[CV]]:
        weights = []

        colvar = rounds.get_collective_variable()
        sti: StaticMdInfo | None = None
        sp: list[SystemParams] = []
        nl: list[NeighbourList] | None = [] if new_r_cut is not None else None
        cv: list[CV] = []

        for round, traj in rounds.iter(stop=None, num=num, c=cv_round):
            if sti is None:
                sti = round.tic

            sp0 = traj.ti.sp
            nl0 = (
                sp0.get_neighbour_list(
                    r_cut=new_r_cut,
                    z_array=round.tic.atomic_numbers,
                )
                if new_r_cut is not None
                else None
            )

            if (cv0 := traj.ti.CV) is None:
                bias = traj.get_bias()
                if colvar is None:
                    colvar = bias.collective_variable

                if new_r_cut != round.tic.r_cut:
                    nlr = (
                        sp0.get_neighbour_list(
                            r_cut=round.tic.r_cut,
                            z_array=round.tic.atomic_numbers,
                        )
                        if round.tic.r_cut is not None
                        else None
                    )
                else:
                    nlr = nl0

                cv0, _ = bias.collective_variable.compute_cv(sp=sp0, nl=nlr)

            e = None

            if filter_bias:
                if (b0 := traj.ti.e_bias) is None:
                    # map cvs
                    bias = traj.get_bias()

                    b0, _ = bias.compute_from_cv(cvs=cv0)

                e = b0

            if filter_energy:
                if (e0 := traj.ti.e_pot) is None:
                    raise ValueError("e_pot is None")

                if e is None:
                    e = e0
                else:
                    e = e + e0

            sp.append(sp0)
            if nl is not None:
                assert nl0 is not None
                nl.append(nl0)
            cv.append(cv0)

            if e is None:
                weights.append(None)
            else:
                beta = 1 / round.tic.T
                weight = jnp.exp(-beta * e)
                weights.append(weight)

        assert sti is not None
        assert len(sp) != 0
        if nl is not None:
            assert len(nl) == len(sp)

        def choose(key, probs: Array, len: int):
            if len is None:
                len = probs.shape[0]

            key, key_return = split(key, 2)

            indices = choice(
                key=key,
                a=len,
                shape=(int(out),),
                p=probs,
                replace=True,
            )

            return key_return, indices

        key = PRNGKey(0)

        out_sp: list[SystemParams] = []
        out_nl: list[NeighbourList] | None = [] if nl is not None else None
        out_cv: list[CV] = []

        if split_data:
            for n, wi in enumerate(weights):
                if wi is None:
                    probs = None
                else:
                    probs = wi / jnp.sum(wi)
                key, indices = choose(key, probs)

                out_sp.append(sp[n][indices])
                if nl is not None:
                    assert out_nl is not None
                    out_nl.append(nl[n][indices])
                out_cv.append(cv[n][indices])

        else:
            if weights[0] is None:
                probs = None
            else:
                probs = jnp.hstack(weights)
                probs = probs / jnp.sum(probs)

            key, indices = choose(key, probs, len=sum([sp_n.shape[0] for sp_n in sp]))

            indices = jnp.sort(indices)

            count = 0

            sp_trimmed = []
            nl_trimmed = [] if nl is not None else None
            cv_trimmed = []

            for sp_n, nl_n, cv_n in zip(sp, nl, cv):
                n_i = sp_n.shape[0]

                index = indices[jnp.logical_and(count <= indices, indices < count + n_i)] - count
                sp_trimmed.append(sp_n[index])
                if nl is not None:
                    nl_trimmed.append(nl_n[index])
                cv_trimmed.append(cv_n[index])
                count += n_i

            if len(sp) >= 1:
                out_sp.append(sum(sp_trimmed[1:], sp_trimmed[0]))
                if nl is not None:
                    assert out_nl is not None
                    out_nl.append(sum(nl_trimmed[1:], nl_trimmed[0]))
                out_cv.append(CV.stack(*cv_trimmed))

            else:
                out_sp.append(sp_trimmed[0][indices])
                if nl is not None:
                    assert out_nl is not None
                    out_nl.append(nl_trimmed[0][indices])
                out_cv.append(cv_trimmed[0][indices])

        return (out_sp, out_nl, colvar, sti, out_cv)

    def compute(
        self,
        rounds: Rounds,
        num_rounds=4,
        samples=1e4,
        plot=True,
        new_r_cut=None,
        chunk_size=None,
        split_data=False,
        name=None,
        **kwargs,
    ) -> CollectiveVariable:
        (sp_list, nl_list, old_collective_variable, sti, cvs_old) = self.data_loader(
            num=num_rounds,
            out=samples,
            rounds=rounds,
            new_r_cut=new_r_cut,
            split_data=split_data,
        )

        cvs_new, new_collective_variable = self.transformer.fit(
            sp_list,
            nl_list,
            chunk_size=chunk_size,
            **kwargs,
        )

        # todo: use cv data instead of reaculating

        if plot:
            sp = sum(sp_list[1:], sp_list[0])
            nl = sum(nl_list[1:], nl_list[0]) if nl_list is not None else None
            ind = np.random.choice(
                a=sp.shape[0],
                size=min(1e4, sp.shape[0]),
                replace=False,
            )

            folder = rounds.path(c=rounds.cv, r=rounds.round)

            CVDiscovery.plot_app(
                name=str(folder / "cvdiscovery"),
                old_cv=old_collective_variable,
                new_cv=new_collective_variable,
                sps=sp[ind],
                nl=nl[ind] if nl is not None else None,
                chunk_size=chunk_size,
                cv_data_old=CV.stack(*cvs_old)[ind],
                cv_data_new=cvs_new[ind],
            )

        return new_collective_variable

    @staticmethod
    def plot_app(
        sps: SystemParams,
        nl: NeighbourList,
        old_cv: CollectiveVariable,
        new_cv: CollectiveVariable,
        name: str | Path,
        labels=None,
        chunk_size: int | None = None,
        cv_data_old: CV | None = None,
        cv_data_new: CV | None = None,
    ):
        """Plot the app for the CV discovery. all 1d and 2d plots are plotted directly, 3d or higher are plotted as 2d slices."""

        cv_data: list[CV] = []

        cvs = [old_cv, new_cv]
        if cv_data_old is None:
            cv_data.append(old_cv.compute_cv(sps, nl, chunk_size=chunk_size)[0])
        else:
            cv_data.append(cv_data_old)

        if cv_data_new is None:
            cv_data.append(new_cv.compute_cv(sps, nl, chunk_size=chunk_size)[0])
        else:
            cv_data.append(cv_data_new)

        # plot setting
        kwargs = {"s": 0.2}

        if labels is None:
            labels = [
                ["cv in 1", "cv in 2", "cv in 3"],
                ["cv out 1", "cv out 2", "cv out 3"],
            ]

            indim = cvs[0].n
            outdim = cvs[1].n

            fig = plt.figure()

            for in_out_color, ls, rs in CVDiscovery._grid_spec_iterator(
                fig=fig,
                indim=indim,
                outdim=outdim,
            ):
                if in_out_color == 0:
                    dim = indim
                    color_data = cv_data[0].cv
                else:
                    dim = outdim
                    color_data = cv_data[1].cv

                max_val = jnp.max(color_data, axis=0)
                min_val = jnp.min(color_data, axis=0)

                data_col = (color_data - min_val) / (max_val - min_val)

                # https://www.hsluv.org/
                # hue 0-360 sat 0-100 lighness 0-1000

                if dim == 1:  # skip luminance and set to 0.5. green/red = blue/yellow
                    lab = jnp.ones((data_col.shape[0], 3))
                    lab = lab.at[:, 0].set(data_col[:, 0] * 360)
                    lab = lab.at[:, 1].set(75)
                    lab = lab.at[:, 2].set(40)

                if dim == 2:
                    lab = jnp.ones((data_col.shape[0], 3))
                    lab = lab.at[:, 0].set(data_col[:, 0] * 360)
                    lab = lab.at[:, 1].set(75)
                    lab = lab.at[:, 2].set(data_col[:, 1] * 100)

                if dim == 3:
                    lab = jnp.ones((data_col.shape[0], 3))
                    lab = lab.at[:, 0].set(data_col[:, 0] * 360)
                    lab = lab.at[:, 1].set(data_col[:, 1] * 100)
                    lab = lab.at[:, 2].set(data_col[:, 2] * 100)

                rgb = []

                for s in lab:
                    rgb.append(hsluv_to_rgb(s))

                rgb = jnp.array(rgb)

                def do_plot(dim, in_out, axes):
                    data_proc = cv_data[in_out].cv
                    if dim == 1:
                        data_proc = jnp.hstack([data_proc, rgb[:, [0]]])

                        f = CVDiscovery._plot_1d
                    elif dim == 2:
                        f = CVDiscovery._plot_2d
                    elif dim == 3:
                        f = CVDiscovery._plot_3d

                    f(fig, axes, data_proc, rgb, labels[0][0:dim], **kwargs)

                do_plot(indim, 0, ls)
                do_plot(outdim, 1, rs)

            name = Path(name)
            if name.suffix != ".pdf":
                name = Path(
                    f"{name}.pdf",
                )

            name.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(name)

    @staticmethod
    def _grid_spec_iterator(
        fig: Figure,
        indim,
        outdim,
    ) -> Iterator[tuple[list[int], int, list[plt.Axes], list[plt.Axes]]]:
        width_in = 1 if indim < 3 else 2
        width_out = 1 if outdim < 3 else 2

        spec = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[width_in, width_out])
        yield 0, spec[0, 0], spec[0, 1]
        yield 1, spec[1, 0], spec[1, 1]

    @staticmethod
    def _plot_1d(fig: Figure, grid: gridspec, data, colors, labels, **scatter_kwargs):
        gs = grid.subgridspec(ncols=1, nrows=2, height_ratios=[1, 4])

        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0])

        # create inset
        ax.scatter(
            data[:, 0],
            data[:, 1],
            c=colors,
        )

        ax.set_xlabel(labels)
        ax.set_ylabel("count")

        ax_histx.hist(data[:, 0])

    @staticmethod
    def _plot_2d(fig: Figure, grid: gridspec, data, colors, labels, **scatter_kwargs):
        gs = grid.subgridspec(
            ncols=2,
            nrows=2,
            width_ratios=[4, 1],
            height_ratios=[1, 4],
        )
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0])
        ax_histy = fig.add_subplot(gs[1, 1])

        ax.scatter(
            *[data[:, l] for l in range(2)],
            c=colors,
            **scatter_kwargs,
        )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        ax_histx.hist(data[:, 0])
        ax_histy.hist(data[:, 1], orientation="horizontal")

    @staticmethod
    def _plot_3d(fig: Figure, grid: gridspec, data, colors, labels, **scatter_kwargs):
        gs = grid.subgridspec(
            ncols=2,
            nrows=1,
            width_ratios=[1, 1],
            height_ratios=[1],
        )

        # ?https://matplotlib.org/3.2.1/gallery/axisartist/demo_floating_axes.html

        ax0 = fig.add_subplot(gs[0, 0], projection="3d")
        ax1 = fig.add_subplot(gs[0, 1], projection="3d")
        axes = [ax0, ax1]
        for ax in axes:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])

            ax.view_init(elev=20, azim=45)

        # create  3d scatter plot with 2D histogram on side
        ax0.scatter(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            **scatter_kwargs,
            c=colors,
            zorder=1,
        )

        for a, b, z in [
            [0, 1, "z"],
            [0, 2, "y"],
            [1, 2, "x"],
        ]:
            Z, X, Y = np.histogram2d(data[:, a], data[:, b])

            X = (X[1:] + X[:-1]) / 2
            Y = (Y[1:] + Y[:-1]) / 2

            X, Y = np.meshgrid(X, Y)

            Z = (Z - Z.min()) / (Z.max() - Z.min())

            kw = {
                "facecolors": plt.cm.Greys(Z),
                "shade": True,
                "alpha": 1.0,
                "zorder": 0,
            }

            zz = np.zeros(X.shape) - 1.1

            if z == "z":
                ax0.plot_surface(X, Y, zz, **kw)
            elif z == "y":
                ax0.plot_surface(X, zz, Y, **kw)
            else:
                ax0.plot_surface(zz, X, Y, **kw)

        # scatter 2d projections
        zz = np.zeros(data[:, 0].shape)
        for z in ["x", "y", "z"]:
            if z == "z":
                ax1.scatter(
                    data[:, 0],
                    data[:, 1],
                    zz,
                    **scatter_kwargs,
                    zorder=1,
                    c=colors,
                )
            elif z == "y":
                ax1.scatter(
                    data[:, 0],
                    zz,
                    data[:, 2],
                    **scatter_kwargs,
                    zorder=1,
                    c=colors,
                )
            else:
                ax1.scatter(
                    zz,
                    data[:, 1],
                    data[:, 2],
                    **scatter_kwargs,
                    zorder=1,
                    c=colors,
                )
