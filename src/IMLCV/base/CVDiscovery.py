import itertools
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvFlow
from IMLCV.base.CV import CvMetric
from IMLCV.base.CV import CvTrans
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CV import SystemParams
from IMLCV.base.MdEngine import StaticMdInfo
from IMLCV.base.rounds import Rounds
from IMLCV.implementations.CV import distance_descriptor
from IMLCV.implementations.CV import sb_descriptor
from IMLCV.implementations.CV import scale_cv_trans
from jax import Array
from jax import jacrev
from jax import vmap
from jax.random import choice
from jax.random import PRNGKey
from jax.random import split
from matplotlib import gridspec
from matplotlib.colors import hsv_to_rgb


class Transformer:
    def __init__(
        self,
        outdim,
        periodicity=None,
        bounding_box=None,
        descriptor="sb",
        descriptor_kwargs={},
    ) -> None:
        self.outdim = outdim

        if periodicity is None:
            periodicity = [False for _ in range(self.outdim)]
        if bounding_box is None:
            bounding_box = np.array([[0.0, 10.0] for _ in periodicity])

        self.periodicity = periodicity
        self.bounding_box = bounding_box

        self.descriptor: CvFlow

        if descriptor == "sb":
            self.descriptor = sb_descriptor(**descriptor_kwargs)
        elif descriptor == "distance":
            self.descriptor = distance_descriptor(**descriptor_kwargs)
        else:
            raise NotImplementedError

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
            g = scale_cv_trans(CV.stack(*x))
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
        *fit_args,
        **fit_kwargs,
    ) -> tuple[CV, CollectiveVariable]:
        # for i,spi in enumerate(sp_list):
        #     nli = nl_list[i] if nl_list is not None else None

        # sp = sum(sp_list[1:], sp_list[0])
        # nl = sum(nl_list[1:], nl_list[0]) if nl_list is not None else None

        x, f = self.pre_fit(
            sp_list,
            nl_list,
            scale=prescale,
            chunk_size=chunk_size,
        )

        y, g = self._fit(
            x,
            nl_list,
            *fit_args,
            **fit_kwargs,
        )

        z, h = self.post_fit(y, scale=postscale)

        cv = CollectiveVariable(
            f=f * g * h,
            metric=CvMetric(periodicities=self.periodicity),
            jac=jac,
        )

        return z, cv

    def _fit(
        self,
        x: list[CV],
        nl: list[NeighbourList] | None,
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

    def data_loader(
        self,
        rounds: Rounds,
        num=4,
        out=-1,
        split_data=False,
        new_r_cut=None,
    ) -> tuple[list[SystemParams], list[NeighbourList] | None, CollectiveVariable, StaticMdInfo]:
        weights = []

        colvar = rounds.get_collective_variable()
        sti: StaticMdInfo | None = None
        sp: list[SystemParams] = []
        nl: list[NeighbourList] | None = [] if new_r_cut is not None else None

        for round, traj in rounds.iter(stop=None, num=num):
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

            if (b0 := traj.ti.e_bias) is None:
                # map cvs
                bias = traj.get_bias()

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

                if (cv0 := traj.ti.CV) is None:
                    if colvar is None:
                        colvar = bias.collective_variable

                    cv0, _ = bias.collective_variable.compute_cv(sp=sp0, nl=nlr)

                b0, _ = bias.compute_from_cv(cvs=cv0)

            sp.append(sp0)
            if nl is not None:
                assert nl0 is not None
                nl.append(nl0)

            beta = 1 / round.tic.T
            weight = jnp.exp(beta * b0)
            weights.append(weight)

        assert sti is not None
        assert len(sp) != 0
        if nl is not None:
            assert len(nl) == len(sp)

        def choose(key, probs: Array):
            key, key_return = split(key, 2)

            indices = choice(
                key=key,
                a=probs.shape[0],
                shape=(int(out),),
                # p=probs,
                replace=False,
            )

            return key_return, indices

        key = PRNGKey(0)

        out_sp: list[SystemParams] = []
        out_nl: list[NeighbourList] | None = [] if nl is not None else None

        if split_data:
            for n, wi in enumerate(weights):
                probs = wi / jnp.sum(wi)
                key, indices = choose(key, probs)

                out_sp.append(sp[n][indices])
                if nl is not None:
                    assert out_nl is not None
                    out_nl.append(nl[n][indices])

        else:
            probs = jnp.hstack(weights)
            probs = probs / jnp.sum(probs)

            key, indices = choose(key, probs)

            out_sp.append(sum(sp[1:], sp[0])[indices])
            if nl is not None:
                assert out_nl is not None
                out_nl.append(sum(nl[1:], nl[0])[indices])

        return (out_sp, out_nl, colvar, sti)

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
        (sp_list, nl_list, cv_old, sti) = self.data_loader(
            num=num_rounds,
            out=samples,
            rounds=rounds,
            new_r_cut=new_r_cut,
            split_data=split_data,
        )

        cvs_new, new_cv = self.transformer.fit(
            sp_list,
            nl_list,
            chunk_size=chunk_size,
            **kwargs,
        )

        if plot:
            sp = sum(sp_list[1:], sp_list[0])
            nl = sum(nl_list[1:], nl_list[0]) if nl_list is not None else None
            ind = np.random.choice(
                a=sp.shape[0],
                size=min(1000, sp.shape[0]),
                replace=False,
            )

            CVDiscovery.plot_app(
                name=str(rounds.folder / f"round_{rounds.round}" / "cvdiscovery"),
                old_cv=cv_old,
                new_cv=new_cv,
                sps=sp[ind],
                nl=nl[ind] if nl is not None else None,
                chunk_size=chunk_size,
            )

        return new_cv

    @staticmethod
    def plot_app(
        sps: SystemParams,
        nl: NeighbourList,
        old_cv: CollectiveVariable,
        new_cv: CollectiveVariable,
        name,
        labels=None,
        chunk_size: int | None = None,
    ):
        def color(c, per):
            c2 = (c - c.min()) / (c.max() - c.min())
            if not per:
                c2 *= 330.0 / 360.0

            col = np.ones((len(c), 3))
            col[:, 0] = c2

            return hsv_to_rgb(col)

        cv_data = []
        cv_data_mapped = []

        # raise "add neighlist"

        cvs = [old_cv, new_cv]
        for cv in cvs:
            cvd = cv.compute_cv(sps, nl, chunk_size=chunk_size)[0].cv
            cvdm = vmap(cv.metric.map)(cvd)

            cv_data.append(np.array(cvd))
            cv_data_mapped.append(np.array(cvdm))

        # for z, data in enumerate([cv_data, cv_data_mapped]):
        for z, data in enumerate([cv_data]):
            # plot setting
            kwargs = {"s": 0.2}

            if labels is None:
                labels = [
                    ["cv in 1", "cv in 2", "cv in 3"],
                    ["cv out 1", "cv out 2", "cv out 3"],
                ]

            # labels = [[r"$\Phi$", r"$\Psi$"], ["umap 1", "umap 2", "umap 3"]]
            for [i, j] in [[0, 1], [1, 0]]:  # order
                indim = cvs[i].n

                if indim == 1:
                    continue

                outdim = cvs[j].n

                if outdim == 2:
                    proj = None
                    wr = 1
                elif outdim == 3:
                    proj = "3d"
                    wr = 1
                else:
                    continue

                indim_pairs = list(itertools.combinations(range(indim), r=2))
                print(indim_pairs)

                fig = plt.figure()

                if outdim == 2:
                    spec = gridspec.GridSpec(
                        nrows=len(indim_pairs) * 2,
                        ncols=2,
                        width_ratios=[1, wr],
                        wspace=0.5,
                    )
                elif outdim == 3:
                    spec = gridspec.GridSpec(nrows=len(indim_pairs) * 2, ncols=3)

                for id, inpair in enumerate(indim_pairs):
                    for cc in range(2):
                        print(f"cc={cc}")

                        col = color(
                            data[i][:, inpair[cc]],
                            cvs[i].metric.periodicities[inpair[cc]],
                        )

                        if outdim == 2:
                            l = fig.add_subplot(spec[id * 2 + cc, 0])
                            r = fig.add_subplot(spec[id * 2 + cc, 1], projection=proj)
                        elif outdim == 3:
                            l = fig.add_subplot(spec[id * 2 + cc, 0])
                            r = [
                                fig.add_subplot(spec[id * 3 + cc, 1], projection=proj),
                                fig.add_subplot(spec[id * 3 + cc, 2], projection=proj),
                            ]

                        print(f"scatter={cc}")
                        l.scatter(*[data[i][:, l] for l in inpair], c=col, **kwargs)

                        l.set_xlabel(labels[i][inpair[0]])
                        l.set_ylabel(labels[i][inpair[1]])

                        if outdim == 2:
                            print("plot r 2d")
                            r.scatter(
                                *[data[j][:, l] for l in range(2)],
                                c=col,
                                **kwargs,
                            )
                            r.set_xlabel(labels[j][0])
                            r.set_ylabel(labels[j][1])

                        elif outdim == 3:
                            print("plot r 3d")

                            def plot3d(data, ax, colors=None, labels=labels[j], mode=0):
                                ax.set_xlabel(labels[0])
                                ax.set_ylabel(labels[1])
                                ax.set_zlabel(labels[2])

                                if mode == 0:
                                    ax.scatter(
                                        data[:, 0],
                                        data[:, 1],
                                        data[:, 2],
                                        **kwargs,
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

                                        # im = NonUniformImage(ax, interpolation='bilinear')

                                        zz = np.zeros(X.shape) - 0.1
                                        # zz = - Z

                                        if z == "z":
                                            ax.plot_surface(X, Y, zz, **kw)
                                        elif z == "y":
                                            ax.plot_surface(X, zz, Y, **kw)
                                        else:
                                            ax.plot_surface(zz, X, Y, **kw)

                                else:
                                    zz = np.zeros(data[:, 0].shape)
                                    for z in ["x", "y", "z"]:
                                        if z == "z":
                                            ax.scatter(
                                                data[:, 0],
                                                data[:, 1],
                                                zz,
                                                **kwargs,
                                                zorder=1,
                                                c=colors,
                                            )
                                        elif z == "y":
                                            ax.scatter(
                                                data[:, 0],
                                                zz,
                                                data[:, 2],
                                                **kwargs,
                                                zorder=1,
                                                c=colors,
                                            )
                                        else:
                                            ax.scatter(
                                                zz,
                                                data[:, 1],
                                                data[:, 2],
                                                **kwargs,
                                                zorder=1,
                                                c=colors,
                                            )

                                ax.view_init(elev=20, azim=45)

                            plot3d(data=data[j], colors=col, ax=r[0], mode=0)
                            plot3d(data=data[j], colors=col, ax=r[1], mode=1)

                # fig.set_size_inches([10, 16])

                n = Path(
                    f"{name}_{ 'mapped' if z==1 else ''}_{'old_new' if i == 0 else 'new_old'}.pdf",
                )

                n.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(n)

                # outputs.append(File(str(n)))
