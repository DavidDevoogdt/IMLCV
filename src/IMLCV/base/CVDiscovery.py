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
from IMLCV.base.MdEngine import StaticTrajectoryInfo
from IMLCV.base.rounds import Rounds
from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.implementations.CV import distance_descriptor
from IMLCV.implementations.CV import sb_descriptor
from IMLCV.implementations.CV import scale_cv_trans
from jax import jacrev
from jax import vmap
from jax.random import choice
from jax.random import PRNGKey
from matplotlib import gridspec
from matplotlib.colors import hsv_to_rgb
from parsl.data_provider.files import File

plt.rcParams["text.usetex"] = True


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
        z: SystemParams,
        nl: NeighbourList | None,
        chunk_size=None,
        scale=True,
    ) -> tuple[CV, CvFlow]:
        f = self.descriptor
        x = f.compute_cv_flow(z, nl, chunk_size=chunk_size)

        if scale:
            g = scale_cv_trans(x)
            x, _ = g.compute_cv_trans(x)

            f = f * g

        return x, f

    def fit(
        self,
        sp: SystemParams,
        nl: NeighbourList | None,
        chunk_size=None,
        prescale=True,
        postscale=True,
        *fit_args,
        **fit_kwargs,
    ) -> tuple[CV, CollectiveVariable]:
        x, f = self.pre_fit(
            sp,
            nl,
            scale=prescale,
            chunk_size=chunk_size,
        )
        y, g = self._fit(
            x,
            # indices,
            *fit_args,
            **fit_kwargs,
        )
        z, h = self.post_fit(y, scale=postscale)

        cv = CollectiveVariable(
            f=f * g * h,
            metric=CvMetric(periodicities=self.periodicity),
            jac=jacrev,
        )

        return z, cv

    def _fit(
        self,
        x: CV,
        **kwargs,
    ) -> tuple[CV, CvFlow]:
        raise NotImplementedError

    def post_fit(self, y: CV, scale) -> tuple[CV, CvTrans]:
        if not scale:
            return y, CvTrans.from_cv_function(lambda x, y: x)
        h = scale_cv_trans(y)
        return h.compute_cv_trans(y)[0], h


class CVDiscovery:
    """convert set of coordinates to good collective variables."""

    def __init__(self, transformer: Transformer) -> None:
        # self.rounds = rounds
        self.transformer = transformer

    def _get_data(
        self,
        rounds: Rounds,
        num=4,
        out=1e4,
        chunk_size=None,
    ) -> tuple[SystemParams, CV, CollectiveVariable, StaticTrajectoryInfo]:
        weights = []

        colvar: CollectiveVariable | None = None
        sti: StaticTrajectoryInfo | None = None

        sp: SystemParams | None = None
        cv: CV | None = None

        for round, traj in rounds.iter(stop=None, num=num):
            if sti is None:
                sti = round.tic

            # map cvs
            bias = traj.get_bias()

            if colvar is None:
                colvar = bias.collective_variable

            sp0 = traj.ti.sp

            if traj.ti.cv is not None:
                cv0 = traj.ti.CV
            else:
                nl0 = sp0.get_neighbour_list(
                    r_cut=round.tic.r_cut,
                    z_array=round.tic.atomic_numbers,
                )
                cv0, _ = bias.collective_variable.compute_cv(sp=sp0, nl=nl0)
            if sp is None:
                sp = sp0
                cv = cv0
            else:
                sp += sp0
                cv = CV.stack(cv, cv0)  # type:ignore

            biases, _ = bias.compute_from_cv(cvs=cv0)

            beta = 1 / round.tic.T
            weight = jnp.exp(beta * biases)

            weights.append(weight)

        assert sp is not None
        assert cv is not None
        assert colvar is not None
        assert sti is not None

        # todo modify probability
        probs = jnp.hstack(weights)
        probs = probs / jnp.sum(probs)

        key = PRNGKey(0)

        indices = choice(
            key=key,
            a=probs.shape[0],
            shape=(int(out),),
            # p=probs,
            replace=False,
        )

        return (
            sp[indices],
            cv,
            colvar,
            sti,
        )

    def compute(
        self,
        rounds: Rounds,
        num_rounds=4,
        samples=3e3,
        plot=True,
        r_cut=None,
        chunk_size=None,
        name=None,
        **kwargs,
    ) -> CollectiveVariable:
        (
            sps,
            _,
            cv_old,
            sti,
        ) = self._get_data(
            num=num_rounds,
            out=samples,
            rounds=rounds,
            chunk_size=chunk_size,
        )

        nls = sps.get_neighbour_list(r_cut=r_cut, z_array=sti.atomic_numbers)

        cvs_new, new_cv = self.transformer.fit(
            sps,
            nls,
            sti=sti,
            chunk_size=chunk_size,
            **kwargs,
        )

        if plot:
            ind = np.random.choice(
                a=sps.shape[0],
                size=min(3000, sps.shape[0]),
                replace=False,
            )

            fut = plot_app(
                name="cvdiscovery",
                old_cv=cv_old,
                new_cv=new_cv,
                sps=sps[ind],
                nl=nls[ind] if nls is not None else None,
                execution_folder=f"{rounds.folder}/round_{rounds.round}",
            )

            fut.result()

        return new_cv


@bash_app_python(executors=["default"])
def plot_app(
    sps,
    nl: NeighbourList,
    old_cv: CollectiveVariable,
    new_cv: CollectiveVariable,
    name,
    outputs=[],
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
        cvd = cv.compute_cv(sps, nl)[0].cv
        cvdm = vmap(cv.metric.map)(cvd)

        cv_data.append(np.array(cvd))
        cv_data_mapped.append(np.array(cvdm))

    # for z, data in enumerate([cv_data, cv_data_mapped]):
    for z, data in enumerate([cv_data]):
        # plot setting
        kwargs = {"s": 0.2}

        labels = [[r"$\Phi$", r"$\Psi$"], ["umap 1", "umap 2", "umap 3"]]
        for [i, j] in [[0, 1], [1, 0]]:  # order
            indim = cvs[i].n
            outdim = cvs[j].n

            if outdim == 2:
                proj = None
                wr = 1
            elif outdim == 3:
                proj = "3d"
                wr = 1
            else:
                raise NotImplementedError

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
                        r.scatter(*[data[j][:, l] for l in range(2)], c=col, **kwargs)
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

                                for a, b, z in [[0, 1, "z"], [0, 2, "y"], [1, 2, "x"]]:
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

            outputs.append(File(str(n)))
