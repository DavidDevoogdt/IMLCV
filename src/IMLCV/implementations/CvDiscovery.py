from functools import partial

import jax
import jax.numpy as jnp
import optax

from flax import linen as nn
from flax.training import train_state
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvFun
from IMLCV.base.CV import CvTrans
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CVDiscovery import Transformer
from IMLCV.implementations.CV import get_non_constant_trans
from IMLCV.implementations.CV import trunc_svd
from IMLCV.implementations.CV import un_atomize
from jax import Array
from jax import jit
from jax import random
from jax import vmap
from molmod.units import nanosecond
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from IMLCV.base.rounds import data_loader_output

# https://arxiv.org/pdf/1602.08776.pdf appendix b


def shrink(S: Array, n: int, shrinkage="OAS"):
    if shrinkage == "None":
        return S
    # Todo https://papers.nips.cc/paper_files/paper/2014/file/fa83a11a198d5a7f0bf77a1987bcd006-Paper.pdf
    # todo: paper Covariance shrinkage for autocorrelated data
    assert shrinkage in ["RBLW", "OAS", "BC"]

    p = S.shape[0]
    F = jnp.trace(S) / p * jnp.eye(p)

    tr_s2 = jnp.trace(S**2)
    tr2_s = jnp.trace(S) ** 2

    if shrinkage == "RBLW":
        rho = ((n - 2) / n * tr_s2 + tr2_s) / ((n + 2) * (tr_s2 - tr2_s / p))
    elif shrinkage == "OAS":
        # use oracle https://arxiv.org/pdf/0907.4698.pdf, eq 23
        rho = ((1 - 2 / p) * tr_s2 + tr2_s) / ((n + 1 - 2 / p) * (tr_s2 - tr2_s / p))

    elif shrinkage == "BC":
        # https://proceedings.neurips.cc/paper_files/paper/2014/file/fa83a11a198d5a7f0bf77a1987bcd006-Paper.pdf
        pass
        # shrinkage based on  X
        # n = X.shape[0]
        # p = X.shape[1]
        # b = 20

        # u = X - pi_x
        # v = Y - pi_y

        # S_0 = jnp.einsum("ti,tj->ij", u, u) / (n - 1)
        # S_1 = jnp.einsum("ti,tj->ij", u, v) / (n - 1)

        # T_0 = jnp.trace(S_0) / p * jnp.eye(p)
        # T_1 = jnp.trace(S_1) / p * jnp.eye(p)

        # def gamma(s, u, v, S):
        #     return (
        #         jnp.einsum(
        #             "ti,tj,ti,tj,t->ij", u[: n - s, :], w[: n - s], v[: n - s, :], u[s:, :], w[s:], v[s:, :], w
        #         )
        #         - jnp.sum(w[: n - s]) / jnp.sum(w) * S**2
        #     )

        # var_BC = gamma(0)
        # for i in range(1, b + 1):
        #     var_BC += 2 * gamma(i)
        # var_BC /= n - 1 - 2 * b + b * (b + 1) / n

        # lambda_BC = jnp.einsum("ij,ij", var_BC, var_BC) / jnp.einsum("ij,ij", S_0 - T_0, S_0 - T_0)

        # lambda_BC = jnp.clip(lambda_BC, 0, 1)

        # print(f"lambda_BC = {lambda_BC}")

        #

    if rho > 1:
        rho = 1

    print(f"{rho=}")

    def f(C):
        return rho * F + (1 - rho) * C

    return f


class Encoder(nn.Module):
    latents: int
    layers: int
    nunits: int
    dim: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.layers):
            x = nn.Dense(
                self.nunits,
                name=f"encoder_{i}",
                # bias_init=jax.nn.initializers.normal(),
                kernel_init=jax.nn.initializers.xavier_normal(),
            )(x)
            x = nn.tanh(x)
        mean_x = nn.Dense(
            self.latents,
            name="fc2_mean",
            # bias_init=jax.nn.initializers.normal(),
            # kernel_init=jax.nn.initializers.normal(),
        )(x)
        logvar_x = nn.Dense(
            self.latents,
            name="fc2_logvar",
            # bias_init=jax.nn.initializers.normal(),
            # kernel_init=jax.nn.initializers.normal(),
        )(x)
        return mean_x, logvar_x


class Decoder(nn.Module):
    latents: int
    layers: int
    nunits: int
    dim: int

    @nn.compact
    def __call__(self, z):
        for i in range(self.layers):
            z = nn.Dense(
                self.nunits,
                name=f"decoder_{i}",
                # bias_init=jax.nn.initializers.normal(),
                kernel_init=jax.nn.initializers.xavier_normal(),
            )(z)
            z = nn.tanh(z)
        z = nn.Dense(
            self.dim,
            name="fc2",
            # bias_init=jax.nn.initializers.normal(),
            kernel_init=jax.nn.initializers.normal(),
        )(z)
        return z


class VAE(nn.Module):
    latents: int
    layers: int
    nunits: int
    dim: int

    def setup(self):
        self.encoder = Encoder(self.latents, self.layers, self.nunits, self.dim)
        self.decoder = Decoder(self.latents, self.layers, self.nunits, self.dim)

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = VAE.reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    # def generate(self, z):
    #     return nn.sigmoid(self.decoder(z))

    def encode(self, x):
        mean, _ = self.encoder(x)
        return mean

    @classmethod
    def reparameterize(cls, rng, mean, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, logvar.shape)
        return mean + eps * std


class TranformerAutoEncoder(Transformer):
    def _fit(
        self,
        cv: list[CV],
        cv_t: list[CV] | None,
        dlo: data_loader_output,
        nunits=250,
        nlayers=3,
        lr=1e-4,
        num_epochs=100,
        batch_size=32,
        chunk_size=None,
        verbose=True,
        **kwargs,
    ):
        # import wandb

        # wandb.init(
        #     project="TranformerAutoEncode",
        #     config={
        #         "cv": cv,
        #         "nunits": nunits,
        #         "nlayers": nlayers,
        #         "lr": lr,
        #         "num_epochs": num_epochs,
        #         "batch_size": batch_size,
        #         "outdim": self.outdim,
        #         **self.fit_kwargs,
        #         **kwargs,
        #     },
        # )

        cv = CV.stack(*cv)
        cv = un_atomize.compute_cv_trans(cv, None)[0]

        rng = random.PRNGKey(0)

        dim = cv.shape[1]

        vae_args = {
            "latents": self.outdim,
            "layers": nlayers,
            "nunits": nunits,
            "dim": dim,
        }

        @jax.vmap
        def kl_divergence(mean, logvar):
            return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

        @jax.vmap
        def mean_Squared_error(x1, x2):
            return 0.5 * jnp.linalg.norm(x1 - x2) ** 2

        def compute_metrics(recon_x, x, mean, logvar):
            bce_loss = mean_Squared_error(recon_x, x).mean()
            kld_loss = 0.01 * kl_divergence(mean, logvar).mean()
            return {"bce": bce_loss, "kld": kld_loss, "loss": bce_loss + kld_loss}

        # @jax.vmap
        # def binary_cross_entropy_with_logits(logits, labels):
        #     logits = nn.log_sigmoid(logits)
        #     return -jnp.sum(
        #         labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits))
        #     )

        @jax.jit
        def train_step(state: optax.TraceState, batch, z_rng):
            def loss_fn(params):
                recon_x, mean, logvar = VAE(**vae_args).apply(
                    {"params": params},
                    batch,
                    z_rng,
                )

                bce_loss = mean_Squared_error(recon_x, batch).mean()
                # bce_loss = mean_Squared_error(recon_x, batch)
                kld_loss = kl_divergence(mean, logvar).mean()
                loss = bce_loss + kld_loss
                return loss

            grads = jax.grad(loss_fn)(state.params)

            return state.apply_gradients(grads=grads)

        @jax.jit
        def eval(params, x, z_rng):
            def eval_model(vae):
                recon_x, mean, logvar = vae(x, z_rng)
                comparison = jnp.stack([x, recon_x])

                metrics = compute_metrics(recon_x, x, mean, logvar)
                return metrics, comparison

            return nn.apply(eval_model, VAE(**vae_args))({"params": params})

        # Test encoder implementation
        # Random key for initialization
        key, rng = jax.random.split(rng, 2)

        init_data = jax.random.normal(key, (batch_size, dim), jnp.float32)

        key, rng = jax.random.split(rng, 2)

        state = train_state.TrainState.create(
            apply_fn=VAE(**vae_args).apply,
            params=VAE(**vae_args).init(key, init_data, rng)["params"],
            tx=optax.adam(lr),
        )

        rng, key, eval_rng = random.split(rng, 3)

        x = cv.cv
        x = random.permutation(key, x)

        split = x.shape[0] // 10

        x_train = x[0:-split, :]
        x_test = x[-split:, :]

        steps_per_epoch = x_train.shape[0] // batch_size

        for epoch in range(num_epochs):
            rng, key = random.split(rng)
            indices = jax.random.choice(
                key=key,
                a=int(x.shape[0]),
                shape=(steps_per_epoch, batch_size),
                replace=False,
            )

            for n, index in enumerate(indices):
                rng, key = random.split(rng)
                state = train_step(state, x_train[index, :], key)

                if n % 5000:
                    metrics, comparison = eval(state.params, x_test, eval_rng)
                    # wandb.log(
                    #     {
                    #         "loss": metrics["loss"],
                    #         "kld": metrics["kld"],
                    #         "distance": metrics["bce"],
                    #     }
                    # )

                    print(
                        "eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}".format(
                            epoch + 1,
                            metrics["loss"],
                            metrics["bce"],
                            metrics["kld"],
                        ),
                    )

            # @partial(jit, static_argnums=(0,))

        def forward(x: CV, nl, y: list[CV] | None = None, _=None):
            assert y is None
            encoded: Array = VAE(**vae_args).apply(
                {"params": state.params},
                x.cv,
                method=VAE.encode,
            )
            return x.replace(cv=encoded)

        f_enc = CvTrans(trans=(CvFun(forward=forward),))

        cv = f_enc.compute_cv_trans(cv)[0].unstack()
        if cv_t is not None:
            cv_t = f_enc.compute_cv_trans(cv_t)[0].unstack()

        return cv, cv_t, un_atomize * f_enc, None


def _LDA_trans(cv: CV, nl: NeighbourList | None, _, alpha, outdim, solver):
    if solver == "eigen":

        def f(cv, scalings):
            return cv @ scalings

        f = partial(f, scalings=jnp.array(alpha.scalings_)[:, :outdim])

    elif solver == "svd":

        def f(cv, scalings, xbar):
            return (cv - xbar) @ scalings

        f = partial(
            f,
            scalings=jnp.array(alpha.scalings_),
            xbar=jnp.array(alpha.xbar_),
        )

    else:
        raise NotImplementedError

    return CV(
        cv=f(cv.cv),
        _stack_dims=cv._stack_dims,
        _combine_dims=cv._combine_dims,
        atomic=cv.atomic,
        mapped=cv.mapped,
    )


def _LDA_rescale(cv: CV, nl: NeighbourList | None, _, mean):
    return CV(
        cv=(cv.cv - mean[0]) / (mean[1] - mean[0]),
        _stack_dims=cv._stack_dims,
        _combine_dims=cv._combine_dims,
        atomic=cv.atomic,
        mapped=cv.mapped,
    )


def _scale_trans(cv: CV, nl: NeighbourList | None, _, alpha, scale_factor):
    return CV(
        (alpha.T @ cv.cv - scale_factor[0, :]) / (scale_factor[1, :] - scale_factor[0, :]),
        _stack_dims=cv._stack_dims,
        _combine_dims=cv._combine_dims,
        atomic=cv.atomic,
        mapped=cv.mapped,
    )


class TransoformerLDA(Transformer):
    def __init__(
        self,
        outdim: int,
        kernel=False,
        optimizer=None,
        solver="eigen",
        method="pymanopt",
        harmonic=True,
        min_gradient_norm: float = 1e-3,
        min_step_size: float = 1e-3,
        max_iterations=25,
        **kwargs,
    ):
        super().__init__(
            outdim=outdim,
            kernel=kernel,
            optimizer=optimizer,
            solver=solver,
            method=method,
            harmonic=harmonic,
            min_gradient_norm=min_gradient_norm,
            min_step_size=min_step_size,
            max_iterations=max_iterations,
            **kwargs,
        )

    def _fit(
        self,
        cv_list: list[CV],
        cv_t: list[CV] | None,
        dlo: data_loader_output,
        kernel=False,
        optimizer=None,
        chunk_size=None,
        solver="eigen",
        method="pymanopt",
        harmonic=True,
        min_gradient_norm: float = 1e-3,
        min_step_size: float = 1e-3,
        max_iterations=25,
        verbose=True,
        **kwargs,
    ):
        # nl_list = dlo.nl

        if kernel:
            raise NotImplementedError("kernel not implemented for lda")

        cv = CV.stack(*cv_list)
        # nl = NeighbourList.stack(*nl_list)
        cv, _, _ = un_atomize.compute_cv_trans(cv)

        if method == "sklearn":
            labels = []
            for i, cvi in enumerate(cv_list):
                labels.append(jnp.full(cvi.shape[0], i))

            labels = jnp.hstack(labels)

            alpha = LDA(n_components=self.outdim, solver=solver, shrinkage="auto").fit(
                cv.cv,
                labels,
            )

            lda_cv = CvTrans.from_cv_function(_LDA_trans, alpha=alpha, outdim=self.outdim, solver=solver)
            cv, _, _ = lda_cv.compute_cv_trans(cv)

            cvs = CV.unstack(cv)

            mean = []
            for i, cvs_i in enumerate(cvs):
                mean.append(jnp.mean(cvs_i.cv, axis=0))
            mean = jnp.array(mean)

            assert self.outdim == 1

            lda_rescale = CvTrans.from_cv_function(_LDA_rescale, mean=mean)
            cv, _, _ = lda_rescale.compute_cv_trans(cv)

            full_trans = un_atomize * lda_cv * lda_rescale

        elif method == "pymanopt":
            import pymanopt

            assert isinstance(cv_list, list)
            if optimizer is None:
                optimizer = pymanopt.optimizers.TrustRegions(
                    max_iterations=max_iterations,
                    min_gradient_norm=min_gradient_norm,
                    min_step_size=min_step_size,
                )

            cv, _f = trunc_svd(cv)
            normed_cv_u = CV.unstack(cv)

            mu_i = [jnp.mean(x.cv, axis=0) for x in normed_cv_u]
            mu = jnp.einsum(
                "i,ij->j",
                jnp.array(cv.stack_dims) / jnp.sum(jnp.array(cv.stack_dims)),
                jnp.array(mu_i),
            )

            obersevations_within = CV.stack(*[cv_i - mu_i for mu_i, cv_i in zip(mu_i, normed_cv_u)])
            observations_between = CV.stack(*[cv_i - mu for cv_i in normed_cv_u])

            cov_w = LedoitWolf(assume_centered=True).fit(obersevations_within.cv).covariance_
            cov_b = LedoitWolf(assume_centered=True).fit(observations_between.cv).covariance_

            manifold = pymanopt.manifolds.stiefel.Stiefel(n=mu.shape[0], p=self.outdim)

            @pymanopt.function.jax(manifold)
            @jit
            def cost(x):
                a = jnp.trace(x.T @ cov_w @ x)
                b = jnp.trace(x.T @ cov_b @ x)

                if harmonic:
                    out = a / b
                else:
                    out = -(b / a)

                return out

            problem = pymanopt.Problem(manifold, cost)
            result = optimizer.run(problem)

            alpha = result.point

            scale_factor = vmap(lambda x: alpha.T @ x)(jnp.array(mu_i))

            _g = CvTrans.from_cv_function(_scale_trans, alpha=alpha, scale_factor=scale_factor)

            cv = _g.compute_cv_trans(cv)[0]
            if cv_t is not None:
                cv_t = _g.compute_cv_trans(cv_t)[0]

            full_trans = un_atomize * _f * _g

        return cv.unstack(), cv_t.unstack(), full_trans, None


def _transform(cv: CV, nl: NeighbourList | None, _, mask):
    return cv.replace(cv=cv.cv[mask])


def _cv_slice(cv: CV, nl: NeighbourList, _, indices):
    return cv.replace(cv=jnp.take(cv.cv, indices, axis=-1), _combine_dims=None)


def _tranform_maf(cv, nl, _, pi, add_1, q):
    cv_v = (cv.cv - pi) @ q

    if add_1:
        cv_v = jnp.hstack([cv_v, jnp.array([1])])

    return CV(cv=cv_v, _stack_dims=cv._stack_dims)


def _tica_selection(cv: CV, nl: NeighbourList | None, _, pi_eq, alpha):
    return CV(cv=(cv.cv - pi_eq) @ alpha, _stack_dims=cv._stack_dims)


def _tica_selection_2(cv: CV, nl: NeighbourList | None, _, u):
    return cv @ u


class TransformerMAF(Transformer):
    # Maximum Autocorrelation Factors

    def _fit(
        self,
        x: list[CV],
        x_t: list[CV] | None,
        dlo: data_loader_output,
        correct_bias=False,
        pre_selction_epsilon=1e-10,
        max_features=2000,
        max_functions=2500,
        koopman_weighting=False,
        kinetic_distance=False,
        method="tcca",
        use_ground_bias=False,
        T_scale=3,
        verbose=True,
        **fit_kwargs,
    ) -> tuple[CV, CvTrans]:
        assert dlo.time_series
        assert x_t is not None

        print("stacking")

        cv_0 = CV.stack(*x)
        cv_tau = CV.stack(*x_t)

        trans = un_atomize

        print("unatomizing")

        cv_0, _, _ = un_atomize.compute_cv_trans(cv_0)
        cv_tau, _, _ = un_atomize.compute_cv_trans(cv_tau)

        cv_0, transform = get_non_constant_trans(cv_0, epsilon=pre_selction_epsilon, max_functions=max_functions)
        cv_tau, _, _ = transform.compute_cv_trans(cv_tau)

        print("getting feature cov")

        # cv_0, cv_tau, transform = get_feature_cov(
        #     cv_0,
        #     cv_tau,
        #     max_functions=max_functions,
        #     epsilon=pre_selction_epsilon,
        #     abs_val=False,
        # )

        trans *= transform

        w = None

        print("getting weights")

        if correct_bias:
            # koopman according to idnicator functions in orginal CV space before FES reweighing
            w = dlo.weights(
                correct_U=True,
                correct_FES=False,
                koopman=False,
                n_max=60,
                T_scale=1,
            )

        print("getting koopman")

        km = dlo.koopman_model(
            cv_0=cv_0.unstack(),
            cv_tau=cv_tau.unstack(),
            eps=1e-10,
            method=method,
            max_features=max_features,
            w=w,
            koopman_weight=koopman_weighting,
        )

        ts = km.timescales() / nanosecond

        print(f"timescales {  ts[: min(self.outdim+5,len(ts))  ]   } ns")

        trans *= km.f(out_dim=self.outdim)

        return trans.compute_cv_trans(x=cv_0)[0].unstack(), trans.compute_cv_trans(x=cv_tau)[0].unstack(), trans, km.w
