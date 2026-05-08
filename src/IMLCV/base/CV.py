from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import jax.lax
import jax.numpy as jnp
import jsonpickle
from jax.tree_util import tree_flatten

from IMLCV import unpickler
from IMLCV.base.dataobjects import CV, CvMetric, NeighbourList, ShmapKwargs, SystemParams, X, padded_vmap
from IMLCV.base.datastructures import MyPyTreeNode, field

if TYPE_CHECKING:
    pass

from functools import partial

from IMLCV.base.datastructures import Partial_decorator, jit_decorator

######################################
#       CV tranformations            #
######################################


INITIALIZERS = {
    "lecun_normal": jax.nn.initializers.lecun_normal,
    "lecun_uniform": jax.nn.initializers.lecun_uniform,
    "glorot_normal": jax.nn.initializers.glorot_normal,
    "glorot_uniform": jax.nn.initializers.glorot_uniform,
    "he_normal": jax.nn.initializers.he_normal,
    "he_uniform": jax.nn.initializers.he_uniform,
    "normal": jax.nn.initializers.normal,
    "uniform": jax.nn.initializers.uniform,
    "zeros": lambda: jax.nn.initializers.zeros,
    "ones": lambda: jax.nn.initializers.ones,
    "orthogonal": jax.nn.initializers.orthogonal,
}


class CvFunBase(ABC, MyPyTreeNode):
    kwargs: dict = field(pytree_node=True, default_factory=dict)
    static_kwargs: dict = field(pytree_node=False, default_factory=dict)
    learnable_kwargs: tuple[str] | None = field(pytree_node=False, default=None)
    apply_rule: Callable[[dict, dict], dict] | None = field(pytree_node=False, default=None)
    custom_getstate: Callable[[CvFunBase], dict] | None = field(pytree_node=False, default=None)
    custom_setstate: Callable[[dict], dict] | None = field(pytree_node=False, default=None)
    initializers: dict[str, str | tuple[str, dict]] | Callable[[Any], dict] | None = field(
        pytree_node=False, default=None
    )

    @partial(
        jit_decorator,
        static_argnames=(
            "reverse",
            "shmap",
            # "shmap_kwargs",
            "chunk_size",
            "jacobian",
        ),
    )
    def compute_cv(
        self,
        x: CV | SystemParams,
        nl: NeighbourList | None = None,
        chunk_size=None,
        jacobian=False,
        reverse=False,
        shmap=False,
        shmap_kwargs=ShmapKwargs.create(),
        learnable_kwargs: dict | None = None,
    ) -> tuple[CV, CV | None]:
        if x.batched:
            # if nl is not None:
            #     assert nl.batched

            _f = padded_vmap(
                Partial_decorator(
                    self.compute_cv,
                    jacobian=jacobian,
                    chunk_size=None,
                    shmap=False,
                    reverse=reverse,
                    learnable_kwargs=learnable_kwargs,
                ),
                chunk_size=chunk_size,
                shmap=shmap,
            )

            # if shmap:
            #     _f = padded_shard_map(_f, shmap_kwargs)

            return _f(x, nl)

        def f(x):
            return self._compute_cv(
                x,
                nl,
                shmap=shmap,
                shmap_kwargs=shmap_kwargs,
                learnable_kwargs=learnable_kwargs,
            )[0]

        y = f(x)
        if jacobian:
            dy = jax.jacrev(f)(x)

            dy = dy.cv
        else:
            dy = None

        return y, dy

    @partial(
        jit_decorator,
        static_argnames=(
            "reverse",
            "shmap",
            # "shmap_kwargs",
        ),
    )
    @abstractmethod
    def _compute_cv(
        self,
        x: CV | SystemParams,
        nl: NeighbourList | None = None,
        reverse=False,
        shmap=False,
        shmap_kwargs=ShmapKwargs.create(),
        learnable_kwargs: dict | None = None,
    ) -> CV:
        raise

    def __getstate__(self):
        # print(f"{self.custom_getstate=}")
        if self.custom_getstate is not None:
            # print("using custom getstate")
            _d = self.custom_getstate(self)

        else:
            _d = self.__dict__

        # print(f"getstate dict keys: {_d.keys()}")

        return _d

    def __setstate__(self, statedict: dict):
        # print(f"{statedict.keys()=}")

        setstate_fun = statedict.get("custom_setstate", None)

        if setstate_fun is not None:
            # print(f"before custom setstate {statedict.keys()=}")
            statedict = setstate_fun(statedict)
            # print(f"after custom setstate {statedict=}")

        if "static_kwarg_names" in statedict:
            statedict["static_kwargs"] = {k: statedict.pop(k) for k in statedict["static_kwarg_names"]}

            statedict.pop("static_kwarg_names")

        self.__init__(**statedict)

    @property
    def learnable_params_shape(self):
        if self.learnable_kwargs is None:
            return None

        out = {}

        for k in self.learnable_kwargs:
            v = self.kwargs[k]

            if isinstance(v, jax.Array):
                out[k] = v.shape
            else:
                out[k] = jax.tree_util.tree_map(lambda x: x.shape, v)

        return out

    @property
    def num_learnable_params(self):
        p = self.learnable_params_shape

        if p is None:
            return 0

        n = 0

        x, _ = tree_flatten(p)
        for xi in x:
            if xi is not None:
                n += jnp.prod(jnp.array(xi))

        return n

    def apply_learnable_kwargs(self, learnable_kwargs: dict) -> CvFunBase:
        if self.learnable_kwargs is None:
            return self

        # print(f"{self.kwargs=} {learnable_kwargs=}")

        kw = self.kwargs.copy()

        for k in self.learnable_kwargs:
            if learnable_kwargs[k] is None:
                continue

            kw[k] = learnable_kwargs[k]

        if self.apply_rule is not None:
            kw = self.apply_rule(kw, self.static_kwargs)

        return self.replace(kwargs=kw)

    def init_learnable_params(self, key=jax.random.PRNGKey(0), initializer=jax.nn.initializers.lecun_normal()):
        if self.learnable_kwargs is None:
            return None

        if isinstance(self.initializers, Callable):
            print(f"initializing learnable params with {self.initializers=}")
            return self.initializers(
                key,
                self.kwargs,
                self.static_kwargs,
                self.learnable_kwargs,
            )

        kwargs = {}

        # print(f"{self.learnable_kwargs=}")
        for k in self.learnable_kwargs:
            v = self.kwargs[k]

            key, subkey = jax.random.split(key)

            def do_init(v):
                _initializer = None

                if self.initializers is not None:
                    if k in self.initializers:
                        _initializer = self.initializers[k]

                        if isinstance(_initializer, str):
                            name = _initializer.lower()
                            kwargs = {}
                            print(f"initializer for {k} is {name}")
                        elif isinstance(_initializer, tuple):
                            name, kwargs = _initializer
                            name = name.lower()

                            print(f"initializer for {k} is {name} with kwargs {kwargs}")

                        else:
                            raise ValueError(
                                f"Initializer for {k} must be either a string or a tuple of (string, dict). Got {self.initializers[k]}"
                            )

                        if name in INITIALIZERS:
                            _initializer = INITIALIZERS[name](**kwargs)
                        else:
                            raise ValueError(f"Initializer {name} not recognized.")

                if _initializer is None:
                    if v.ndim <= 1:
                        _initializer = INITIALIZERS["zeros"]()
                    else:
                        _initializer = initializer

                try:
                    # print(f"{k=} {v.shape=} {_initializer=}")

                    v_out = _initializer(subkey, v.shape, v.dtype)
                except Exception as e:
                    print(f"Error initializing parameter {k} with shape {v.shape} and dtype {v.dtype}: {e}")

                    v_out = jnp.zeros_like(v)

                return v_out

            if not isinstance(v, jax.Array):
                v_out = jax.tree_util.tree_map(do_init, v)
            else:
                v_out = do_init(v)

            kwargs[k] = v_out

        if self.apply_rule is not None:
            kwargs = self.apply_rule(kwargs, self.static_kwargs)

        return kwargs

    def get_learnable_params(self):
        if self.learnable_kwargs is None:
            return None

        kwargs = {}

        # print(f"{self.learnable_kwargs=}")
        for k in self.learnable_kwargs:
            v = self.kwargs[k]
            kwargs[k] = v

        return kwargs


class CvFun(CvFunBase):
    # __: KW_ONLY
    forward: Callable[[CV, NeighbourList | None, CV | None], CV] | None = field(pytree_node=False, default=None)
    backward: Callable[[CV, NeighbourList | None, CV | None], CV] | None = field(pytree_node=False, default=None)

    @partial(
        jit_decorator,
        static_argnames=(
            "reverse",
            "shmap",
            # "shmap_kwargs",
        ),
    )
    def _compute_cv(
        self,
        x: CV | SystemParams,
        nl: NeighbourList | None = None,
        reverse=False,
        shmap=False,
        shmap_kwargs=ShmapKwargs.create(),
        learnable_kwargs: dict | None = None,
    ) -> CV:
        kwargs = self.kwargs

        if learnable_kwargs is not None and self.learnable_kwargs is not None:
            print(f"applying learnable kwargs {learnable_kwargs=}")

            for k in self.learnable_kwargs:
                kwargs[k] = learnable_kwargs[k]

        if reverse:
            assert self.backward is not None
            return Partial_decorator(
                self.backward,
                shmap=shmap,
                shmap_kwargs=shmap_kwargs,
                **self.static_kwargs,
            )(x, nl, **self.kwargs)
        else:
            assert self.forward is not None
            return Partial_decorator(
                self.forward,
                shmap=shmap,
                shmap_kwargs=shmap_kwargs,
                **self.static_kwargs,
            )(x, nl, **self.kwargs)


##################
##  all CvFunBase are chained
##################


class _SerialCvTrans(MyPyTreeNode):
    """f can either be a single CV tranformation or a list of transformations"""

    trans: tuple[_SerialCvTrans | _ParralelCvTrans | CvFun, ...]

    @partial(
        jit_decorator,
        static_argnames=(
            "reverse",
            "shmap",
            # "shmap_kwargs",
        ),
    )
    def _compute_cv(
        self,
        x: CV | SystemParams,
        nl: NeighbourList | None = None,
        reverse=False,
        shmap=False,
        shmap_kwargs=ShmapKwargs.create(),
        learnable_kwargs: dict = {},
    ) -> CV:
        ordered = reversed(self.trans) if reverse else self.trans

        assert len(self.trans) > 0

        for i, tr in enumerate(ordered):
            x = tr._compute_cv(
                x=x,
                nl=nl,
                reverse=reverse,
                shmap=shmap,
                shmap_kwargs=shmap_kwargs,
            )

        return x  # type: ignore

    @property
    def learnable_params_shape(self):
        return [tr.learnable_params_shape for tr in self.trans]

    def apply_learnable_kwargs(self, learnable_kwargs) -> _SerialCvTrans:
        return self.replace(trans=tuple(tr.apply_learnable_kwargs(lp) for tr, lp in zip(self.trans, learnable_kwargs)))

    def init_learnable_params(self, key: jax.Array | None, initializer=jax.nn.initializers.lecun_normal()) -> dict:
        out = []

        if key is None:
            key = jax.random.PRNGKey(42)

        for tr in self.trans:
            key, subkey = jax.random.split(key)
            out.append(tr.init_learnable_params(subkey, initializer=initializer))
        return tuple(out)

    def get_learnable_params(self) -> dict:
        out = []

        for tr in self.trans:
            out.append(tr.get_learnable_params())
        return tuple(out)

    @property
    def num_learnable_params(self):
        n = 0

        for tr in self.trans:
            n += tr.num_learnable_params
        return n


class _ParralelCvTrans(MyPyTreeNode):
    trans: tuple[_SerialCvTrans | _ParralelCvTrans, ...]

    @partial(
        jit_decorator,
        static_argnames=(
            "reverse",
            "shmap",
            # "shmap_kwargs",
        ),
    )
    def _compute_cv(
        self,
        x: CV | SystemParams,
        nl: NeighbourList | None = None,
        reverse=False,
        shmap=False,
        shmap_kwargs=ShmapKwargs.create(),
    ) -> CV:
        def order(a):
            return reversed(a) if reverse else a

        o_trans = order(self.trans)

        out = []

        for _tr in o_trans:
            _x = _tr._compute_cv(
                x=x,
                nl=nl,
                reverse=reverse,
                shmap=shmap,
                shmap_kwargs=shmap_kwargs,
            )

            out.append(_x)

        # jax.debug.print("parallel cv  {out}", out=out)

        return CV(cv=jnp.hstack([cvi.cv for cvi in out]))

    @property
    def learnable_params_shape(self):
        return [tr.learnable_params_shape for tr in self.trans]

    @property
    def num_learnable_params(self):
        n = 0

        for tr in self.trans:
            n += tr.num_learnable_params
        return n

    def apply_learnable_kwargs(self, learnable_kwargs) -> _ParralelCvTrans:
        return self.replace(trans=tuple(tr.apply_learnable_kwargs(lp) for tr, lp in zip(self.trans, learnable_kwargs)))

    def init_learnable_params(self, key: jax.Array | None, initializer=jax.nn.initializers.lecun_normal()) -> dict:
        out = []

        if key is None:
            key = jax.random.PRNGKey(42)

        for tr in self.trans:
            key, subkey = jax.random.split(key)
            out.append(tr.init_learnable_params(subkey, initializer=initializer))

        return tuple(out)

    def get_learnable_params(self) -> dict:
        out = []

        for tr in self.trans:
            out.append(tr.get_learnable_params())

        return tuple(out)


class CvTrans(MyPyTreeNode):
    trans: _ParralelCvTrans | _SerialCvTrans

    @staticmethod
    def from_cv_function(
        f: Callable,
        # jacfun: Callable = None,
        static_argnames=None,
        check_input: bool = True,
        learnable_argnames: tuple[str, ...] | None = None,
        apply_rule: Callable[[dict, dict], dict] | None = None,
        initializers: dict[str, str] | Callable | None = None,
        custom_getstate: Callable[[CvFunBase], dict] | None = None,
        custom_setstate: Callable[[dict], dict] | None = None,
        **kwargs,
    ) -> CvTrans:
        static_kwargs = {}

        if static_argnames is not None:
            for a in static_argnames:
                static_kwargs[a] = kwargs.pop(a)

        kw = dict(
            forward=f,
            kwargs=kwargs,
            static_kwargs=static_kwargs,
            learnable_kwargs=learnable_argnames,
            apply_rule=apply_rule,
            initializers=initializers,
            custom_getstate=custom_getstate,
            custom_setstate=custom_setstate,
        )

        if check_input:

            @jit_decorator
            def _f(x):
                return x

            for k, v in list(kw["kwargs"].items()):
                try:
                    _f(v)
                except Exception as e:
                    print(
                        f"Error: {k} of type {type(v)} is not a valid jax type and should be added to static_argnames."
                    )
                    raise e

                # check if

            if learnable_argnames is not None:
                for a in learnable_argnames:
                    assert a in kw["kwargs"], f"learnable arg {a} not in kwargs"

            for k, v in list(kw["static_kwargs"].items()):
                try:
                    hash(v)
                    assert v == v
                except Exception as e:
                    print(
                        f"Error: {k} of type {type(v)} is not hashable, consider to make it hashable or not a static argument. exception: {e}"
                    )
                    raise

            if initializers is not None:
                if isinstance(initializers, Callable):
                    pass

                elif isinstance(initializers, dict):
                    for k in initializers.keys():
                        assert k in kw["kwargs"], f"initializer arg {k} not in kwargs"

                        p = initializers[k]
                        if isinstance(p, str):
                            p = p.lower()
                        elif isinstance(p, tuple):
                            p = p[0].lower()

                        assert p in INITIALIZERS, f"initializer {initializers[k]} not recognized."
                else:
                    raise ValueError("initializers must be either a dict or a callable")

        return CvTrans(trans=_SerialCvTrans(trans=(CvFun(**kw),)))  # type: ignore

    @partial(
        jit_decorator,
        static_argnames=(
            "reverse",
            "shmap",
            # "shmap_kwargs",
            "chunk_size",
            "jacobian",
        ),
    )
    def compute_cv(
        self,
        x: X,
        nl: NeighbourList | None = None,
        chunk_size=None,
        jacobian=False,
        reverse=False,
        shmap=False,
        shmap_kwargs=ShmapKwargs.create(),
    ) -> tuple[CV, X | None]:
        if x.batched:
            _f = padded_vmap(
                Partial_decorator(
                    self.compute_cv,
                    jacobian=jacobian,
                    chunk_size=None,
                    shmap=False,
                    reverse=reverse,
                ),
                chunk_size=chunk_size,
                shmap=shmap,
            )

            # if shmap:
            #     _f = padded_shard_map(_f, shmap_kwargs)

            return _f(x, nl)

        def f(x):
            return self.trans._compute_cv(x, nl, shmap=shmap, shmap_kwargs=shmap_kwargs)

        y = f(x)
        if jacobian:
            dy = jax.jacrev(f)(x)

            dy = dy.cv
        else:
            dy = None

        return y, dy

    def compute_cv_eval(
        self,
        x: X,
        nl: NeighbourList | None = None,
        chunk_size=None,
        reverse=False,
        shmap=False,
        shmap_kwargs=ShmapKwargs.create(),
    ) -> CV:
        return jax.eval_shape(
            self.compute_cv,
            x,
            nl,
            chunk_size=chunk_size,
            reverse=reverse,
            shmap=shmap,
            shmap_kwargs=shmap_kwargs,
        )

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        self.__init__(**statedict)

    def __mul__(self: CvTrans, other: CvTrans):
        return CvTrans(trans=_SerialCvTrans(trans=(self.trans, other.trans)))

    def __add__(self: CvTrans, other: CvTrans):
        return CvTrans(trans=_ParralelCvTrans(trans=(self.trans, other.trans)))

    def __radd__(self: CvTrans, other: CvTrans | None):
        if other is None:
            return self

        return self + other

    @property
    def learnable_params_shape(self):
        return [a.learnable_params_shape for a in self.trans.trans]

    def init_learnable_params(self, key: jax.Array | None, initializer=jax.nn.initializers.lecun_normal()):
        return self.trans.init_learnable_params(key, initializer=initializer)

    def apply_learnable_params(self, learnable_params: dict) -> CvTrans:
        return self.replace(trans=self.trans.apply_learnable_kwargs(learnable_params))

    def get_learnable_params(self) -> dict:
        return self.trans.get_learnable_params()

    @property
    def num_learnable_params(self):
        return self.trans.num_learnable_params


######################################
#       Collective variable          #
######################################


class CollectiveVariable(MyPyTreeNode):
    f: CvTrans
    metric: CvMetric
    jac: Callable = field(pytree_node=False, default=jax.jacrev)
    name: str = field(pytree_node=False, default="")
    cvs_name: tuple[str, ...] | None = field(pytree_node=False, default=None)
    extra_info: tuple[str, ...] | None = field(pytree_node=False, default=None)

    @partial(
        jit_decorator,
        static_argnames=[
            "chunk_size",
            "jacobian",
            "shmap",
            "push_jac",
            # "shmap_kwargs",
        ],
    )
    def compute_cv(
        self,
        sp: SystemParams,
        nl: NeighbourList | None = None,
        jacobian=False,
        chunk_size: int | None = None,
        shmap=False,
        push_jac=False,
        shmap_kwargs=ShmapKwargs.create(),
        learnable_kwargs=None,
    ) -> tuple[CV, SystemParams | None]:
        if push_jac:
            raise ValueError("push_jax not supported")

        return self.f.compute_cv(
            x=sp,
            nl=nl,
            jacobian=jacobian,
            chunk_size=chunk_size,
            shmap=shmap,
            shmap_kwargs=shmap_kwargs,
        )

    @property
    def n(self):
        return self.metric.ndim

    def save(self, file):
        filename = Path(file)

        if filename.suffix == ".json":
            with open(filename, "w") as f:
                f.writelines(jsonpickle.encode(self, indent=1, use_base85=True))  # type: ignore
        else:
            import cloudpickle

            with open(filename, "wb") as f:
                cloudpickle.dump(self, f)

    @staticmethod
    def load(file, **kwargs) -> CollectiveVariable:
        filename = Path(file)

        # print("loading CV")

        if filename.suffix == ".json":
            with open(filename) as f:
                self = jsonpickle.decode(f.read(), context=unpickler)
        else:
            import cloudpickle

            with open(filename, "rb") as f:
                self = cloudpickle.load(f)

        assert isinstance(self, CollectiveVariable)

        for key in kwargs.keys():
            self.__setattr__(key, kwargs[key])

        return self

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        self.__init__(**statedict)

    def __getitem__(self, tup):
        from IMLCV.implementations.CV import _cv_slice

        assert isinstance(tup, tuple)

        return CollectiveVariable(
            f=self.f
            * CvTrans.from_cv_function(
                _cv_slice,
                indices=jnp.array(tup).reshape((-1,)),
            ),
            jac=self.jac,
            metric=self.metric[tup],
            name=self.name,
            cvs_name=tuple(self.cvs_name[i] for i in tup) if self.cvs_name is not None else None,
            extra_info=tuple(self.extra_info[i] for i in tup) if self.extra_info is not None else None,
        )
