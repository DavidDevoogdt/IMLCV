from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence, TypeVar

from flax.struct import dataclass as flax_dataclass
from flax.struct import field as _field
from jax import custom_jvp, jit, vmap
from typing_extensions import (
    dataclass_transform,  # pytype: disable=not-supported-yet
)

#        Data types                  #
######################################s

TNode = TypeVar("TNode", bound="MyPyTreeNode")
from functools import partial

# def my_dataclass(cls):
#     return flax_dataclass(
#         cls,
#         field_specifiers=(_field,),
#         kw_only_default=True,
#     )


@dataclass_transform(
    field_specifiers=(_field,),
    kw_only_default=True,
)  # type: ignore[literal-required]
class MyPyTreeNode:
    """Base class for dataclasses that should act like a JAX pytree node."""

    def __init_subclass__(cls, **kwargs):
        # print(f"init subclass {cls=}  {kwargs=}")

        kwargs.update(
            {
                "kw_only": True,
                "frozen": False,
            }
        )

        flax_dataclass(cls, **kwargs)  # pytype: disable=wrong-arg-types

    def __init__(self, *args, **kwargs):
        # stub for pytype
        raise NotImplementedError

    def replace(self: TNode, **overrides) -> TNode:
        # stub for pytype
        raise NotImplementedError


field = _field


from typing import ParamSpec

P = ParamSpec("P")
P2 = ParamSpec("P2")
T = TypeVar("T")


def jit_decorator(
    f: Callable[P, T],
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
):
    def _f(*args: P.args, **kwargs: P.kwargs) -> T:
        return jit(f, static_argnames=static_argnames, static_argnums=static_argnums)(*args, **kwargs)

    return _f


def vmap_decorator(
    f: Callable[P, T],
    in_axes: int | Sequence[Any] | None = 0,
    out_axes: Any = 0,
):
    def _f(*args: P.args, **kwargs: P.kwargs) -> T:
        return vmap(f, in_axes=in_axes, out_axes=out_axes)(*args, **kwargs)

    return _f


def custom_jvp_decorator(
    f: Callable[P, T],
    # f_jvp: Callable,
    nondiff_argnums: Sequence[int] = (),
) -> Callable[P, T]:
    @partial(custom_jvp, nondiff_argnums=nondiff_argnums)
    def _f(*args: P.args, **kwargs: P.kwargs) -> T:
        return f(*args, **kwargs)

    return _f


from jax.tree_util import Partial


def Partial_decorator(
    f: Callable[P, T],
    *partial_args,
    **partial_kwargs,
):
    _g = Partial(f, *partial_args, **partial_kwargs)

    def _f(*args, **kwargs) -> T:
        return _g(*args, **kwargs)

    return _f
