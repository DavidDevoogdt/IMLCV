from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence, TypeVar

from flax.struct import dataclass as flax_dataclass
from flax.struct import field as _field
from jax import custom_jvp, jit, vmap
from typing_extensions import (
    dataclass_transform,  # pytype: disable=not-supported-yet
)

import dataclasses
import jax.numpy as jnp

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


def _format(value, indent=0):
    pad = "  " * indent
    child_pad = "  " * (indent + 1)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        lines = [f"{type(value).__name__}("]
        for f in dataclasses.fields(value):
            child = getattr(value, f.name)
            lines.append(f"{child_pad}{f.name}={_format(child, indent + 1)},")
        lines.append(f"{pad})")
        return "\n".join(lines)

    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = ["{"]
        for k, v in value.items():
            lines.append(f"{child_pad}{k!r}: {_format(v, indent + 1)},")
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    if isinstance(value, (list, tuple)):
        # treat as a leaf if it's a plain shape-like tuple of ints
        if isinstance(value, tuple) and all(isinstance(x, int) for x in value):
            return repr(value)
        open_b, close_b = ("[", "]") if isinstance(value, list) else ("(", ")")
        if not value:
            return open_b + close_b
        lines = [open_b]
        for v in value:
            lines.append(f"{child_pad}{_format(v, indent + 1)},")
        lines.append(f"{pad}{close_b}")
        return "\n".join(lines)

    # leaf: show shape if it looks like an array, else repr
    if hasattr(value, "shape"):
        return f"{type(value).__name__}{tuple(value.shape)}"

    return repr(value)


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

    def __str__(self):
        return _format(self)


field = _field


from typing import ParamSpec

P = ParamSpec("P")
P2 = ParamSpec("P2")
T = TypeVar("T")


def jit_decorator(
    f: Callable[P, T],
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    **jit_kwargs,
):
    def _f(*args: P.args, **kwargs: P.kwargs) -> T:
        return jit(f, static_argnames=static_argnames, static_argnums=static_argnums, **jit_kwargs)(*args, **kwargs)

    return _f


def vmap_decorator(
    f: Callable[P, T],
    in_axes: int | Sequence[Any] | None = 0,
    out_axes: Any = 0,
):
    def _f(*args: P.args, **kwargs: P.kwargs) -> T:
        # print(f"vmap_decorator: in_axes={in_axes}, out_axes={out_axes}")

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
