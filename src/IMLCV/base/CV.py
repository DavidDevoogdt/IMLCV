from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Concatenate

import jax.lax
import jax.numpy as jnp
import jax.sharding as jshard
import jsonpickle
from jax import Array, jit
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec
from jax.tree_util import tree_flatten, tree_unflatten
from typing_extensions import ParamSpec

from IMLCV import unpickler
from IMLCV.base.datastructures import MyPyTreeNode, field

if TYPE_CHECKING:
    from IMLCV.base.MdEngine import StaticMdInfo

from functools import partial
from typing import TypeVar, cast

from IMLCV.base.datastructures import Partial_decorator, jit_decorator, vmap_decorator

T = TypeVar("T")
S = TypeVar("S")

P = ParamSpec("P")
P2 = ParamSpec("P2")


######################################
#        Data types                  #
######################################s


class ShmapKwargs(MyPyTreeNode):
    axis: int = 0
    out_axes: int = 0
    axis_name: str | None = field(pytree_node=False, default="i")
    n_devices: int | None = None
    pmap: bool = False
    explicit_shmap: bool = False
    verbose: bool = False
    device_get: bool = True
    devices: tuple[Any] = field(pytree_node=False, default=None)
    mesh: Mesh | None = field(pytree_node=False, default=None)

    @staticmethod
    def create(
        axis: int = 0,
        out_axes: int = 0,
        axis_name: str | None = "i",
        n_devices: int | None = None,
        pmap: bool = False,
        explicit_shmap: bool = False,
        verbose: bool = False,
        device_get: bool = True,
        devices=None,
        mesh=None,
    ):
        if devices is None:
            devices = tuple(jax.devices("cpu"))

        if mesh is None:
            mesh = Mesh(devices, axis_names=(axis_name,))

        return ShmapKwargs(
            axis=axis,
            out_axes=out_axes,
            axis_name=axis_name,
            n_devices=n_devices,
            pmap=pmap,
            explicit_shmap=explicit_shmap,
            verbose=verbose,
            device_get=device_get,
            devices=devices,
            mesh=mesh,
        )


@partial(
    jit_decorator,
    static_argnames=(
        "p",
        "axis",
        "chunk_size",
        "reshape",
        "move_axis",
        "n_chunk_move",
    ),
)
def _n_pad(
    x,
    axis,
    p,
    chunk_size,
    reshape=False,
    move_axis=True,
    n_chunk_move=False,
):
    if x is None:
        return x, None

    if p != 0:
        pad_shape = [[0, p if i == axis else 0] for i in range(x.ndim)]

        x_padded = jnp.pad(x, pad_shape)
    else:
        x_padded = x

    if reshape:
        shape = x_padded.shape

        x_padded = jnp.reshape(x_padded, (*shape[:axis], chunk_size, -1, *shape[axis + 1 :]))

    if move_axis:
        if n_chunk_move:
            x_padded = jnp.moveaxis(x_padded, axis + 1, 0)
        else:
            x_padded = jnp.moveaxis(x_padded, axis, 0)

    return x_padded


@partial(
    jit_decorator,
    static_argnames=(
        "axis",
        "reshape",
        "shape",
        "move_axis",
        "trim",
        "n_chunk_move",
    ),
)
def _n_unpad(x, axis, shape, reshape=False, move_axis=True, trim=True, n_chunk_move=False):
    if move_axis:
        if n_chunk_move:
            x = jnp.moveaxis(x, 0, axis + 1)
        else:
            x = jnp.moveaxis(x, 0, axis)

    if reshape:
        x = jnp.reshape(x, (*x.shape[0:axis], -1, *x.shape[axis + 2 :]))

    if trim:
        x = jnp.apply_along_axis(lambda x: x[:shape], axis, x)

    return x


def _shard(x_padded, axis, axis_name, mesh, put=True, unflatten=True):
    shardings = []
    specs = []

    for i in range(len(x_padded)):
        ps = [None] * x_padded[i].ndim
        ps[axis] = axis_name
        spec = PartitionSpec(*ps)
        sharding = jshard.NamedSharding(mesh, spec)

        if put:
            x_padded[i] = jax.device_put(x_padded[i], sharding)

        shardings.append(sharding)
        specs.append(spec)

    specs = tuple(specs)

    return x_padded, shardings, specs


def _shard_out(out_tree_def, axis, axis_name, mesh):
    shardings = []

    specs = []

    for out_tree_i in out_tree_def:
        ps = [None] * len(out_tree_i.shape)
        ps[axis] = axis_name
        spec = PartitionSpec(*ps)

        sharding = jshard.NamedSharding(mesh, spec)

        shardings.append(sharding)
        specs.append(spec)

    specs = tuple(specs)

    return specs, shardings


def padded_shard_map(
    f: Callable[P, T],
    kwargs: ShmapKwargs = ShmapKwargs.create(),
):
    # helper function to pad pytree, apply pmap/shmap and unpad the result
    # strategy: unmap pytree to arrays, pad the arrays, reshape sush that the axis in front equals number of threads, apply pmap/shmap, unpad the arrays, remap to output pytree

    def apply_pmap_fn(
        # kwargs,
        *args: P.args,
        **f_kwargs: P.kwargs,
    ):
        # print("no shmap")

        return f(*args, **f_kwargs)

        axis = kwargs.axis
        n_devices = kwargs.n_devices
        axis_name = kwargs.axis_name
        pmap = kwargs.pmap
        explicit_shmap = kwargs.explicit_shmap
        verbose = kwargs.verbose
        # device_get = kwargs.device_get
        devices = kwargs.devices
        mesh = kwargs.mesh
        reshape_shmap = False
        move_axis = False

        n_devices = len(devices)

        if n_devices == 1:
            if verbose:
                print(f"no shmap, only 1 devices {jax.devices()=}")
            return f(*args)

        in_tree_flat, tree_def = tree_flatten(args)
        out_tree_eval = jax.eval_shape(f, *args)

        out_tree_flat_eval, out_tree_def = tree_flatten(out_tree_eval)

        shape = int(in_tree_flat[0].shape[axis])

        rem = shape % n_devices

        if rem != 0:
            p = n_devices - rem
        else:
            p = 0

        # mesh = mesh_utils.create_device_mesh((n_devices,), devices)

        if verbose:
            print(
                f"sharding: dividing {shape} in to chunks of size {n_devices} + {rem}, {in_tree_flat[0].shape} {axis=}"
            )

        def f_inner(*args):
            # print(f"inside: {args[0].shape=}")

            return f(*args)

        @jit_decorator
        def f_flat(*tree_flat):
            if pmap:
                _f = f_inner
            elif reshape_shmap:
                if not explicit_shmap:
                    _f = vmap_decorator(f_inner, in_axes=0 if move_axis else axis)  # vmap_decorator over sharded axis
                else:
                    tree_flat = [
                        jnp.sqeeze(x, 0 if move_axis else axis) for x in tree_flat
                    ]  # sharding keeps axis, pmap removes it
                    _f = f_inner
            else:
                _f = f_inner

            args = tree_unflatten(tree_def, tree_flat)

            out = _f(*args)

            out = tree_flatten(out)[0]

            if reshape_shmap and not pmap and explicit_shmap:
                out = [jnp.expand_dims(x, 0) for x in out]  # sharding keeps axis, pmap removes it

            return tuple(out)

        @jit_decorator
        def flatten(in_tree_flat):
            return [
                Partial_decorator(
                    _n_pad,
                    axis=axis,
                    p=p,
                    chunk_size=n_devices,
                    reshape=pmap or reshape_shmap,
                    move_axis=move_axis,
                )(in_tree_flat_a)
                for in_tree_flat_a in in_tree_flat
            ]

        in_tree_flat_padded = flatten(in_tree_flat)

        if pmap:
            print("WARNING: do not use pmap")

            shard_fun = jax.pmap(
                f_flat,
                in_axes=0 if move_axis else axis,
                out_axes=0,
            )

        else:
            # assert mesh is not None
            # mesh = kwargs.get_mesh()

            _, sharding, specs = _shard(
                in_tree_flat_padded,
                axis=0,  # axis is moved to 0
                axis_name=axis_name,
                mesh=mesh,
                put=False,
            )

            if verbose:
                print(f"sharding  {specs=} ")

            specs_out, sharding_out = _shard_out(
                out_tree_flat_eval,
                axis=axis,
                axis_name=axis_name,
                mesh=mesh,
            )

            # explicit: sharding is done by the user
            # otherwise the compiler figures it out
            # print(".", end="")
            if explicit_shmap:
                # print(f"explicit shmap")

                shard_fun = jit_decorator(
                    shard_map(
                        f_flat,
                        mesh=mesh,
                        in_specs=tuple(specs),  # already sharded
                        out_specs=specs_out,
                        check_rep=False,
                    )
                )

            else:
                # WARNING: might result in random crashes, probably related to https://github.com/jax-ml/jax/issues/19691
                shard_fun = jit_decorator(
                    f_flat,
                    in_shardings=sharding,
                    out_shardings=sharding_out,
                )

        out_flat = shard_fun(*in_tree_flat_padded)

        @jit_decorator
        def unflatten(out_flat):
            out_flat = [
                _n_unpad(
                    x=out_flat_a,
                    axis=axis,
                    shape=shape,
                    reshape=pmap or reshape_shmap,
                    move_axis=move_axis,
                )
                for out_flat_a in out_flat
            ]

            return tree_unflatten(out_tree_def, out_flat)

        out = unflatten(out_flat)
        if verbose:
            print(f"done sharding {f=}")

        return out

    return apply_pmap_fn


P = ParamSpec("P")
T = TypeVar("T")


def padded_vmap(
    f: Callable[P, T],
    chunk_size=None,
    axis=0,
    out_axes: int = 0,
    vmap=True,
    verbose=False,
):
    axis = axis
    chunk_size = chunk_size
    out_axes = out_axes
    vmap = vmap
    function = f

    def apply_vmap_fn(
        # function: Callable[P, T] = f,
        # axis: int = 0,
        # out_axes: int = 0,
        # chunk_size: int | None = None,
        # vmap: bool = True,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        # assert vmap_decorator

        if chunk_size is None:
            if vmap:
                _function = vmap_decorator(function, in_axes=axis, out_axes=out_axes)
            else:
                _function = function

            return _function(*args, **kwargs)

        in_tree_flat, tree_def = tree_flatten(args)
        shape = int(in_tree_flat[0].shape[axis])

        if shape < chunk_size:
            if vmap:
                _function = vmap_decorator(function, in_axes=axis, out_axes=out_axes)
            else:
                _function = function

            return _function(*args, **kwargs)

        rem = shape % chunk_size

        if rem != 0:
            p = chunk_size - rem
        else:
            p = 0

        @jit_decorator
        def pad_tree(args):
            return jax.tree.map(
                Partial_decorator(
                    _n_pad,
                    axis=axis,
                    p=p,
                    chunk_size=chunk_size,
                    reshape=True,
                    move_axis=True,
                    n_chunk_move=True,
                ),
                args,
            )

        in_tree_padded = pad_tree(args)

        # print(f"chunked into {in_tree_padded[0].shape[0]} pieces")

        if verbose:
            print(
                f"padded_vmap: shape divided in {in_tree_padded[0].shape[0]} pieces of size {chunk_size} + {p}, calling function {function}"
            )

        def _apply_fn(args):
            if verbose:
                print(f"inside: {args[0].shape[0]=}, vmap_decorator over axis 0")
            return vmap_decorator(function, in_axes=0)(*args, **kwargs)

        out = jax.lax.map(
            _apply_fn,
            in_tree_padded,
        )

        @jit_decorator
        def unpad_tree(out):
            return jax.tree.map(
                Partial_decorator(
                    _n_unpad,
                    shape=shape,
                    reshape=True,
                    axis=axis,
                    move_axis=True,
                    n_chunk_move=True,
                ),
                out,
            )

        out_reshaped = unpad_tree(out)
        print("padded_vmap done")
        return out_reshaped

    return apply_vmap_fn


T = TypeVar("T")
X = TypeVar("X", "CV", "SystemParams", "NeighbourList")
X2 = TypeVar("X2", "CV", "SystemParams", "NeighbourList")


def macro_chunk_map_fun(
    f: Callable[[X, NeighbourList | None], X2],
    y: list[X],
    w: list[Array] | None = None,
    nl: list[NeighbourList] | NeighbourList | None = None,
    ft: Callable[[X, NeighbourList | None], X2] | None = None,
    y_t: list[X] | None = None,
    nl_t: list[NeighbourList] | NeighbourList | None = None,
    macro_chunk: int | None = 1000,
    verbose=False,
    chunk_func: Callable[[T, X2, X2 | None, Array | None, Array | None], T] | None = None,
    chunk_func_init_args: T = None,
    w_t: list[Array] | None = None,
    print_every=10,
    jit_f=True,
) -> T:
    return _macro_chunk_map(
        f=f,
        y=y,
        nl=nl,
        ft=ft,
        y_t=y_t,
        nl_t=nl_t,
        macro_chunk=macro_chunk,
        verbose=verbose,
        chunk_func=chunk_func,
        chunk_func_init_args=chunk_func_init_args,
        w=w,
        w_t=w_t,
        print_every=print_every,
        jit_f=jit_f,
    )  # type: ignore


def macro_chunk_map(
    f: Callable[[X, NeighbourList | None], X2],
    y: list[X],
    nl: list[NeighbourList] | NeighbourList | None = None,
    ft: Callable[[X, NeighbourList | None], X2] | None = None,
    y_t: list[X] | None = None,
    nl_t: list[NeighbourList] | NeighbourList | None = None,
    macro_chunk: int | None = 1000,
    verbose=False,
    print_every=10,
    jit_f=True,
) -> list[X2] | tuple[list[X2], list[X2] | None]:
    return _macro_chunk_map(
        f=f,
        y=y,
        nl=nl,
        ft=ft,
        y_t=y_t,
        nl_t=nl_t,
        macro_chunk=macro_chunk,
        verbose=verbose,
        print_every=print_every,
        jit_f=jit_f,
    )  # type: ignore


def _macro_chunk_map(
    f: Callable[[X, NeighbourList | None], X2],
    y: list[X],
    w: list[Array] | None = None,
    nl: list[NeighbourList] | NeighbourList | None = None,
    ft: Callable[[X, NeighbourList | None], X2] | None = None,
    y_t: list[X] | None = None,
    nl_t: list[NeighbourList] | NeighbourList | None = None,
    macro_chunk: int | None = 1000,
    verbose=False,
    chunk_func: Callable[[T, X2, X2 | None, Array | None, Array | None], T] | None = None,
    chunk_func_init_args: T = None,
    w_t: list[Array] | None = None,
    print_every=10,
    jit_f=True,
):
    # helper method to apply a function to list of SystemParams or CVs, chunked in groups of macro_chunk

    compute_t = y_t is not None

    if isinstance(y[0], CV):
        op = cast(Callable[..., X], CV.stack)
    elif isinstance(y[0], SystemParams):
        op = cast(Callable[..., X], SystemParams.stack)
    elif isinstance(y[0], NeighbourList):
        op = cast(Callable[..., X], NeighbourList.stack)
    else:
        raise ValueError(f"{type(y[0])=} is not supported")

    if ft is None:
        ft = f

    if jit_f:
        f_cache = 0

        f = cast(Callable[[X, NeighbourList | None], X2], jit(f))

        if ft is not None:
            ft = cast(Callable[[X, NeighbourList | None], X2], jit(ft))

        if chunk_func is not None:
            cf_cache = 0
            chunk_func = cast(Callable[[T, X2, X2 | None, Array | None, Array | None], T], jit(chunk_func))

    def single_chunk():
        # print("performing single chunk")

        stack_dims = tuple([yi.shape[0] for yi in y])

        z = f(
            op(*y),
            NeighbourList.stack(*nl) if isinstance(nl, list) else nl,
        )

        zt = None

        if compute_t:
            assert y_t is not None
            zt = ft(
                op(*y_t),
                NeighbourList.stack(*nl_t) if isinstance(nl_t, list) else nl_t,
            )

        if chunk_func is None:
            assert not isinstance(z, SystemParams)

            z = z.replace(_stack_dims=stack_dims).unstack()

            if compute_t:
                assert not isinstance(zt, SystemParams)
                assert zt is not None
                zt = zt.replace(_stack_dims=stack_dims).unstack()

            if not compute_t:
                return z

            return z, zt

        if w is not None:
            w_stack = jnp.hstack(w)
        else:
            w_stack = None

        if compute_t and w_t is not None:
            wt_stack = jnp.hstack(w_t)
        else:
            wt_stack = None

        return chunk_func(
            chunk_func_init_args,
            z,
            zt if compute_t else None,
            w_stack,
            wt_stack,
        )

    if macro_chunk is None:
        return single_chunk()

    # n_prev = 0
    n = 0

    y_chunk: list[X] = []

    nl_chunk: list[NeighbourList] | None = [] if isinstance(nl, list) else None
    last_chunk_y: X | None = None
    w_chunk: list[Array] | None
    last_chunk_nl: NeighbourList | None = None

    if w is not None:
        w_chunk = []
    else:
        w_chunk = None

    tot_chunk = 0
    stack_dims_chunk = []

    tot = sum(y_chunk.shape[0] for y_chunk in y)

    if tot <= macro_chunk:
        return single_chunk()

    tot_running = 0

    z = []
    last_z: X2 | None = None

    n_iter = 0

    # if compute_t:
    yt_chunk: list[X] = []
    nlt_chunk: list[NeighbourList] | None = [] if isinstance(nl_t, list) else None
    zt = []
    zt_chunk: X2 | None = None
    last_zt: X2 | None = None
    last_chunk_yt: X | None = None
    wt_chunk: list[Array] | None
    wt_chunk = [] if w_t is not None else None
    last_chunk_nlt: NeighbourList | None = None
    yt_stack: X | None = None
    nlt_stack: NeighbourList | None = None

    dt0: datetime | None = None
    n_chunks: int | None = None
    rem: int | None = None

    if verbose:
        n_chunks = tot // macro_chunk
        rem = tot - n_chunks * macro_chunk

        print(f"aplying cv func to {n_chunks} chunks of size {macro_chunk} + remainder of size {rem} ")

        dt0 = datetime.now()

        print(f"start time: {dt0:%H:%M:%S}.{dt0.microsecond // 1000:03d}", end="", flush=True)

    while tot_running < tot:
        # print(",", end="", flush=True)

        # print(f"pre {tot_running=} {tot=}")

        if (tot_chunk < macro_chunk) and (tot_running + tot_chunk != tot):
            # print("adding elem")
            # the last element from previous cycle might be bigger tan the chunk size
            s = y[n].shape[0]

            y_chunk.append(y[n])
            if nl_chunk is not None and nl is not None:
                nl_chunk.append(nl[n])
            if w_chunk is not None and w is not None:
                w_chunk.append(w[n])

            if compute_t:
                yt_chunk.append(y_t[n]) if compute_t else None
                if nlt_chunk is not None and nl_t is not None:
                    nlt_chunk.append(nl_t[n])
                if wt_chunk is not None and w_t is not None:
                    wt_chunk.append(w_t[n])

            tot_chunk += s
            stack_dims_chunk.append(s)

            # print("added elem")

        if tot_chunk >= macro_chunk or (tot_running + tot_chunk == tot):
            # split last element
            split_last = tot_chunk > macro_chunk

            if split_last:
                y_last = y_chunk[-1]

                yls = y_last.shape[0]
                s_last = yls - (tot_chunk - macro_chunk)

                last_chunk_y = y_last[s_last:yls]
                y_chunk[-1] = y_last[0:s_last]

                if compute_t:
                    yt_last = yt_chunk[-1]
                    last_chunk_yt = yt_last[s_last:yls]
                    yt_chunk[-1] = yt_last[0:s_last]

                if nl is not None:
                    if nl_chunk is not None and nl is not None:
                        nl_last = nl_chunk[-1]
                        last_chunk_nl = nl_last[s_last:yls]

                        nl_chunk[-1] = nl_last[0:s_last]

                    if compute_t:
                        if nlt_chunk is not None and nl_t is not None:
                            nlt_last = nlt_chunk[-1]
                            last_chunk_nlt = nlt_last[s_last:yls]
                            nlt_chunk[-1] = nlt_last[0:s_last]

                if w is not None and w_chunk is not None:
                    w_last = w_chunk[-1]

                    last_chunk_w = w_last[s_last:yls]
                    w_chunk[-1] = w_last[0:s_last]

                if compute_t and w_t is not None:
                    if w_t is not None and wt_chunk is not None:
                        wt_last = wt_chunk[-1]

                        last_chunk_wt = wt_last[s_last:yls]
                        wt_chunk[-1] = wt_last[0:s_last]

                tot_chunk -= y_last.shape[0] - s_last
                stack_dims_chunk[-1] = s_last

            # if verbose:
            #     print("stacking")

            y_stack = op(*y_chunk)
            nl_stack = NeighbourList.stack(*nl_chunk) if nl_chunk is not None else None

            if compute_t:
                yt_stack = op(*yt_chunk)
                nlt_stack = NeighbourList.stack(*nlt_chunk) if nlt_chunk is not None else None

            w_stack = jnp.hstack(w_chunk) if w is not None else None  # type:ignore

            if compute_t and w_t is not None:
                wt_stack = jnp.hstack(wt_chunk)  # type: ignore

            # remove the stack dims from the CVs and NLs

            if isinstance(y_stack, CV):
                y_stack.stack_dims = None

            if isinstance(nl, list):
                nl_stack.update.stack_dims = None  # type: ignore

            _z_chunk = f(
                y_stack,
                nl_stack if isinstance(nl, list) else nl,
            )

            _zt_chunk = None

            if compute_t:
                #
                yt_stack = cast(X, yt_stack)

                if isinstance(yt_stack, CV):
                    yt_stack.stack_dims = None

                if isinstance(nlt_stack, list):
                    nlt_stack.update.stack_dims = None

                assert ft is not None

                _zt_chunk = ft(
                    yt_stack,
                    nlt_stack if isinstance(nl_t, list) else nl_t,
                )

            if verbose:
                print(".", end="", flush=True)

                n_iter += 1
                if n_iter % print_every == 0:
                    dt = datetime.now()

                    assert dt0 is not None
                    assert rem is not None
                    assert n_chunks is not None

                    dte = dt0 + (dt - dt0) / (n_iter) * (n_chunks + (rem != 0))

                    print(
                        f"\ntime: {dt:%H:%M:%S}.{dt.microsecond // 1000:03d}, estimated end time {dte:%H:%M:%S}.{dte.microsecond // 1000:03d}:, {n_iter:>3}/{n_chunks + (rem != 0)}:",  # type: ignore
                        end="",
                        flush=True,
                    )

            # keep track of recompilation cache. Frequent recompilation indicates a problem
            if jit_f:
                if (cs := f._cache_size()) != f_cache:  # type: ignore
                    if cs == 1 and verbose:
                        print("compiled f")
                        pass

                    elif cs > 1 and verbose:
                        if tot_chunk == macro_chunk:
                            print("WARNING: recompiled f")
                        else:
                            print("recompiled f for last chunk")
                            # pass
                    f_cache = cs

            tot_running += tot_chunk

            if chunk_func is None:
                assert not isinstance(_z_chunk, SystemParams)
                _z_chunk.stack_dims = tuple(stack_dims_chunk)
                z_chunk = cast(list[X2], _z_chunk.unstack())

                if compute_t:
                    assert not isinstance(_zt_chunk, SystemParams)
                    _zt_chunk.stack_dims = tuple(stack_dims_chunk)
                    zt_chunk = cast(list[X2], _zt_chunk.unstack())

                if last_z is not None:
                    z_chunk[0] = last_z.stack(last_z, z_chunk[0])

                    if isinstance(z_chunk[0], CV):
                        z_chunk[0].stack_dims = None

                    elif isinstance(z_chunk[0], NeighbourList):
                        z_chunk[0].update.stack_dims = None

                    if compute_t:
                        assert last_zt is not None
                        # last_zt = cast(CV, last_zt)
                        assert zt_chunk is not None
                        # zt_chunk = cast(list[CV], zt_chunk)

                        zt_chunk[0] = last_zt.stack(last_zt, zt_chunk[0])
                        if isinstance(zt_chunk[0], CV):
                            zt_chunk[0].stack_dims = None

                        elif isinstance(zt_chunk[0], NeighbourList):
                            zt_chunk[0].update.stack_dims = None

                if split_last:
                    last_z = z_chunk[-1]
                    z_chunk = z_chunk[:-1]

                    last_chunk_y = cast(X, last_chunk_y)

                    y_chunk = [last_chunk_y]

                    if last_chunk_nl is not None:
                        nl_chunk = [last_chunk_nl]

                    if compute_t:
                        assert zt_chunk is not None

                        last_zt = zt_chunk[-1]
                        zt_chunk = zt_chunk[:-1]

                        last_chunk_yt = cast(X, last_chunk_yt)
                        yt_chunk = [last_chunk_yt]

                        if last_chunk_nlt is not None:
                            nlt_chunk = [last_chunk_nlt]

                    tot_chunk = last_chunk_y.shape[0]
                    stack_dims_chunk = [tot_chunk]

                    # n_prev = n

                else:
                    last_z = None
                    y_chunk = []
                    nl_chunk = [] if isinstance(nl, list) else None

                    tot_chunk = 0
                    stack_dims_chunk = []

                    last_chunk_y = None
                    last_chunk_nl = None

                    if compute_t:
                        last_zt = None
                        yt_chunk = []
                        nlt_chunk = [] if isinstance(nl_t, list) else None

                        last_chunk_yt = None
                        last_chunk_nlt = None

                    # n_prev = n + 1

                w_chunk = [] if w is not None else None

                z.extend(z_chunk)
                if compute_t:
                    zt.extend(zt_chunk)  # type: ignore
                    wt_chunk = [] if w_t is not None else None

                # print("exiting")
            else:
                # assert w_stack is not None

                chunk_func_init_args = chunk_func(
                    chunk_func_init_args,
                    _z_chunk,
                    _zt_chunk if compute_t else None,
                    w_stack,
                    wt_stack if compute_t else None,
                )

                if jit_f:
                    if (cs := chunk_func._cache_size()) != cf_cache:  # type: ignore
                        if cs == 1 and verbose:
                            print("compiled chunk func")

                        elif cs > 1 and verbose:
                            if tot_chunk == macro_chunk:
                                print("WARNING: recompiled chunk func")
                            else:
                                print("recompiled chunk func for last chunk")

                        cf_cache = cs

                if split_last:
                    assert last_chunk_y is not None
                    last_chunk_y = cast(X, last_chunk_y)

                    y_chunk = [last_chunk_y]
                    nl_chunk = [last_chunk_nl] if last_chunk_nl is not None else None
                    if w is not None:
                        assert last_chunk_w is not None
                        w_chunk = [last_chunk_w]

                    if compute_t:
                        assert last_chunk_yt is not None
                        last_chunk_yt = cast(X, last_chunk_yt)

                        yt_chunk = [last_chunk_yt]
                        nlt_chunk = [last_chunk_nlt] if last_chunk_nlt is not None else None

                        if w_t is not None:
                            assert last_chunk_wt is not None
                            wt_chunk = [last_chunk_wt]

                    tot_chunk = last_chunk_y.shape[0]
                    stack_dims_chunk = [tot_chunk]

                    # n_prev = n

                else:
                    y_chunk = []
                    nl_chunk = [] if isinstance(nl, list) else None
                    w_chunk = [] if w is not None else None

                    last_chunk_y = None
                    last_chunk_nl = None
                    last_chunk_w = None

                    tot_chunk = 0
                    stack_dims_chunk = []

                    if compute_t:
                        yt_chunk = []
                        nlt_chunk = [] if isinstance(nl_t, list) else None
                        wt_chunk = [] if w_t is not None else None

                        last_chunk_yt = None
                        last_chunk_nlt = None

        if tot_chunk < macro_chunk and (tot_running + tot_chunk != tot):
            n += 1

    # print("exited")
    if verbose:
        dt = datetime.now()
        print(f"\nfinished at: {dt:%H:%M:%S}.{dt.microsecond // 1000:03d}")

    if chunk_func is None:
        if compute_t:
            return z, zt

        return z

    return chunk_func_init_args


class SystemParams(MyPyTreeNode):
    coordinates: Array
    cell: Array | None = field(default=None)

    def __getitem__(self, slices) -> SystemParams:
        if not self.batched:
            self = self.batch()

        return SystemParams(
            coordinates=self.coordinates[slices, :, :],
            cell=(self.cell[slices, :, :] if self.cell is not None else None),
        )

    def __iter__(self):
        if not self.batched:
            yield self
            return
        for i in range(self.coordinates.shape[0]):
            yield SystemParams(
                coordinates=self.coordinates[i, :, :],
                cell=self.cell[i, :, :] if self.cell is not None else None,
            )
        return

    @property
    def batched(self):
        return len(self.coordinates.shape) == 3

    @property
    def batch_dim(self):
        if self.batched:
            return self.shape[0]
        return 1

    @property
    def shape(self):
        return self.coordinates.shape

    def __add__(self, other):
        return SystemParams.stack(self, other)

    @staticmethod
    def stack(*sps: SystemParams) -> SystemParams:
        sps = tuple([sp.batch() for sp in sps])
        has_cell = all([x.cell is not None for x in sps])
        s = sps[0].shape[1:]

        for sp in sps:
            assert (sp.cell is not None) == has_cell
            assert sp.shape[1:] == s

        return SystemParams(
            coordinates=jnp.vstack([s.coordinates for s in sps]),
            cell=jnp.vstack([s.cell for s in sps]) if has_cell else None,  # type:ignore
        )

    def batch(self) -> SystemParams:
        if self.batched:
            return self

        return SystemParams(
            coordinates=jnp.array([self.coordinates]),
            cell=self.cell if self.cell is None else jnp.array([self.cell]),
        )

    def unbatch(self) -> SystemParams:
        if not self.batched:
            return self
        assert self.shape[0] == 1
        return SystemParams(
            coordinates=self.coordinates[0, :],
            cell=self.cell if self.cell is None else self.cell[0, :],
        )

    def angles(self, deg=True) -> Array:
        @partial(vmap_decorator, in_axes=(None, 0))
        @partial(vmap_decorator, in_axes=(0, None))
        def ang(x1, x2):
            a = jnp.arccos(jnp.dot(x1, x2) / (jnp.dot(x1, x1) * jnp.dot(x2, x2)) ** 0.5)
            if deg:
                a *= 180 / jnp.pi
            return a

        return ang(self.cell, self.cell)

    def _get_neighbour_list(
        self,
        info: NeighbourListInfo,
        update: NeighbourListUpdate | None = None,
        chunk_size: int | None = 100,
        verbose=False,
        chunk_size_inner: int | None = 100,
        shmap=False,
        shmap_kwargs: ShmapKwargs = ShmapKwargs.create(),
        only_update=False,
    ):
        if info.r_cut is None:
            return False, None, None, None

        if verbose:
            print("canonicalize")

        sp, (op_cell, op_coor, op_qr) = SystemParams.canonicalize(self, chunk_size=chunk_size)

        b = True

        if update is not None:
            nxyz, num_neighs = update.nxyz, update.num_neighs
        else:
            nxyz, num_neighs = None, None

        def _get_num_per_images(cell, r_cut):
            # orthogonal distance for number of blocks
            v1, v2, v3 = cell[0, :], cell[1, :], cell[2, :]

            # sorted from short to long
            e3 = v3 / jnp.linalg.norm(v3)
            e1 = jnp.cross(v2, e3)
            e1 /= jnp.linalg.norm(e1)
            e2 = jnp.cross(e3, e1)

            proj = vmap_decorator(vmap_decorator(jnp.dot, in_axes=(0, None)), in_axes=(None, 0))(
                jnp.array([e1, e2, e3]),
                jnp.array([v1, v2, v3]),
            )

            bounds = jnp.ceil(jnp.sum(jnp.abs(jnp.linalg.inv(proj)) * r_cut, axis=1))

            bounds = jnp.nan_to_num(bounds)

            return bounds

        if verbose:
            print("num neighbour cells")

        # cannot be jitted
        if sp.cell is not None:
            if sp.batched:
                new_nxyz = jnp.max(
                    vmap_decorator(
                        lambda sp: _get_num_per_images(sp.cell, info.r_cut + info.r_skin),
                    )(sp),  # type: ignore
                    axis=0,
                )

            else:
                new_nxyz = _get_num_per_images(sp.cell, info.r_cut + info.r_skin)

            if nxyz is None:
                nxyz = [int(i) for i in new_nxyz.tolist()]
            else:
                b = b and (jnp.array(new_nxyz) <= jnp.array(nxyz)).all()  # only check r_cut, because we are retracing

            nx, ny, nz = nxyz  # type: ignore
            bx = jnp.arange(-nx, nx + 1)
            by = jnp.arange(-ny, ny + 1)
            bz = jnp.arange(-nz, nz + 1)
        else:
            bx, by, bz = None, None, None
            new_nxyz = None

        # this is batchable

        @partial(vmap_decorator, in_axes=(0, None, None, None))
        @partial(jit_decorator, static_argnames=["take_num", "shmap"])
        def res(center_coordinates, sp: SystemParams, take_num=0, shmap=False):
            if center_coordinates is not None:
                sp_center = SystemParams(coordinates=sp.coordinates - center_coordinates, cell=sp.cell)
            else:
                sp_center = SystemParams(coordinates=sp.coordinates, cell=sp.cell)

            sp_center, center_op = sp_center.wrap_positions(min=True)

            # only return the needed values
            @jit_decorator
            def func(r_ij, index, i, j, k):
                return (
                    jnp.linalg.norm(r_ij),
                    jnp.array(index) if take_num != 0 else None,
                    jnp.array([i, j, k]) if (sp_center.cell is not None and take_num != 0) else None,
                )

            def _apply_g_inner(
                sp: SystemParams,
                func,
                r_cut,
                ijk=(None, None, None),
                exclude_self=True,
            ):
                i, j, k = ijk

                if ijk != (None, None, None):
                    assert sp.cell is not None
                    pos = sp.coordinates + i * sp.cell[0, :] + j * sp.cell[1, :] + k * sp.cell[2, :]
                else:
                    pos = sp.coordinates

                norm2 = jnp.sum(pos**2, axis=1)
                index_j = jnp.ones_like(norm2, dtype=jnp.int64).cumsum() - 1

                bools = norm2 < r_cut**2

                if exclude_self:
                    bools = jnp.logical_and(
                        bools,
                        jnp.logical_not(
                            vmap_decorator(jnp.allclose, in_axes=(0, None))(pos, jnp.zeros((3,))),
                        ),
                    )

                true_val = vmap_decorator(func, in_axes=(0, 0, None, None, None))(pos, index_j, i, j, k)
                false_val = jax.tree.map(
                    jnp.zeros_like,
                    true_val,
                )

                val = vmap_decorator(
                    lambda b, x, y: jax.tree.map(lambda t, f: jnp.where(b, t, f), x, y),
                )(bools, true_val, false_val)

                return bools, val

            if sp_center.cell is None:
                _, (r, atoms, indices) = _apply_g_inner(
                    sp=sp_center,
                    func=func,
                    r_cut=jnp.inf,
                    exclude_self=False,
                )

            else:

                @partial(padded_vmap, chunk_size=chunk_size_inner)
                def __f(i, j, k):
                    return _apply_g_inner(
                        sp=sp_center,
                        func=func,
                        r_cut=jnp.inf,
                        ijk=(i, j, k),
                        exclude_self=False,
                    )

                grid_x, grid_y, grid_z = jnp.meshgrid(bx, by, bz, indexing="ij")  # type: ignore

                grid_x = jnp.reshape(grid_x, (-1,))
                grid_y = jnp.reshape(grid_y, (-1,))
                grid_z = jnp.reshape(grid_z, (-1,))

                _, (r, atoms, indices) = __f(grid_x, grid_y, grid_z)

                r = jnp.reshape(r, (-1,))
                atoms = jnp.reshape(atoms, (-1)) if atoms is not None else None
                indices = jnp.reshape(indices, (-1, 3)) if indices is not None else None

            n = jnp.sum(r < info.r_cut + info.r_skin)

            if take_num == 0:
                return n

            idx = jnp.argsort(r)[0:take_num]

            return (
                n,
                r[idx],
                atoms[idx] if atoms is not None else None,
                indices[idx, :] if indices is not None else None,
                center_op,
            )

        @partial(jit_decorator, static_argnames=["take_num", "shmap"])
        def _f(sp: SystemParams, take_num, shmap=False):
            def _res(coor):
                return res(coor, sp, take_num, shmap)

            if shmap:
                _res = padded_shard_map(_res, shmap_kwargs)

            if take_num == 0:
                n = _res(sp.coordinates)
                num_neighs = jnp.max(n)  # type: ignore

                return num_neighs

            n, r, a, ijk, center_op = _res(sp.coordinates)
            num_neighs = jnp.max(n)

            return num_neighs, r, a, ijk, center_op

        def get_f(
            take_num,
        ):
            f = Partial_decorator(_f, take_num=take_num, shmap=shmap and not sp.batched)
            if sp.batched:
                f = padded_vmap(f, chunk_size=chunk_size)

                if shmap:
                    f = padded_shard_map(f, shmap_kwargs)

            return f

        if verbose:
            print(f"obtaining num neihgs { new_nxyz=}")

        # not jittable
        if num_neighs is None:
            nn = get_f(0)(sp)
            if sp.batched:
                nn = jnp.max(nn)  # type: ignore

            num_neighs = int(nn)  # type: ignore

        if only_update:
            return (
                b,
                num_neighs,
                new_nxyz,
                None,
            )

        if verbose:
            print(f"obtaining neighs {num_neighs=}")

        nn, _, a, ijk, center_op = get_f(num_neighs)(sp)

        if sp.batched:
            nn = jnp.max(nn)

        if verbose:
            print("got new Neighbourlist")

        b = jnp.logical_and(b, nn <= num_neighs)

        if update is None:
            new_update = NeighbourListUpdate.create(
                num_neighs=nn,  # type:ignore
                nxyz=new_nxyz,
            )
            update = new_update

        return (
            b,
            nn,
            new_nxyz,
            NeighbourList(
                update=update,
                info=info,
                atom_indices=a,
                ijk_indices=ijk,
                sp_orig=self,
                op_cell=op_cell,
                op_coor=op_coor,
                op_center=center_op,
            ),
        )

    def get_neighbour_list(
        self,
        info: NeighbourListInfo,
        chunk_size: int | None = None,
        chunk_size_inner=100,
        verbose=False,
        shmap=False,
        shmap_kwargs: ShmapKwargs = ShmapKwargs.create(),
        only_update=False,
    ) -> NeighbourList | None:
        if info.r_cut is None:
            return None

        # print(f"{shmap=} {shmap_kwargs=}")

        _, nn, _, nl = self._get_neighbour_list(
            info=info,
            chunk_size=chunk_size,
            chunk_size_inner=chunk_size_inner,
            verbose=verbose,
            shmap=shmap,
            shmap_kwargs=shmap_kwargs,
            only_update=only_update,
        )

        if nn <= 1:
            raise ValueError("No neighbours found")

        return nl

    @jit_decorator
    def minkowski_reduce(self) -> tuple[SystemParams, Array]:
        """base on code from ASE: https://wiki.fysik.dtu.dk/ase/_modules/ase/geometry/minkowski_reduction.html"""
        if self.cell is None:
            return self, jnp.eye(3)

        import itertools

        TOL = 1e-12

        @partial(jit_decorator, static_argnums=(0,))
        def cycle_checker(d):
            assert d in [2, 3]
            max_cycle_length = {2: 60, 3: 3960}[d]
            return jnp.zeros((max_cycle_length, 3 * d), dtype=int)

        @jit_decorator
        def add_site(visited, H):
            # flatten array for simplicity
            H = H.ravel()

            # check if site exists
            found = (visited == H).all(axis=1).any()

            # shift all visited sites down and place current site at the top
            visited = jnp.roll(visited, 1, axis=0)
            visited = visited.at[0].set(H)
            return visited, found

        @jit_decorator
        def reduction_gauss(B, hu, hv):
            """Calculate a Gauss-reduced lattice basis (2D reduction)."""
            visited = cycle_checker(2)
            u = hu @ B
            v = hv @ B

            def body(vals):
                u, v, hu, hv, visited, found, i = vals

                x = jnp.array(jnp.round(jnp.dot(u, v) / jnp.dot(u, u)), dtype=jnp.int64)
                hu, hv = hv - x * hu, hu
                u = hu @ B
                v = hv @ B
                site = jnp.array([hu, hv])

                visited, found = add_site(visited=visited, H=site)

                return (u, v, hu, hv, visited, found, i + 1)

            def cond_fun(vals):
                u, v, hu, hv, visited, found, i = vals

                return jnp.logical_not(
                    jnp.logical_and(
                        jnp.logical_or(jnp.dot(u, u) >= jnp.dot(v, v), found),
                        i != 0,
                    ),
                )

            u, v, hu, hv, visited, found, i = jax.lax.while_loop(
                cond_fun=cond_fun,
                body_fun=body,
                init_val=(u, v, hu, hv, visited, False, 0),
            )

            return hv, hu

        @jit_decorator
        def relevant_vectors_2D(u, v):
            cs = jnp.array(list(itertools.product([-1, 0, 1], repeat=2)))
            vs = cs @ jnp.array([u, v])
            indices = jnp.argsort(jnp.linalg.norm(vs, axis=1))[:7]
            return vs[indices], cs[indices]

        @jit_decorator
        def closest_vector(t0, u, v):
            t = t0
            a = jnp.zeros(2, dtype=int)
            rs, cs = relevant_vectors_2D(u, v)

            dprev = float("inf")
            ds = jnp.linalg.norm(rs + t, axis=1)
            index = jnp.argmin(ds)

            def body_fun(vals):
                ds, index, a, _, t, i = vals

                dprev = ds[index]

                r = rs[index]
                kopt = jnp.array(
                    jnp.round(-jnp.dot(t, r) / jnp.dot(r, r)),
                    dtype=jnp.int64,
                )
                a += kopt * cs[index]
                t = t0 + a[0] * u + a[1] * v

                ds = jnp.linalg.norm(rs + t, axis=1)
                index = jnp.argmin(ds)

                return ds, index, a, dprev, t, i + 1

            def cond_fun(vals):
                ds, index, a, dprev, t, i = vals

                return jnp.logical_not(jnp.logical_or(index == 0, ds[index] >= dprev))

            ds, index, a, dprev, t, i = jax.lax.while_loop(
                cond_fun=cond_fun,
                body_fun=body_fun,
                init_val=(ds, index, a, dprev, t0, 0),
            )

            return a

        @jit_decorator
        def reduction_full(B):
            """Calculate a Minkowski-reduced lattice basis (3D reduction)."""
            # init
            visited = cycle_checker(d=3)
            H = jnp.eye(3, dtype=int)
            norms = jnp.linalg.norm(B, axis=1)

            def body(vals):
                H, norms, visited, _, i = vals
                # for it in range(MAX_IT):
                # Sort vectors by norm
                H = H[jnp.argsort(norms)]

                # Gauss-reduce smallest two vectors
                hw = H[2]
                hu, hv = reduction_gauss(B, H[0], H[1])
                H = jnp.array([hu, hv, hw])
                R = H @ B

                # Orthogonalize vectors using Gram-Schmidt
                u, v, _ = R
                X = u / jnp.linalg.norm(u)
                Y = v - X * jnp.dot(v, X)
                Y /= jnp.linalg.norm(Y)

                # Find closest vector to last element of R
                pu, pv, pw = R @ jnp.array([X, Y]).T
                nb = closest_vector(pw, pu, pv)

                # Update basis
                H = H.at[2].set(jnp.array([nb[0], nb[1], 1]) @ H)
                R = H @ B

                norms = jnp.linalg.norm(R, axis=1)

                visited, found = add_site(visited, H)

                return H, norms, visited, found, i + 1

            def cond_fun(vals):
                _, norms, _, found, i = vals

                return jnp.logical_not(
                    jnp.logical_and(
                        jnp.logical_or(
                            norms[2] >= norms[1],
                            found,
                        ),
                        i != 0,
                    ),
                )

            H, norms, visited, found, i = jax.lax.while_loop(
                cond_fun,
                body,
                (H, norms, visited, False, 0),
            )

            return H @ B, H

        @jit_decorator
        def is_minkowski_reduced(cell):
            """Tests if a cell is Minkowski-reduced.

            Parameters:

            cell: array
                The lattice basis to test (in row-vector format).
            pbc: array, optional
                The periodic boundary conditions of the cell (Default `True`).
                If `pbc` is provided, only periodic cell vectors are tested.

            Returns:

            is_reduced: bool
                True if cell is Minkowski-reduced, False otherwise.
            """

            """These conditions are due to Minkowski, but a nice description in English
            can be found in the thesis of Carine Jaber: "Algorithmic approaches to
            Siegel's fundamental domain", https://www.theses.fr/2017UBFCK006.pdf
            This is also good background reading for Minkowski reduction.

            0D and 1D cells are trivially reduced. For 2D cells, the conditions which
            an already-reduced basis fulfil are:
            |b1| ≤ |b2|
            |b2| ≤ |b1 - b2|
            |b2| ≤ |b1 + b2|

            For 3D cells, the conditions which an already-reduced basis fulfil are:
            |b1| ≤ |b2| ≤ |b3|

            |b1 + b2|      ≥ |b2|
            |b1 + b3|      ≥ |b3|
            |b2 + b3|      ≥ |b3|
            |b1 - b2|      ≥ |b2|
            |b1 - b3|      ≥ |b3|
            |b2 - b3|      ≥ |b3|
            |b1 + b2 + b3| ≥ |b3|
            |b1 - b2 + b3| ≥ |b3|
            |b1 + b2 - b3| ≥ |b3|
            |b1 - b2 - b3| ≥ |b3|
            """

            A = jnp.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, -1, 0],
                    [1, 0, -1],
                    [0, 1, -1],
                    [1, 1, 1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [1, -1, -1],
                ],
            )
            lhs = jnp.linalg.norm(A @ cell, axis=1)
            norms = jnp.linalg.norm(cell, axis=1)
            rhs = norms[jnp.array([0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2])]

            return (lhs >= rhs - TOL).all()

        @jit_decorator
        def minkowski_reduce(cell):
            """Calculate a Minkowski-reduced lattice basis.  The reduced basis
            has the shortest possible vector lengths and has
            norm(a) <= norm(b) <= norm(c).

            Implements the method described in:

            Low-dimensional Lattice Basis Reduction Revisited
            Nguyen, Phong Q. and Stehlé, Damien,
            ACM Trans. Algorithms 5(4) 46:1--46:48, 2009
            :doi:`10.1145/1597036.1597050`

            Parameters:

            cell: array
                The lattice basis to reduce (in row-vector format).
            pbc: array, optional
                The periodic boundary conditions of the cell (Default `True`).
                If `pbc` is provided, only periodic cell vectors are reduced.

            Returns:

            rcell: array
                The reduced lattice basis.
            op: array
                The unimodular matrix transformation (rcell = op @ cell).
            """

            # reduced, op = jnp.where(
            #     is_minkowski_reduced(cell=cell),
            #     (cell, jnp.eye(3, dtype=int)),
            reduced, op = reduction_full(cell)
            # )

            # reduced, op = jax.lax.cond(
            #     is_minkowski_reduced(cell=cell),
            #     lambda: (cell, jnp.eye(3, dtype=int)),
            #     lambda: reduction_full(cell),
            # )

            change_handedness = jnp.sign(jnp.linalg.det(cell)) * jnp.sign(jnp.linalg.det(reduced))
            # jax.debug.print("handedness change {}", change_handedness)

            op = change_handedness * op

            return op @ cell, op

        if self.batched:
            cell, op = vmap_decorator(minkowski_reduce)(self.cell)
        else:
            cell, op = minkowski_reduce(self.cell)

        op = jnp.round(op).astype(jnp.int64)

        return SystemParams(coordinates=self.coordinates, cell=cell), op

    @jit_decorator
    def apply_minkowski_reduction(self, op):
        assert not self.batched, "apply vamp"

        if self.cell is None:
            return self

        if op is None:
            return self

        return SystemParams(coordinates=self.coordinates, cell=op @ self.cell)

    @jit_decorator
    def rotate_cell(self) -> tuple[SystemParams, tuple[Array, Array] | None]:
        if self.cell is None:
            return self, None

        if self.batched:
            return vmap_decorator(SystemParams.rotate_cell)(self)

        q, r = jnp.linalg.qr(self.cell.T)

        # make diagonals positive
        signs = jnp.sign(jnp.diag(r))

        # do not flip sign of new coordinates, as these are absolute
        new_cell = jnp.diag(signs) @ self.cell @ q
        new_coordinates = self.coordinates @ q

        return SystemParams(coordinates=new_coordinates, cell=new_cell), (signs, q)

    @jit_decorator
    def apply_rotation(self, op):
        signs, q = op
        sp = self
        if sp.cell is None:
            return sp

        assert not sp.batched, "apply vmap_decorator"

        new_cell = jnp.diag(signs) @ self.cell @ q
        new_coordinates = self.coordinates @ q

        return SystemParams(coordinates=new_coordinates, cell=new_cell)

    @jit_decorator
    def to_relative(self) -> tuple[SystemParams, jax.Array | None]:
        if self.cell is None:
            return self, None

        if self.batched:
            return vmap_decorator(Partial_decorator(SystemParams.to_relative))(self)

        c_inv = jnp.linalg.inv(self.cell)

        out = jnp.einsum("ij,ni->nj", c_inv, self.coordinates)

        return self.replace(coordinates=out), c_inv

    @jit_decorator
    def to_absolute(self) -> SystemParams:
        if self.cell is None:
            return self

        if self.batched:
            return vmap_decorator(Partial_decorator(SystemParams.to_absolute))(self)

        @partial(vmap_decorator, in_axes=(0, None), out_axes=0)
        def deproj(x, y):
            return jnp.einsum("i,ij->j", x, y)

        out = deproj(self.coordinates, self.cell)

        return self.replace(coordinates=out)

    @partial(jit_decorator, static_argnames=["min"])
    def wrap_positions(self, min=False) -> tuple[SystemParams, Array]:
        """wrap pos to lie within unit cell"""

        if self.batched:
            return vmap_decorator(Partial_decorator(SystemParams.wrap_positions, min=min))(self)

        cell = self.cell
        coordinates = self.coordinates

        if self.cell is None:
            return self, jnp.zeros_like(coordinates, dtype=jnp.int64)

        trans = vmap_decorator(vmap_decorator(jnp.dot, in_axes=(0, None)), in_axes=(None, 0))(
            vmap_decorator(lambda x: x / jnp.linalg.norm(x))(cell),
            jnp.eye(3),
        )

        @partial(vmap_decorator)
        def proj(x):
            return jnp.linalg.inv(trans) @ x

        # @partial(vmap_decorator)
        # def deproj(x):
        #     return trans @ x

        scaled = proj(coordinates)

        norms = jnp.linalg.norm(cell, axis=1)

        x0 = scaled / norms
        x0 = jnp.where(min, x0 + 0.5, x0)
        a = jnp.mod(x0, 1)
        # b = jnp.where(min, (a - 0.5) * norms, a * norms)

        op = a - x0
        # reduced = b

        op = jnp.round(op).astype(jnp.int64)

        return SystemParams(coordinates=coordinates + op @ cell, cell=cell), op

    @jit_decorator
    def apply_wrap(sp: SystemParams, wrap_op: Array) -> SystemParams:
        if sp.cell is None:
            return sp

        assert not sp.batched, "apply vmap_decorator"

        return SystemParams(coordinates=sp.coordinates + wrap_op @ sp.cell, cell=sp.cell)

    @partial(jit_decorator, static_argnames=["min", "qr", "chunk_size"])
    def canonicalize(
        self, min=False, qr=False, chunk_size=None
    ) -> tuple[SystemParams, tuple[Array, Array, tuple[Array, Array] | None]]:
        if self.batched:
            return padded_vmap(
                Partial_decorator(SystemParams.canonicalize, min=min, qr=qr),
                chunk_size=chunk_size,
            )(self)

        mr, op_cell = self.minkowski_reduce()

        op_qr = None

        if qr:
            mr, op_qr = mr.rotate_cell()

        mr, op_coor = mr.wrap_positions(min)

        return mr, (op_cell, op_coor, op_qr)

    @jit_decorator
    def apply_canonicalize(self, ops):
        op_cell, op_coor, op_qr = ops
        assert not self.batched, "apply vmap_decorator"

        sp: SystemParams = self.apply_minkowski_reduction(op_cell)

        if op_qr is not None:
            sp: SystemParams = sp.apply_rotation(op_qr)

        sp: SystemParams = sp.apply_wrap(op_coor)

        return sp

    def min_distance(self, index_1, index_2):
        assert self.batched is False

        assert self.cell is not None

        sp, _ = self.canonicalize()  # necessary if cell is skewed

        coor1 = sp.coordinates[index_1, :]
        coor2 = sp.coordinates[index_2, :]

        @partial(vmap_decorator, in_axes=(None, None, 0))
        @partial(vmap_decorator, in_axes=(None, 0, None))
        @partial(vmap_decorator, in_axes=(0, None, None))
        def dist(n0, n1, n2):
            assert sp.cell is not None

            return jnp.linalg.norm(
                coor2 - coor1 + n0 * sp.cell[0, :] + n1 * sp.cell[1, :] + n2 * sp.cell[2, :],
            )

        ind = jnp.array([-1, 0, 1])
        return jnp.min(dist(ind, ind, ind))

    def super_cell(
        self, n: int | list[int], info: NeighbourListInfo | None = None
    ) -> tuple[SystemParams, NeighbourListInfo | None]:
        if self.cell is None:
            return self, None

        if self.batched:
            return vmap_decorator(Partial_decorator(SystemParams.super_cell, n=n, info=info))(self)

        if isinstance(n, int):
            n = [n, n, n]

        coor = []

        if info is not None:
            z_list = []

            assert info.z_array is not None

        for a, b, c in itertools.product(range(n[0]), range(n[1]), range(n[2])):
            coor.append(self.coordinates + a * self.cell[0, :] + b * self.cell[1, :] + c * self.cell[2, :])
            z_list.extend(info.z_array)  # type:ignore

        if info is not None:
            info = NeighbourListInfo.create(r_cut=info.r_cut, z_array=z_list, r_skin=info.r_skin)

        print(f"{jnp.array(n).shape=} {self.cell.shape} {jnp.array(n) *self.cell=}")

        return SystemParams(
            coordinates=jnp.concatenate(coor, axis=0),
            cell=jnp.array(n).reshape((-1,)) * self.cell,
        ), info

    def volume(self):
        if self.cell is None:
            return None

        if self.batched:
            return vmap_decorator(SystemParams.volume)(self)

        return jnp.abs(jnp.linalg.det(self.cell))

    def to_ase(self, static_trajectory_info: StaticMdInfo):
        from ase import Atoms

        from IMLCV.base.UnitsConstants import angstrom

        return Atoms(
            numbers=static_trajectory_info.atomic_numbers,
            positions=self.coordinates / angstrom,
            cell=self.cell / angstrom if self.cell is not None else None,
            pbc=self.cell is not None,
        )


class NeighbourListInfo(MyPyTreeNode):
    # esssential information to create a neighbour list
    r_cut: float = field(pytree_node=False)
    r_skin: float = field(pytree_node=False)

    z_array: tuple[int] | None = field(pytree_node=False, default=None)
    z_unique: tuple[int] | None = field(pytree_node=False, default=None)
    num_z_unique: tuple[int] | None = field(pytree_node=False, default=None)

    @staticmethod
    def create(
        r_cut,
        z_array,
        r_skin=None,
    ):
        def to_tuple(a) -> tuple[int] | None:
            if a is None:
                return None
            return tuple([int(ai) for ai in a])  # type: ignore

        zu = jnp.unique(jnp.array(z_array)) if z_array is not None else None
        nzu = vmap_decorator(lambda zu: jnp.sum(jnp.array(z_array) == zu))(zu) if zu is not None else None

        if r_skin is None:
            r_skin = 0.0

        return NeighbourListInfo(
            r_cut=float(r_cut),
            r_skin=float(r_skin),
            z_array=to_tuple(z_array),
            z_unique=to_tuple(zu),
            num_z_unique=to_tuple(nzu),
        )

    def nl_split_z(self, p: T) -> tuple[jax.Array, list[jax.Array], list[T]]:
        assert self.z_unique is not None
        assert self.num_z_unique is not None

        bool_masks = [jnp.array(self.z_array) == zu for zu in self.z_unique]

        arg_split = [jnp.argsort(~bm, stable=True)[0:nzu] for bm, nzu in zip(bool_masks, self.num_z_unique)]
        p = [jax.tree.map(lambda pi: pi[a], tree=p) for a in arg_split]  # type: ignore

        return jnp.array(bool_masks), arg_split, p  # type: ignore

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        if isinstance(state["z_array"], list):
            # print("z_array is list, correcting")
            state["z_array"] = tuple(state["z_array"])

        if isinstance(state["z_unique"], list):
            # print("z_unique is list, correcting")
            state["z_unique"] = tuple(state["z_unique"])

        if isinstance(state["num_z_unique"], list):
            # print("num_z_unique is list, correcting")
            state["num_z_unique"] = tuple(state["num_z_unique"])

        self.__init__(**state)


class NeighbourListUpdate(MyPyTreeNode):
    # system specific information to update a neighbour list
    nxyz: tuple[int] | None = field(pytree_node=False, default=None)
    stack_dims: tuple[int] | None = field(pytree_node=False, default=None)
    num_neighs: int | None = field(pytree_node=False, default=None)

    @staticmethod
    def create(
        nxyz=None,
        stack_dims=None,
        num_neighs: int | None = None,
    ):
        def to_tuple(a) -> tuple[int] | None:
            if a is None:
                return None
            return tuple([int(ai) for ai in a])  # type: ignore

        return NeighbourListUpdate(
            nxyz=to_tuple(nxyz),
            stack_dims=to_tuple(stack_dims),
            num_neighs=int(num_neighs) if num_neighs is not None else None,
        )

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        if isinstance(state["nxyz"], list):
            print("nxyz is list, correcting")
            state["nxyz"] = tuple(state["nxyz"])

        if isinstance(state["stack_dims"], list):
            print("stack_dims is list, correcting")
            state["stack_dims"] = tuple(state["stack_dims"])

        self.__init__(**state)


class NeighbourList(MyPyTreeNode):
    sp_orig: SystemParams | None

    info: NeighbourListInfo
    update: NeighbourListUpdate

    atom_indices: Array | None = field(default=None)
    op_cell: Array | None = field(default=None)
    op_coor: Array | None = field(default=None)
    op_center: Array | None = field(default=None)

    ijk_indices: Array | None = field(default=None)
    _padding_bools: Array | None = field(default=None)

    @staticmethod
    def create(
        r_cut,
        sp_orig,
        atom_indices=None,
        r_skin=None,
        ijk_indices=None,
        z_array=None,
        nxyz=None,
        op_cell=None,
        op_coor=None,
        op_center=None,
        stack_dims=None,
        num_neighs=None,
    ):
        info = NeighbourListInfo.create(r_cut=r_cut, z_array=z_array, r_skin=r_skin)

        if num_neighs is None:
            num_neighs = sp_orig.shape[-1]

        update = NeighbourListUpdate.create(
            nxyz=nxyz,
            stack_dims=stack_dims,
            num_neighs=num_neighs,
        )

        return NeighbourList(
            info=info,
            update=update,
            atom_indices=atom_indices,
            ijk_indices=ijk_indices,
            op_cell=op_cell,
            op_coor=op_coor,
            op_center=op_center,
            sp_orig=sp_orig,
        )

    @jit_decorator
    def nneighs(self, sp=None):
        if sp is None:
            sp = self.sp_orig

        assert sp is not None

        return vmap_decorator(lambda x: jnp.sum(jnp.sum(x**2, axis=1) < self.info.r_cut**2))(self.neighbour_pos(sp))

    @property
    def needs_calculation(self):
        return self.atom_indices is None

    @property
    def padding_bools(self):
        if self._padding_bools is None:
            return self.atom_indices != -1
        return self._padding_bools

    @jit_decorator
    def canonicalized_sp(self, sp: SystemParams) -> SystemParams:
        app_can = SystemParams.apply_canonicalize

        if self.batched:
            assert sp.batched
            app_can = vmap_decorator(app_can)

        return app_can(sp, (self.op_cell, self.op_coor, None))

    @jit_decorator
    def neighbour_pos(self, sp_orig: SystemParams) -> jax.Array:
        @partial(vmap_decorator, in_axes=(0, 0, None, None))
        def neighbour_translate(
            ijk: jax.Array | None, a: jax.Array, sp_centered_on_atom: jax.Array, cell: jax.Array | None
        ):
            # a: index of atom
            out = sp_centered_on_atom[a]
            if ijk is not None:
                assert cell is not None
                (i, j, k) = cast(tuple[int, int, int], ijk)
                out = out + i * cell[0, :] + j * cell[1, :] + k * cell[2, :]
            return out

        @partial(vmap_decorator, in_axes=(0, 0, 0, None, 0))
        def vmap_atoms(
            atom_center_coordinates: Array,
            ijk_indices: Array | None,
            atom_indices: Array,
            canon_sp: SystemParams,
            op_center: Array,
        ):
            # center on atom

            sp_centered_on_atom = SystemParams(
                coordinates=canon_sp.coordinates - atom_center_coordinates, cell=can_sp.cell
            )
            sp_centered_on_atom = SystemParams.apply_wrap(sp_centered_on_atom, op_center)

            # transform centered neighbours
            return neighbour_translate(
                ijk_indices,
                atom_indices,
                sp_centered_on_atom.coordinates,
                canon_sp.cell,
            )

        if sp_orig.batched:
            assert self.batched
            vmap_atoms = vmap_decorator(vmap_atoms)

        can_sp = self.canonicalized_sp(sp_orig)

        assert self.atom_indices is not None
        assert self.op_center is not None

        return vmap_atoms(
            can_sp.coordinates,
            self.ijk_indices,
            self.atom_indices,
            can_sp,
            self.op_center,
        )

    def apply_fun_neighbour(
        self,
        sp: SystemParams,
        func: Callable[[jax.Array, jax.Array], T],
        r_cut: float | None = None,
        fill_value=0,
        reduce="full",  # or 'z' or 'none'
        exclude_self: bool = False,
        chunk_size_neigbourgs: int | None = None,
        chunk_size_atoms: int | None = None,
        chunk_size_batch: int | None = None,
        shmap=False,
        split_z=False,
        shmap_kwargs=ShmapKwargs.create(),
    ):
        if sp.batched:
            _g: Callable[[NeighbourList, SystemParams], tuple[jax.Array, T]] = padded_vmap(
                Partial_decorator(
                    NeighbourList.apply_fun_neighbour,
                    func=func,
                    r_cut=r_cut,
                    fill_value=fill_value,
                    reduce=reduce,
                    exclude_self=exclude_self,
                    chunk_size_neigbourgs=chunk_size_neigbourgs,
                    chunk_size_atoms=chunk_size_atoms,
                    chunk_size_batch=None,
                    shmap=False,
                    split_z=split_z,
                ),
                chunk_size=chunk_size_batch,
            )

            if shmap:
                _g = padded_shard_map(_g, shmap_kwargs)

            return _g(self, sp)
        # calculate nl on the fly
        if self.needs_calculation:
            _, self = self.update_nl(sp, shmap=shmap, chunk_size_inner=chunk_size_batch, verbose=True)
            assert self is not None

        if r_cut is None:
            r_cut = self.info.r_cut

        pos = self.neighbour_pos(sp)
        ind = self.atom_indices

        i_vec = jnp.arange(pos.shape[0])

        @partial(padded_vmap, chunk_size=chunk_size_neigbourgs)
        def _f(j: jax.Array, pos_ij: jax.Array, ind_ij: jax.Array, pb_ij: jax.Array):
            r_ij = jnp.linalg.norm(pos_ij, axis=-1)

            b = jnp.logical_and(pb_ij, r_ij < r_cut)

            if exclude_self:
                b = jnp.logical_and(b, r_ij > 1e-16)

            out = func(
                pos_ij,
                ind_ij,
            )

            out: T = jax.tree.map(lambda x: jnp.where(b, x, jnp.zeros_like(x) + fill_value), out)

            return (b, out), jnp.array(self.info.z_array)[ind_ij]

        @partial(padded_vmap, chunk_size=chunk_size_atoms)
        def _apply_fun(i: jax.Array, pos_i: jax.Array, ind_i: jax.Array, pb_i: jax.Array):
            j = jnp.arange(pos_i.shape[0])

            out_tree_n, zj_n = _f(j, pos_i, ind_i, pb_i)

            if reduce == "full":
                out_tree_n: tuple[Array, T] = jax.tree.map(lambda x: jnp.sum(x, axis=0), out_tree_n)
                return out_tree_n

            if reduce == "z":

                @vmap_decorator
                def _red(zj_ref: Array) -> tuple[Array, T]:
                    a = jnp.array(zj_n == zj_ref, dtype=jnp.int64)

                    return jax.tree.map(lambda x: jnp.einsum("n,n...->...", a, x), out_tree_n)

                out_tree_n: tuple[Array, T] = _red(jnp.array(self.info.z_unique))

                return out_tree_n

            if reduce == "none":
                return out_tree_n

            raise ValueError(
                "unknown value {reduce} for reduce argument of neighbourghfunction, try 'none','z' or 'full'"
            )

        if shmap:
            _apply_fun = padded_shard_map(_apply_fun, shmap_kwargs)

        assert ind is not None

        out = _apply_fun(i_vec, pos, ind, self.padding_bools)

        if split_z:
            # sum atom indices to  value per z
            b_split, _, _ = self.info.nl_split_z(out)

            @vmap_decorator
            def _split(b_split) -> tuple[Array, T]:
                return jax.tree.map(
                    lambda x: jnp.sum(jnp.where(b_split, x, jnp.zeros_like(x) + fill_value)),  # type:ignore
                    out,
                )

            out = _split(b_split)

        return out

    def apply_fun_neighbour_pair(
        self,
        sp: SystemParams,
        func_double: Callable[[jax.Array, jax.Array, T, jax.Array, jax.Array, T], S],
        func_single: Callable[[jax.Array, jax.Array], T] = lambda x, y: None,
        r_cut=None,
        fill_value=0.0,
        reduce="full",  # or 'z' or 'none'
        split_z=False,  #
        exclude_self=True,
        unique=True,
        chunk_size_neigbourgs=None,
        chunk_size_atoms=None,
        chunk_size_batch=None,
        shmap=False,
        shmap_kwargs=ShmapKwargs.create(),
    ) -> tuple[Array, S]:
        """
        Args:
        ______
        func_single(r_ij, atom_index_j) = ..
        func_double( p_ij, atom_index_j, data_j, p_ik, atom_index_k, data_k) = ...
        """

        if sp.batched:
            _f: Callable[[NeighbourList, SystemParams], tuple[Array, S]] = padded_vmap(
                Partial_decorator(
                    NeighbourList.apply_fun_neighbour_pair,
                    # self=nl,
                    # sp=sp,
                    func_single=func_single,
                    func_double=func_double,
                    # args_single=args_single,
                    # args_double=args_double,
                    r_cut=r_cut,
                    fill_value=fill_value,
                    reduce=reduce,
                    split_z=split_z,
                    exclude_self=exclude_self,
                    unique=unique,
                    chunk_size_neigbourgs=chunk_size_neigbourgs,
                    chunk_size_atoms=chunk_size_atoms,
                    chunk_size_batch=chunk_size_batch,
                    shmap=False,
                ),
                chunk_size=chunk_size_batch,
            )
            if shmap:
                _f = padded_shard_map(_f, shmap_kwargs)

            return _f(self, sp)

        # calculate nl on the fly
        if self.needs_calculation:
            _, self = self.update_nl(
                sp,
                shmap=shmap,
                chunk_size_inner=chunk_size_batch,
            )

            assert self is not None

        pos = self.neighbour_pos(sp)
        ind = self.atom_indices

        i_vec = jnp.arange(pos.shape[0])

        @partial(padded_vmap, chunk_size=chunk_size_atoms)
        def f2(i, pos_i, ind_i, padding_bools_i):
            # first do single data
            n = jnp.arange(pos_i.shape[0])

            @partial(padded_vmap, chunk_size=chunk_size_neigbourgs)
            def _f1(j: jax.Array, pos_j: jax.Array, ind_j: jax.Array, padding_bools_j: jax.Array):
                r_j = jnp.linalg.norm(pos_j, axis=-1)
                b = jnp.logical_and(padding_bools_j, r_j < r_cut)

                if exclude_self:
                    b = jnp.logical_and(b, r_j > 1e-16)

                if func_single is not None:
                    out = func_single(pos_j, ind_j)
                else:
                    out = None

                out: T | None = jax.tree.map(lambda x: jnp.where(b, x, jnp.zeros_like(x) + fill_value), out)

                return (b, out)

            bools_i, data_single_i = _f1(
                j=n,  # type: ignore
                pos_j=pos_i,
                ind_j=ind_i,
                padding_bools_j=padding_bools_i,
            )

            # two site

            nj, nk = jnp.meshgrid(n, n, indexing="ij")
            nj, nk = jnp.reshape(nj, -1), jnp.reshape(nk, -1)

            @partial(padded_vmap, chunk_size=chunk_size_neigbourgs)
            def _f2(
                j: jax.Array,
                k: jax.Array,
                bools_j: jax.Array,
                bools_k: jax.Array,
                pos_j: jax.Array,
                pos_k: jax.Array,
                ind_j: jax.Array,
                ind_k: jax.Array,
                data_single_j: T,
                data_single_k: T,
            ):
                b = jnp.logical_and(
                    bools_j,
                    bools_k,
                )

                if unique:
                    b = jnp.logical_and(b, j != k)

                out = func_double(pos_j, ind_j, data_single_j, pos_k, ind_k, data_single_k)

                out: S = jax.tree.map(lambda x: jnp.where(b, x, jnp.zeros_like(x) + fill_value), out)

                return (b, out)

            out_tree_n = _f2(
                nj,
                nk,
                bools_i[nj],
                bools_i[nk],
                pos_i[nj],
                pos_i[nk],
                ind_i[nj],
                ind_i[nk],
                data_single_i[nj],
                data_single_i[nk],
            )

            # replace vals with fill_value if bools is False
            if reduce == "full":
                out: tuple[Array, S] = jax.tree.map(lambda x: jnp.sum(x, axis=0), out_tree_n)

            elif reduce == "z":
                zj_n, zk_n = (
                    jnp.array(self.info.z_array)[ind_i[nj]],
                    jnp.array(self.info.z_array)[ind_i[nk]],
                )

                def _red(zj_ref, zk_ref):
                    b = jnp.logical_and(zj_n == zj_ref, zk_n == zk_ref)
                    return jax.tree.map(lambda x: jnp.einsum("n,n...->...", b, x), out_tree_n)

                _red = vmap_decorator(_red, in_axes=(0, None))
                _red = vmap_decorator(_red, in_axes=(None, 0))

                out = _red(jnp.array(self.info.z_unique), jnp.array(self.info.z_unique))

            elif reduce == "none":
                out = jax.tree.map(lambda x: x.reshape((*nj.shape, *x.shape[1:])), out_tree_n)
            else:
                raise ValueError(
                    f"unknown value {reduce=} for reduce argument of neighbourghfunction, try 'none','z' or 'full'"
                )

            return out

        if shmap:
            f2 = padded_shard_map(f2, shmap_kwargs)

        out = f2(i_vec, pos, ind, self.padding_bools)

        if split_z:
            # sum atom indices to  value per z
            b_split, _, _ = self.info.nl_split_z(out)

            @vmap_decorator
            def _split(b_split) -> tuple[Array, S]:
                return jax.tree.map(lambda x: jnp.einsum("i,i...->...", b_split, x), out)

            out = _split(b_split)

        return out

    @property
    def batched(self):
        assert self.sp_orig is not None

        return self.sp_orig.batched

    @property
    def batch_dim(self):
        assert self.sp_orig is not None

        return self.sp_orig.batch_dim

    @property
    def shape(self):
        assert self.sp_orig is not None

        return self.sp_orig.shape

    def __getitem__(self, slices):
        if not self.batched:
            self = self.batch()

        # assert self.batched

        return NeighbourList(
            info=self.info,
            update=self.update.replace(stack_dims=None),
            atom_indices=self.atom_indices[slices, :, :] if self.atom_indices is not None else None,
            ijk_indices=self.ijk_indices[slices, :, :] if self.ijk_indices is not None else None,
            op_cell=self.op_cell[slices, :, :] if self.op_cell is not None else None,
            op_coor=self.op_coor[slices, :, :] if self.op_coor is not None else None,
            op_center=self.op_center[slices, :] if self.op_center is not None else None,
            sp_orig=self.sp_orig[slices] if self.sp_orig is not None else None,
            _padding_bools=self._padding_bools[slices, :] if self._padding_bools is not None else None,
        )

    @jit_decorator
    def needs_update(self, sp: SystemParams) -> bool:
        if self.sp_orig is None:
            return True

        max_displacement = jnp.max(
            jnp.linalg.norm(self.neighbour_pos(self.sp_orig) - self.neighbour_pos(sp), axis=-1),
        )

        return max_displacement > self.info.r_skin / 2  # type:ignore

    @partial(
        jit_decorator,
        static_argnames=(
            "chunk_size",
            "shmap",
            "verbose",
            "chunk_size_inner",
            # "shmap_kwargs",
        ),
    )
    def update_nl(
        self,
        sp: SystemParams,
        chunk_size: int | None = None,
        chunk_size_inner: int | None = 10,
        shmap=False,
        shmap_kwargs=ShmapKwargs.create(),
        verbose=False,
    ):
        a, __, __, b = sp._get_neighbour_list(
            info=self.info,
            update=self.update,
            chunk_size=chunk_size,
            chunk_size_inner=chunk_size_inner,
            shmap=shmap,
            shmap_kwargs=shmap_kwargs,
            verbose=verbose,
        )

        return a, b

    def nl_split_z(self, p):
        f = self.info.nl_split_z

        # if self.batched:
        #     return vmap_decorator(f)

        return f(p)

    def batch(self):
        if self.batched:
            return self

        return NeighbourList(
            atom_indices=jnp.expand_dims(self.atom_indices, axis=0) if self.atom_indices is not None else None,
            ijk_indices=jnp.expand_dims(self.ijk_indices, axis=0) if self.ijk_indices is not None else None,
            op_cell=jnp.expand_dims(self.op_cell, axis=0) if self.op_cell is not None else None,
            op_coor=jnp.expand_dims(self.op_coor, axis=0) if self.op_coor is not None else None,
            op_center=jnp.expand_dims(self.op_center, axis=0) if self.op_center is not None else None,
            _padding_bools=jnp.expand_dims(self._padding_bools, axis=0) if self._padding_bools is not None else None,
            info=self.info,
            update=self.update,
            sp_orig=self.sp_orig.batch() if self.sp_orig is not None else None,
        )

    def __add__(self, other):
        return NeighbourList.stack(self, other)

    @staticmethod
    def stack(*nls: NeighbourList) -> NeighbourList:
        nls = tuple([nli.batch() for nli in nls])

        nl_0 = nls[0]

        # nxyz_none = nl_0.update.nxyz is None

        nxyz = nl_0.update.nxyz

        z_array = nl_0.info.z_array
        z_unique = nl_0.info.z_unique
        num_z_unique = nl_0.info.num_z_unique

        ijk_indices_none = nl_0.ijk_indices is None
        op_cell_none = nl_0.op_cell is None
        op_coor_none = nl_0.op_coor is None
        op_center_none = nl_0.op_center is None
        atom_indices_none = nl_0.atom_indices is None
        sp_orig_none = nl_0.sp_orig is None

        m = nl_0.update.num_neighs

        assert m is not None

        # assert m is not None

        r_cut = jnp.max(jnp.array([nli.info.r_cut for nli in nls]))

        def c(a, b):
            if a is None:
                assert b is None
            else:
                assert jnp.all(a == b)

        _stack_dims = []

        # consistency checks
        for nl_i in nls:
            assert nl_i.info.r_cut + nl_i.info.r_skin >= r_cut

            c(z_array, nl_i.info.z_array)
            c(z_unique, nl_i.info.z_unique)
            c(num_z_unique, nl_i.info.num_z_unique)

            assert ijk_indices_none == (nl_i.ijk_indices is None)
            assert op_cell_none == (nl_i.op_cell is None)
            assert op_coor_none == (nl_i.op_coor is None)
            assert op_center_none == (nl_i.op_center is None)
            assert atom_indices_none == (nl_i.atom_indices is None)
            assert sp_orig_none == (nl_i.sp_orig is None)

            assert nl_i.update.num_neighs is not None

            _stack_dims.append(nl_i.shape[0])

            if nxyz is not None:
                assert nl_i.update.nxyz is not None
                nxyz = [max(a, b) for a, b in zip(nxyz, nl_i.update.nxyz)]

            m = jnp.max(
                jnp.array([m, nl_i.update.num_neighs]),
            )

        m = int(m)

        r_skin = jnp.min(jnp.array([nli.info.r_cut + nli.info.r_skin - r_cut for nli in nls]))

        @partial(vmap_decorator, in_axes=(0, None))
        def _p(a: jax.Array, constant_value=None):
            # pad to size of largest neighbourlist
            n = m - a.shape[-1]

            return jnp.pad(
                array=a,
                pad_width=((0, 0), (0, n)),
                constant_values=constant_value if constant_value is not None else a[0, 0],
            )

        op_cell = None if atom_indices_none else []
        op_coor = None if op_coor_none else []
        op_center = None if op_center_none else []
        ijk_indices = None if ijk_indices_none else []
        atom_indices: list[jax.Array] | None = None if atom_indices_none else []
        padding_bools = None if atom_indices_none else []

        for nl_i in nls:
            if not atom_indices_none:
                atom_indices.append(_p(nl_i.atom_indices, None))  # type: ignore
                padding_bools.append(_p(nl_i.padding_bools, False))  # type: ignore

            if not ijk_indices_none:
                ijk_indices.append(  # type: ignore
                    vmap_decorator(
                        _p,
                        in_axes=-1,
                        out_axes=-1,
                    )(nl_i.ijk_indices, None),  # type: ignore
                )

            if not op_cell_none:
                op_cell.append(nl_i.op_cell)  # type: ignore

            if not op_coor_none:
                op_coor.append(nl_i.op_coor)  # type: ignore

            if not op_center_none:
                op_center.append(nl_i.op_center)  # type: ignore

        if not sp_orig_none:
            sp_orig = SystemParams.stack(*[nl_i.sp_orig for nl_i in nls])  # type:ignore
        else:
            sp_orig = None

        return NeighbourList(
            info=NeighbourListInfo.create(
                r_cut,
                z_array,
                r_skin=r_skin,
            ),
            update=NeighbourListUpdate.create(
                stack_dims=_stack_dims,
                nxyz=nxyz,
                num_neighs=m,
            ),
            atom_indices=jnp.vstack(atom_indices) if not atom_indices_none else None,  # type: ignore
            sp_orig=sp_orig,
            ijk_indices=jnp.vstack(ijk_indices) if not ijk_indices_none else None,  # type: ignore
            op_cell=jnp.vstack(op_cell) if not op_cell_none else None,  # type: ignore
            op_coor=jnp.vstack(op_coor) if not op_coor_none else None,  # type: ignore
            op_center=jnp.vstack(op_center) if not op_center_none else None,  # type: ignore
            _padding_bools=jnp.vstack(padding_bools) if not atom_indices_none else None,  # type: ignore
        )

    def unstack(self) -> list[NeighbourList]:
        if self.update.stack_dims is None:
            return [self]

        out = []

        t = 0

        for i in self.update.stack_dims:
            out.append(
                NeighbourList(
                    info=self.info,
                    update=self.update.replace(
                        stack_dims=None,
                    ),
                    atom_indices=self.atom_indices[t : t + i, :, :] if self.atom_indices is not None else None,
                    ijk_indices=self.ijk_indices[t : t + i, :, :] if self.ijk_indices is not None else None,
                    op_cell=self.op_cell[t : t + i, :, :] if self.op_cell is not None else None,
                    op_coor=self.op_coor[t : t + i, :, :] if self.op_coor is not None else None,
                    op_center=self.op_center[t : t + i, :] if self.op_center is not None else None,
                    sp_orig=self.sp_orig[t : t + i] if self.sp_orig is not None else None,
                    _padding_bools=self._padding_bools[t : t + i, :] if self._padding_bools is not None else None,
                )
            )

        return out

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        sd = statedict

        if "r_cut" in sd:
            r_cut = sd.pop("r_cut")
            r_skin = sd.pop("r_skin")
            z_array = sd.pop("z_array")
            _ = sd.pop("z_unique")
            _ = sd.pop("num_z_unique")

            sd["info"] = NeighbourListInfo.create(r_cut, z_array, r_skin=r_skin)

        if "nxyz" in sd:
            nxyz = sd.pop("nxyz")
            stack_dims = sd.pop("_stack_dims") if "_stack_dims" in sd else None

            sd["update"] = NeighbourListUpdate.create(
                nxyz=nxyz,
                stack_dims=stack_dims,
                num_neighs=sd["atom_indices"].shape[-1] if "atom_indices" in sd else None,
            )

        self.__init__(**sd)

    @property
    def stack_dims(self):
        return self.update.stack_dims

    @stack_dims.setter
    def stack_dims(self, value):
        self.update = self.update.replace(stack_dims=value)


class CV(MyPyTreeNode):
    cv: Array = field(pytree_node=True)
    mapped: bool = field(pytree_node=False, default=False)
    atomic: bool = field(pytree_node=False, default=False)
    _combine_dims: tuple[int | Any] | None = field(pytree_node=False, default=None)
    _stack_dims: tuple[int] | None = field(pytree_node=False, default=None)

    @staticmethod
    def create(
        cv: jax.Array,
        mapped=False,
        atomic=False,
        combine_dims=None,
        stack_dims=None,
    ):
        return CV(
            cv=cv,
            mapped=mapped,
            atomic=atomic,
            _combine_dims=tuple(combine_dims) if combine_dims is not None else None,
            _stack_dims=tuple(stack_dims) if stack_dims is not None else None,
        )

    @property
    def batched(self):
        if self.atomic:
            return len(self.cv.shape) == 3

        return len(self.cv.shape) == 2

    @property
    def batch_dim(self):
        if self.batched:
            return self.shape[0]
        return 1

    @property
    def dim(self):
        if self.cv.shape == ():
            return 1
        return self.cv.shape[-1]

    @property
    def size(self):
        return self.cv.size

    @property
    def shape(self):
        return self.cv.shape

    @property
    def combine_dims(self):
        if self._combine_dims is None:
            # if self.atomic:
            #     return self.shape[-2]

            return self.shape[-1]
        return self._combine_dims

    @property
    def stack_dims(self):
        if self._stack_dims is None:
            return [self.batch_dim]
        return self._stack_dims

    @stack_dims.setter
    def stack_dims(self, value):
        self._stack_dims = tuple(value) if value is not None else None

    def __add__(self, other) -> CV:
        assert isinstance(other, Array)

        return self.replace(cv=self.cv + other)

    def __radd__(self, other) -> CV:
        return other + self

    def __sub__(self, other) -> CV:
        assert isinstance(other, Array)
        return self.replace(cv=self.cv - other)

    def __rsub__(self, other) -> CV:
        return other - self

    def __mul__(self, other) -> CV:
        assert isinstance(other, Array)
        return self.replace(cv=self.cv * other)

    def __rmul__(self, other) -> CV:
        return other * self

    def __matmul__(self, other) -> CV:
        assert isinstance(other, Array)
        return self.replace(cv=self.cv @ other, _combine_dims=None)

    def __rmatmul__(self, other) -> CV:
        return other @ self

    def __div__(self, other) -> CV:
        assert isinstance(other, Array)
        return self.replace(cv=self.cv / other)

    def batch(self) -> CV:
        if self.batched:
            return self
        return CV(
            cv=jnp.array([self.cv]),
            mapped=self.mapped,
            atomic=self.atomic,
            _stack_dims=self._stack_dims,
            _combine_dims=self._combine_dims,
        )

    def __iter__(self):
        if not self.batched:
            yield self
            return

        for i in range(self.cv.shape[0]):
            yield self[i]
        return

    def __getitem__(self, idx):
        assert self.batched
        return CV(
            cv=self.cv[idx, :],
            mapped=self.mapped,
            atomic=self.atomic,
            _stack_dims=None,
            _combine_dims=self._combine_dims,
        )

    def unbatch(self) -> CV:
        if not self.batched:
            return self
        assert self.cv.shape[0] == 1
        return CV(
            cv=self.cv[0, :],
            mapped=self.mapped,
            atomic=self.atomic,
            _stack_dims=self._stack_dims,
            _combine_dims=self._combine_dims,
        )

    @staticmethod
    def stack(*cvs: CV) -> CV:
        """stacks a list of CVs into a single CV. The dimenisions are stored such that it can later be unstacked into separated CVs. The CVs are stacked over the batch dimension"""

        assert len(cvs) != 0
        atomic = cvs[0].atomic

        in_dims = None
        mapped = None

        cv_arr = []
        stack_dims = []

        for cv in cvs:
            assert atomic == cv.atomic

            # assert isinstance(cv, CV)
            if in_dims is None:
                in_dims = cv._combine_dims
                mapped = cv.mapped
            else:
                assert cv._combine_dims == in_dims
                assert cv.mapped == mapped

            cv_arr.append(cv.batch().cv)
            stack_dims += cv.stack_dims

        assert mapped is not None

        return CV(
            cv=jnp.vstack(cv_arr),
            _combine_dims=in_dims,
            _stack_dims=tuple(stack_dims),
            atomic=atomic,
        )

    def unstack(self) -> list[CV]:
        i = 0

        out: list[CV] = []

        for j in self.stack_dims:
            if j == 0:
                # print("skipping empty stack")
                continue

            out += [
                CV(
                    cv=self.cv[i : i + j, :],
                    mapped=self.mapped,
                    _combine_dims=self._combine_dims,
                    atomic=self.atomic,
                ),
            ]
            i += j
        return out

    def split(self, flatten=False) -> list[CV]:
        """inverse operation of combine"""

        if self._combine_dims is None:
            return [CV(cv=self.cv, mapped=self.mapped, atomic=self.atomic)]

        def broaden_tree(subtree):
            if isinstance(subtree, int):
                return (subtree,)
            num = []
            for leaf in subtree:
                num += broaden_tree(leaf)

            return tuple(num)

        if not flatten:
            sz = tuple([sum(broaden_tree(a)) for a in self._combine_dims])
            out_dim = self._combine_dims
        else:
            sz = broaden_tree(self._combine_dims)
            out_dim = sz

        out = []
        running = 0

        for i, s in enumerate(sz):
            assert s != 0

            x = jnp.apply_along_axis(
                lambda x: x[running : running + s],
                -1,  # if not self.atomic else -2,
                self.cv,
            )

            out.append(
                CV(
                    cv=x,
                    _combine_dims=out_dim[i] if isinstance(out_dim[i], tuple) else None,  # type:ignore
                    mapped=self.mapped,
                    atomic=self.atomic,
                    _stack_dims=self._stack_dims,
                )
            )

            running += s

        return out

    @staticmethod
    def combine(*cvs: CV, flatten=False) -> CV:
        """merges a list of CVs into a single CV. The dimenisions are stored such that it can later be split into separated CVs. The CVs are combined over the last dimension"""

        out_cv: list[Array] = []
        out_dim: list[int] = []

        mapped = cvs[0].mapped
        batched = cvs[0].batched
        atomic = cvs[0].atomic
        bdim = cvs[0].batch_dim

        stack_dims = cvs[0].stack_dims

        assert len(cvs) != 0
        if len(cvs) == 1:
            return cvs[0]

        def _inner(
            cv: CV,
            batched,
            mapped,
            atomic,
            bdim,
            stack_dims=None,
        ) -> tuple[list[Array], tuple[int | tuple[int | Any]]]:
            assert mapped == cv.mapped

            if batched is None:
                batched = cv.batched
                if batched:
                    bdim = cv.batch_dim
            else:
                assert batched == cv.batched
                if batched:
                    assert bdim == cv.batch_dim

            if atomic is None:
                atomic = cv.atomic
            else:
                assert atomic == cv.atomic

            assert stack_dims == cv.stack_dims

            def simple(cv: CV):
                return [cv.cv], [cv.combine_dims]

            if cv._combine_dims is not None and flatten:
                cv_split = cv.split()

                cvi = []
                dimi = []

                for ii in cv_split:
                    if ii._combine_dims is None:
                        a, b = simple(ii)
                    else:
                        a, b = _inner(ii, batched, mapped, atomic, bdim, stack_dims)

                    cvi += a
                    dimi += b

            else:
                cvi, dimi = simple(cv)

            return cvi, tuple(dimi)  # type:ignore

        for cv in cvs:
            a, b = _inner(cv, batched, mapped, atomic, bdim, stack_dims)
            out_cv += a
            out_dim += b  # type:ignore

        out_dim = tuple(out_dim)  # type:ignore

        return CV(
            cv=jnp.concatenate(out_cv, axis=-1),
            mapped=mapped,
            _combine_dims=out_dim,  # type:ignore
            _stack_dims=cvs[0]._stack_dims,
            atomic=atomic,
        )

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        sd = statedict

        def _convert_to_tuple(sd):
            out = []
            for i in sd:
                if isinstance(i, list):
                    i = _convert_to_tuple(i)
                out.append(i)

            return tuple(out)

        if "_combine_dims" in sd:
            if isinstance(sd["_combine_dims"], list):
                # print(f"converting combine dims to tuple  { sd['_combine_dims']}")
                sd["_combine_dims"] = _convert_to_tuple(sd["_combine_dims"])
                # print(f"converted combine dims to tuple  { sd['_combine_dims']}")

        if "_stack_dims" in sd:
            if isinstance(sd["_stack_dims"], list):
                # print("converting combine dims to tuple")
                sd["_stack_dims"] = _convert_to_tuple(sd["_stack_dims"])

        self.__init__(**sd)


class CvMetric(MyPyTreeNode):
    """class to keep track of topology of given CV. Identifies the periodicitie of CVs and maps to unit square with correct peridicities"""

    bounding_box: jax.Array
    periodicities: jax.Array

    @classmethod
    def create(
        cls,
        periodicities=None,
        bounding_box=None,
        # map_meshgrids=None,
    ) -> CvMetric:
        if periodicities is None:
            assert bounding_box is not None

            periodicities = [False for _ in bounding_box]

        if isinstance(periodicities, list):
            periodicities = jnp.array(periodicities, dtype=jnp.bool)

        assert periodicities.ndim == 1

        if bounding_box is None:
            assert periodicities is not None

            bounding_box = jnp.zeros((len(periodicities), 2), dtype=jnp.float64)
            bounding_box = bounding_box.at[:, 1].set(1.0)
        else:
            if isinstance(bounding_box, list):
                bounding_box = jnp.array(bounding_box, dtype=jnp.float64)

            if bounding_box.ndim == 1:
                bounding_box = jnp.reshape(bounding_box, (1, 2))

        return CvMetric(bounding_box=bounding_box, periodicities=periodicities)

    def norm(self, x1: CV, x2: CV, k=1.0):
        diff = self.difference(x1=x1, x2=x2) * k
        return jnp.linalg.norm(diff)

    @partial(jit_decorator, static_argnums=(2))
    def periodic_wrap(self, x: CV, min=False) -> CV:
        out = self.__periodic_wrap(self.map(x.cv), min=min)

        if not x.mapped:
            out = self.unmap(out)

        return CV(cv=out, mapped=True)

    @jit_decorator
    def difference(self, x1: CV, x2: CV) -> Array:
        assert not x1.mapped
        assert not x2.mapped

        return self.min_cv(
            x2.cv - x1.cv,
        )

    def min_cv(self, cv: Array):
        mapped = self.map(cv, displace=False)
        wrapped = self.__periodic_wrap(mapped, min=True)

        return self.unmap(
            wrapped,
            displace=False,
        )

    @partial(jit_decorator, static_argnums=(2))
    def __periodic_wrap(self, xs: Array, min=False):
        """Translate cvs such over unit cell.

        min=True calculates distances, False translates one vector inside box
        """

        coor = jnp.mod(xs, 1)  # between 0 and 1
        if min:
            coor = jnp.where(coor > 0.5, coor - 1, coor)  # between [-0.5,0.5]

        return jnp.where(self.periodicities, coor, xs)

    @partial(jit_decorator, static_argnums=(2))
    def map(self, x: Array, displace=True) -> Array:
        """transform CVs to lie in unit square."""

        if displace:
            x -= self.bounding_box[:, 0]

        y = x / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        return y

    @partial(jit_decorator, static_argnums=(2))
    def unmap(self, x: Array, displace=True) -> Array:
        """transform CVs to lie in unit square."""

        y = x * (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        if displace:
            x += self.bounding_box[:, 0]

        return y

    def __add__(self, other):
        assert isinstance(self, CvMetric)
        if other is None:
            return self

        assert isinstance(other, CvMetric)

        periodicities = jnp.hstack((self.periodicities, other.periodicities))
        bounding_box = jnp.vstack((self.bounding_box, other.bounding_box))

        return CvMetric(
            periodicities=periodicities,
            bounding_box=bounding_box,
        )

    @staticmethod
    def get_n(samples_per_bin, samples, n_dims, max_bins=None, max_bins_per_dim=30):
        f = samples / samples_per_bin
        if max_bins is not None:
            f = jnp.min(jnp.array([f, max_bins]))

        out = int(f ** (1 / n_dims))

        if max_bins_per_dim is not None:
            out = min(out, max_bins_per_dim)

        return out

    def grid(self, n=30, bounds=None, margin=0.1, indexing="ij"):
        """forms regular grid in mapped space. If coordinate is periodic, last rows are ommited.

        Args:
            n: number of points in each dim
            map: boolean. True: work in mapped space (default), False: calculate grid in space without metric
            endpoints: if

        Returns:
            meshgrid and vector with distances between points

        """

        if bounds is None:
            b = self.bounding_box

            if margin is not None:
                diff = b[:, 1] - b[:, 0]

                diff = jnp.where(diff < 1e-12, 1, diff * margin)

                diff = jnp.where(self.periodicities, 0, diff)  # if periodic, do not add bounds

                b = b.at[:, 0].set(b[:, 0] - diff)
                b = b.at[:, 1].set(b[:, 1] + diff)

        else:
            b = bounds

        grid = [jnp.linspace(row[0], row[1], n, endpoint=True) for i, row in enumerate(b)]

        # turn meshgrid into linear cv
        cv = CV(cv=jnp.reshape(jnp.array(jnp.meshgrid(*grid, indexing=indexing)), (len(grid), -1)).T)

        # get the midpoints

        mid = [a[:-1] + (a[1:] - a[:-1]) / 2 for a in grid]
        cv_mid = CV.combine(*[CV(cv=j.reshape(-1, 1)) for j in jnp.meshgrid(*mid, indexing=indexing)])

        return grid, cv, cv_mid, b

    @property
    def ndim(self):
        return len(self.periodicities)

    @staticmethod
    def bounds_from_cv(
        cv_0: list[CV],
        percentile=0.1,
        weights: list[Array] | None = None,
        rho: list[Array] | None = None,
        margin=None,
        chunk_size: int | None = None,
        n=400,
        macro_chunk: int | None = 5000,
        verbose=True,
    ):
        n = int(n)

        if margin is None:
            margin = percentile / 100

        # first find absolute bounds

        mini = jnp.min(cv_0[0].cv, axis=0)
        maxi = jnp.max(cv_0[0].cv, axis=0)

        for cvi in cv_0:
            mini = jnp.minimum(mini, jnp.min(cvi.cv, axis=0))
            maxi = jnp.maximum(maxi, jnp.max(cvi.cv, axis=0))

        bounding_box = jnp.vstack((mini, maxi)).T

        if verbose:
            print(f"bounding box pre: {bounding_box=}")

        ndim = bounding_box.shape[0]

        constants = False

        for i in range(ndim):
            if jnp.abs(bounding_box[i, 0] - bounding_box[i, 1]) <= 1e-12:
                print(f"WARNING: CV in dimension {i} is constant, increase margin to avoid division by zero.")

                bounding_box = bounding_box.at[i, 0].set(bounding_box[i, 0] - 0.5)
                bounding_box = bounding_box.at[i, 1].set(bounding_box[i, 1] + 0.5)

                constants = True

        # do bin count over range

        from IMLCV.base.rounds import DataLoaderOutput
        from IMLCV.implementations.CV import _cv_slice

        bounds = jnp.zeros((cv_0[0].shape[1], 2))

        if rho and weights is not None:
            w_tot_log = [jnp.log(a) + jnp.log(b) for a, b in zip(weights, rho)]

            w_log_tot = jnp.hstack(w_tot_log)
            w_log_tot_max = jnp.max(w_log_tot)
            w_log_tot_norm = jnp.log(jnp.sum(jnp.exp(w_log_tot - w_log_tot_max))) + w_log_tot_max

            w_tot = [jnp.exp(a - w_log_tot_norm) for a in w_tot_log]
        else:
            w_tot = weights

        for dim in range(ndim):
            print(f"new iterated bounds {dim=}")

            cv_mid, nums, bins, closest_trans, get_histo = DataLoaderOutput._histogram(
                metric=CvMetric.create(
                    periodicities=[False],
                    bounding_box=bounding_box[[dim], :],
                ),
                grid_bounds=bounding_box[[dim], :],
                n_grid=n,
            )

            print("getting grid nums")

            a = CvTrans.from_cv_function(
                _cv_slice,
                indices=jnp.array([dim]),
            )

            b = closest_trans

            _f = (a * b).compute_cv

            def f(x, nl):
                return _f(
                    x,
                    nl,
                    chunk_size=chunk_size,
                    shmap=False,
                )[0]

            f = jit_decorator(f)

            hist = get_histo(
                cv_0,
                weights=w_tot,
                f_func=f,
            )
            # hist = jnp.reshape(hist, (n - 1,) * len(mini))

            hist /= jnp.sum(hist)  # type: ignore

            cummul = jnp.cumsum(hist)

            # print(f"{hist=} {cummul=}")

            v0 = jnp.argwhere(cummul > percentile / 100)
            if len(v0) == 0:
                n_min = 0
            else:
                n_min = jnp.min(v0)

                # lower end
                # if n_min > 0:
                #     n_min -= 1

            v0 = jnp.argwhere(cummul < 1 - percentile / 100)

            if len(v0) == 0:
                n_max = n
            else:
                n_max = jnp.max(v0) + 1

                # # higher end
                # if n_max < n - 1:
                #     n_max += 1

            # bins, not bins mid

            bounds = bounds.at[dim, 0].set(bins[0][n_min])
            bounds = bounds.at[dim, 1].set(bins[0][n_max])

        # update bounds with margin

        bounds_margin = (bounds[:, 1] - bounds[:, 0]) * margin
        # bounds_margin = jnp.where(   self.periodicities, )
        bounds = bounds.at[:, 0].set(bounds[:, 0] - bounds_margin)
        bounds = bounds.at[:, 1].set(bounds[:, 1] + bounds_margin)

        if verbose:
            print(f"{bounds=}")

        @partial(CvTrans.from_cv_function, bounds=bounds)
        def get_mask(x: CV, nl, shmap, shmap_kwargs, bounds):
            b = jnp.logical_and(jnp.all(x.cv > bounds[:, 0]), jnp.all(x.cv < bounds[:, 1]))

            return x.replace(cv=jnp.reshape(b, (1,)), _combine_dims=None)

        mask, _ = DataLoaderOutput.apply_cv(
            x=cv_0,
            f=get_mask,
            macro_chunk=macro_chunk,
            verbose=verbose,
            chunk_size=chunk_size,
        )

        return bounds, [mask.cv.reshape(-1) for mask in mask], constants

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        self.__init__(**statedict)

    def __getitem__(self, idx):
        idx = jnp.array(idx)
        return CvMetric(
            bounding_box=self.bounding_box[idx],
            periodicities=self.periodicities[idx],
        )


######################################
#       CV tranformations            #
######################################


class CvFunBase(ABC, MyPyTreeNode):
    # __: KW_ONLY
    kwargs: dict = field(pytree_node=True, default_factory=dict)
    static_kwargs: dict = field(pytree_node=False, default_factory=dict)
    # jacfun: Callable = field(pytree_node=False, default=jax.jacfwd)

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
                ),
                chunk_size=chunk_size,
            )

            if shmap:
                _f = padded_shard_map(_f, shmap_kwargs)

            return _f(x, nl)

        def f(x):
            return self._compute_cv(x, nl, shmap=shmap, shmap_kwargs=shmap_kwargs)[0]

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
    ) -> CV:
        raise

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        if "static_kwarg_names" in statedict:
            statedict["static_kwargs"] = {k: statedict.pop(k) for k in statedict["static_kwarg_names"]}

            statedict.pop("static_kwarg_names")

        self.__init__(**statedict)


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
    ) -> CV:
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
    ) -> CV:
        ordered = reversed(self.trans) if reverse else self.trans

        assert len(self.trans) > 0

        for tr in ordered:
            x = tr._compute_cv(
                x=x,
                nl=nl,
                reverse=reverse,
                shmap=shmap,
                shmap_kwargs=shmap_kwargs,
            )

        return x  # type: ignore


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

        return CV(cv=jnp.hstack([cvi.cv for cvi in out]))


class CvTrans(MyPyTreeNode):
    trans: _ParralelCvTrans | _SerialCvTrans

    @staticmethod
    def from_cv_function(
        f: Callable,
        # jacfun: Callable = None,
        static_argnames=None,
        check_input: bool = True,
        **kwargs,
    ) -> CvTrans:
        static_kwargs = {}

        if static_argnames is not None:
            for a in static_argnames:
                static_kwargs[a] = kwargs.pop(a)

        kw = dict(forward=f, kwargs=kwargs, static_kwargs=static_kwargs)

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

            for k, v in list(kw["static_kwargs"].items()):
                try:
                    hash(v)
                    assert v == v
                except Exception as e:
                    print(
                        f"Error: {k} of type {type(v)} is not hashable, consider to make it hashable or not a static argument. exception: {e}"
                    )
                    raise

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
            )

            if shmap:
                _f = padded_shard_map(_f, shmap_kwargs)

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

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        self.__init__(**statedict)

    def __mul__(self: CvTrans, other: CvTrans):
        return CvTrans(trans=_SerialCvTrans(trans=(self.trans, other.trans)))

    def __add__(self: CvTrans, other: CvTrans):
        return CvTrans(trans=_ParralelCvTrans(trans=(self.trans, other.trans)))


######################################
#       Collective variable          #
######################################


class CollectiveVariable(MyPyTreeNode):
    f: CvTrans
    metric: CvMetric
    jac: Callable = field(pytree_node=False, default=jax.jacrev)  # jacfwd is generally faster, but not always supported

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

        return CollectiveVariable(
            f=self.f
            * CvTrans.from_cv_function(
                _cv_slice,
                indices=jnp.array(tup).reshape((-1,)),
            ),
            jac=self.jac,
            metric=self.metric[tup],
        )
