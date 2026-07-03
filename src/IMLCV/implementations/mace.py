"""Node-feature-only MACE.

A stripped-down MACE that returns only ``{'node_feats': ...}`` for use as a
descriptor / CV backbone. Removed relative to the full model:

  * energy / readouts (atomic_energies, AtomicEnergiesBlock, all readout blocks,
    scatter_sum over graphs, contributions/interaction_energy)
  * LAMMPS plumbing (lammps_mliap/lammps_class, _apply_lammps_exchange, n_real
    slicing, ghost-atom truncation)
  * pair repulsion (ZBLBasis)
  * joint / generic embeddings and the embedding readout
  * multi-head support (single implicit head)
  * the add_output_interface force/stress wrapper (__call__ now just accepts and
    ignores compute_force / compute_stress)

The one substantive change vs. the original irreps math: ``interaction_irreps``
is ordered the way cuequivariance's descriptor orders irreps -- ascending
(l, p), i.e. ODD-before-EVEN within each l -- instead of e3nn's natural-parity-
first ``.sort()``. e3nn's order puts ``0e`` before ``0o``; the cue descriptor
puts ``0o`` before ``0e``; ``collapse_ir_mul_segments`` zips the two positionally
and raises "Descriptor and target irreps mismatch: 0o vs 0e". Matching cue's
order here lines the zip up.

NOTE: ``hidden_irreps`` (the product-block targets, and what your apply wrapper
reads back) must be ordered the SAME way, which is set in graph_neural_network():
drop the ``.regroup()`` so the inversion build keeps its native odd-first order.
With hidden in cue order, ``hidden_irreps[0]`` is the 0o block, so the collapsed
final layer carries pseudoscalars -- fine for a CV, parity-invalid for energy
(which is gone here anyway).

This class is a drop-in replacement for the MACE class in
mace_jax/modules/models.py; reuse that file's existing imports for the blocks.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
from flax import nnx
from e3nn_jax import Irrep, Irreps


import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irrep, Irreps
from flax import nnx

from mace_jax.adapters.e3nn.math import (
    estimate_normalize2mom_const,
    register_normalize2mom_const,
)
from mace_jax.adapters.e3nn.o3 import SphericalHarmonics
from mace_jax.adapters.nnx.torch import nxx_auto_import_from_torch
from mace_jax.modules.embeddings import GenericJointEmbedding
from mace_jax.modules.radial import ZBLBasis
from mace_jax.nnx_config import ConfigVar
from mace_jax.tools.dtype import default_dtype
from mace_jax.tools.lammps_exchange import forward_exchange as lammps_forward_exchange
from mace_jax.tools.scatter import scatter_sum

from mace_jax.modules.blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from mace_jax.modules.utils import add_output_interface, prepare_graph


def _as_tuple(x, n: int) -> tuple:
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x,) * n


def _cue_sort(irreps) -> Irreps:
    """Order irreps the way cuequivariance's descriptor does: ascending (l, p),
    so within each l the odd (p=-1) block precedes the even (p=+1) one.

    e3nn's own .sort() is natural-parity-first (0e before 0o, 1o before 1e),
    which mismatches the cue descriptor at even l and triggers the
    collapse_ir_mul_segments assertion. Sorting by (l, p) reproduces cue's
    convention so the adapter's positional merge aligns.
    """
    merged = Irreps(irreps).simplify()
    ordered = sorted(merged, key=lambda mi: (mi.ir.l, mi.ir.p))
    return Irreps([(mi.mul, mi.ir) for mi in ordered])


class MACE(nnx.Module):
    def __init__(
        self,
        *,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls,
        interaction_cls_first,
        atomic_numbers: Sequence[int],
        num_interactions: int = 3,
        num_elements: int = 1,
        hidden_irreps: Irreps = Irreps("1x0e"),
        avg_num_neighbors: float = 1.0,
        correlation: int | Sequence[int] = 1,
        distance_transform: str = "None",
        radial_type: str = "bessel",
        radial_MLP: Sequence[int] | None = None,
        use_so3: bool = False,
        use_reduced_cg: bool = True,
        use_agnostic_product: bool = False,
        collapse_hidden_irreps: bool = True,
        rngs: nnx.Rngs,
        # swallow the kwargs the wrapper / _graph_neural_network_model still pass
        # but that the node-feature-only model does not use (atomic_energies,
        # readout_cls, MLP_irreps, gate, cueq_config, pair_repulsion, heads, ...)
        **_ignored,
    ) -> None:
        self.num_interactions = num_interactions
        self.num_elements = num_elements
        self.max_ell = max_ell
        self.use_so3 = use_so3

        hidden_irreps = hidden_irreps if isinstance(hidden_irreps, Irreps) else Irreps(hidden_irreps)
        self._hidden_irreps = hidden_irreps

        correlation = _as_tuple(correlation, num_interactions)
        if len(correlation) != num_interactions:
            raise ValueError(f"correlation length {len(correlation)} != num_interactions {num_interactions}")

        self._atomic_numbers = jnp.asarray(tuple(atomic_numbers), dtype=jnp.int32)

        # ---- irreps bookkeeping ------------------------------------------------
        node_attr_irreps = Irreps([(num_elements, (0, 1))])
        num_features = hidden_irreps.count(Irrep(0, 1))  # number of 0e channels
        node_feats_irreps = Irreps([(num_features, (0, 1))])

        if not use_so3:
            sh_irreps = Irreps.spherical_harmonics(max_ell)
        else:
            sh_irreps = Irreps.spherical_harmonics(max_ell, p=1)

        def _generate_irreps(l_val: int) -> Irreps:
            return Irreps("+".join(f"1x{i}e+1x{i}o" for i in range(l_val + 1)))

        # mixed-parity edge features once odd scalars are requested in hidden
        sh_irreps_inter = sh_irreps
        if hidden_irreps.count(Irrep(0, -1)) > 0:
            sh_irreps_inter = _generate_irreps(max_ell)

        # cue-ordered (odd-before-even) targets -> no collapse_ir_mul_segments clash
        interaction_irreps = _cue_sort(sh_irreps_inter * num_features)
        interaction_irreps_first = _cue_sort(sh_irreps * num_features)

        hidden_irreps_out = (
            Irreps(str(hidden_irreps[0])) if num_interactions == 1 and collapse_hidden_irreps else hidden_irreps
        )

        # ---- embeddings --------------------------------------------------------
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=None,
            rngs=rngs,
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=True,
            rngs=rngs,
        )
        edge_feats_irreps = Irreps(f"{self.radial_embedding.out_dim}x0e")
        self.spherical_harmonics = SphericalHarmonics(sh_irreps, normalize=True, normalization="component")

        radial_mlp = list(radial_MLP) if radial_MLP is not None else [64, 64, 64]

        # ---- interaction + product stack (no readouts) -------------------------
        interactions: list = []
        products: list = []

        interaction_first = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps_first,
            hidden_irreps=hidden_irreps_out,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_mlp,
            cueq_config=None,
            rngs=rngs,
        )
        interactions.append(interaction_first)
        products.append(
            EquivariantProductBasisBlock(
                node_feats_irreps=interaction_first.target_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[0],
                num_elements=num_elements,
                use_sc="Residual" in interaction_cls_first.__name__,
                cueq_config=None,
                use_reduced_cg=use_reduced_cg,
                use_agnostic_product=use_agnostic_product,
                rngs=rngs,
            )
        )

        for idx in range(num_interactions - 1):
            h_out = Irreps(str(hidden_irreps[0])) if idx == num_interactions - 2 else hidden_irreps
            interactions.append(
                interaction_cls(
                    node_attrs_irreps=node_attr_irreps,
                    node_feats_irreps=hidden_irreps,
                    edge_attrs_irreps=sh_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    target_irreps=interaction_irreps,
                    hidden_irreps=h_out,
                    avg_num_neighbors=avg_num_neighbors,
                    edge_irreps=None,
                    radial_MLP=radial_mlp,
                    cueq_config=None,
                    rngs=rngs,
                )
            )
            products.append(
                EquivariantProductBasisBlock(
                    node_feats_irreps=interaction_irreps,
                    target_irreps=h_out,
                    correlation=correlation[idx + 1],
                    num_elements=num_elements,
                    use_sc=True,
                    cueq_config=None,
                    use_reduced_cg=use_reduced_cg,
                    use_agnostic_product=use_agnostic_product,
                    rngs=rngs,
                )
            )

        self.interactions = nnx.List(interactions)
        self.products = nnx.List(products)

    def __call__(
        self,
        data: dict,
        *,
        compute_force: bool = False,  # accepted + ignored (no energy here)
        compute_stress: bool = False,  # accepted + ignored
        compute_node_feats: bool = True,
        **_ignored,
    ) -> dict:
        ctx = prepare_graph(data)
        node_attrs = data["node_attrs"]

        node_feats = self.node_embedding(node_attrs)
        edge_attrs = self.spherical_harmonics(ctx.vectors)
        edge_feats, cutoff = self.radial_embedding(
            ctx.lengths,
            node_attrs,
            data["edge_index"],
            self._atomic_numbers,
            node_attrs_index=None,
        )

        feats: list = []
        for interaction, product in zip(self.interactions, self.products):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                n_real=None,
                first_layer=(len(feats) == 0),
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs,
                node_attrs_index=None,
            )
            feats.append(node_feats)

        node_feats_out = jnp.concatenate(feats, axis=-1) if compute_node_feats else None

        return {
            "energy": None,
            "node_energy": None,
            "contributions": None,
            "node_feats": node_feats_out,
            "interaction_energy": None,
            "displacement": getattr(ctx, "displacement", None),
            "lammps_natoms": None,
        }
