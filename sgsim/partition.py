"""Neighbor list construction and update for efficient pairwise interactions.

Implements a cell-list based neighbor list that is JIT-compatible via JAX.
The neighbor list stores pairs of particles within a cutoff + skin distance,
enabling O(N) force evaluation instead of brute-force O(N^2).
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


class NeighborList(NamedTuple):
    """Dense neighbor list: each particle stores up to max_neighbors neighbors.

    Attributes:
        neighbors: (N, max_neighbors) int32 — neighbor particle indices.
                   Padded with N (out-of-range sentinel) for unused slots.
        mask: (N, max_neighbors) bool — True where neighbor is valid.
        overflow: scalar bool — True if any particle exceeded max_neighbors.
        reference_positions: (N, 3) — positions at last rebuild (for skin check).
    """
    neighbors: jnp.ndarray
    mask: jnp.ndarray
    overflow: jnp.ndarray
    reference_positions: jnp.ndarray


def build_neighbor_list(
    positions: jnp.ndarray,
    box_size: jnp.ndarray,
    cutoff: float,
    skin: float = 2.0,
    max_neighbors: int = 128,
    batch_size: int = None,
) -> NeighborList:
    """Build a neighbor list using batched pairwise distance computation.

    Processes particles in batches to avoid allocating the full (N, N, 3)
    distance tensor at once.  The batch size is chosen automatically to keep
    peak memory near 256 MB; pass ``batch_size`` explicitly to override (useful
    for testing).

    Args:
        positions: (N, 3) particle positions
        box_size: (3,) box dimensions for PBC
        cutoff: interaction cutoff distance (nm)
        skin: skin distance beyond cutoff (nm)
        max_neighbors: maximum neighbors per particle
        batch_size: rows of the distance matrix computed per iteration.
                    Defaults to an adaptive value targeting ~256 MB peak.

    Returns:
        NeighborList with neighbor indices and validity mask.
        Check ``nl.overflow`` to detect when max_neighbors was too small.
    """
    n = positions.shape[0]
    r_list = cutoff + skin

    # Adaptive batch size: target ~256 MB peak for the (B, N, 3) float32 slab.
    if batch_size is None:
        target_bytes = 256 * 1024 * 1024
        batch_size = max(1, target_bytes // max(n * 12, 1))
        batch_size = min(batch_size, n)

    sentinel_dist = jnp.float32(1e6)
    all_neighbors = []
    all_masks = []
    overflow = jnp.bool_(False)

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_pos = positions[batch_start:batch_end]  # (B, 3)

        # Pairwise displacements: this batch vs all N particles  (B, N, 3)
        dr = batch_pos[:, None, :] - positions[None, :, :]
        dr = dr - box_size * jnp.round(dr / box_size)
        dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-10)  # (B, N)

        # Exclude self-pairs
        batch_indices = jnp.arange(batch_start, batch_end)[:, None]  # (B, 1)
        full_indices = jnp.arange(n)[None, :]                         # (1, N)
        is_self = batch_indices == full_indices                         # (B, N)
        is_neighbor = (dist < r_list) & ~is_self                       # (B, N)

        # Sort each row by distance, keep the closest max_neighbors
        masked_dist = jnp.where(is_neighbor, dist, sentinel_dist)
        sorted_indices = jnp.argsort(masked_dist, axis=-1)
        batch_neighbors = sorted_indices[:, :max_neighbors]            # (B, K)
        sorted_dist = jnp.take_along_axis(masked_dist, sorted_indices, axis=-1)
        batch_mask = sorted_dist[:, :max_neighbors] < r_list           # (B, K)
        batch_neighbors = jnp.where(batch_mask, batch_neighbors, n)

        # Accumulate overflow flag
        neighbor_counts = jnp.sum(is_neighbor, axis=-1)                # (B,)
        overflow = overflow | jnp.any(neighbor_counts > max_neighbors)

        all_neighbors.append(batch_neighbors)
        all_masks.append(batch_mask)

    neighbors = jnp.concatenate(all_neighbors, axis=0)  # (N, max_neighbors)
    mask = jnp.concatenate(all_masks, axis=0)           # (N, max_neighbors)

    return NeighborList(
        neighbors=neighbors.astype(jnp.int32),
        mask=mask,
        overflow=overflow,
        reference_positions=positions,
    )


def needs_rebuild(
    positions: jnp.ndarray,
    reference_positions: jnp.ndarray,
    box_size: jnp.ndarray,
    skin: float = 2.0,
) -> jnp.ndarray:
    """Check if the neighbor list needs rebuilding based on particle displacement.

    The neighbor list is valid as long as no particle has moved more than skin/2
    from its reference position. This guarantees that no new neighbor pair has
    been missed.

    Args:
        positions: (N, 3) current positions
        reference_positions: (N, 3) positions at last NL build
        box_size: (3,) for PBC
        skin: skin distance used when building

    Returns:
        Scalar bool: True if rebuild is needed
    """
    dr = positions - reference_positions
    dr = dr - box_size * jnp.round(dr / box_size)
    max_displacement = jnp.max(jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-10))
    # Rebuild if any particle moved more than skin/2
    return max_displacement > skin / 2.0


def neighbor_pairs(neighbor_list: NeighborList) -> tuple:
    """Extract unique (i, j) pairs with i < j from the neighbor list.

    Returns:
        pair_i: (N_pairs,) int32 — first particle index
        pair_j: (N_pairs,) int32 — second particle index
        pair_mask: (N_pairs,) bool — valid pair mask
    """
    n = neighbor_list.neighbors.shape[0]
    max_nb = neighbor_list.neighbors.shape[1]

    # Particle indices: (N, max_neighbors)
    i_idx = jnp.arange(n)[:, None] * jnp.ones((1, max_nb), dtype=jnp.int32)
    j_idx = neighbor_list.neighbors  # (N, max_neighbors)

    # Flatten
    i_flat = i_idx.reshape(-1)  # (N * max_neighbors,)
    j_flat = j_idx.reshape(-1)
    mask_flat = neighbor_list.mask.reshape(-1)

    # Keep only i < j to avoid double counting
    valid = mask_flat & (i_flat < j_flat)

    return i_flat, j_flat, valid
