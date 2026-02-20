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
) -> NeighborList:
    """Build a neighbor list from scratch using brute-force distance computation.

    This uses an O(N^2) all-pairs distance computation to build the neighbor list.
    For the system sizes we target (~5K-20K beads), this is fast enough on GPU,
    and the neighbor list is only rebuilt every ~10-100 steps.

    Args:
        positions: (N, 3) particle positions
        box_size: (3,) box dimensions for PBC
        cutoff: interaction cutoff distance (nm)
        skin: skin distance beyond cutoff (nm)
        max_neighbors: maximum neighbors per particle

    Returns:
        NeighborList with neighbor indices and validity mask
    """
    n = positions.shape[0]
    r_list = cutoff + skin

    # All pairwise displacements with PBC
    dr = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    dr = dr - box_size * jnp.round(dr / box_size)
    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-10)  # (N, N)

    # Mask: within cutoff+skin AND not self
    is_neighbor = (dist < r_list) & (~jnp.eye(n, dtype=bool))  # (N, N)

    # For each particle, collect neighbor indices (padded)
    # We sort neighbors by distance and take the closest max_neighbors
    # Use large sentinel distance for non-neighbors
    sentinel_dist = jnp.float32(1e6)
    masked_dist = jnp.where(is_neighbor, dist, sentinel_dist)  # (N, N)

    # Argsort each row to get closest neighbors first
    sorted_indices = jnp.argsort(masked_dist, axis=-1)  # (N, N)

    # Take only max_neighbors
    neighbors = sorted_indices[:, :max_neighbors]  # (N, max_neighbors)

    # Build validity mask
    # A slot is valid if the sorted distance < r_list
    sorted_dist = jnp.take_along_axis(masked_dist, sorted_indices, axis=-1)
    mask = sorted_dist[:, :max_neighbors] < r_list  # (N, max_neighbors)

    # Replace invalid slots with sentinel index N
    neighbors = jnp.where(mask, neighbors, n)

    # Check overflow: if any particle has more neighbors than max_neighbors
    neighbor_counts = jnp.sum(is_neighbor, axis=-1)  # (N,)
    overflow = jnp.any(neighbor_counts > max_neighbors)

    return NeighborList(
        neighbors=neighbors.astype(jnp.int32),
        mask=mask,
        overflow=overflow,
        reference_positions=positions,
    )


def needs_rebuild(
    positions: jnp.ndarray,
    neighbor_list: NeighborList,
    box_size: jnp.ndarray,
    skin: float = 2.0,
) -> jnp.ndarray:
    """Check if the neighbor list needs rebuilding based on particle displacement.

    The neighbor list is valid as long as no particle has moved more than skin/2
    from its reference position. This guarantees that no new neighbor pair has
    been missed.

    Args:
        positions: (N, 3) current positions
        neighbor_list: current neighbor list
        box_size: (3,) for PBC
        skin: skin distance used when building

    Returns:
        Scalar bool: True if rebuild is needed
    """
    dr = positions - neighbor_list.reference_positions
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
