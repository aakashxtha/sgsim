"""Periodic boundary conditions: displacement and shift functions.

Provides the core spatial operations for the simulation box.
Can use JAX-MD's space module if available, or standalone implementation.
"""

import jax
import jax.numpy as jnp
from functools import partial


def periodic_displacement(box_size: jnp.ndarray):
    """Create a displacement function for periodic boundaries.

    Returns a function: displacement(ra, rb) -> dr
    where dr is the minimum image displacement vector from rb to ra.
    """
    @jax.jit
    def displacement(ra: jnp.ndarray, rb: jnp.ndarray) -> jnp.ndarray:
        dr = ra - rb
        dr = dr - box_size * jnp.round(dr / box_size)
        return dr

    return displacement


def periodic_shift(box_size: jnp.ndarray):
    """Create a shift function that wraps positions into the box.

    Returns a function: shift(positions, displacement) -> new_positions
    Both positions and new_positions are wrapped into [0, box_size).
    """
    @jax.jit
    def shift(positions: jnp.ndarray, dr: jnp.ndarray) -> jnp.ndarray:
        new_pos = positions + dr
        new_pos = new_pos % box_size
        return new_pos

    return shift


def periodic_wrap(positions: jnp.ndarray, box_size: jnp.ndarray) -> jnp.ndarray:
    """Wrap positions into the primary box [0, box_size)."""
    return positions % box_size


@partial(jax.jit, static_argnums=())
def pairwise_displacement(positions: jnp.ndarray, box_size: jnp.ndarray) -> jnp.ndarray:
    """Compute all pairwise displacement vectors with minimum image convention.

    Args:
        positions: (N, 3) particle positions
        box_size: (3,) box dimensions

    Returns:
        (N, N, 3) displacement matrix where dr[i, j] = r_i - r_j (minimum image)
    """
    # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    dr = positions[:, None, :] - positions[None, :, :]
    dr = dr - box_size * jnp.round(dr / box_size)
    return dr


@partial(jax.jit, static_argnums=())
def pairwise_distances(positions: jnp.ndarray, box_size: jnp.ndarray) -> jnp.ndarray:
    """Compute all pairwise distances with minimum image convention.

    Args:
        positions: (N, 3) particle positions
        box_size: (3,) box dimensions

    Returns:
        (N, N) distance matrix
    """
    dr = pairwise_displacement(positions, box_size)
    return jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-10)
