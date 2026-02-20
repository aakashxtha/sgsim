"""Utility functions: smooth switching, helpers.

All functions are JAX-compatible and JIT-friendly.
"""

import jax.numpy as jnp


def smooth_step(x: jnp.ndarray, x0: float, width: float) -> jnp.ndarray:
    """Smooth step function transitioning from 0 to 1 around x0.

    Uses a sigmoid: 1 / (1 + exp(-(x - x0) / width))
    """
    return 1.0 / (1.0 + jnp.exp(-(x - x0) / width))


def smooth_cutoff(r: jnp.ndarray, r_cut: float, width: float = 0.5) -> jnp.ndarray:
    """Smooth cutoff function: 1 at r=0, smoothly -> 0 at r=r_cut.

    Uses a cosine switch: 0.5 * (1 + cos(pi * (r - r_start) / (r_cut - r_start)))
    for r in [r_start, r_cut], where r_start = r_cut - width.
    Returns 1 for r < r_start, 0 for r > r_cut.
    """
    r_start = r_cut - width
    # Normalized position in the switching region
    t = (r - r_start) / (r_cut - r_start)
    t = jnp.clip(t, 0.0, 1.0)
    switch = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    return jnp.where(r < r_start, 1.0, switch)


def safe_norm(dr: jnp.ndarray, epsilon: float = 1e-7) -> jnp.ndarray:
    """Compute vector norm with numerical safety to avoid NaN gradients at r=0."""
    return jnp.sqrt(jnp.sum(dr ** 2) + epsilon)


def safe_norm_batch(dr: jnp.ndarray, epsilon: float = 1e-7) -> jnp.ndarray:
    """Batch version: dr is (N, 3), returns (N,) norms."""
    return jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + epsilon)
