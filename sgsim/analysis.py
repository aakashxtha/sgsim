"""Analysis tools for stress granule simulations.

All functions operate on JAX arrays and are JIT-compatible where possible.

- RDF: radial distribution function between particle type pairs
- MSD: mean squared displacement from trajectory
- Cluster detection: label propagation on neighbor graph (GPU-friendly)
- Condensate detection: largest cluster properties
- Binding statistics: occupancy fractions
- Order parameter: fraction of particles in largest cluster
"""

import jax
import jax.numpy as jnp
from functools import partial


def compute_rdf(
    positions: jnp.ndarray,
    box_size: jnp.ndarray,
    type_a: jnp.ndarray,
    type_b: jnp.ndarray,
    particle_types: jnp.ndarray,
    n_bins: int = 100,
    r_max: float = 20.0,
) -> tuple:
    """Compute radial distribution function g(r) between two particle types.

    Args:
        positions: (N, 3)
        box_size: (3,)
        type_a: int or array of ints — particle types for species A
        type_b: int or array of ints — particle types for species B
        particle_types: (N,) int32
        n_bins: number of histogram bins
        r_max: maximum distance (nm)

    Returns:
        (r_bins, g_r): bin centers and g(r) values
    """
    n = positions.shape[0]
    dr_bins = r_max / n_bins
    r_edges = jnp.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    # Mask for type A and type B particles
    mask_a = jnp.isin(particle_types, type_a)  # (N,)
    mask_b = jnp.isin(particle_types, type_b)  # (N,)

    # All pairwise distances with PBC
    dr = positions[:, None, :] - positions[None, :, :]
    dr = dr - box_size * jnp.round(dr / box_size)
    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-10)

    # Pair mask: (A, B) pairs, excluding self
    pair_mask = mask_a[:, None] & mask_b[None, :] & (~jnp.eye(n, dtype=bool))

    # Histogram
    hist = jnp.zeros(n_bins)
    for i in range(n_bins):
        in_bin = (dist >= r_edges[i]) & (dist < r_edges[i + 1]) & pair_mask
        hist = hist.at[i].set(jnp.sum(in_bin))

    # Normalization
    n_a = jnp.sum(mask_a)
    n_b = jnp.sum(mask_b)
    volume = jnp.prod(box_size)
    rho_b = n_b / volume

    # Shell volumes
    shell_vol = (4.0 / 3.0) * jnp.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)

    # g(r) = hist / (N_a * rho_b * shell_vol)
    g_r = hist / (n_a * rho_b * shell_vol + 1e-10)

    return r_centers, g_r


def compute_msd(
    trajectory: list,
    box_size: jnp.ndarray,
) -> jnp.ndarray:
    """Compute mean squared displacement from a trajectory.

    Uses the unwrapped displacement from the first frame.

    Args:
        trajectory: list of (N, 3) position arrays
        box_size: (3,) for PBC unwrapping

    Returns:
        (n_frames,) MSD values
    """
    ref = trajectory[0]
    n_frames = len(trajectory)
    msd = jnp.zeros(n_frames)

    for t in range(n_frames):
        dr = trajectory[t] - ref
        dr = dr - box_size * jnp.round(dr / box_size)
        msd = msd.at[t].set(jnp.mean(jnp.sum(dr ** 2, axis=-1)))

    return msd


def detect_clusters(
    positions: jnp.ndarray,
    box_size: jnp.ndarray,
    cutoff: float = 5.0,
    max_iterations: int = 50,
) -> jnp.ndarray:
    """Detect clusters via iterative label propagation.

    GPU-friendly alternative to scipy connected components.
    Each particle starts with its own label, then iteratively
    adopts the minimum label among its neighbors.

    Args:
        positions: (N, 3)
        box_size: (3,)
        cutoff: distance cutoff for connectivity
        max_iterations: maximum propagation iterations

    Returns:
        (N,) int32 — cluster labels (0-based)
    """
    n = positions.shape[0]

    # Build adjacency: within cutoff and not self
    dr = positions[:, None, :] - positions[None, :, :]
    dr = dr - box_size * jnp.round(dr / box_size)
    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-10)
    adj = (dist < cutoff) & (~jnp.eye(n, dtype=bool))

    # Initialize labels
    labels = jnp.arange(n, dtype=jnp.int32)

    def propagate_step(labels, _):
        # For each particle, take minimum label among self + neighbors
        # Replace non-neighbor labels with large sentinel
        neighbor_labels = jnp.where(adj, labels[None, :], n)  # (N, N)
        min_neighbor = jnp.min(neighbor_labels, axis=-1)  # (N,)
        new_labels = jnp.minimum(labels, min_neighbor)
        return new_labels, None

    labels, _ = jax.lax.scan(propagate_step, labels, None, length=max_iterations)

    return labels


def cluster_statistics(
    labels: jnp.ndarray,
    positions: jnp.ndarray,
    box_size: jnp.ndarray,
    particle_types: jnp.ndarray,
) -> dict:
    """Compute statistics for detected clusters.

    Args:
        labels: (N,) int32 — cluster labels from detect_clusters
        positions: (N, 3)
        box_size: (3,)
        particle_types: (N,) int32

    Returns:
        dict with:
            n_clusters: number of unique clusters
            largest_cluster_size: number of particles in largest cluster
            largest_cluster_fraction: fraction of particles in largest cluster
            cluster_sizes: sorted array of cluster sizes (descending)
    """
    n = labels.shape[0]
    unique_labels = jnp.unique(labels, size=n, fill_value=-1)

    # Count size of each cluster
    sizes = jnp.zeros(n, dtype=jnp.int32)
    for i in range(n):
        mask = (labels == unique_labels[i])
        sizes = sizes.at[i].set(jnp.sum(mask))

    # Filter out empty clusters (fill_value=-1)
    valid = sizes > 0
    n_clusters = jnp.sum(valid)
    sorted_sizes = jnp.sort(sizes)[::-1]

    largest = sorted_sizes[0]
    fraction = largest / n

    return {
        "n_clusters": n_clusters,
        "largest_cluster_size": largest,
        "largest_cluster_fraction": fraction,
        "cluster_sizes": sorted_sizes,
    }


def binding_occupancy(binding_state, site_types: jnp.ndarray, n_site_types: int) -> jnp.ndarray:
    """Compute binding site occupancy fraction per site type.

    Args:
        binding_state: BindingState
        site_types: (N_sites,) int32
        n_site_types: number of distinct site types

    Returns:
        (n_site_types,) float32 — fraction occupied per type
    """
    occupancy = jnp.zeros(n_site_types, dtype=jnp.float32)
    total = jnp.zeros(n_site_types, dtype=jnp.float32)

    for t in range(n_site_types):
        mask = site_types == t
        n_type = jnp.sum(mask)
        n_occupied = jnp.sum(binding_state.site_occupied * mask)
        occupancy = occupancy.at[t].set(
            jnp.where(n_type > 0, n_occupied / n_type, 0.0)
        )
        total = total.at[t].set(n_type)

    return occupancy


def compute_density_profile(
    positions: jnp.ndarray,
    box_size: jnp.ndarray,
    n_bins: int = 50,
    axis: int = 0,
) -> tuple:
    """Compute 1D density profile along an axis.

    Args:
        positions: (N, 3)
        box_size: (3,)
        n_bins: number of bins along the axis
        axis: 0=x, 1=y, 2=z

    Returns:
        (bin_centers, density): bin centers and number density in each slab
    """
    L = box_size[axis]
    bin_edges = jnp.linspace(0, L, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = L / n_bins

    # Cross-sectional area
    axes = [0, 1, 2]
    axes.remove(axis)
    area = box_size[axes[0]] * box_size[axes[1]]
    slab_volume = area * bin_width

    coords = positions[:, axis]
    hist = jnp.zeros(n_bins)
    for i in range(n_bins):
        in_bin = (coords >= bin_edges[i]) & (coords < bin_edges[i + 1])
        hist = hist.at[i].set(jnp.sum(in_bin))

    density = hist / slab_volume

    return bin_centers, density
