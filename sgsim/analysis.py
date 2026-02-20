"""Analysis tools for stress granule simulations.

All functions operate on JAX arrays and are JIT-compatible where possible.

- RDF: radial distribution function between particle type pairs
- MSD: mean squared displacement from trajectory
- Cluster detection: label propagation on sparse neighbor list (scalable)
- Condensate detection: largest cluster properties
- Binding statistics: occupancy fractions
- Order parameter: fraction of particles in largest cluster
"""

import jax
import jax.numpy as jnp
from functools import partial

from .partition import build_neighbor_list


def compute_rdf(
    positions: jnp.ndarray,
    box_size: jnp.ndarray,
    type_a: jnp.ndarray,
    type_b: jnp.ndarray,
    particle_types: jnp.ndarray,
    n_bins: int = 100,
    r_max: float = 20.0,
    max_neighbors: int = 256,
) -> tuple:
    """Compute radial distribution function g(r) between two particle types.

    Uses a neighbor list internally to avoid holding a full N x N distance
    matrix. For small systems (N < 2000) this has some overhead from NL
    construction, but for large systems it prevents OOM.

    Args:
        positions: (N, 3)
        box_size: (3,)
        type_a: int or array of ints — particle types for species A
        type_b: int or array of ints — particle types for species B
        particle_types: (N,) int32
        n_bins: number of histogram bins
        r_max: maximum distance (nm)
        max_neighbors: max neighbors in NL (increase if overflow)

    Returns:
        (r_bins, g_r): bin centers and g(r) values
    """
    n = positions.shape[0]
    r_edges = jnp.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    mask_a = jnp.isin(particle_types, type_a)  # (N,)
    mask_b = jnp.isin(particle_types, type_b)  # (N,)

    # Build neighbor list with cutoff = r_max
    nl = build_neighbor_list(positions, box_size, r_max, skin=0.0,
                             max_neighbors=max_neighbors)
    neighbors = nl.neighbors  # (N, max_neighbors)
    nl_mask = nl.mask          # (N, max_neighbors)

    # Compute distances for NL pairs only: (N, max_neighbors)
    dr = positions[:, None, :] - positions[neighbors]  # (N, max_nb, 3)
    dr = dr - box_size * jnp.round(dr / box_size)
    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-10)  # (N, max_nb)

    # Pair type mask: particle i is type_a, neighbor j is type_b
    i_is_a = mask_a[:, None]                                     # (N, 1)
    j_is_b = mask_b[jnp.clip(neighbors, 0, n - 1)]              # (N, max_nb)
    pair_mask = i_is_a & j_is_b & nl_mask                        # (N, max_nb)

    # Histogram via vectorized bin assignment
    bin_idx = jnp.floor(dist / (r_max / n_bins)).astype(jnp.int32)  # (N, max_nb)
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)

    # One-hot accumulation
    hist = jnp.zeros(n_bins, dtype=jnp.float32)
    valid_bins = jnp.where(pair_mask, bin_idx, -1)  # -1 for invalid
    # Flatten and scatter-add
    flat_bins = valid_bins.reshape(-1)
    flat_mask = pair_mask.reshape(-1)
    hist = jnp.zeros(n_bins, dtype=jnp.float32).at[flat_bins].add(
        flat_mask.astype(jnp.float32)
    )

    # Normalization
    n_a = jnp.sum(mask_a)
    n_b = jnp.sum(mask_b)
    volume = jnp.prod(box_size)
    rho_b = n_b / volume
    shell_vol = (4.0 / 3.0) * jnp.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    g_r = hist / (n_a * rho_b * shell_vol + 1e-10)

    return r_centers, g_r


def compute_msd(
    trajectory: list,
    box_size: jnp.ndarray,
) -> jnp.ndarray:
    """Compute mean squared displacement from a trajectory.

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
    max_neighbors: int = 128,
) -> jnp.ndarray:
    """Detect clusters via label propagation on a sparse neighbor list.

    Scalable to large systems (N > 10,000). The neighbor list build is
    O(N^2) but temporary; label propagation is O(N * max_neighbors * iters)
    instead of O(N^2 * iters).

    Args:
        positions: (N, 3)
        box_size: (3,)
        cutoff: distance cutoff for connectivity
        max_iterations: maximum propagation iterations
        max_neighbors: max neighbors per particle for NL

    Returns:
        (N,) int32 — cluster labels (0-based, min index in cluster)
    """
    n = positions.shape[0]

    # Build sparse neighbor list (skin=0: exact cutoff)
    nl = build_neighbor_list(positions, box_size, cutoff, skin=0.0,
                             max_neighbors=max_neighbors)
    neighbors = nl.neighbors  # (N, max_neighbors) int32
    nl_mask = nl.mask          # (N, max_neighbors) bool

    # Label propagation on sparse structure
    labels = jnp.arange(n, dtype=jnp.int32)

    def propagate_step(labels, _):
        # Gather neighbor labels: (N, max_neighbors)
        neighbor_labels = labels[neighbors]
        # Mask invalid neighbors with sentinel
        neighbor_labels = jnp.where(nl_mask, neighbor_labels, n)
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

    Fully vectorized — no Python loops over N.

    Args:
        labels: (N,) int32 — cluster labels from detect_clusters
        positions: (N, 3)
        box_size: (3,)
        particle_types: (N,) int32

    Returns:
        dict with n_clusters, largest_cluster_size, largest_cluster_fraction,
        cluster_sizes (descending)
    """
    n = labels.shape[0]

    # Count particles per label using scatter-add
    # labels are in range [0, N), each cluster's label = min particle index
    sizes = jnp.zeros(n, dtype=jnp.int32).at[labels].add(1)

    # Number of clusters = number of labels with count > 0
    n_clusters = jnp.sum(sizes > 0)

    # Largest cluster
    largest = jnp.max(sizes)
    fraction = largest / n

    # Sorted sizes (descending)
    sorted_sizes = jnp.sort(sizes)[::-1]

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

    for t in range(n_site_types):
        mask = site_types == t
        n_type = jnp.sum(mask)
        n_occupied = jnp.sum(binding_state.site_occupied * mask)
        occupancy = occupancy.at[t].set(
            jnp.where(n_type > 0, n_occupied / n_type, 0.0)
        )

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

    # Vectorized bin assignment
    coords = positions[:, axis]
    bin_idx = jnp.floor(coords / bin_width).astype(jnp.int32)
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)

    hist = jnp.zeros(n_bins, dtype=jnp.float32).at[bin_idx].add(1.0)
    density = hist / slab_volume

    return bin_centers, density
