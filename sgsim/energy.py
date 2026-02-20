"""Energy functions for the coarse-grained stress granule model.

Each potential is a pure function returning a scalar energy value.
Forces are obtained via jax.grad(energy_fn, argnums=0) on positions.
All functions are designed for JIT compilation.
"""

import jax
import jax.numpy as jnp
from functools import partial

from .utils import safe_norm


# ============================================================================
# Bonded potentials
# ============================================================================


def harmonic_bond_energy(
    positions: jnp.ndarray,
    bond_pairs: jnp.ndarray,
    bond_types: jnp.ndarray,
    bond_k: jnp.ndarray,
    bond_r0: jnp.ndarray,
    box_size: jnp.ndarray,
) -> jnp.ndarray:
    """Harmonic bond energy: E = sum_bonds 0.5 * k * (r - r0)^2

    Args:
        positions: (N, 3) particle positions
        bond_pairs: (N_bonds, 2) int32, particle index pairs
        bond_types: (N_bonds,) int32, bond type indices
        bond_k: (N_bond_types,) spring constants
        bond_r0: (N_bond_types,) equilibrium lengths
        box_size: (3,) box dimensions for PBC

    Returns:
        Scalar total bond energy
    """
    if bond_pairs.shape[0] == 0:
        return jnp.float32(0.0)

    # Get positions of bonded particles
    r_i = positions[bond_pairs[:, 0]]  # (N_bonds, 3)
    r_j = positions[bond_pairs[:, 1]]  # (N_bonds, 3)

    # Minimum image displacement
    dr = r_i - r_j
    dr = dr - box_size * jnp.round(dr / box_size)

    # Distances
    r = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-10)  # (N_bonds,)

    # Per-bond parameters
    k = bond_k[bond_types]     # (N_bonds,)
    r0 = bond_r0[bond_types]   # (N_bonds,)

    # Energy
    energies = 0.5 * k * (r - r0) ** 2
    return jnp.sum(energies)


def harmonic_angle_energy(
    positions: jnp.ndarray,
    angle_triples: jnp.ndarray,
    angle_types: jnp.ndarray,
    angle_k: jnp.ndarray,
    angle_theta0: jnp.ndarray,
    box_size: jnp.ndarray,
) -> jnp.ndarray:
    """Harmonic angle energy: E = sum_angles 0.5 * k * (theta - theta0)^2

    Args:
        positions: (N, 3) particle positions
        angle_triples: (N_angles, 3) int32, particle index triples (i, j, k)
                       where j is the central particle
        angle_types: (N_angles,) int32
        angle_k: (N_angle_types,) bending stiffness
        angle_theta0: (N_angle_types,) equilibrium angles
        box_size: (3,) box dimensions

    Returns:
        Scalar total angle energy
    """
    if angle_triples.shape[0] == 0:
        return jnp.float32(0.0)

    # Positions of angle particles
    r_i = positions[angle_triples[:, 0]]  # (N_angles, 3)
    r_j = positions[angle_triples[:, 1]]  # (N_angles, 3) -- central
    r_k = positions[angle_triples[:, 2]]  # (N_angles, 3)

    # Displacement vectors with PBC
    dr_ji = r_i - r_j
    dr_ji = dr_ji - box_size * jnp.round(dr_ji / box_size)

    dr_jk = r_k - r_j
    dr_jk = dr_jk - box_size * jnp.round(dr_jk / box_size)

    # Compute angle via dot product
    cos_theta = jnp.sum(dr_ji * dr_jk, axis=-1) / (
        jnp.sqrt(jnp.sum(dr_ji ** 2, axis=-1) + 1e-10) *
        jnp.sqrt(jnp.sum(dr_jk ** 2, axis=-1) + 1e-10)
    )
    # Clamp for numerical safety
    cos_theta = jnp.clip(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = jnp.arccos(cos_theta)  # (N_angles,)

    # Per-angle parameters
    k = angle_k[angle_types]
    theta0 = angle_theta0[angle_types]

    # Energy
    energies = 0.5 * k * (theta - theta0) ** 2
    return jnp.sum(energies)


# ============================================================================
# Non-bonded pair potentials
# ============================================================================


def wca_energy_pair(
    dr: jnp.ndarray,
    sigma_ij: jnp.ndarray,
) -> jnp.ndarray:
    """WCA (purely repulsive LJ) energy for a single pair.

    E = 4 * eps * [(sigma/r)^12 - (sigma/r)^6] + eps  for r < r_cut
    E = 0  for r >= r_cut
    where r_cut = sigma * 2^(1/6), eps = 1.0 (in kT units)

    Args:
        dr: (3,) displacement vector
        sigma_ij: effective diameter

    Returns:
        Scalar energy
    """
    r = safe_norm(dr)
    r_cut = sigma_ij * (2.0 ** (1.0 / 6.0))
    inv_r = sigma_ij / jnp.maximum(r, 0.1)  # prevent division by zero
    inv_r6 = inv_r ** 6
    inv_r12 = inv_r6 ** 2
    e_lj = 4.0 * (inv_r12 - inv_r6) + 1.0
    return jnp.where(r < r_cut, e_lj, 0.0)


def yukawa_energy_pair(
    dr: jnp.ndarray,
    epsilon_ij: jnp.ndarray,
    sigma_ij: jnp.ndarray,
    kappa: jnp.ndarray,
) -> jnp.ndarray:
    """Short-range Yukawa attraction for a single pair.

    E = -epsilon * (sigma/r) * exp(-kappa * (r - sigma))  for r > sigma
    E = 0  for r <= sigma (overlap region handled by WCA)

    Args:
        dr: (3,) displacement vector
        epsilon_ij: attraction strength (kT)
        sigma_ij: effective diameter (nm)
        kappa: inverse screening length (1/nm)

    Returns:
        Scalar energy
    """
    r = safe_norm(dr)
    e_yukawa = -epsilon_ij * (sigma_ij / jnp.maximum(r, 0.1)) * jnp.exp(-kappa * (r - sigma_ij))
    return jnp.where((r > sigma_ij) & (epsilon_ij > 0.0), e_yukawa, 0.0)


def debye_huckel_energy_pair(
    dr: jnp.ndarray,
    q_i: jnp.ndarray,
    q_j: jnp.ndarray,
    debye_length: jnp.ndarray,
    sigma_ij: jnp.ndarray,
) -> jnp.ndarray:
    """Screened electrostatic (Debye-Huckel) energy for a single pair.

    E = (q_i * q_j * lB / r) * exp(-r / lambda_D)  for r > sigma
    where lB ~ 0.7 nm is the Bjerrum length in water at 300K.

    Args:
        dr: (3,) displacement vector
        q_i, q_j: charges (elementary charges)
        debye_length: screening length (nm)
        sigma_ij: minimum approach distance

    Returns:
        Scalar energy (kT)
    """
    r = safe_norm(dr)
    lB = 0.7  # Bjerrum length in nm (water, 300K)
    e_dh = q_i * q_j * lB / jnp.maximum(r, 0.1) * jnp.exp(-r / debye_length)
    # Only apply beyond contact distance and when charges are nonzero
    has_charge = (jnp.abs(q_i) > 0.01) & (jnp.abs(q_j) > 0.01)
    return jnp.where((r > sigma_ij) & has_charge, e_dh, 0.0)


# ============================================================================
# Pairwise non-bonded energy (over all pairs from neighbor list or brute force)
# ============================================================================


def nonbonded_energy_bruteforce(
    positions: jnp.ndarray,
    particle_types: jnp.ndarray,
    particle_charges: jnp.ndarray,
    epsilon_attract: jnp.ndarray,
    sigma: jnp.ndarray,
    kappa: jnp.ndarray,
    debye_length: jnp.ndarray,
    box_size: jnp.ndarray,
    cutoff: float,
    bond_pairs: jnp.ndarray,
) -> jnp.ndarray:
    """Compute total non-bonded energy using brute-force O(N^2) pairwise evaluation.

    This is used for small systems and testing. For production, use neighbor-list
    based evaluation (to be added in Phase B).

    Args:
        positions: (N, 3)
        particle_types: (N,) int32
        particle_charges: (N,) float32
        epsilon_attract: (N_types, N_types) attraction strengths
        sigma: (N_types, N_types) effective diameters
        kappa: scalar, Yukawa screening
        debye_length: scalar, Debye length
        box_size: (3,)
        cutoff: non-bonded cutoff distance (nm)
        bond_pairs: (N_bonds, 2) -- pairs to exclude

    Returns:
        Scalar total non-bonded energy
    """
    n = positions.shape[0]

    # All pairwise displacements with PBC
    dr_all = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    dr_all = dr_all - box_size * jnp.round(dr_all / box_size)
    dist_all = jnp.sqrt(jnp.sum(dr_all ** 2, axis=-1) + 1e-10)  # (N, N)

    # Type-pair lookups
    types_i = particle_types[:, None]  # (N, 1)
    types_j = particle_types[None, :]  # (1, N)
    eps_ij = epsilon_attract[types_i, types_j]  # (N, N)
    sig_ij = sigma[types_i, types_j]  # (N, N)

    # Charges
    q_i = particle_charges[:, None]  # (N, 1)
    q_j = particle_charges[None, :]  # (1, N)

    # --- WCA ---
    r_cut_wca = sig_ij * (2.0 ** (1.0 / 6.0))
    inv_r = sig_ij / jnp.maximum(dist_all, 0.1)
    inv_r6 = inv_r ** 6
    e_wca = 4.0 * (inv_r6 ** 2 - inv_r6) + 1.0
    e_wca = jnp.where(dist_all < r_cut_wca, e_wca, 0.0)

    # --- Yukawa attraction ---
    e_yukawa = -eps_ij * (sig_ij / jnp.maximum(dist_all, 0.1)) * jnp.exp(-kappa * (dist_all - sig_ij))
    e_yukawa = jnp.where((dist_all > sig_ij) & (eps_ij > 0.0), e_yukawa, 0.0)

    # --- Debye-Huckel ---
    lB = 0.7
    e_dh = q_i * q_j * lB / jnp.maximum(dist_all, 0.1) * jnp.exp(-dist_all / debye_length)
    has_charge = (jnp.abs(q_i) > 0.01) & (jnp.abs(q_j) > 0.01)
    e_dh = jnp.where((dist_all > sig_ij) & has_charge, e_dh, 0.0)

    # Total pairwise energy
    e_pair = e_wca + e_yukawa + e_dh

    # Apply cutoff
    e_pair = jnp.where(dist_all < cutoff, e_pair, 0.0)

    # Zero self-interactions (diagonal)
    e_pair = jnp.where(jnp.eye(n, dtype=bool), 0.0, e_pair)

    # Build exclusion mask for bonded pairs
    if bond_pairs.shape[0] > 0:
        excl_mask = jnp.zeros((n, n), dtype=bool)
        excl_mask = excl_mask.at[bond_pairs[:, 0], bond_pairs[:, 1]].set(True)
        excl_mask = excl_mask.at[bond_pairs[:, 1], bond_pairs[:, 0]].set(True)
        e_pair = jnp.where(excl_mask, 0.0, e_pair)

    # Sum upper triangle only (avoid double counting)
    mask_upper = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    return jnp.sum(e_pair * mask_upper)


# ============================================================================
# Neighbor-list based non-bonded energy
# ============================================================================


def nonbonded_energy_neighborlist(
    positions: jnp.ndarray,
    particle_types: jnp.ndarray,
    particle_charges: jnp.ndarray,
    epsilon_attract: jnp.ndarray,
    sigma: jnp.ndarray,
    kappa: jnp.ndarray,
    debye_length: jnp.ndarray,
    box_size: jnp.ndarray,
    cutoff: float,
    bond_exclusion_mask: jnp.ndarray,
    pair_i: jnp.ndarray,
    pair_j: jnp.ndarray,
    pair_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute total non-bonded energy using neighbor list pairs.

    This evaluates WCA + Yukawa + Debye-Huckel over pre-extracted neighbor pairs.
    Much faster than brute-force for large systems.

    Args:
        positions: (N, 3)
        particle_types: (N,) int32
        particle_charges: (N,) float32
        epsilon_attract: (N_types, N_types) attraction matrix
        sigma: (N_types, N_types) diameter matrix
        kappa: scalar, Yukawa screening
        debye_length: scalar, Debye length
        box_size: (3,)
        cutoff: interaction cutoff distance (nm)
        bond_exclusion_mask: (N, N) bool — True for bonded pairs to exclude.
                             Can be a sparse representation in future.
        pair_i: (N_pairs,) int32 — first particle in pair
        pair_j: (N_pairs,) int32 — second particle in pair
        pair_mask: (N_pairs,) bool — which pairs are valid

    Returns:
        Scalar total non-bonded energy
    """
    # Displacements for all pairs
    r_i = positions[pair_i]  # (N_pairs, 3)
    r_j = positions[pair_j]  # (N_pairs, 3)
    dr = r_i - r_j
    dr = dr - box_size * jnp.round(dr / box_size)
    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-10)  # (N_pairs,)

    # Type lookups
    type_i = particle_types[pair_i]
    type_j = particle_types[pair_j]
    eps_ij = epsilon_attract[type_i, type_j]
    sig_ij = sigma[type_i, type_j]
    q_i = particle_charges[pair_i]
    q_j = particle_charges[pair_j]

    # --- WCA ---
    r_cut_wca = sig_ij * (2.0 ** (1.0 / 6.0))
    inv_r = sig_ij / jnp.maximum(dist, 0.1)
    inv_r6 = inv_r ** 6
    e_wca = 4.0 * (inv_r6 ** 2 - inv_r6) + 1.0
    e_wca = jnp.where(dist < r_cut_wca, e_wca, 0.0)

    # --- Yukawa attraction ---
    e_yukawa = -eps_ij * (sig_ij / jnp.maximum(dist, 0.1)) * jnp.exp(-kappa * (dist - sig_ij))
    e_yukawa = jnp.where((dist > sig_ij) & (eps_ij > 0.0), e_yukawa, 0.0)

    # --- Debye-Huckel ---
    lB = 0.7
    e_dh = q_i * q_j * lB / jnp.maximum(dist, 0.1) * jnp.exp(-dist / debye_length)
    has_charge = (jnp.abs(q_i) > 0.01) & (jnp.abs(q_j) > 0.01)
    e_dh = jnp.where((dist > sig_ij) & has_charge, e_dh, 0.0)

    # Total per-pair energy
    e_pair = e_wca + e_yukawa + e_dh

    # Apply distance cutoff
    e_pair = jnp.where(dist < cutoff, e_pair, 0.0)

    # Exclude bonded pairs
    is_excluded = bond_exclusion_mask[pair_i, pair_j]
    e_pair = jnp.where(is_excluded, 0.0, e_pair)

    # Apply pair validity mask
    e_pair = jnp.where(pair_mask, e_pair, 0.0)

    return jnp.sum(e_pair)


def build_bond_exclusion_mask(
    n_particles: int,
    bond_pairs: jnp.ndarray,
) -> jnp.ndarray:
    """Build a boolean matrix marking bonded pairs for exclusion.

    Args:
        n_particles: total number of particles
        bond_pairs: (N_bonds, 2) int32

    Returns:
        (N, N) bool mask, True for pairs that should be excluded
    """
    mask = jnp.zeros((n_particles, n_particles), dtype=bool)
    if bond_pairs.shape[0] > 0:
        mask = mask.at[bond_pairs[:, 0], bond_pairs[:, 1]].set(True)
        mask = mask.at[bond_pairs[:, 1], bond_pairs[:, 0]].set(True)
    return mask


# ============================================================================
# Total energy
# ============================================================================


def total_energy(
    positions: jnp.ndarray,
    particle_types: jnp.ndarray,
    particle_charges: jnp.ndarray,
    topology,
    params,
    box_size: jnp.ndarray,
    cutoff: float = 10.0,
    neighbor_data: tuple = None,
) -> jnp.ndarray:
    """Total potential energy of the system (bonded + non-bonded).

    Args:
        positions: (N, 3) particle positions
        particle_types: (N,) int32 particle type indices
        particle_charges: (N,) float32 particle charges
        topology: Topology NamedTuple
        params: InteractionParams NamedTuple
        box_size: (3,) simulation box size
        cutoff: non-bonded cutoff (nm)
        neighbor_data: optional (pair_i, pair_j, pair_mask, bond_excl_mask)
                       from neighbor list. If None, uses brute force.

    Returns:
        Scalar total energy (kT)
    """
    e_bond = harmonic_bond_energy(
        positions,
        topology.bond_pairs,
        topology.bond_types,
        params.bond_k,
        params.bond_r0,
        box_size,
    )

    e_angle = harmonic_angle_energy(
        positions,
        topology.angle_triples,
        topology.angle_types,
        params.angle_k,
        params.angle_theta0,
        box_size,
    )

    if neighbor_data is not None:
        pair_i, pair_j, pair_mask, bond_excl_mask = neighbor_data
        e_nonbond = nonbonded_energy_neighborlist(
            positions,
            particle_types,
            particle_charges,
            params.epsilon_attract,
            params.sigma,
            params.kappa,
            params.debye_length,
            box_size,
            cutoff,
            bond_excl_mask,
            pair_i,
            pair_j,
            pair_mask,
        )
    else:
        e_nonbond = nonbonded_energy_bruteforce(
            positions,
            particle_types,
            particle_charges,
            params.epsilon_attract,
            params.sigma,
            params.kappa,
            params.debye_length,
            box_size,
            cutoff,
            topology.bond_pairs,
        )

    return e_bond + e_angle + e_nonbond
