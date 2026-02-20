"""Conformational switching for G3BP1: compact ↔ expanded.

G3BP1's openness parameter phi ∈ [0, 1] is an auxiliary continuous variable
updated via overdamped Langevin dynamics in phi-space:

  dphi/dt = -(dE/dphi) / gamma_phi + sqrt(2 * kT * dt / gamma_phi) * noise

Energy landscape (purely phi-dependent):
  - Compact state (phi=0) favored by intrinsic autoinhibition bias:
    E_bias = k_compact * phi
  - Expanded state (phi=1) favored by RNA binding to RG sites:
    E_expand = -eps_expand * phi * n_rna_bound
  - phi is clamped to [0, 1] after each update

The spatial compaction force (acidic IDR ↔ RG IDR electrostatic attraction)
is handled by Debye-Huckel in the main BD force computation, keeping the
conformational variable cleanly separated from position dynamics.
"""

import jax
import jax.numpy as jnp

from .types import ConformationalState


def init_conformational_state(
    n_switchable: int,
    molecule_indices: jnp.ndarray,
    acidic_bead_indices: jnp.ndarray,
    rg_bead_indices: jnp.ndarray,
    initial_openness: float = 0.0,
) -> ConformationalState:
    """Initialize conformational state for switchable molecules.

    Args:
        n_switchable: number of switchable molecules (G3BP1 dimers)
        molecule_indices: (n_switchable,) int32 — molecule IDs
        acidic_bead_indices: (n_switchable,) int32 — particle index of acidic IDR bead
        rg_bead_indices: (n_switchable,) int32 — particle index of first RG IDR bead
        initial_openness: initial phi value (0=compact)

    Returns:
        ConformationalState
    """
    return ConformationalState(
        openness=jnp.full(n_switchable, initial_openness, dtype=jnp.float32),
        molecule_indices=molecule_indices,
        acidic_bead_indices=acidic_bead_indices,
        rg_bead_indices=rg_bead_indices,
    )


def conformational_energy(
    positions: jnp.ndarray,
    box_size: jnp.ndarray,
    conf_state: ConformationalState,
    k_compact: float,
    r0_compact: float,
    eps_rna_expand: float,
    n_rna_bound: jnp.ndarray,
) -> jnp.ndarray:
    """Compute conformational energy for all switchable molecules.

    E = sum_i [ k_compact * phi_i - eps * phi_i * n_rna_i ]

    Purely phi-dependent. Physical compaction is handled by Debye-Huckel
    electrostatics between negatively charged acidic IDR and positively
    charged RG IDR in the main force computation.

    Args:
        positions: (N, 3) particle positions (unused, kept for API compat)
        box_size: (3,) (unused, kept for API compat)
        conf_state: ConformationalState
        k_compact: compact bias strength (kT)
        r0_compact: unused, kept for API compatibility
        eps_rna_expand: RNA-expansion coupling (kT per RNA contact)
        n_rna_bound: (n_switchable,) number of RNA sites bound per molecule

    Returns:
        Scalar conformational energy
    """
    phi = conf_state.openness  # (n_switchable,)

    # Intrinsic compact bias: phi=0 favored by electrostatic autoinhibition
    e_bias = k_compact * phi

    # RNA expansion coupling: negative energy for expanded state with RNA
    e_expand = -eps_rna_expand * phi * n_rna_bound

    return jnp.sum(e_bias + e_expand)


def update_conformational_state(
    conf_state: ConformationalState,
    positions: jnp.ndarray,
    box_size: jnp.ndarray,
    k_compact: float,
    r0_compact: float,
    eps_rna_expand: float,
    n_rna_bound: jnp.ndarray,
    gamma_phi: float,
    kT: float,
    dt: float,
    rng_key: jnp.ndarray,
) -> tuple:
    """Update phi via overdamped Langevin in phi-space.

    dphi = -(dE/dphi) / gamma_phi * dt + sqrt(2 * kT * dt / gamma_phi) * noise

    Args:
        conf_state: current ConformationalState
        positions: (N, 3)
        box_size: (3,)
        k_compact, r0_compact, eps_rna_expand: energy params
        n_rna_bound: (n_switchable,) RNA contacts per molecule
        gamma_phi: friction in phi-space
        kT: thermal energy
        dt: timestep
        rng_key: JAX PRNG key

    Returns:
        (new_conf_state, new_rng_key)
    """
    phi = conf_state.openness
    n = phi.shape[0]

    # Compute dE/dphi analytically
    # E_bias = k * phi → dE/dphi = k
    # E_expand = -eps * phi * n_rna → dE/dphi = -eps * n_rna
    dE_dphi = k_compact - eps_rna_expand * n_rna_bound  # (n_switchable,)

    # Overdamped Langevin update
    key, noise_key = jax.random.split(rng_key)
    drift = -(dE_dphi / gamma_phi) * dt
    noise_scale = jnp.sqrt(2.0 * kT * dt / gamma_phi)
    noise = jax.random.normal(noise_key, (n,)) * noise_scale

    new_phi = phi + drift + noise

    # Clamp to [0, 1]
    new_phi = jnp.clip(new_phi, 0.0, 1.0)

    new_state = ConformationalState(
        openness=new_phi,
        molecule_indices=conf_state.molecule_indices,
        acidic_bead_indices=conf_state.acidic_bead_indices,
        rg_bead_indices=conf_state.rg_bead_indices,
    )

    return new_state, key


def count_rna_bound_per_molecule(
    binding_state,
    site_types: jnp.ndarray,
    site_molecule: jnp.ndarray,
    molecule_indices: jnp.ndarray,
    rna_site_type: int,
) -> jnp.ndarray:
    """Count RNA binding sites bound per switchable molecule.

    Args:
        binding_state: BindingState
        site_types: (N_sites,) int32
        site_molecule: (N_sites,) int32
        molecule_indices: (n_switchable,) int32 — the molecule IDs to count for
        rna_site_type: int — the RNA_BINDING_SITE type index

    Returns:
        (n_switchable,) int32 — number of RNA sites bound per molecule
    """
    n_switchable = molecule_indices.shape[0]

    # For each active bound pair, check if one site is RNA_BINDING_SITE
    # and the other belongs to a switchable molecule
    pair_site_i = binding_state.bound_pairs[:, 0]
    pair_site_j = binding_state.bound_pairs[:, 1]

    # Which pairs involve an RNA site?
    is_rna_i = (site_types[pair_site_i] == rna_site_type)
    is_rna_j = (site_types[pair_site_j] == rna_site_type)

    # For each switchable molecule, count RNA bonds
    def count_for_mol(mol_id):
        # Pairs where one partner is in this molecule and the other is RNA
        in_mol_i = (site_molecule[pair_site_i] == mol_id)
        in_mol_j = (site_molecule[pair_site_j] == mol_id)

        # Count: (mol_partner on i side, RNA on j side) OR (RNA on i side, mol_partner on j)
        count = jnp.sum(
            ((in_mol_i & is_rna_j) | (in_mol_j & is_rna_i)) & binding_state.bound_mask
        )
        return count

    counts = jax.vmap(count_for_mol)(molecule_indices)
    return counts
