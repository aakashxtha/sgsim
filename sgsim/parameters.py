"""Default interaction parameters for stress granule proteins.

Parameters are organized as lookup tables indexed by particle type or
binding site type. Values are in reduced units:
  - Length: nm
  - Energy: kT (at 300K, kT ~ 2.494 kJ/mol)
  - Charge: elementary charges (e)

These defaults are based on literature values from:
  - Guillen-Boixet et al., Cell 2020
  - Sanders et al., Cell 2020
  - Schulte et al., Open Biology 2016
"""

import jax.numpy as jnp
from .types import (
    N_PARTICLE_TYPES, N_BINDING_SITE_TYPES, N_BOND_TYPES, N_ANGLE_TYPES,
    PARTICLE_TYPES as PT, BINDING_SITE_TYPES as BST,
    InteractionParams,
)


def default_params() -> InteractionParams:
    """Build the default interaction parameter set."""

    # --- Non-bonded attraction strengths (kT) ---
    # Symmetric matrix: epsilon_attract[type_i, type_j]
    # Most pairs have zero explicit attraction (only WCA repulsion).
    # Specific domain-domain attractions are set below.
    eps = jnp.zeros((N_PARTICLE_TYPES, N_PARTICLE_TYPES), dtype=jnp.float32)

    # RRM-RNA: moderate specific binding
    eps = eps.at[PT["RRM"], PT["RNA_BEAD"]].set(4.0)
    eps = eps.at[PT["RNA_BEAD"], PT["RRM"]].set(4.0)

    # RG_IDR-RNA: moderate promiscuous binding
    eps = eps.at[PT["RG_IDR"], PT["RNA_BEAD"]].set(3.0)
    eps = eps.at[PT["RNA_BEAD"], PT["RG_IDR"]].set(3.0)

    # RG_IDR-RG_IDR: weak homotypic (drives clustering)
    eps = eps.at[PT["RG_IDR"], PT["RG_IDR"]].set(1.5)

    # UBAP2L_RGG-RNA
    eps = eps.at[PT["UBAP2L_RGG"], PT["RNA_BEAD"]].set(3.0)
    eps = eps.at[PT["RNA_BEAD"], PT["UBAP2L_RGG"]].set(3.0)

    # UBAP2L_IDR self-association (weak, critical for node function)
    eps = eps.at[PT["UBAP2L_IDR"], PT["UBAP2L_IDR"]].set(1.5)

    # CAPRIN1_RGG-RNA
    eps = eps.at[PT["CAPRIN1_RGG"], PT["RNA_BEAD"]].set(3.0)
    eps = eps.at[PT["RNA_BEAD"], PT["CAPRIN1_RGG"]].set(3.0)

    # FXR1_KH-RNA
    eps = eps.at[PT["FXR1_KH"], PT["RNA_BEAD"]].set(3.5)
    eps = eps.at[PT["RNA_BEAD"], PT["FXR1_KH"]].set(3.5)

    # FXR1_RGG-RNA
    eps = eps.at[PT["FXR1_RGG"], PT["RNA_BEAD"]].set(2.5)
    eps = eps.at[PT["RNA_BEAD"], PT["FXR1_RGG"]].set(2.5)

    # TIA1_RRM-RNA (3 RRMs, strong binding)
    eps = eps.at[PT["TIA1_RRM"], PT["RNA_BEAD"]].set(4.0)
    eps = eps.at[PT["RNA_BEAD"], PT["TIA1_RRM"]].set(4.0)

    # --- Effective bead diameters (nm) ---
    # Sigma matrix: sigma[type_i, type_j] = (sigma_i + sigma_j) / 2
    # Individual bead radii
    radii = jnp.zeros(N_PARTICLE_TYPES, dtype=jnp.float32)
    radii = radii.at[PT["NTF2"]].set(2.5)
    radii = radii.at[PT["ACIDIC_IDR"]].set(1.5)
    radii = radii.at[PT["PXXP"]].set(1.0)
    radii = radii.at[PT["RRM"]].set(2.0)
    radii = radii.at[PT["RG_IDR"]].set(1.5)
    radii = radii.at[PT["UBAP2L_UBA"]].set(2.0)
    radii = radii.at[PT["UBAP2L_RGG"]].set(1.5)
    radii = radii.at[PT["UBAP2L_IDR"]].set(1.5)
    radii = radii.at[PT["CAPRIN1_NIM"]].set(1.5)
    radii = radii.at[PT["CAPRIN1_RGG"]].set(1.5)
    radii = radii.at[PT["USP10_NIM"]].set(1.5)
    radii = radii.at[PT["USP10_BODY"]].set(2.0)
    radii = radii.at[PT["FXR1_DIM"]].set(2.0)
    radii = radii.at[PT["FXR1_KH"]].set(2.0)
    radii = radii.at[PT["FXR1_RGG"]].set(1.5)
    radii = radii.at[PT["TIA1_RRM"]].set(2.0)
    radii = radii.at[PT["TIA1_QIDR"]].set(1.5)
    radii = radii.at[PT["RNA_BEAD"]].set(1.5)

    # Combined sigma: arithmetic mean
    sigma = (radii[:, None] + radii[None, :]) / 2.0

    # --- Bond parameters ---
    bond_k = jnp.array([
        100.0,   # DOMAIN_LINKER_STIFF
        20.0,    # IDR_LINKER_FLEX
        200.0,   # DIMER_BOND (very stiff)
        50.0,    # RNA_BACKBONE
    ], dtype=jnp.float32)

    bond_r0 = jnp.array([
        3.0,     # DOMAIN_LINKER_STIFF (nm)
        2.5,     # IDR_LINKER_FLEX (nm)
        3.5,     # DIMER_BOND (nm) -- NTF2 dimer center-to-center
        2.0,     # RNA_BACKBONE (nm)
    ], dtype=jnp.float32)

    # --- Angle parameters ---
    angle_k = jnp.array([
        5.0,    # DOMAIN_ANGLE (semi-rigid)
        1.0,    # IDR_ANGLE (very flexible)
        3.0,    # RNA_ANGLE
    ], dtype=jnp.float32)

    angle_theta0 = jnp.array([
        jnp.pi,     # DOMAIN_ANGLE (~180 deg, extended)
        jnp.pi,     # IDR_ANGLE (~180 deg, but very flexible)
        jnp.pi,     # RNA_ANGLE
    ], dtype=jnp.float32)

    # --- Binding site parameters ---
    # Binding energy (kT): negative = favorable
    bind_energy = jnp.zeros((N_BINDING_SITE_TYPES, N_BINDING_SITE_TYPES), dtype=jnp.float32)

    # NTF2_POCKET <-> partner NIMs (competitive)
    bind_energy = bind_energy.at[BST["NTF2_POCKET"], BST["USP10_NIM_SITE"]].set(-8.0)
    bind_energy = bind_energy.at[BST["USP10_NIM_SITE"], BST["NTF2_POCKET"]].set(-8.0)
    bind_energy = bind_energy.at[BST["NTF2_POCKET"], BST["CAPRIN1_NIM_SITE"]].set(-6.0)
    bind_energy = bind_energy.at[BST["CAPRIN1_NIM_SITE"], BST["NTF2_POCKET"]].set(-6.0)
    bind_energy = bind_energy.at[BST["NTF2_POCKET"], BST["UBAP2L_NTF2_SITE"]].set(-6.0)
    bind_energy = bind_energy.at[BST["UBAP2L_NTF2_SITE"], BST["NTF2_POCKET"]].set(-6.0)

    # RRM <-> RNA
    bind_energy = bind_energy.at[BST["RRM_RNA"], BST["RNA_BINDING_SITE"]].set(-5.0)
    bind_energy = bind_energy.at[BST["RNA_BINDING_SITE"], BST["RRM_RNA"]].set(-5.0)

    # RG <-> RNA
    bind_energy = bind_energy.at[BST["RG_RNA"], BST["RNA_BINDING_SITE"]].set(-3.5)
    bind_energy = bind_energy.at[BST["RNA_BINDING_SITE"], BST["RG_RNA"]].set(-3.5)

    # RG <-> RG (homotypic)
    bind_energy = bind_energy.at[BST["RG_RG"], BST["RG_RG"]].set(-1.5)

    # UBAP2L self-association
    bind_energy = bind_energy.at[BST["UBAP2L_SELF"], BST["UBAP2L_SELF"]].set(-1.5)

    # FXR1 <-> UBAP2L
    bind_energy = bind_energy.at[BST["FXR1_UBAP2L"], BST["UBAP2L_NTF2_SITE"]].set(-5.0)
    bind_energy = bind_energy.at[BST["UBAP2L_NTF2_SITE"], BST["FXR1_UBAP2L"]].set(-5.0)

    # Binding cutoff distances (nm)
    bind_cutoff = jnp.where(bind_energy != 0.0, 5.0, 0.0).astype(jnp.float32)

    # Compatibility matrix (which site types can bind)
    bind_compat = (bind_energy != 0.0).astype(jnp.float32)

    # --- Conformational switching ---
    k_compact = jnp.float32(10.0)      # kT/nm^2
    r0_compact = jnp.float32(2.0)      # nm (compact distance)
    eps_rna_expand = jnp.float32(3.0)  # kT per RNA contact

    return InteractionParams(
        epsilon_attract=eps,
        sigma=sigma,
        kappa=jnp.float32(0.5),            # 1/nm, Yukawa screening
        debye_length=jnp.float32(1.0),     # nm, ~150 mM salt
        dielectric=jnp.float32(80.0),      # water
        bond_k=bond_k,
        bond_r0=bond_r0,
        angle_k=angle_k,
        angle_theta0=angle_theta0,
        binding_energy=bind_energy,
        binding_cutoff=bind_cutoff,
        binding_compatibility=bind_compat,
        k_compact=k_compact,
        r0_compact=r0_compact,
        eps_rna_expand=eps_rna_expand,
    )


def default_radii() -> jnp.ndarray:
    """Return per-type bead radii (nm) as an array indexed by particle type."""
    radii = jnp.zeros(N_PARTICLE_TYPES, dtype=jnp.float32)
    radii = radii.at[PT["NTF2"]].set(2.5)
    radii = radii.at[PT["ACIDIC_IDR"]].set(1.5)
    radii = radii.at[PT["PXXP"]].set(1.0)
    radii = radii.at[PT["RRM"]].set(2.0)
    radii = radii.at[PT["RG_IDR"]].set(1.5)
    radii = radii.at[PT["UBAP2L_UBA"]].set(2.0)
    radii = radii.at[PT["UBAP2L_RGG"]].set(1.5)
    radii = radii.at[PT["UBAP2L_IDR"]].set(1.5)
    radii = radii.at[PT["CAPRIN1_NIM"]].set(1.5)
    radii = radii.at[PT["CAPRIN1_RGG"]].set(1.5)
    radii = radii.at[PT["USP10_NIM"]].set(1.5)
    radii = radii.at[PT["USP10_BODY"]].set(2.0)
    radii = radii.at[PT["FXR1_DIM"]].set(2.0)
    radii = radii.at[PT["FXR1_KH"]].set(2.0)
    radii = radii.at[PT["FXR1_RGG"]].set(1.5)
    radii = radii.at[PT["TIA1_RRM"]].set(2.0)
    radii = radii.at[PT["TIA1_QIDR"]].set(1.5)
    radii = radii.at[PT["RNA_BEAD"]].set(1.5)
    return radii


def default_charges() -> jnp.ndarray:
    """Return per-type bead charges (elementary charges) as an array."""
    charges = jnp.zeros(N_PARTICLE_TYPES, dtype=jnp.float32)
    charges = charges.at[PT["NTF2"]].set(0.0)
    charges = charges.at[PT["ACIDIC_IDR"]].set(-8.0)   # Glutamate-rich
    charges = charges.at[PT["PXXP"]].set(0.0)
    charges = charges.at[PT["RRM"]].set(0.0)
    charges = charges.at[PT["RG_IDR"]].set(+5.0)       # Arginine-rich
    charges = charges.at[PT["RNA_BEAD"]].set(-4.0)      # Phosphate backbone
    charges = charges.at[PT["UBAP2L_RGG"]].set(+3.0)
    charges = charges.at[PT["CAPRIN1_RGG"]].set(+2.0)
    charges = charges.at[PT["TIA1_QIDR"]].set(0.0)
    return charges
