"""Core data structures for sgsim.

All state is stored in immutable JAX arrays inside NamedTuples,
ensuring JIT-compatibility throughout the simulation pipeline.
"""

from typing import NamedTuple, Optional
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Particle type enumeration
# ---------------------------------------------------------------------------

PARTICLE_TYPES = {
    "NTF2": 0,
    "ACIDIC_IDR": 1,
    "PXXP": 2,
    "RRM": 3,
    "RG_IDR": 4,
    "UBAP2L_UBA": 5,
    "UBAP2L_RGG": 6,
    "UBAP2L_IDR": 7,
    "CAPRIN1_NIM": 8,
    "CAPRIN1_RGG": 9,
    "USP10_NIM": 10,
    "USP10_BODY": 11,
    "FXR1_DIM": 12,
    "FXR1_KH": 13,
    "FXR1_RGG": 14,
    "TIA1_RRM": 15,
    "TIA1_QIDR": 16,
    "RNA_BEAD": 17,
}

N_PARTICLE_TYPES = len(PARTICLE_TYPES)

# Reverse lookup
PARTICLE_TYPE_NAMES = {v: k for k, v in PARTICLE_TYPES.items()}

# ---------------------------------------------------------------------------
# Binding site type enumeration
# ---------------------------------------------------------------------------

BINDING_SITE_TYPES = {
    "NTF2_POCKET": 0,        # Accepts CAPRIN1_NIM, USP10_NIM, or UBAP2L
    "NTF2_DIMER": 1,         # NTF2-NTF2 dimerization (permanent)
    "RRM_RNA": 2,             # RRM binds RNA (specific)
    "RG_RNA": 3,              # RG-rich IDR binds RNA (promiscuous)
    "RG_RG": 4,               # RG-RG homotypic protein-protein
    "CAPRIN1_NIM_SITE": 5,    # CAPRIN1's NTF2-binding motif
    "USP10_NIM_SITE": 6,      # USP10's NTF2-binding motif
    "UBAP2L_NTF2_SITE": 7,    # UBAP2L's G3BP-binding region
    "RNA_BINDING_SITE": 8,    # Binding site on RNA beads
    "UBAP2L_SELF": 9,         # UBAP2L self-association IDR
    "FXR1_UBAP2L": 10,        # FXR1-UBAP2L interaction site
}

N_BINDING_SITE_TYPES = len(BINDING_SITE_TYPES)

# ---------------------------------------------------------------------------
# Bond type enumeration
# ---------------------------------------------------------------------------

BOND_TYPES = {
    "DOMAIN_LINKER_STIFF": 0,   # Between folded domains (stiff)
    "IDR_LINKER_FLEX": 1,       # Within IDR chains (flexible)
    "DIMER_BOND": 2,            # NTF2-NTF2 permanent dimerization
    "RNA_BACKBONE": 3,          # Between consecutive RNA beads
}

N_BOND_TYPES = len(BOND_TYPES)

# ---------------------------------------------------------------------------
# Angle type enumeration
# ---------------------------------------------------------------------------

ANGLE_TYPES = {
    "DOMAIN_ANGLE": 0,     # Between folded domains (semi-rigid)
    "IDR_ANGLE": 1,        # Within IDR (very flexible)
    "RNA_ANGLE": 2,        # RNA backbone (semi-flexible)
}

N_ANGLE_TYPES = len(ANGLE_TYPES)

# ---------------------------------------------------------------------------
# State containers (all JAX-compatible NamedTuples)
# ---------------------------------------------------------------------------


class SystemState(NamedTuple):
    """Per-particle physical state."""
    positions: jnp.ndarray          # (N, 3) float32
    particle_types: jnp.ndarray     # (N,) int32
    particle_charges: jnp.ndarray   # (N,) float32 -- net charge per bead
    particle_radii: jnp.ndarray     # (N,) float32 -- effective radius (nm)
    molecule_ids: jnp.ndarray       # (N,) int32 -- which molecule
    box_size: jnp.ndarray           # (3,) float32


class Topology(NamedTuple):
    """Static connectivity (bonds, angles, binding sites)."""
    bond_pairs: jnp.ndarray             # (N_bonds, 2) int32 -- particle indices
    bond_types: jnp.ndarray             # (N_bonds,) int32
    angle_triples: jnp.ndarray          # (N_angles, 3) int32 -- i, j, k
    angle_types: jnp.ndarray            # (N_angles,) int32
    binding_site_particle: jnp.ndarray  # (N_sites,) int32 -- owning particle
    binding_site_type: jnp.ndarray      # (N_sites,) int32
    binding_site_molecule: jnp.ndarray  # (N_sites,) int32 -- owning molecule
    n_particles: int
    n_bonds: int
    n_angles: int
    n_sites: int


class BindingState(NamedTuple):
    """Mutable binding state, updated by MC moves."""
    # Sparse representation of bound pairs for efficiency
    bound_pairs: jnp.ndarray     # (max_bonds, 2) int32 -- site index pairs
    bound_mask: jnp.ndarray      # (max_bonds,) bool -- which entries are active
    n_bound: jnp.ndarray         # scalar int32 -- current number of bonds
    # Per-site occupancy: 1 if bound, 0 if free
    site_occupied: jnp.ndarray   # (N_sites,) int32
    # Partner site index (-1 if unbound)
    site_partner: jnp.ndarray    # (N_sites,) int32


class ConformationalState(NamedTuple):
    """Per-molecule conformational variables for switchable proteins."""
    # Openness parameter: 0 = compact/autoinhibited, 1 = fully expanded
    openness: jnp.ndarray           # (N_switchable,) float32
    # Mapping: which molecules are switchable and their bead indices
    molecule_indices: jnp.ndarray   # (N_switchable,) int32
    # Indices of acidic IDR beads for each switchable molecule
    acidic_bead_indices: jnp.ndarray  # (N_switchable,) int32
    # Indices of RG IDR beads for each switchable molecule
    rg_bead_indices: jnp.ndarray      # (N_switchable,) int32


class InteractionParams(NamedTuple):
    """All interaction parameters, stored as arrays for fast lookup."""
    # Non-bonded: (N_types, N_types) symmetric matrices
    epsilon_attract: jnp.ndarray   # Attraction strength (kT)
    sigma: jnp.ndarray             # Effective diameter (nm)
    # Yukawa screening
    kappa: jnp.ndarray             # scalar, inverse range (1/nm)
    # Electrostatics
    debye_length: jnp.ndarray     # scalar (nm)
    dielectric: jnp.ndarray       # scalar, relative permittivity
    # Bond parameters
    bond_k: jnp.ndarray           # (N_bond_types,) spring constant (kT/nm^2)
    bond_r0: jnp.ndarray          # (N_bond_types,) equilibrium length (nm)
    # Angle parameters
    angle_k: jnp.ndarray          # (N_angle_types,) bending stiffness (kT/rad^2)
    angle_theta0: jnp.ndarray     # (N_angle_types,) equilibrium angle (rad)
    # Binding site parameters
    binding_energy: jnp.ndarray   # (N_site_types, N_site_types) free energy (kT)
    binding_cutoff: jnp.ndarray   # (N_site_types, N_site_types) distance (nm)
    # Compatibility matrix: which site types can bind
    binding_compatibility: jnp.ndarray  # (N_site_types, N_site_types) bool
    # Conformational switching
    k_compact: jnp.ndarray        # scalar, compact restraint stiffness
    r0_compact: jnp.ndarray       # scalar, compact equilibrium distance
    eps_rna_expand: jnp.ndarray   # scalar, RNA-expansion coupling strength


class SimulationState(NamedTuple):
    """Complete simulation state passed through the integrator."""
    system: SystemState
    topology: Topology
    binding: BindingState
    conformation: ConformationalState
    params: InteractionParams
    rng_key: jnp.ndarray          # JAX PRNG key
    step: jnp.ndarray             # scalar int32


class SimulationConfig(NamedTuple):
    """Simulation hyperparameters (not part of the differentiable state)."""
    dt: float                      # timestep (reduced units)
    kT: float                      # thermal energy
    gamma: float                   # base friction coefficient
    n_steps: int                   # total steps
    save_every: int                # frames between saves
    binding_interval: int          # MD steps between binding MC attempts
    n_binding_attempts: int        # MC attempts per binding update
    neighbor_skin: float           # neighbor list skin distance (nm)
    neighbor_update_every: int     # steps between neighbor list rebuilds
    cutoff: float                  # non-bonded interaction cutoff (nm)
