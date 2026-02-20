"""System builders: molecule constructors and system assembly.

Each protein/RNA builder creates a MoleculeTopology with particles,
bonds, angles, and binding sites. The build_system function places
all molecules in a box and returns the complete simulation state.

Molecule models:
  G3BP1 monomer (6 beads): NTF2 - Acidic_IDR - PxxP - RRM - RG_IDR_1 - RG_IDR_2
  G3BP1 dimer (12 beads): two monomers + permanent NTF2-NTF2 bond
  RNA chain (variable): n_beads with tunable exposure
  USP10 (3 beads): NIM - linker - body
  CAPRIN1 (4 beads): NIM - linker - RGG - IDR
  UBAP2L (5 beads): UBA - linker - RGG - IDR_1 - IDR_2
  FXR1 dimer (8 beads): 2x (DIM - KH - RGG - linker)
  TIA1 (4 beads): RRM_1 - RRM_2 - RRM_3 - Q_IDR
"""

import jax
import jax.numpy as jnp

from .types import PARTICLE_TYPES as PT
from .topology import MoleculeTopology, merge_molecules
from .parameters import default_params, default_radii, default_charges
from .space import periodic_wrap


# ---------------------------------------------------------------------------
# Molecule builders
# ---------------------------------------------------------------------------


def build_g3bp1_monomer():
    """Build a single G3BP1 monomer (6 beads).

    Bead order: NTF2 - Acidic_IDR - PxxP - RRM - RG_IDR_1 - RG_IDR_2
    """
    radii_table = default_radii()
    charges_table = default_charges()

    types = [PT["NTF2"], PT["ACIDIC_IDR"], PT["PXXP"], PT["RRM"], PT["RG_IDR"], PT["RG_IDR"]]
    charges = [float(charges_table[t]) for t in types]
    radii = [float(radii_table[t]) for t in types]

    mol = MoleculeTopology(6, types, charges, radii)

    # Sequential bonds
    mol.add_bond(0, 1, "DOMAIN_LINKER_STIFF")  # NTF2 - Acidic_IDR
    mol.add_bond(1, 2, "IDR_LINKER_FLEX")       # Acidic_IDR - PxxP
    mol.add_bond(2, 3, "DOMAIN_LINKER_STIFF")  # PxxP - RRM
    mol.add_bond(3, 4, "DOMAIN_LINKER_STIFF")  # RRM - RG_IDR_1
    mol.add_bond(4, 5, "IDR_LINKER_FLEX")       # RG_IDR_1 - RG_IDR_2

    # Angles along the chain
    mol.add_angle(0, 1, 2, "DOMAIN_ANGLE")
    mol.add_angle(1, 2, 3, "IDR_ANGLE")
    mol.add_angle(2, 3, 4, "DOMAIN_ANGLE")
    mol.add_angle(3, 4, 5, "IDR_ANGLE")

    # Binding sites
    mol.add_binding_site(0, "NTF2_POCKET")   # Competitive PPI pocket
    mol.add_binding_site(0, "NTF2_DIMER")    # Dimerization interface
    mol.add_binding_site(3, "RRM_RNA")        # RRM-RNA binding
    mol.add_binding_site(4, "RG_RNA")         # RG-RNA binding (bead 1)
    mol.add_binding_site(5, "RG_RNA")         # RG-RNA binding (bead 2)
    mol.add_binding_site(4, "RG_RG")          # RG-RG homotypic
    mol.add_binding_site(5, "RG_RG")          # RG-RG homotypic

    return mol


def build_g3bp1_dimer():
    """Build a G3BP1 dimer (12 beads): two monomers + permanent NTF2-NTF2 bond."""
    radii_table = default_radii()
    charges_table = default_charges()

    # Two copies of monomer
    mono_types = [PT["NTF2"], PT["ACIDIC_IDR"], PT["PXXP"], PT["RRM"], PT["RG_IDR"], PT["RG_IDR"]]
    types = mono_types + mono_types
    charges = [float(charges_table[t]) for t in types]
    radii = [float(radii_table[t]) for t in types]

    mol = MoleculeTopology(12, types, charges, radii)

    # Monomer 1 bonds (beads 0-5)
    mol.add_bond(0, 1, "DOMAIN_LINKER_STIFF")
    mol.add_bond(1, 2, "IDR_LINKER_FLEX")
    mol.add_bond(2, 3, "DOMAIN_LINKER_STIFF")
    mol.add_bond(3, 4, "DOMAIN_LINKER_STIFF")
    mol.add_bond(4, 5, "IDR_LINKER_FLEX")

    # Monomer 2 bonds (beads 6-11)
    mol.add_bond(6, 7, "DOMAIN_LINKER_STIFF")
    mol.add_bond(7, 8, "IDR_LINKER_FLEX")
    mol.add_bond(8, 9, "DOMAIN_LINKER_STIFF")
    mol.add_bond(9, 10, "DOMAIN_LINKER_STIFF")
    mol.add_bond(10, 11, "IDR_LINKER_FLEX")

    # Dimer bond: NTF2_1 (bead 0) — NTF2_2 (bead 6)
    mol.add_bond(0, 6, "DIMER_BOND")

    # Angles for each monomer
    for offset in [0, 6]:
        mol.add_angle(offset + 0, offset + 1, offset + 2, "DOMAIN_ANGLE")
        mol.add_angle(offset + 1, offset + 2, offset + 3, "IDR_ANGLE")
        mol.add_angle(offset + 2, offset + 3, offset + 4, "DOMAIN_ANGLE")
        mol.add_angle(offset + 3, offset + 4, offset + 5, "IDR_ANGLE")

    # Binding sites for each monomer
    for offset in [0, 6]:
        mol.add_binding_site(offset + 0, "NTF2_POCKET")
        mol.add_binding_site(offset + 3, "RRM_RNA")
        mol.add_binding_site(offset + 4, "RG_RNA")
        mol.add_binding_site(offset + 5, "RG_RNA")
        mol.add_binding_site(offset + 4, "RG_RG")
        mol.add_binding_site(offset + 5, "RG_RG")

    return mol


def build_rna_chain(n_beads=10, exposure=1.0):
    """Build an RNA chain.

    Args:
        n_beads: number of RNA beads (~20-50 nt per bead)
        exposure: fraction of binding sites exposed (0=folded, 1=unfolded)

    Returns:
        MoleculeTopology
    """
    radii_table = default_radii()
    charges_table = default_charges()

    t = PT["RNA_BEAD"]
    types = [t] * n_beads
    charges = [float(charges_table[t])] * n_beads
    radii = [float(radii_table[t])] * n_beads

    mol = MoleculeTopology(n_beads, types, charges, radii)

    # Backbone bonds
    for i in range(n_beads - 1):
        mol.add_bond(i, i + 1, "RNA_BACKBONE")

    # Backbone angles
    for i in range(n_beads - 2):
        mol.add_angle(i, i + 1, i + 2, "RNA_ANGLE")

    # Binding sites: each bead has an RNA_BINDING_SITE
    # exposure controls how many are active (for folded RNA, fewer sites)
    n_exposed = max(1, int(n_beads * exposure))
    for i in range(n_exposed):
        mol.add_binding_site(i, "RNA_BINDING_SITE")

    return mol


def build_usp10():
    """Build USP10 (3 beads): NIM - linker_body - body.

    USP10 is a cap protein (v=1): binds NTF2 pocket but has no RNA-binding domain.
    """
    radii_table = default_radii()
    charges_table = default_charges()

    types = [PT["USP10_NIM"], PT["USP10_BODY"], PT["USP10_BODY"]]
    charges = [float(charges_table[t]) for t in types]
    radii = [float(radii_table[t]) for t in types]

    mol = MoleculeTopology(3, types, charges, radii)

    mol.add_bond(0, 1, "DOMAIN_LINKER_STIFF")
    mol.add_bond(1, 2, "IDR_LINKER_FLEX")
    mol.add_angle(0, 1, 2, "DOMAIN_ANGLE")

    mol.add_binding_site(0, "USP10_NIM_SITE")

    return mol


def build_caprin1():
    """Build CAPRIN1 (4 beads): NIM - linker - RGG - IDR.

    CAPRIN1 is a bridge: binds G3BP NTF2 via NIM and binds RNA via RGG.
    """
    radii_table = default_radii()
    charges_table = default_charges()

    types = [PT["CAPRIN1_NIM"], PT["PXXP"], PT["CAPRIN1_RGG"], PT["RG_IDR"]]
    charges = [float(charges_table[t]) for t in types]
    radii = [float(radii_table[t]) for t in types]

    mol = MoleculeTopology(4, types, charges, radii)

    mol.add_bond(0, 1, "DOMAIN_LINKER_STIFF")
    mol.add_bond(1, 2, "IDR_LINKER_FLEX")
    mol.add_bond(2, 3, "IDR_LINKER_FLEX")

    mol.add_angle(0, 1, 2, "DOMAIN_ANGLE")
    mol.add_angle(1, 2, 3, "IDR_ANGLE")

    mol.add_binding_site(0, "CAPRIN1_NIM_SITE")
    mol.add_binding_site(2, "RG_RNA")
    mol.add_binding_site(3, "RG_RNA")

    return mol


def build_ubap2l():
    """Build UBAP2L (5 beads): UBA - linker - RGG - IDR_1 - IDR_2.

    UBAP2L is a critical node: binds G3BP NTF2, has RGG for RNA,
    and self-associates via its IDR.
    """
    radii_table = default_radii()
    charges_table = default_charges()

    types = [PT["UBAP2L_UBA"], PT["PXXP"], PT["UBAP2L_RGG"],
             PT["UBAP2L_IDR"], PT["UBAP2L_IDR"]]
    charges = [float(charges_table[t]) for t in types]
    radii = [float(radii_table[t]) for t in types]

    mol = MoleculeTopology(5, types, charges, radii)

    mol.add_bond(0, 1, "DOMAIN_LINKER_STIFF")
    mol.add_bond(1, 2, "IDR_LINKER_FLEX")
    mol.add_bond(2, 3, "IDR_LINKER_FLEX")
    mol.add_bond(3, 4, "IDR_LINKER_FLEX")

    mol.add_angle(0, 1, 2, "DOMAIN_ANGLE")
    mol.add_angle(1, 2, 3, "IDR_ANGLE")
    mol.add_angle(2, 3, 4, "IDR_ANGLE")

    mol.add_binding_site(0, "UBAP2L_NTF2_SITE")
    mol.add_binding_site(2, "RG_RNA")
    mol.add_binding_site(3, "UBAP2L_SELF")
    mol.add_binding_site(4, "UBAP2L_SELF")

    return mol


def build_fxr1_dimer():
    """Build FXR1 dimer (8 beads): 2x (DIM - KH - RGG - linker).

    FXR1 is a dimeric RBP that interacts with UBAP2L.
    """
    radii_table = default_radii()
    charges_table = default_charges()

    mono_types = [PT["FXR1_DIM"], PT["FXR1_KH"], PT["FXR1_RGG"], PT["PXXP"]]
    types = mono_types + mono_types
    charges = [float(charges_table[t]) for t in types]
    radii = [float(radii_table[t]) for t in types]

    mol = MoleculeTopology(8, types, charges, radii)

    # Monomer 1
    mol.add_bond(0, 1, "DOMAIN_LINKER_STIFF")
    mol.add_bond(1, 2, "DOMAIN_LINKER_STIFF")
    mol.add_bond(2, 3, "IDR_LINKER_FLEX")

    # Monomer 2
    mol.add_bond(4, 5, "DOMAIN_LINKER_STIFF")
    mol.add_bond(5, 6, "DOMAIN_LINKER_STIFF")
    mol.add_bond(6, 7, "IDR_LINKER_FLEX")

    # Dimer bond
    mol.add_bond(0, 4, "DIMER_BOND")

    # Angles
    for offset in [0, 4]:
        mol.add_angle(offset, offset + 1, offset + 2, "DOMAIN_ANGLE")
        mol.add_angle(offset + 1, offset + 2, offset + 3, "IDR_ANGLE")

    # Binding sites
    for offset in [0, 4]:
        mol.add_binding_site(offset + 1, "RG_RNA")     # KH-RNA
        mol.add_binding_site(offset + 2, "RG_RNA")     # RGG-RNA
        mol.add_binding_site(offset, "FXR1_UBAP2L")    # FXR1-UBAP2L

    return mol


def build_tia1():
    """Build TIA1 (4 beads): RRM_1 - RRM_2 - RRM_3 - Q_IDR.

    TIA1 has 3 RRM domains for strong RNA binding + a Q-rich IDR.
    """
    radii_table = default_radii()
    charges_table = default_charges()

    types = [PT["TIA1_RRM"], PT["TIA1_RRM"], PT["TIA1_RRM"], PT["TIA1_QIDR"]]
    charges = [float(charges_table[t]) for t in types]
    radii = [float(radii_table[t]) for t in types]

    mol = MoleculeTopology(4, types, charges, radii)

    mol.add_bond(0, 1, "DOMAIN_LINKER_STIFF")
    mol.add_bond(1, 2, "DOMAIN_LINKER_STIFF")
    mol.add_bond(2, 3, "DOMAIN_LINKER_STIFF")

    mol.add_angle(0, 1, 2, "DOMAIN_ANGLE")
    mol.add_angle(1, 2, 3, "DOMAIN_ANGLE")

    mol.add_binding_site(0, "RRM_RNA")
    mol.add_binding_site(1, "RRM_RNA")
    mol.add_binding_site(2, "RRM_RNA")

    return mol


# ---------------------------------------------------------------------------
# System assembly
# ---------------------------------------------------------------------------

# Registry of molecule builders
MOLECULE_BUILDERS = {
    "g3bp1_monomer": build_g3bp1_monomer,
    "g3bp1_dimer": build_g3bp1_dimer,
    "rna": build_rna_chain,
    "usp10": build_usp10,
    "caprin1": build_caprin1,
    "ubap2l": build_ubap2l,
    "fxr1_dimer": build_fxr1_dimer,
    "tia1": build_tia1,
}


def build_system(
    composition: dict,
    box_size: float | jnp.ndarray,
    rng_key: jnp.ndarray,
    rna_beads: int = 10,
    rna_exposure: float = 1.0,
) -> dict:
    """Build a complete system from a composition dictionary.

    Args:
        composition: dict of {"molecule_name": count}, e.g.
                     {"g3bp1_dimer": 10, "rna": 20, "usp10": 5}
        box_size: scalar or (3,) array for simulation box
        rng_key: JAX PRNG key for random placement
        rna_beads: number of beads per RNA chain
        rna_exposure: fraction of exposed RNA binding sites

    Returns:
        dict with keys: positions, particle_types, particle_charges,
        particle_radii, molecule_ids, topology, box_size
    """
    if isinstance(box_size, (int, float)):
        box_size = jnp.array([box_size, box_size, box_size], dtype=jnp.float32)
    else:
        box_size = jnp.asarray(box_size, dtype=jnp.float32)

    molecules = []
    for mol_name, count in composition.items():
        for _ in range(count):
            if mol_name == "rna":
                mol = build_rna_chain(n_beads=rna_beads, exposure=rna_exposure)
            else:
                builder = MOLECULE_BUILDERS[mol_name]
                mol = builder()
            molecules.append(mol)

    # Merge topologies
    merged = merge_molecules(molecules)

    n = merged["topology"].n_particles
    params = default_params()

    # Place molecules randomly in the box
    # For each molecule, generate a random center and place beads in a line
    key = rng_key
    positions = jnp.zeros((n, 3), dtype=jnp.float32)
    offset = 0

    for mol in molecules:
        key, subkey = jax.random.split(key)
        # Random center for molecule
        center = jax.random.uniform(subkey, (3,)) * box_size

        # Place beads in a line along x, spacing = 2 nm
        for i in range(mol.n_particles):
            pos_i = center + jnp.array([i * 2.0, 0.0, 0.0])
            positions = positions.at[offset + i].set(pos_i)
        offset += mol.n_particles

    # Wrap into box
    positions = periodic_wrap(positions, box_size)

    return {
        "positions": positions,
        "particle_types": merged["particle_types"],
        "particle_charges": merged["particle_charges"],
        "particle_radii": merged["particle_radii"],
        "molecule_ids": merged["molecule_ids"],
        "topology": merged["topology"],
        "box_size": box_size,
    }
