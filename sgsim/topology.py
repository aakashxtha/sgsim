"""Topology builders: bonds, angles, and binding site registries.

Provides helper functions to construct Topology NamedTuples from lists
of molecules. Each molecule builder returns local topology (bonds, angles,
binding sites) which are then merged into a global Topology.
"""

import jax.numpy as jnp
from .types import (
    Topology, BOND_TYPES, ANGLE_TYPES, BINDING_SITE_TYPES,
    PARTICLE_TYPES as PT,
)


class MoleculeTopology:
    """Intermediate container for a single molecule's topology.

    All indices are local (0-based within the molecule) and get
    offset when merged into the global Topology.
    """

    def __init__(self, n_particles, particle_types, particle_charges, particle_radii):
        self.n_particles = n_particles
        self.particle_types = list(particle_types)
        self.particle_charges = list(particle_charges)
        self.particle_radii = list(particle_radii)
        self.bonds = []        # list of (i, j, bond_type_str)
        self.angles = []       # list of (i, j, k, angle_type_str)
        self.binding_sites = []  # list of (particle_idx, site_type_str)

    def add_bond(self, i, j, bond_type):
        """Add a bond between local particle indices i and j."""
        self.bonds.append((i, j, bond_type))

    def add_angle(self, i, j, k, angle_type):
        """Add an angle i-j-k where j is the center particle."""
        self.angles.append((i, j, k, angle_type))

    def add_binding_site(self, particle_idx, site_type):
        """Register a binding site on the given particle."""
        self.binding_sites.append((particle_idx, site_type))


def merge_molecules(molecules, molecule_ids=None):
    """Merge multiple MoleculeTopology objects into arrays for Topology.

    Args:
        molecules: list of MoleculeTopology objects
        molecule_ids: optional list of molecule IDs (one per molecule).
                      If None, auto-assigned 0, 1, 2, ...

    Returns:
        dict with keys: positions_types, charges, radii, molecule_ids,
        bond_pairs, bond_types, angle_triples, angle_types,
        binding_site_particle, binding_site_type, binding_site_molecule,
        n_particles, n_bonds, n_angles, n_sites
    """
    all_particle_types = []
    all_charges = []
    all_radii = []
    all_mol_ids = []
    all_bond_pairs = []
    all_bond_types = []
    all_angle_triples = []
    all_angle_types = []
    all_site_particle = []
    all_site_type = []
    all_site_molecule = []

    offset = 0
    for mol_idx, mol in enumerate(molecules):
        mol_id = mol_idx if molecule_ids is None else molecule_ids[mol_idx]

        # Particles
        all_particle_types.extend(mol.particle_types)
        all_charges.extend(mol.particle_charges)
        all_radii.extend(mol.particle_radii)
        all_mol_ids.extend([mol_id] * mol.n_particles)

        # Bonds (offset indices)
        for i, j, btype in mol.bonds:
            all_bond_pairs.append((i + offset, j + offset))
            all_bond_types.append(BOND_TYPES[btype])

        # Angles (offset indices)
        for i, j, k, atype in mol.angles:
            all_angle_triples.append((i + offset, j + offset, k + offset))
            all_angle_types.append(ANGLE_TYPES[atype])

        # Binding sites
        for pidx, stype in mol.binding_sites:
            all_site_particle.append(pidx + offset)
            all_site_type.append(BINDING_SITE_TYPES[stype])
            all_site_molecule.append(mol_id)

        offset += mol.n_particles

    n_particles = offset
    n_bonds = len(all_bond_pairs)
    n_angles = len(all_angle_triples)
    n_sites = len(all_site_particle)

    # Convert to JAX arrays
    bond_pairs = jnp.array(all_bond_pairs, dtype=jnp.int32) if n_bonds > 0 \
        else jnp.zeros((0, 2), dtype=jnp.int32)
    bond_types = jnp.array(all_bond_types, dtype=jnp.int32) if n_bonds > 0 \
        else jnp.zeros(0, dtype=jnp.int32)
    angle_triples = jnp.array(all_angle_triples, dtype=jnp.int32) if n_angles > 0 \
        else jnp.zeros((0, 3), dtype=jnp.int32)
    angle_types = jnp.array(all_angle_types, dtype=jnp.int32) if n_angles > 0 \
        else jnp.zeros(0, dtype=jnp.int32)
    site_particle = jnp.array(all_site_particle, dtype=jnp.int32) if n_sites > 0 \
        else jnp.zeros(0, dtype=jnp.int32)
    site_type = jnp.array(all_site_type, dtype=jnp.int32) if n_sites > 0 \
        else jnp.zeros(0, dtype=jnp.int32)
    site_molecule = jnp.array(all_site_molecule, dtype=jnp.int32) if n_sites > 0 \
        else jnp.zeros(0, dtype=jnp.int32)

    topology = Topology(
        bond_pairs=bond_pairs,
        bond_types=bond_types,
        angle_triples=angle_triples,
        angle_types=angle_types,
        binding_site_particle=site_particle,
        binding_site_type=site_type,
        binding_site_molecule=site_molecule,
        n_particles=n_particles,
        n_bonds=n_bonds,
        n_angles=n_angles,
        n_sites=n_sites,
    )

    return {
        "particle_types": jnp.array(all_particle_types, dtype=jnp.int32),
        "particle_charges": jnp.array(all_charges, dtype=jnp.float32),
        "particle_radii": jnp.array(all_radii, dtype=jnp.float32),
        "molecule_ids": jnp.array(all_mol_ids, dtype=jnp.int32),
        "topology": topology,
    }
