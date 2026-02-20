"""Tests for topology builders and system setup.

Validates:
- Molecule builders produce correct bead counts and topology
- Topology merging with correct index offsets
- System assembly places all particles in the box
- Integration: built system can be simulated
"""

import jax
import jax.numpy as jnp
import pytest

from sgsim.setup import (
    build_g3bp1_monomer, build_g3bp1_dimer, build_rna_chain,
    build_usp10, build_caprin1, build_ubap2l, build_fxr1_dimer, build_tia1,
    build_system,
)
from sgsim.topology import merge_molecules
from sgsim.types import PARTICLE_TYPES as PT, BINDING_SITE_TYPES as BST
from sgsim.simulate import init_fn, make_step_fn


class TestMoleculeBuilders:
    """Tests for individual molecule builders."""

    def test_g3bp1_monomer_beads(self):
        """G3BP1 monomer should have 6 beads."""
        mol = build_g3bp1_monomer()
        assert mol.n_particles == 6

    def test_g3bp1_monomer_bonds(self):
        """G3BP1 monomer should have 5 sequential bonds."""
        mol = build_g3bp1_monomer()
        assert len(mol.bonds) == 5

    def test_g3bp1_monomer_binding_sites(self):
        """G3BP1 monomer should have NTF2_POCKET, NTF2_DIMER, RRM_RNA, RG sites."""
        mol = build_g3bp1_monomer()
        site_types = [s[1] for s in mol.binding_sites]
        assert "NTF2_POCKET" in site_types
        assert "NTF2_DIMER" in site_types
        assert "RRM_RNA" in site_types
        assert "RG_RNA" in site_types

    def test_g3bp1_dimer_beads(self):
        """G3BP1 dimer should have 12 beads."""
        mol = build_g3bp1_dimer()
        assert mol.n_particles == 12

    def test_g3bp1_dimer_has_dimer_bond(self):
        """G3BP1 dimer should have a DIMER_BOND between NTF2 beads."""
        mol = build_g3bp1_dimer()
        dimer_bonds = [(i, j) for i, j, t in mol.bonds if t == "DIMER_BOND"]
        assert len(dimer_bonds) == 1
        assert dimer_bonds[0] == (0, 6), "Dimer bond should connect NTF2 beads 0 and 6"

    def test_g3bp1_dimer_two_ntf2_pockets(self):
        """G3BP1 dimer should have 2 NTF2_POCKET sites (one per monomer)."""
        mol = build_g3bp1_dimer()
        pockets = [s for s in mol.binding_sites if s[1] == "NTF2_POCKET"]
        assert len(pockets) == 2

    def test_rna_chain_length(self):
        """RNA chain should have the requested number of beads."""
        for n in [5, 10, 20]:
            mol = build_rna_chain(n_beads=n)
            assert mol.n_particles == n
            assert len(mol.bonds) == n - 1

    def test_rna_exposure(self):
        """RNA exposure should control number of binding sites."""
        mol_full = build_rna_chain(n_beads=10, exposure=1.0)
        mol_half = build_rna_chain(n_beads=10, exposure=0.5)
        assert len(mol_full.binding_sites) == 10
        assert len(mol_half.binding_sites) == 5

    def test_usp10_beads(self):
        """USP10 should have 3 beads and 1 NIM binding site."""
        mol = build_usp10()
        assert mol.n_particles == 3
        site_types = [s[1] for s in mol.binding_sites]
        assert "USP10_NIM_SITE" in site_types
        # USP10 has no RNA binding → no RRM_RNA or RG_RNA sites
        assert "RRM_RNA" not in site_types
        assert "RG_RNA" not in site_types

    def test_caprin1_beads(self):
        """CAPRIN1 should have 4 beads."""
        mol = build_caprin1()
        assert mol.n_particles == 4

    def test_ubap2l_beads(self):
        """UBAP2L should have 5 beads with self-association sites."""
        mol = build_ubap2l()
        assert mol.n_particles == 5
        site_types = [s[1] for s in mol.binding_sites]
        assert "UBAP2L_SELF" in site_types

    def test_fxr1_dimer_beads(self):
        """FXR1 dimer should have 8 beads."""
        mol = build_fxr1_dimer()
        assert mol.n_particles == 8

    def test_tia1_beads(self):
        """TIA1 should have 4 beads with 3 RRM sites."""
        mol = build_tia1()
        assert mol.n_particles == 4
        rrm_sites = [s for s in mol.binding_sites if s[1] == "RRM_RNA"]
        assert len(rrm_sites) == 3


class TestTopologyMerge:
    """Tests for merging multiple molecules."""

    def test_merge_particle_count(self):
        """Merged topology should have correct total particle count."""
        mol1 = build_g3bp1_monomer()  # 6
        mol2 = build_rna_chain(n_beads=5)  # 5
        result = merge_molecules([mol1, mol2])
        assert result["topology"].n_particles == 11

    def test_merge_bond_offset(self):
        """Bond indices should be correctly offset after merging."""
        mol1 = build_g3bp1_monomer()  # 6 beads
        mol2 = build_usp10()  # 3 beads
        result = merge_molecules([mol1, mol2])

        bond_pairs = result["topology"].bond_pairs
        # mol2 bond 0-1 should become 6-7 after offset
        mol2_bonds = bond_pairs[len(mol1.bonds):]
        assert jnp.any(mol2_bonds[:, 0] >= 6), "Second molecule bonds should be offset"

    def test_merge_molecule_ids(self):
        """Each particle should get the correct molecule ID."""
        mol1 = build_g3bp1_monomer()  # 6
        mol2 = build_rna_chain(n_beads=3)  # 3
        result = merge_molecules([mol1, mol2])

        mol_ids = result["molecule_ids"]
        assert jnp.all(mol_ids[:6] == 0), "First molecule particles should have mol_id=0"
        assert jnp.all(mol_ids[6:] == 1), "Second molecule particles should have mol_id=1"


class TestBuildSystem:
    """Tests for the full system builder."""

    def test_build_simple_system(self):
        """Build a minimal system and check particle count."""
        key = jax.random.PRNGKey(0)
        system = build_system(
            {"g3bp1_dimer": 2, "rna": 3},
            box_size=50.0, rng_key=key, rna_beads=5,
        )

        expected_n = 2 * 12 + 3 * 5  # 24 + 15 = 39
        assert system["positions"].shape == (expected_n, 3)
        assert system["particle_types"].shape == (expected_n,)
        assert system["topology"].n_particles == expected_n

    def test_positions_in_box(self):
        """All positions should be within the simulation box."""
        key = jax.random.PRNGKey(1)
        system = build_system(
            {"g3bp1_dimer": 5, "rna": 10},
            box_size=80.0, rng_key=key, rna_beads=8,
        )
        pos = system["positions"]
        box = system["box_size"]
        assert jnp.all(pos >= 0.0), "Positions should be >= 0"
        assert jnp.all(pos < box), "Positions should be < box_size"

    def test_system_can_be_simulated(self):
        """A built system should be simulatable with the integrator."""
        key = jax.random.PRNGKey(2)
        k1, k2 = jax.random.split(key)

        system = build_system(
            {"g3bp1_dimer": 2, "rna": 3},
            box_size=50.0, rng_key=k1, rna_beads=5,
        )

        from sgsim.parameters import default_params
        params = default_params()

        state = init_fn(
            system["positions"],
            system["particle_types"],
            system["particle_charges"],
            system["particle_radii"],
            system["box_size"],
            system["topology"],
            params,
            k2,
            cutoff=10.0,
        )

        step_fn = make_step_fn(system["topology"], params, dt=0.001, kT=1.0, cutoff=10.0)

        # Run a few steps
        for _ in range(5):
            state = step_fn(state)

        assert jnp.all(jnp.isfinite(state.positions)), "Positions should be finite after simulation"
        assert state.step == 5

    def test_multicomponent_system(self):
        """Build a system with all protein types."""
        key = jax.random.PRNGKey(3)
        system = build_system(
            {
                "g3bp1_dimer": 2,
                "rna": 5,
                "usp10": 1,
                "caprin1": 1,
                "ubap2l": 1,
                "tia1": 1,
            },
            box_size=80.0, rng_key=key, rna_beads=8,
        )

        expected_n = 2*12 + 5*8 + 1*3 + 1*4 + 1*5 + 1*4  # 24+40+3+4+5+4 = 80
        assert system["topology"].n_particles == expected_n
