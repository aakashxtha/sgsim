"""Tests for conformational switching (compact ↔ expanded).

Validates:
- Initialization and state construction
- Conformational energy: compact restraint + RNA expansion coupling
- Phi dynamics: overdamped Langevin updates with clamping
- RNA counting per molecule
"""

import jax
import jax.numpy as jnp
import pytest

from sgsim.conformational import (
    init_conformational_state,
    conformational_energy,
    update_conformational_state,
    count_rna_bound_per_molecule,
)
from sgsim.binding import init_binding_state, _apply_bind
from sgsim.types import BINDING_SITE_TYPES as BST


def _make_single_switchable():
    """Create a minimal system with one switchable molecule.

    Particle 0 = acidic IDR, particle 1 = RG IDR, separated by 5 nm.
    """
    positions = jnp.array([
        [10.0, 25.0, 25.0],  # acidic IDR
        [15.0, 25.0, 25.0],  # RG IDR (5 nm apart)
    ])
    box_size = jnp.array([50.0, 50.0, 50.0])

    conf_state = init_conformational_state(
        n_switchable=1,
        molecule_indices=jnp.array([0], dtype=jnp.int32),
        acidic_bead_indices=jnp.array([0], dtype=jnp.int32),
        rg_bead_indices=jnp.array([1], dtype=jnp.int32),
        initial_openness=0.0,
    )

    return positions, box_size, conf_state


class TestConformationalInit:
    """Tests for conformational state initialization."""

    def test_init_compact(self):
        """Initial state should be compact (phi=0)."""
        _, _, state = _make_single_switchable()
        assert jnp.allclose(state.openness, 0.0)

    def test_init_custom_openness(self):
        """Should support custom initial openness."""
        state = init_conformational_state(
            n_switchable=3,
            molecule_indices=jnp.array([0, 1, 2], dtype=jnp.int32),
            acidic_bead_indices=jnp.array([0, 2, 4], dtype=jnp.int32),
            rg_bead_indices=jnp.array([1, 3, 5], dtype=jnp.int32),
            initial_openness=0.5,
        )
        assert jnp.allclose(state.openness, 0.5)


class TestConformationalEnergy:
    """Tests for the conformational energy function."""

    def test_zero_energy_at_compact(self):
        """Energy should be 0 at phi=0 (compact, no RNA)."""
        positions, box_size, state = _make_single_switchable()

        e = conformational_energy(
            positions, box_size, state,
            k_compact=10.0, r0_compact=2.0, eps_rna_expand=3.0,
            n_rna_bound=jnp.array([0]),
        )
        assert jnp.abs(e) < 0.01, f"Energy at phi=0 should be ~0, got {e}"

    def test_positive_energy_at_expanded(self):
        """Energy should be positive at phi=1 without RNA (bias penalty)."""
        positions, box_size, _ = _make_single_switchable()
        state = init_conformational_state(
            1, jnp.array([0], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            jnp.array([1], dtype=jnp.int32),
            initial_openness=1.0,
        )

        e = conformational_energy(
            positions, box_size, state,
            k_compact=10.0, r0_compact=2.0, eps_rna_expand=3.0,
            n_rna_bound=jnp.array([0]),
        )
        assert e > 0, f"Energy at phi=1 without RNA should be positive, got {e}"
        assert jnp.abs(e - 10.0) < 0.1, f"Expected k_compact=10, got {e}"

    def test_expanded_higher_energy_without_rna(self):
        """Without RNA, expanded state should have higher energy than compact."""
        positions, box_size, _ = _make_single_switchable()

        state_compact = init_conformational_state(
            1, jnp.array([0], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            jnp.array([1], dtype=jnp.int32),
            initial_openness=0.0,
        )
        state_expanded = init_conformational_state(
            1, jnp.array([0], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            jnp.array([1], dtype=jnp.int32),
            initial_openness=1.0,
        )

        e_compact = conformational_energy(
            positions, box_size, state_compact,
            k_compact=10.0, r0_compact=2.0, eps_rna_expand=3.0,
            n_rna_bound=jnp.array([0]),
        )
        e_expanded = conformational_energy(
            positions, box_size, state_expanded,
            k_compact=10.0, r0_compact=2.0, eps_rna_expand=3.0,
            n_rna_bound=jnp.array([0]),
        )

        assert e_expanded > e_compact, \
            f"Without RNA, expanded ({e_expanded}) should have higher E than compact ({e_compact})"

    def test_rna_coupling_favors_expansion(self):
        """RNA binding should lower energy for expanded state."""
        positions, box_size, _ = _make_single_switchable()

        state = init_conformational_state(
            1, jnp.array([0], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            jnp.array([1], dtype=jnp.int32),
            initial_openness=1.0,  # fully expanded
        )

        e_no_rna = conformational_energy(
            positions, box_size, state,
            k_compact=10.0, r0_compact=2.0, eps_rna_expand=3.0,
            n_rna_bound=jnp.array([0]),
        )
        e_with_rna = conformational_energy(
            positions, box_size, state,
            k_compact=10.0, r0_compact=2.0, eps_rna_expand=3.0,
            n_rna_bound=jnp.array([5]),  # 5 RNA > k/eps=3.3 → net expansion
        )

        assert e_with_rna < e_no_rna, \
            f"RNA should stabilize expanded state: {e_with_rna} vs {e_no_rna}"


class TestPhiDynamics:
    """Tests for phi update dynamics."""

    def test_phi_stays_clamped(self):
        """Phi should remain in [0, 1] after updates."""
        positions, box_size, state = _make_single_switchable()
        key = jax.random.PRNGKey(0)

        for _ in range(100):
            key, subkey = jax.random.split(key)
            state, _ = update_conformational_state(
                state, positions, box_size,
                k_compact=10.0, r0_compact=2.0, eps_rna_expand=3.0,
                n_rna_bound=jnp.array([0]),
                gamma_phi=1.0, kT=1.0, dt=0.1,
                rng_key=subkey,
            )

        assert jnp.all(state.openness >= 0.0), "Phi should be >= 0"
        assert jnp.all(state.openness <= 1.0), "Phi should be <= 1"

    def test_compact_without_rna(self):
        """Without RNA and beads near r0, bias drives phi toward compact.

        When beads are at r0, the compact spring term vanishes and the
        intrinsic bias (k_compact * phi) dominates, driving phi -> 0.
        """
        # Beads at r0 = 2 nm so compact spring is neutral
        positions = jnp.array([
            [10.0, 25.0, 25.0],
            [12.0, 25.0, 25.0],
        ])
        box_size = jnp.array([50.0, 50.0, 50.0])

        state = init_conformational_state(
            1, jnp.array([0], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            jnp.array([1], dtype=jnp.int32),
            initial_openness=0.8,
        )

        key = jax.random.PRNGKey(42)
        for _ in range(200):
            key, subkey = jax.random.split(key)
            state, _ = update_conformational_state(
                state, positions, box_size,
                k_compact=10.0, r0_compact=2.0, eps_rna_expand=3.0,
                n_rna_bound=jnp.array([0]),
                gamma_phi=1.0, kT=0.1, dt=0.01,
                rng_key=subkey,
            )

        assert state.openness[0] < 0.5, \
            f"Without RNA, phi should relax toward compact, got {state.openness[0]}"

    def test_expand_with_rna(self):
        """With strong RNA coupling, phi should move toward 1 (expanded)."""
        positions, box_size, _ = _make_single_switchable()

        # Start compact
        state = init_conformational_state(
            1, jnp.array([0], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            jnp.array([1], dtype=jnp.int32),
            initial_openness=0.2,
        )

        key = jax.random.PRNGKey(99)
        for _ in range(200):
            key, subkey = jax.random.split(key)
            state, _ = update_conformational_state(
                state, positions, box_size,
                k_compact=5.0, r0_compact=2.0, eps_rna_expand=10.0,
                n_rna_bound=jnp.array([5]),  # Strong RNA coupling
                gamma_phi=1.0, kT=0.1, dt=0.01,
                rng_key=subkey,
            )

        assert state.openness[0] > 0.5, \
            f"With strong RNA coupling, phi should expand, got {state.openness[0]}"

    def test_phi_update_jit(self):
        """Phi update should be JIT-compilable."""
        positions, box_size, state = _make_single_switchable()
        key = jax.random.PRNGKey(0)

        @jax.jit
        def do_update(s, k):
            return update_conformational_state(
                s, positions, box_size,
                k_compact=10.0, r0_compact=2.0, eps_rna_expand=3.0,
                n_rna_bound=jnp.array([0]),
                gamma_phi=1.0, kT=1.0, dt=0.01,
                rng_key=k,
            )

        new_state, new_key = do_update(state, key)
        assert new_state.openness.shape == (1,)


class TestRNACounting:
    """Tests for RNA bond counting per molecule."""

    def test_zero_with_no_bonds(self):
        """No bonds means zero RNA contacts."""
        binding_state = init_binding_state(n_sites=4, max_bonds=10)

        site_types = jnp.array([
            BST["NTF2_POCKET"], BST["RG_RNA"],
            BST["RNA_BINDING_SITE"], BST["RNA_BINDING_SITE"],
        ], dtype=jnp.int32)
        site_molecule = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        mol_indices = jnp.array([0], dtype=jnp.int32)

        counts = count_rna_bound_per_molecule(
            binding_state, site_types, site_molecule,
            mol_indices, BST["RNA_BINDING_SITE"],
        )
        assert counts[0] == 0

    def test_count_with_rna_bond(self):
        """Should count RNA bonds correctly."""
        binding_state = init_binding_state(n_sites=4, max_bonds=10)
        # Bind site 1 (RG_RNA on mol 0) to site 2 (RNA_BINDING_SITE on mol 1)
        binding_state = _apply_bind(binding_state, 1, 2)

        site_types = jnp.array([
            BST["NTF2_POCKET"], BST["RG_RNA"],
            BST["RNA_BINDING_SITE"], BST["RNA_BINDING_SITE"],
        ], dtype=jnp.int32)
        site_molecule = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        mol_indices = jnp.array([0], dtype=jnp.int32)

        counts = count_rna_bound_per_molecule(
            binding_state, site_types, site_molecule,
            mol_indices, BST["RNA_BINDING_SITE"],
        )
        assert counts[0] == 1, f"Should count 1 RNA bond, got {counts[0]}"
