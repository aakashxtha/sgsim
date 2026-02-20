"""Tests for energy functions.

Validates:
- Harmonic bond energy and forces
- Harmonic angle energy and forces
- WCA repulsion properties
- Yukawa attraction decay
- Force-energy consistency (analytical vs numerical gradient)
"""

import jax
import jax.numpy as jnp
import pytest

from sgsim.energy import (
    harmonic_bond_energy,
    harmonic_angle_energy,
    wca_energy_pair,
    yukawa_energy_pair,
    debye_huckel_energy_pair,
    nonbonded_energy_bruteforce,
    nonbonded_energy_neighborlist,
    build_bond_exclusion_mask,
)
from sgsim.partition import build_neighbor_list, neighbor_pairs, needs_rebuild
from sgsim.utils import safe_norm


# ============================================================================
# Bond energy tests
# ============================================================================


class TestHarmonicBond:
    """Tests for harmonic bond potential."""

    def test_zero_energy_at_equilibrium(self):
        """Bond energy should be zero when r = r0."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        bond_pairs = jnp.array([[0, 1]], dtype=jnp.int32)
        bond_types = jnp.array([0], dtype=jnp.int32)
        bond_k = jnp.array([100.0])
        bond_r0 = jnp.array([3.0])
        box_size = jnp.array([100.0, 100.0, 100.0])

        energy = harmonic_bond_energy(positions, bond_pairs, bond_types, bond_k, bond_r0, box_size)
        assert jnp.abs(energy) < 1e-4, f"Energy at equilibrium should be ~0, got {energy}"

    def test_positive_energy_stretched(self):
        """Bond energy should be positive when stretched beyond r0."""
        positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        bond_pairs = jnp.array([[0, 1]], dtype=jnp.int32)
        bond_types = jnp.array([0], dtype=jnp.int32)
        bond_k = jnp.array([100.0])
        bond_r0 = jnp.array([3.0])
        box_size = jnp.array([100.0, 100.0, 100.0])

        energy = harmonic_bond_energy(positions, bond_pairs, bond_types, bond_k, bond_r0, box_size)
        expected = 0.5 * 100.0 * (5.0 - 3.0) ** 2  # = 200.0
        assert jnp.abs(energy - expected) < 1e-3, f"Expected {expected}, got {energy}"

    def test_positive_energy_compressed(self):
        """Bond energy should be positive when compressed below r0."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        bond_pairs = jnp.array([[0, 1]], dtype=jnp.int32)
        bond_types = jnp.array([0], dtype=jnp.int32)
        bond_k = jnp.array([100.0])
        bond_r0 = jnp.array([3.0])
        box_size = jnp.array([100.0, 100.0, 100.0])

        energy = harmonic_bond_energy(positions, bond_pairs, bond_types, bond_k, bond_r0, box_size)
        expected = 0.5 * 100.0 * (1.0 - 3.0) ** 2  # = 200.0
        assert jnp.abs(energy - expected) < 1e-3, f"Expected {expected}, got {energy}"

    def test_pbc_bond(self):
        """Bond across periodic boundary should use minimum image."""
        # Two particles near opposite edges of box
        positions = jnp.array([[1.0, 0.0, 0.0], [99.0, 0.0, 0.0]])
        bond_pairs = jnp.array([[0, 1]], dtype=jnp.int32)
        bond_types = jnp.array([0], dtype=jnp.int32)
        bond_k = jnp.array([100.0])
        bond_r0 = jnp.array([2.0])  # equilibrium at 2 nm
        box_size = jnp.array([100.0, 100.0, 100.0])

        energy = harmonic_bond_energy(positions, bond_pairs, bond_types, bond_k, bond_r0, box_size)
        # Minimum image distance = 2.0 nm (across boundary)
        expected = 0.5 * 100.0 * (2.0 - 2.0) ** 2  # = 0
        assert jnp.abs(energy - expected) < 1e-3

    def test_multiple_bonds(self):
        """Energy should sum correctly over multiple bonds."""
        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ])
        bond_pairs = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
        bond_types = jnp.array([0, 0], dtype=jnp.int32)
        bond_k = jnp.array([100.0])
        bond_r0 = jnp.array([3.0])
        box_size = jnp.array([100.0, 100.0, 100.0])

        energy = harmonic_bond_energy(positions, bond_pairs, bond_types, bond_k, bond_r0, box_size)
        # Both bonds at equilibrium
        assert jnp.abs(energy) < 1e-4

    def test_force_matches_gradient(self):
        """Force from jax.grad should match the negative gradient of energy."""
        positions = jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        bond_pairs = jnp.array([[0, 1]], dtype=jnp.int32)
        bond_types = jnp.array([0], dtype=jnp.int32)
        bond_k = jnp.array([100.0])
        bond_r0 = jnp.array([3.0])
        box_size = jnp.array([100.0, 100.0, 100.0])

        def energy_fn(pos):
            return harmonic_bond_energy(pos, bond_pairs, bond_types, bond_k, bond_r0, box_size)

        # Analytical gradient
        grad_fn = jax.grad(energy_fn)
        grad_analytical = grad_fn(positions)

        # Numerical gradient
        eps = 1e-4
        grad_numerical = jnp.zeros_like(positions)
        for i in range(positions.shape[0]):
            for j in range(3):
                pos_plus = positions.at[i, j].add(eps)
                pos_minus = positions.at[i, j].add(-eps)
                grad_numerical = grad_numerical.at[i, j].set(
                    (energy_fn(pos_plus) - energy_fn(pos_minus)) / (2 * eps)
                )

        assert jnp.allclose(grad_analytical, grad_numerical, atol=0.05), \
            f"Gradient mismatch:\nanalytical={grad_analytical}\nnumerical={grad_numerical}"


# ============================================================================
# Angle energy tests
# ============================================================================


class TestHarmonicAngle:
    """Tests for harmonic angle potential."""

    def test_zero_energy_at_equilibrium(self):
        """Angle energy should be zero when theta = theta0 (pi = straight)."""
        # Three collinear particles: angle = pi
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        angle_triples = jnp.array([[0, 1, 2]], dtype=jnp.int32)
        angle_types = jnp.array([0], dtype=jnp.int32)
        angle_k = jnp.array([5.0])
        angle_theta0 = jnp.array([jnp.pi])
        box_size = jnp.array([100.0, 100.0, 100.0])

        energy = harmonic_angle_energy(positions, angle_triples, angle_types, angle_k, angle_theta0, box_size)
        assert jnp.abs(energy) < 1e-3, f"Energy at equilibrium angle should be ~0, got {energy}"

    def test_positive_energy_bent(self):
        """Angle energy should be positive when bent away from equilibrium."""
        # Right angle (pi/2)
        positions = jnp.array([[3.0, 3.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        angle_triples = jnp.array([[0, 1, 2]], dtype=jnp.int32)
        angle_types = jnp.array([0], dtype=jnp.int32)
        angle_k = jnp.array([5.0])
        angle_theta0 = jnp.array([jnp.pi])
        box_size = jnp.array([100.0, 100.0, 100.0])

        energy = harmonic_angle_energy(positions, angle_triples, angle_types, angle_k, angle_theta0, box_size)
        expected = 0.5 * 5.0 * (jnp.pi / 2.0 - jnp.pi) ** 2
        assert jnp.abs(energy - expected) < 0.1, f"Expected ~{expected}, got {energy}"

    def test_angle_force_gradient(self):
        """Force from jax.grad should be consistent with numerical gradient."""
        positions = jnp.array([[3.0, 2.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        angle_triples = jnp.array([[0, 1, 2]], dtype=jnp.int32)
        angle_types = jnp.array([0], dtype=jnp.int32)
        angle_k = jnp.array([5.0])
        angle_theta0 = jnp.array([jnp.pi])
        box_size = jnp.array([100.0, 100.0, 100.0])

        def energy_fn(pos):
            return harmonic_angle_energy(pos, angle_triples, angle_types, angle_k, angle_theta0, box_size)

        grad_analytical = jax.grad(energy_fn)(positions)

        eps = 1e-4
        grad_numerical = jnp.zeros_like(positions)
        for i in range(positions.shape[0]):
            for j in range(3):
                pos_plus = positions.at[i, j].add(eps)
                pos_minus = positions.at[i, j].add(-eps)
                grad_numerical = grad_numerical.at[i, j].set(
                    (energy_fn(pos_plus) - energy_fn(pos_minus)) / (2 * eps)
                )

        assert jnp.allclose(grad_analytical, grad_numerical, atol=1e-2), \
            f"Gradient mismatch:\nanalytical={grad_analytical}\nnumerical={grad_numerical}"


# ============================================================================
# WCA pair tests
# ============================================================================


class TestWCA:
    """Tests for WCA repulsive potential."""

    def test_zero_beyond_cutoff(self):
        """WCA energy should be exactly zero beyond sigma * 2^(1/6)."""
        sigma = 3.0
        r_cut = sigma * 2.0 ** (1.0 / 6.0)
        dr = jnp.array([r_cut + 0.1, 0.0, 0.0])
        e = wca_energy_pair(dr, jnp.float32(sigma))
        assert jnp.abs(e) < 1e-4, f"WCA should be 0 beyond cutoff, got {e}"

    def test_positive_inside_cutoff(self):
        """WCA energy should be positive inside cutoff."""
        sigma = 3.0
        dr = jnp.array([2.5, 0.0, 0.0])
        e = wca_energy_pair(dr, jnp.float32(sigma))
        assert e > 0.0, f"WCA should be positive inside cutoff, got {e}"

    def test_continuous_at_cutoff(self):
        """WCA should be continuous (approximately zero) at the cutoff."""
        sigma = 3.0
        r_cut = sigma * 2.0 ** (1.0 / 6.0)
        dr_at = jnp.array([r_cut - 0.001, 0.0, 0.0])
        e = wca_energy_pair(dr_at, jnp.float32(sigma))
        assert jnp.abs(e) < 0.1, f"WCA should be ~0 at cutoff, got {e}"


# ============================================================================
# Yukawa pair tests
# ============================================================================


class TestYukawa:
    """Tests for Yukawa attraction."""

    def test_attractive(self):
        """Yukawa should return negative energy for positive epsilon."""
        dr = jnp.array([5.0, 0.0, 0.0])
        e = yukawa_energy_pair(dr, jnp.float32(3.0), jnp.float32(3.0), jnp.float32(0.5))
        assert e < 0.0, f"Yukawa should be attractive (negative), got {e}"

    def test_zero_for_zero_epsilon(self):
        """Yukawa should be zero when epsilon=0 (no attraction)."""
        dr = jnp.array([5.0, 0.0, 0.0])
        e = yukawa_energy_pair(dr, jnp.float32(0.0), jnp.float32(3.0), jnp.float32(0.5))
        assert jnp.abs(e) < 1e-6, f"Yukawa should be 0 for eps=0, got {e}"

    def test_decays_with_distance(self):
        """Yukawa should decay with increasing distance."""
        dr_near = jnp.array([4.0, 0.0, 0.0])
        dr_far = jnp.array([8.0, 0.0, 0.0])
        e_near = yukawa_energy_pair(dr_near, jnp.float32(3.0), jnp.float32(3.0), jnp.float32(0.5))
        e_far = yukawa_energy_pair(dr_far, jnp.float32(3.0), jnp.float32(3.0), jnp.float32(0.5))
        assert e_near < e_far, "Yukawa should be stronger (more negative) at shorter distance"


# ============================================================================
# Debye-Huckel tests
# ============================================================================


class TestDebyeHuckel:
    """Tests for screened electrostatics."""

    def test_like_charges_repulsive(self):
        """Like charges should give positive (repulsive) energy."""
        dr = jnp.array([4.0, 0.0, 0.0])
        e = debye_huckel_energy_pair(dr, jnp.float32(5.0), jnp.float32(5.0),
                                      jnp.float32(1.0), jnp.float32(3.0))
        assert e > 0.0, f"Like charges should repel, got {e}"

    def test_opposite_charges_attractive(self):
        """Opposite charges should give negative (attractive) energy."""
        dr = jnp.array([4.0, 0.0, 0.0])
        e = debye_huckel_energy_pair(dr, jnp.float32(-5.0), jnp.float32(5.0),
                                      jnp.float32(1.0), jnp.float32(3.0))
        assert e < 0.0, f"Opposite charges should attract, got {e}"

    def test_screening_reduces_energy(self):
        """Shorter Debye length should reduce the interaction."""
        dr = jnp.array([5.0, 0.0, 0.0])
        e_long = debye_huckel_energy_pair(dr, jnp.float32(5.0), jnp.float32(5.0),
                                           jnp.float32(2.0), jnp.float32(3.0))
        e_short = debye_huckel_energy_pair(dr, jnp.float32(5.0), jnp.float32(5.0),
                                            jnp.float32(0.5), jnp.float32(3.0))
        assert jnp.abs(e_long) > jnp.abs(e_short), "Shorter screening should reduce energy"


# ============================================================================
# JIT compilation test
# ============================================================================


class TestJIT:
    """Ensure all energy functions can be JIT-compiled."""

    def test_bond_energy_jit(self):
        """Harmonic bond energy should JIT compile without error."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        bond_pairs = jnp.array([[0, 1]], dtype=jnp.int32)
        bond_types = jnp.array([0], dtype=jnp.int32)
        bond_k = jnp.array([100.0])
        bond_r0 = jnp.array([3.0])
        box_size = jnp.array([100.0, 100.0, 100.0])

        jit_fn = jax.jit(harmonic_bond_energy)
        e = jit_fn(positions, bond_pairs, bond_types, bond_k, bond_r0, box_size)
        assert jnp.isfinite(e)

    def test_angle_energy_jit(self):
        """Harmonic angle energy should JIT compile without error."""
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        angle_triples = jnp.array([[0, 1, 2]], dtype=jnp.int32)
        angle_types = jnp.array([0], dtype=jnp.int32)
        angle_k = jnp.array([5.0])
        angle_theta0 = jnp.array([jnp.pi])
        box_size = jnp.array([100.0, 100.0, 100.0])

        jit_fn = jax.jit(harmonic_angle_energy)
        e = jit_fn(positions, angle_triples, angle_types, angle_k, angle_theta0, box_size)
        assert jnp.isfinite(e)

    def test_grad_through_bond_energy(self):
        """jax.grad through bond energy should work."""
        bond_pairs = jnp.array([[0, 1]], dtype=jnp.int32)
        bond_types = jnp.array([0], dtype=jnp.int32)
        bond_k = jnp.array([100.0])
        bond_r0 = jnp.array([3.0])
        box_size = jnp.array([100.0, 100.0, 100.0])

        @jax.jit
        def force_fn(pos):
            return -jax.grad(
                lambda p: harmonic_bond_energy(p, bond_pairs, bond_types, bond_k, bond_r0, box_size)
            )(pos)

        positions = jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        forces = force_fn(positions)
        assert forces.shape == positions.shape
        assert jnp.all(jnp.isfinite(forces))
        # Stretched bond: force on particle 0 should pull toward particle 1 (positive x)
        assert forces[0, 0] > 0.0, "Force should pull particle 0 toward particle 1"


# ============================================================================
# Neighbor list tests (Phase B)
# ============================================================================


class TestNeighborList:
    """Tests for neighbor list construction and queries."""

    def test_build_finds_close_neighbors(self):
        """Neighbor list should include particles within cutoff+skin."""
        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],   # close
            [50.0, 50.0, 50.0],  # far
        ])
        box_size = jnp.array([100.0, 100.0, 100.0])
        nl = build_neighbor_list(positions, box_size, cutoff=5.0, skin=2.0)

        # Particle 0 should have particle 1 as neighbor (dist=3 < 7)
        assert nl.mask[0].any(), "Particle 0 should have at least one neighbor"
        # Particle 0 should NOT have particle 2 as neighbor
        nbrs_0 = nl.neighbors[0][nl.mask[0]]
        assert 1 in nbrs_0, "Particle 1 should be neighbor of particle 0"

    def test_build_pbc_neighbors(self):
        """Neighbor list should find neighbors across periodic boundaries."""
        positions = jnp.array([
            [1.0, 50.0, 50.0],
            [99.0, 50.0, 50.0],  # PBC distance = 2.0
        ])
        box_size = jnp.array([100.0, 100.0, 100.0])
        nl = build_neighbor_list(positions, box_size, cutoff=5.0, skin=2.0)

        # Should find each other across the boundary
        assert nl.mask[0, 0], "PBC neighbor should be found"

    def test_no_self_neighbors(self):
        """No particle should be its own neighbor."""
        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        box_size = jnp.array([100.0, 100.0, 100.0])
        nl = build_neighbor_list(positions, box_size, cutoff=5.0, skin=2.0)

        for i in range(2):
            valid_nbrs = nl.neighbors[i][nl.mask[i]]
            assert i not in valid_nbrs, f"Particle {i} should not be its own neighbor"

    def test_overflow_detection(self):
        """Overflow flag should be set when max_neighbors is too small."""
        # 10 particles all close together
        key = jax.random.PRNGKey(42)
        positions = jax.random.uniform(key, (10, 3)) * 3.0
        box_size = jnp.array([100.0, 100.0, 100.0])

        # max_neighbors=2, but each particle has 9 potential neighbors
        nl = build_neighbor_list(positions, box_size, cutoff=20.0, skin=2.0, max_neighbors=2)
        assert nl.overflow, "Should detect overflow with max_neighbors=2"

    def test_needs_rebuild_false_when_stationary(self):
        """No rebuild needed if particles haven't moved."""
        positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        box_size = jnp.array([100.0, 100.0, 100.0])
        nl = build_neighbor_list(positions, box_size, cutoff=8.0, skin=2.0)

        assert not needs_rebuild(positions, nl, box_size, skin=2.0)

    def test_needs_rebuild_true_after_large_move(self):
        """Rebuild needed if a particle moves more than skin/2."""
        positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        box_size = jnp.array([100.0, 100.0, 100.0])
        nl = build_neighbor_list(positions, box_size, cutoff=8.0, skin=2.0)

        # Move particle 0 by 1.5 nm > skin/2 = 1.0
        new_positions = positions.at[0, 0].set(1.5)
        assert needs_rebuild(new_positions, nl, box_size, skin=2.0)

    def test_neighbor_pairs_unique(self):
        """neighbor_pairs should return unique i<j pairs."""
        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ])
        box_size = jnp.array([100.0, 100.0, 100.0])
        nl = build_neighbor_list(positions, box_size, cutoff=5.0, skin=2.0)
        pair_i, pair_j, pair_mask = neighbor_pairs(nl)

        # Extract valid pairs
        valid_i = pair_i[pair_mask]
        valid_j = pair_j[pair_mask]

        # All pairs should have i < j
        assert jnp.all(valid_i < valid_j), "All pairs should have i < j"

        # Should find pair (0,1) since dist=3 < 7
        pairs = set(zip(valid_i.tolist(), valid_j.tolist()))
        assert (0, 1) in pairs, "Should find pair (0,1)"


class TestNeighborListEnergy:
    """Tests that neighbor-list energy matches brute-force energy."""

    def _setup_system(self, n=8, seed=0):
        """Create a small test system with random positions."""
        key = jax.random.PRNGKey(seed)
        box_size = jnp.array([30.0, 30.0, 30.0])
        positions = jax.random.uniform(key, (n, 3)) * box_size

        # Types: alternate between 0 (NTF2) and 4 (RG_IDR) for attraction
        particle_types = jnp.array([0, 4, 0, 4, 0, 4, 0, 4][:n], dtype=jnp.int32)
        # Charges: some charged beads
        particle_charges = jnp.array([0.0, 5.0, 0.0, 5.0, 0.0, -4.0, 0.0, -4.0][:n])

        # Bonds: 0-1 and 2-3
        bond_pairs = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)

        from sgsim.parameters import default_params
        params = default_params()

        return positions, particle_types, particle_charges, bond_pairs, params, box_size

    def test_nl_matches_bruteforce(self):
        """Neighbor-list energy should match brute-force for same system."""
        positions, ptypes, charges, bonds, params, box_size = self._setup_system()
        cutoff = 10.0
        n = positions.shape[0]

        # Brute force
        e_brute = nonbonded_energy_bruteforce(
            positions, ptypes, charges,
            params.epsilon_attract, params.sigma,
            params.kappa, params.debye_length,
            box_size, cutoff, bonds,
        )

        # Neighbor list (use large cutoff+skin to match brute force)
        nl = build_neighbor_list(positions, box_size, cutoff=cutoff, skin=5.0)
        pair_i, pair_j, pair_mask = neighbor_pairs(nl)
        bond_excl = build_bond_exclusion_mask(n, bonds)

        e_nl = nonbonded_energy_neighborlist(
            positions, ptypes, charges,
            params.epsilon_attract, params.sigma,
            params.kappa, params.debye_length,
            box_size, cutoff, bond_excl,
            pair_i, pair_j, pair_mask,
        )

        assert jnp.allclose(e_brute, e_nl, atol=0.1), \
            f"NL energy {e_nl} should match brute force {e_brute}"

    def test_bond_exclusion(self):
        """Bonded pairs should be excluded from non-bonded energy."""
        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # Very close — large WCA if not excluded
        ])
        box_size = jnp.array([100.0, 100.0, 100.0])
        ptypes = jnp.array([0, 0], dtype=jnp.int32)
        charges = jnp.zeros(2)
        bonds_yes = jnp.array([[0, 1]], dtype=jnp.int32)
        bonds_no = jnp.zeros((0, 2), dtype=jnp.int32)

        from sgsim.parameters import default_params
        params = default_params()

        # With bond exclusion
        e_excluded = nonbonded_energy_bruteforce(
            positions, ptypes, charges,
            params.epsilon_attract, params.sigma,
            params.kappa, params.debye_length,
            box_size, 10.0, bonds_yes,
        )

        # Without bond exclusion
        e_not_excluded = nonbonded_energy_bruteforce(
            positions, ptypes, charges,
            params.epsilon_attract, params.sigma,
            params.kappa, params.debye_length,
            box_size, 10.0, bonds_no,
        )

        assert e_excluded < e_not_excluded, \
            f"Excluding bonded pair should lower energy: {e_excluded} vs {e_not_excluded}"

    def test_nl_energy_jit(self):
        """Neighbor-list energy should be JIT-compilable."""
        positions, ptypes, charges, bonds, params, box_size = self._setup_system()
        cutoff = 10.0
        n = positions.shape[0]

        nl = build_neighbor_list(positions, box_size, cutoff=cutoff, skin=5.0)
        pair_i, pair_j, pair_mask = neighbor_pairs(nl)
        bond_excl = build_bond_exclusion_mask(n, bonds)

        @jax.jit
        def compute_e(pos):
            return nonbonded_energy_neighborlist(
                pos, ptypes, charges,
                params.epsilon_attract, params.sigma,
                params.kappa, params.debye_length,
                box_size, cutoff, bond_excl,
                pair_i, pair_j, pair_mask,
            )

        e = compute_e(positions)
        assert jnp.isfinite(e), f"Energy should be finite, got {e}"

    def test_nl_energy_grad(self):
        """jax.grad through neighbor-list energy should work."""
        positions, ptypes, charges, bonds, params, box_size = self._setup_system()
        cutoff = 10.0
        n = positions.shape[0]

        nl = build_neighbor_list(positions, box_size, cutoff=cutoff, skin=5.0)
        pair_i, pair_j, pair_mask = neighbor_pairs(nl)
        bond_excl = build_bond_exclusion_mask(n, bonds)

        @jax.jit
        def force_fn(pos):
            return -jax.grad(lambda p: nonbonded_energy_neighborlist(
                p, ptypes, charges,
                params.epsilon_attract, params.sigma,
                params.kappa, params.debye_length,
                box_size, cutoff, bond_excl,
                pair_i, pair_j, pair_mask,
            ))(pos)

        forces = force_fn(positions)
        assert forces.shape == positions.shape
        assert jnp.all(jnp.isfinite(forces)), "All forces should be finite"
