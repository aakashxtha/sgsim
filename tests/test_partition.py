"""Tests for neighbor list construction and rebuild logic.

Covers:
- Correctness of batched vs single-pass NL build
- Overflow flag detection
- needs_rebuild trigger condition
"""

import jax
import jax.numpy as jnp
import pytest

from sgsim.partition import build_neighbor_list, needs_rebuild, NeighborList


def _neighbor_sets(nl: NeighborList) -> list:
    """Return the set of valid neighbor indices for every particle."""
    result = []
    for i in range(nl.neighbors.shape[0]):
        valid = {int(nl.neighbors[i, k]) for k in range(nl.mask.shape[1]) if nl.mask[i, k]}
        result.append(valid)
    return result


class TestBuildNeighborList:
    """Correctness tests for build_neighbor_list."""

    def test_close_particles_are_neighbors(self):
        """Two particles within cutoff should appear in each other's neighbor list."""
        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],   # distance 3, within cutoff=5
            [30.0, 0.0, 0.0],  # distance 30, outside cutoff=5
        ])
        box_size = jnp.array([100.0, 100.0, 100.0])
        nl = build_neighbor_list(positions, box_size, cutoff=5.0, skin=1.0, max_neighbors=10)

        sets = _neighbor_sets(nl)
        assert 1 in sets[0], "Particle 1 should be a neighbor of particle 0"
        assert 0 in sets[1], "Particle 0 should be a neighbor of particle 1"
        assert 2 not in sets[0], "Particle 2 should not be a neighbor of particle 0"

    def test_self_not_in_neighbor_list(self):
        """No particle should appear as its own neighbor."""
        key = jax.random.PRNGKey(0)
        positions = jax.random.uniform(key, (8, 3)) * 20.0
        box_size = jnp.array([20.0, 20.0, 20.0])
        nl = build_neighbor_list(positions, box_size, cutoff=5.0, skin=1.0, max_neighbors=20)

        for i in range(8):
            assert i not in _neighbor_sets(nl)[i], \
                f"Particle {i} should not be its own neighbor"

    def test_overflow_flag_set_when_exceeded(self):
        """overflow=True when a particle has more neighbors than max_neighbors."""
        # Pack many particles in a tiny region so every particle neighbors every other
        n = 10
        positions = jnp.zeros((n, 3))  # all at origin
        box_size = jnp.array([100.0, 100.0, 100.0])
        # max_neighbors=3 but every particle has n-1=9 neighbors
        nl = build_neighbor_list(positions, box_size, cutoff=5.0, skin=0.0, max_neighbors=3)
        assert bool(nl.overflow), "Overflow should be detected for a dense cluster"

    def test_overflow_flag_clear_for_sparse_system(self):
        """overflow=False when all particles have fewer neighbors than max_neighbors."""
        # Particles spaced 15 nm apart, cutoff=5 → each has 0 neighbors
        positions = jnp.array([[i * 15.0, 0.0, 0.0] for i in range(8)])
        box_size = jnp.array([200.0, 200.0, 200.0])
        nl = build_neighbor_list(positions, box_size, cutoff=5.0, skin=1.0, max_neighbors=64)
        assert not bool(nl.overflow)

    def test_batched_matches_single_pass(self):
        """Forcing batch_size=2 should give the same neighbor sets as a single batch."""
        key = jax.random.PRNGKey(42)
        n = 12
        positions = jax.random.uniform(key, (n, 3)) * 15.0
        box_size = jnp.array([15.0, 15.0, 15.0])
        cutoff = 5.0
        skin = 1.0
        max_nb = 20

        nl_single = build_neighbor_list(positions, box_size, cutoff, skin, max_nb,
                                        batch_size=n)   # one big batch
        nl_batched = build_neighbor_list(positions, box_size, cutoff, skin, max_nb,
                                         batch_size=2)  # many small batches

        sets_single = _neighbor_sets(nl_single)
        sets_batched = _neighbor_sets(nl_batched)

        for i in range(n):
            assert sets_single[i] == sets_batched[i], \
                f"Particle {i}: single={sets_single[i]} batched={sets_batched[i]}"

    def test_pbc_neighbors_detected(self):
        """Particles near opposite box faces should be neighbors via PBC."""
        positions = jnp.array([
            [0.5, 5.0, 5.0],   # near x=0 face
            [9.5, 5.0, 5.0],   # near x=10 face — distance via PBC = 1.0
        ])
        box_size = jnp.array([10.0, 10.0, 10.0])
        nl = build_neighbor_list(positions, box_size, cutoff=2.0, skin=0.5, max_neighbors=4)

        sets = _neighbor_sets(nl)
        assert 1 in sets[0], "PBC neighbor should be detected across the box boundary"
        assert 0 in sets[1]


class TestNeedsRebuild:
    """Tests for the needs_rebuild helper."""

    def test_no_rebuild_when_stationary(self):
        """Stationary particles should not trigger a rebuild."""
        positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        box_size = jnp.array([50.0, 50.0, 50.0])
        assert not bool(needs_rebuild(positions, positions, box_size, skin=2.0))

    def test_rebuild_triggered_by_large_displacement(self):
        """Moving a particle by more than skin/2 should trigger a rebuild."""
        ref = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        # Move particle 0 by 1.5 nm (> skin/2 = 1.0)
        moved = jnp.array([[1.5, 0.0, 0.0], [5.0, 0.0, 0.0]])
        box_size = jnp.array([50.0, 50.0, 50.0])
        assert bool(needs_rebuild(moved, ref, box_size, skin=2.0))

    def test_no_rebuild_within_skin(self):
        """Small displacement within skin/2 should not trigger a rebuild."""
        ref = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        # Move particle 0 by 0.4 nm (< skin/2 = 1.0)
        moved = jnp.array([[0.4, 0.0, 0.0], [5.0, 0.0, 0.0]])
        box_size = jnp.array([50.0, 50.0, 50.0])
        assert not bool(needs_rebuild(moved, ref, box_size, skin=2.0))
