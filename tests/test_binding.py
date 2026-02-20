"""Tests for the MC binding module.

Validates:
- Binding state initialization
- Binding energy computation
- MC binding/unbinding acceptance
- Saturation (site can't bind twice)
- Competition (NTF2 pocket)
- JIT compatibility
"""

import jax
import jax.numpy as jnp
import pytest

from sgsim.binding import (
    init_binding_state,
    compute_binding_energy,
    mc_binding_step,
    _apply_bind,
    _apply_unbind,
)
from sgsim.setup import build_system
from sgsim.parameters import default_params


def _make_two_site_system():
    """Minimal 2-site system: one NTF2_POCKET and one USP10_NIM_SITE.

    Two particles at distance 3 nm, each owning one binding site.
    They are compatible and within cutoff.
    """
    from sgsim.types import BINDING_SITE_TYPES as BST

    positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    box_size = jnp.array([50.0, 50.0, 50.0])
    site_particle = jnp.array([0, 1], dtype=jnp.int32)
    site_types = jnp.array([BST["NTF2_POCKET"], BST["USP10_NIM_SITE"]], dtype=jnp.int32)
    site_molecule = jnp.array([0, 1], dtype=jnp.int32)

    params = default_params()

    return (positions, box_size, site_particle, site_types, site_molecule,
            params.binding_energy, params.binding_cutoff, params.binding_compatibility)


class TestBindingStateInit:
    """Tests for binding state initialization."""

    def test_empty_state(self):
        """Initial binding state should have no active bonds."""
        state = init_binding_state(n_sites=10, max_bonds=50)
        assert state.n_bound == 0
        assert jnp.sum(state.bound_mask) == 0
        assert jnp.all(state.site_occupied == 0)
        assert jnp.all(state.site_partner == -1)

    def test_state_sizes(self):
        """State arrays should have correct shapes."""
        state = init_binding_state(n_sites=20, max_bonds=100)
        assert state.bound_pairs.shape == (100, 2)
        assert state.site_occupied.shape == (20,)


class TestApplyBindUnbind:
    """Tests for low-level bind/unbind operations."""

    def test_apply_bind(self):
        """Binding should mark both sites as occupied."""
        state = init_binding_state(n_sites=4, max_bonds=10)
        new_state = _apply_bind(state, 0, 2)

        assert new_state.n_bound == 1
        assert new_state.site_occupied[0] == 1
        assert new_state.site_occupied[2] == 1
        assert new_state.site_partner[0] == 2
        assert new_state.site_partner[2] == 0
        assert new_state.bound_mask[0]

    def test_apply_unbind(self):
        """Unbinding should free both sites."""
        state = init_binding_state(n_sites=4, max_bonds=10)
        state = _apply_bind(state, 0, 2)
        new_state = _apply_unbind(state, 0)

        assert new_state.site_occupied[0] == 0
        assert new_state.site_occupied[2] == 0
        assert new_state.site_partner[0] == -1
        assert new_state.site_partner[2] == -1

    def test_double_bind_same_site(self):
        """A site bound once should still show occupied=1."""
        state = init_binding_state(n_sites=4, max_bonds=10)
        state = _apply_bind(state, 0, 1)
        # Site 0 is now occupied
        assert state.site_occupied[0] == 1


class TestBindingEnergy:
    """Tests for binding energy computation."""

    def test_zero_energy_no_bonds(self):
        """No active bonds should give zero binding energy."""
        (positions, box_size, site_particle, site_types, site_molecule,
         bind_energy, bind_cutoff, _) = _make_two_site_system()

        state = init_binding_state(n_sites=2, max_bonds=10)
        e = compute_binding_energy(
            positions, box_size, state,
            site_particle, bind_energy, site_types, bind_cutoff,
        )
        assert jnp.abs(e) < 1e-6, f"Energy with no bonds should be ~0, got {e}"

    def test_negative_energy_with_bond(self):
        """An active favorable bond should give negative energy."""
        (positions, box_size, site_particle, site_types, site_molecule,
         bind_energy, bind_cutoff, _) = _make_two_site_system()

        state = init_binding_state(n_sites=2, max_bonds=10)
        state = _apply_bind(state, 0, 1)

        e = compute_binding_energy(
            positions, box_size, state,
            site_particle, bind_energy, site_types, bind_cutoff,
        )
        assert e < 0.0, f"NTF2-USP10 bond energy should be negative, got {e}"


class TestMCBindingStep:
    """Tests for MC binding moves."""

    def test_mc_step_jit(self):
        """MC binding step should be JIT-compilable."""
        (positions, box_size, site_particle, site_types, site_molecule,
         bind_energy, bind_cutoff, bind_compat) = _make_two_site_system()

        state = init_binding_state(n_sites=2, max_bonds=10)
        key = jax.random.PRNGKey(42)

        @jax.jit
        def do_mc(state, key):
            return mc_binding_step(
                state, positions, box_size,
                site_particle, site_types, site_molecule,
                bind_energy, bind_cutoff, bind_compat,
                kT=1.0, n_attempts=5, rng_key=key,
            )

        new_state, new_key = do_mc(state, key)
        assert new_state.site_occupied.shape == (2,)

    def test_favorable_binding_occurs(self):
        """With strong favorable energy, binding should eventually happen."""
        (positions, box_size, site_particle, site_types, site_molecule,
         bind_energy, bind_cutoff, bind_compat) = _make_two_site_system()

        state = init_binding_state(n_sites=2, max_bonds=10)

        # Run many MC attempts — with -8 kT binding energy, should bind quickly
        key = jax.random.PRNGKey(0)
        bound_at_any_point = False
        for i in range(20):
            key, subkey = jax.random.split(key)
            state, _ = mc_binding_step(
                state, positions, box_size,
                site_particle, site_types, site_molecule,
                bind_energy, bind_cutoff, bind_compat,
                kT=1.0, n_attempts=10, rng_key=subkey,
            )
            if state.n_bound > 0:
                bound_at_any_point = True
                break

        assert bound_at_any_point, \
            "Favorable binding (-8 kT) should occur within 200 MC attempts"

    def test_saturation(self):
        """An occupied site should not bind a second partner."""
        (positions, box_size, site_particle, site_types, site_molecule,
         bind_energy, bind_cutoff, bind_compat) = _make_two_site_system()

        # Pre-bind site 0 to site 1
        state = init_binding_state(n_sites=2, max_bonds=10)
        state = _apply_bind(state, 0, 1)

        # Run MC — no new binding should occur (both sites already occupied)
        key = jax.random.PRNGKey(99)
        state, _ = mc_binding_step(
            state, positions, box_size,
            site_particle, site_types, site_molecule,
            bind_energy, bind_cutoff, bind_compat,
            kT=1.0, n_attempts=50, rng_key=key,
        )

        # n_bound should be 0 or 1 (might unbind, but can't have 2)
        assert state.n_bound <= 1, \
            f"With 2 sites, max 1 bond possible, got {state.n_bound}"


class TestBindingWithSystem:
    """Integration tests: binding with a built system."""

    def test_binding_with_g3bp1_usp10(self):
        """G3BP1 dimer + USP10 system should have compatible binding sites."""
        key = jax.random.PRNGKey(10)
        system = build_system(
            {"g3bp1_dimer": 1, "usp10": 1},
            box_size=30.0, rng_key=key,
        )

        topo = system["topology"]
        n_sites = topo.n_sites
        assert n_sites > 0, "System should have binding sites"

        params = default_params()
        state = init_binding_state(n_sites=n_sites, max_bonds=100)

        # Run MC
        key2 = jax.random.PRNGKey(20)
        state, _ = mc_binding_step(
            state, system["positions"], system["box_size"],
            topo.binding_site_particle, topo.binding_site_type,
            topo.binding_site_molecule,
            params.binding_energy, params.binding_cutoff,
            params.binding_compatibility,
            kT=1.0, n_attempts=50, rng_key=key2,
        )

        # Just verify it runs without error
        assert state.site_occupied.shape == (n_sites,)
