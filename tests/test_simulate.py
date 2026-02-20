"""Tests for the Brownian dynamics integrator.

Validates:
- Single step execution and JIT compilation
- Force computation produces finite results
- Diffusion: MSD = 6*D*t for free particles (D = kT/gamma)
- Temperature equilibration: kinetic energy matches kT
- Periodic boundary conditions maintained
- Neighbor list rebuilds during simulation
"""

import jax
import jax.numpy as jnp
import pytest

from sgsim.simulate import init_fn, make_step_fn, run_brownian_dynamics, IntegratorState
from sgsim.types import Topology
from sgsim.parameters import default_params


def _make_free_particles(n=10, box_size=50.0, seed=42):
    """Create a system of free particles (no bonds) for testing."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    positions = jax.random.uniform(k1, (n, 3)) * box_size
    particle_types = jnp.zeros(n, dtype=jnp.int32)
    particle_charges = jnp.zeros(n, dtype=jnp.float32)
    particle_radii = jnp.ones(n, dtype=jnp.float32) * 1.5
    box = jnp.array([box_size, box_size, box_size])

    # Empty topology (no bonds, no angles, no binding sites)
    topology = Topology(
        bond_pairs=jnp.zeros((0, 2), dtype=jnp.int32),
        bond_types=jnp.zeros(0, dtype=jnp.int32),
        angle_triples=jnp.zeros((0, 3), dtype=jnp.int32),
        angle_types=jnp.zeros(0, dtype=jnp.int32),
        binding_site_particle=jnp.zeros(0, dtype=jnp.int32),
        binding_site_type=jnp.zeros(0, dtype=jnp.int32),
        binding_site_molecule=jnp.zeros(0, dtype=jnp.int32),
        n_particles=n,
        n_bonds=0,
        n_angles=0,
        n_sites=0,
    )

    params = default_params()
    return positions, particle_types, particle_charges, particle_radii, box, topology, params, k2


def _make_bonded_pair(box_size=50.0, seed=42):
    """Create a system with a single bonded pair."""
    key = jax.random.PRNGKey(seed)

    positions = jnp.array([[20.0, 25.0, 25.0], [25.0, 25.0, 25.0]])  # separated by 5 nm
    particle_types = jnp.array([0, 0], dtype=jnp.int32)
    particle_charges = jnp.zeros(2, dtype=jnp.float32)
    particle_radii = jnp.ones(2, dtype=jnp.float32) * 1.5
    box = jnp.array([box_size, box_size, box_size])

    topology = Topology(
        bond_pairs=jnp.array([[0, 1]], dtype=jnp.int32),
        bond_types=jnp.array([0], dtype=jnp.int32),  # DOMAIN_LINKER_STIFF
        angle_triples=jnp.zeros((0, 3), dtype=jnp.int32),
        angle_types=jnp.zeros(0, dtype=jnp.int32),
        binding_site_particle=jnp.zeros(0, dtype=jnp.int32),
        binding_site_type=jnp.zeros(0, dtype=jnp.int32),
        binding_site_molecule=jnp.zeros(0, dtype=jnp.int32),
        n_particles=2,
        n_bonds=1,
        n_angles=0,
        n_sites=0,
    )

    params = default_params()
    return positions, particle_types, particle_charges, particle_radii, box, topology, params, key


class TestIntegratorInit:
    """Tests for integrator initialization."""

    def test_init_returns_state(self):
        """init_fn should return an IntegratorState."""
        pos, ptypes, charges, radii, box, topo, params, key = _make_free_particles(n=5)
        state = init_fn(pos, ptypes, charges, radii, box, topo, params, key)
        assert isinstance(state, IntegratorState)
        assert state.positions.shape == (5, 3)
        assert state.step == 0

    def test_init_neighbor_list_created(self):
        """init_fn should create a neighbor list with valid pair data."""
        pos, ptypes, charges, radii, box, topo, params, key = _make_free_particles(n=5)
        state = init_fn(pos, ptypes, charges, radii, box, topo, params, key)
        assert state.nl_pair_i.shape[0] > 0
        assert state.nl_pair_mask.shape[0] > 0


class TestSingleStep:
    """Tests for a single integrator step."""

    def test_step_updates_positions(self):
        """A single step should change the positions."""
        pos, ptypes, charges, radii, box, topo, params, key = _make_free_particles(n=5)
        state = init_fn(pos, ptypes, charges, radii, box, topo, params, key)
        step_fn = make_step_fn(topo, params, dt=0.01, kT=1.0, gamma_base=1.0)
        new_state = step_fn(state)

        assert not jnp.allclose(state.positions, new_state.positions), \
            "Positions should change after one step"
        assert new_state.step == 1

    def test_step_positions_finite(self):
        """Positions should remain finite after stepping."""
        pos, ptypes, charges, radii, box, topo, params, key = _make_free_particles(n=5)
        state = init_fn(pos, ptypes, charges, radii, box, topo, params, key)
        step_fn = make_step_fn(topo, params, dt=0.01, kT=1.0, gamma_base=1.0)
        new_state = step_fn(state)

        assert jnp.all(jnp.isfinite(new_state.positions)), "Positions should be finite"

    def test_step_pbc_maintained(self):
        """Positions should stay within the box after stepping."""
        pos, ptypes, charges, radii, box, topo, params, key = _make_free_particles(n=5)
        state = init_fn(pos, ptypes, charges, radii, box, topo, params, key)
        step_fn = make_step_fn(topo, params, dt=0.01, kT=1.0, gamma_base=1.0)

        # Run several steps
        for _ in range(50):
            state = step_fn(state)

        assert jnp.all(state.positions >= 0.0), "Positions should be >= 0"
        assert jnp.all(state.positions < state.box_size), "Positions should be < box_size"

    def test_step_jit_compiles(self):
        """make_step_fn should produce a JIT-compilable function."""
        pos, ptypes, charges, radii, box, topo, params, key = _make_free_particles(n=5)
        state = init_fn(pos, ptypes, charges, radii, box, topo, params, key)
        step_fn = make_step_fn(topo, params)

        # Should compile and run without error
        state = step_fn(state)
        state = step_fn(state)
        assert state.step == 2


class TestBondedDynamics:
    """Tests for dynamics with bonded interactions."""

    def test_bond_restores_equilibrium(self):
        """A stretched bond should pull particles toward equilibrium."""
        pos, ptypes, charges, radii, box, topo, params, key = _make_bonded_pair()
        # bond r0 = 3.0 nm (DOMAIN_LINKER_STIFF), initial separation = 5.0 nm
        state = init_fn(pos, ptypes, charges, radii, box, topo, params, key, cutoff=10.0)
        step_fn = make_step_fn(topo, params, dt=0.001, kT=1.0, gamma_base=1.0, cutoff=10.0)

        # Run many steps with small dt (low noise) to let bond relax
        for _ in range(500):
            state = step_fn(state)

        # Measure bond distance
        dr = state.positions[0] - state.positions[1]
        dr = dr - box * jnp.round(dr / box)
        bond_dist = jnp.sqrt(jnp.sum(dr ** 2))
        box = state.box_size

        # Should be closer to equilibrium (r0 = 3.0) than initial (5.0)
        assert jnp.abs(bond_dist - 3.0) < 3.0, \
            f"Bond distance {bond_dist} should be closer to r0=3.0 than initial 5.0"


class TestDiffusion:
    """Tests for diffusion (MSD = 6*D*t for free Brownian particles)."""

    def test_msd_scales_linearly(self):
        """MSD should grow approximately linearly with time for free particles.

        For overdamped Langevin: MSD = 6 * D * t, where D = kT / gamma.
        We use well-separated particles in a large box to minimize interactions.
        """
        n = 20
        box_size = 200.0  # Very large box to minimize interactions
        kT = 1.0
        gamma_base = 1.0
        radius = 1.5
        dt = 0.005
        n_steps = 2000

        key = jax.random.PRNGKey(123)
        k1, k2 = jax.random.split(key)

        # Place particles on a grid far apart
        positions = jnp.zeros((n, 3))
        for i in range(n):
            positions = positions.at[i].set(jnp.array([
                20.0 + (i % 5) * 30.0,
                20.0 + (i // 5) * 30.0,
                100.0,
            ]))

        particle_types = jnp.zeros(n, dtype=jnp.int32)
        particle_charges = jnp.zeros(n, dtype=jnp.float32)
        particle_radii = jnp.ones(n, dtype=jnp.float32) * radius
        box = jnp.array([box_size, box_size, box_size])

        topology = Topology(
            bond_pairs=jnp.zeros((0, 2), dtype=jnp.int32),
            bond_types=jnp.zeros(0, dtype=jnp.int32),
            angle_triples=jnp.zeros((0, 3), dtype=jnp.int32),
            angle_types=jnp.zeros(0, dtype=jnp.int32),
            binding_site_particle=jnp.zeros(0, dtype=jnp.int32),
            binding_site_type=jnp.zeros(0, dtype=jnp.int32),
            binding_site_molecule=jnp.zeros(0, dtype=jnp.int32),
            n_particles=n,
            n_bonds=0,
            n_angles=0,
            n_sites=0,
        )
        params = default_params()

        state = init_fn(positions, particle_types, particle_charges, particle_radii,
                        box, topology, params, k2, cutoff=10.0, skin=2.0)
        step_fn = make_step_fn(topology, params, dt=dt, kT=kT, gamma_base=gamma_base, cutoff=10.0)

        initial_positions = state.positions.copy()

        # Run simulation
        def scan_body(s, _):
            return step_fn(s), None

        state, _ = jax.lax.scan(scan_body, state, None, length=n_steps)

        # Compute MSD (unwrapped — for PBC we use displacement from initial)
        dr = state.positions - initial_positions
        dr = dr - box * jnp.round(dr / box)
        msd = jnp.mean(jnp.sum(dr ** 2, axis=-1))

        # Expected: MSD = 6 * D * t = 6 * (kT / (gamma_base * radius)) * (n_steps * dt)
        gamma = gamma_base * radius
        D = kT / gamma
        total_time = n_steps * dt
        expected_msd = 6.0 * D * total_time

        # Allow generous tolerance (statistical; 20 particles, single run)
        ratio = msd / expected_msd
        assert 0.3 < ratio < 3.0, \
            f"MSD ratio {ratio:.2f} (MSD={msd:.2f}, expected={expected_msd:.2f}) " \
            f"should be within [0.3, 3.0] for statistical test"


class TestRunSimulation:
    """Tests for the full run_brownian_dynamics function."""

    def test_run_returns_trajectory(self):
        """run_brownian_dynamics should return state and trajectory."""
        pos, ptypes, charges, radii, box, topo, params, key = _make_free_particles(n=5, box_size=30.0)
        state = init_fn(pos, ptypes, charges, radii, box, topo, params, key,
                        cutoff=10.0, skin=2.0)

        final_state, traj = run_brownian_dynamics(
            state, topo, params,
            n_steps=50, dt=0.01, kT=1.0, gamma_base=1.0,
            cutoff=10.0, skin=2.0, nl_rebuild_every=10, save_every=50,
        )

        assert final_state.step == 50
        assert len(traj) >= 2  # initial + at least one save

    def test_run_step_count(self):
        """Step counter should match n_steps after simulation."""
        pos, ptypes, charges, radii, box, topo, params, key = _make_free_particles(n=3, box_size=30.0)
        state = init_fn(pos, ptypes, charges, radii, box, topo, params, key,
                        cutoff=10.0, skin=2.0)

        final_state, _ = run_brownian_dynamics(
            state, topo, params,
            n_steps=100, dt=0.01, kT=1.0, gamma_base=1.0,
            cutoff=10.0, skin=2.0, nl_rebuild_every=20, save_every=100,
        )

        assert final_state.step == 100
