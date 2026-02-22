"""Brownian dynamics (overdamped Langevin) integrator.

Implements the init_fn / step_fn pattern:
  state = init_fn(positions, ...)
  state = step_fn(state)

The inner simulation loop uses jax.lax.scan for JIT-compiled stepping
without Python overhead. An outer Python loop handles I/O.

Physics:
  dx = (F / gamma) * dt + sqrt(2 * kT * dt / gamma) * noise
  where gamma_i = gamma_base * radius_i (Stokes drag)
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

from .energy import (
    harmonic_bond_energy,
    harmonic_angle_energy,
    nonbonded_energy_bruteforce,
    nonbonded_energy_neighborlist,
    build_bond_exclusion_mask,
)
from .partition import build_neighbor_list, neighbor_pairs, needs_rebuild
from .space import periodic_wrap


class IntegratorState(NamedTuple):
    """State carried through the integrator loop."""
    positions: jnp.ndarray         # (N, 3)
    particle_types: jnp.ndarray    # (N,) int32
    particle_charges: jnp.ndarray  # (N,) float32
    particle_radii: jnp.ndarray    # (N,) float32
    box_size: jnp.ndarray          # (3,)
    rng_key: jnp.ndarray
    step: jnp.ndarray              # scalar int32
    # Neighbor list data (flattened for scan compatibility)
    nl_pair_i: jnp.ndarray         # (max_pairs,) int32
    nl_pair_j: jnp.ndarray         # (max_pairs,) int32
    nl_pair_mask: jnp.ndarray      # (max_pairs,) bool
    bond_exclusion_mask: jnp.ndarray  # (N, N) bool
    nl_reference_positions: jnp.ndarray  # (N, 3)


def _compute_energy(
    positions, particle_types, particle_charges, topology, params,
    box_size, cutoff, nl_pair_i, nl_pair_j, nl_pair_mask, bond_excl_mask,
):
    """Compute total potential energy (bonded + non-bonded)."""
    e_bond = harmonic_bond_energy(
        positions, topology.bond_pairs, topology.bond_types,
        params.bond_k, params.bond_r0, box_size,
    )
    e_angle = harmonic_angle_energy(
        positions, topology.angle_triples, topology.angle_types,
        params.angle_k, params.angle_theta0, box_size,
    )
    e_nonbond = nonbonded_energy_neighborlist(
        positions, particle_types, particle_charges,
        params.epsilon_attract, params.sigma,
        params.kappa, params.debye_length,
        box_size, cutoff, bond_excl_mask,
        nl_pair_i, nl_pair_j, nl_pair_mask,
    )
    return e_bond + e_angle + e_nonbond


def init_fn(
    positions: jnp.ndarray,
    particle_types: jnp.ndarray,
    particle_charges: jnp.ndarray,
    particle_radii: jnp.ndarray,
    box_size: jnp.ndarray,
    topology,
    params,
    rng_key: jnp.ndarray,
    cutoff: float = 10.0,
    skin: float = 2.0,
    max_neighbors: int = 128,
) -> IntegratorState:
    """Initialize the integrator state.

    Builds the initial neighbor list and prepares all arrays for stepping.

    Args:
        positions: (N, 3) initial particle positions
        particle_types: (N,) int32
        particle_charges: (N,) float32
        particle_radii: (N,) float32 — used for per-particle friction
        box_size: (3,)
        topology: Topology NamedTuple
        params: InteractionParams NamedTuple
        rng_key: JAX PRNG key
        cutoff: interaction cutoff (nm)
        skin: neighbor list skin distance (nm)
        max_neighbors: max neighbors per particle

    Returns:
        IntegratorState ready for stepping
    """
    n = positions.shape[0]

    # Build neighbor list
    nl = build_neighbor_list(positions, box_size, cutoff, skin, max_neighbors)
    pair_i, pair_j, pair_mask = neighbor_pairs(nl)

    # Build bond exclusion mask
    bond_excl = build_bond_exclusion_mask(n, topology.bond_pairs)

    return IntegratorState(
        positions=positions,
        particle_types=particle_types,
        particle_charges=particle_charges,
        particle_radii=particle_radii,
        box_size=box_size,
        rng_key=rng_key,
        step=jnp.int32(0),
        nl_pair_i=pair_i,
        nl_pair_j=pair_j,
        nl_pair_mask=pair_mask,
        bond_exclusion_mask=bond_excl,
        nl_reference_positions=positions,
    )


def make_step_fn(topology, params, dt=0.01, kT=1.0, gamma_base=1.0, cutoff=10.0):
    """Create a JIT-compiled step function for Brownian dynamics.

    Args:
        topology: Topology NamedTuple (static, captured in closure)
        params: InteractionParams NamedTuple (static, captured in closure)
        dt: timestep (reduced units)
        kT: thermal energy
        gamma_base: base friction coefficient (gamma_i = gamma_base * radius_i)
        cutoff: non-bonded cutoff (nm)

    Returns:
        step_fn(state) -> state: a single BD step
    """

    def energy_fn(positions, state):
        """Total energy as a function of positions."""
        return _compute_energy(
            positions, state.particle_types, state.particle_charges,
            topology, params, state.box_size, cutoff,
            state.nl_pair_i, state.nl_pair_j, state.nl_pair_mask,
            state.bond_exclusion_mask,
        )

    @jax.jit
    def step_fn(state: IntegratorState) -> IntegratorState:
        """Perform one Brownian dynamics step.

        dx = (F / gamma) * dt + sqrt(2 * kT * dt / gamma) * noise
        """
        # Compute forces via autodiff
        grad_fn = jax.grad(lambda pos: energy_fn(pos, state))
        forces = -grad_fn(state.positions)  # (N, 3)

        # Per-particle friction: gamma_i = gamma_base * radius_i
        gamma = gamma_base * state.particle_radii[:, None]  # (N, 1) for broadcasting

        # Split RNG key
        key, noise_key = jax.random.split(state.rng_key)

        # Deterministic drift
        drift = (forces / gamma) * dt  # (N, 3)

        # Stochastic noise
        noise_scale = jnp.sqrt(2.0 * kT * dt / gamma)  # (N, 1)
        noise = jax.random.normal(noise_key, state.positions.shape) * noise_scale

        # Position update
        new_positions = state.positions + drift + noise

        # Wrap into periodic box
        new_positions = periodic_wrap(new_positions, state.box_size)

        return IntegratorState(
            positions=new_positions,
            particle_types=state.particle_types,
            particle_charges=state.particle_charges,
            particle_radii=state.particle_radii,
            box_size=state.box_size,
            rng_key=key,
            step=state.step + 1,
            nl_pair_i=state.nl_pair_i,
            nl_pair_j=state.nl_pair_j,
            nl_pair_mask=state.nl_pair_mask,
            bond_exclusion_mask=state.bond_exclusion_mask,
            nl_reference_positions=state.nl_reference_positions,
        )

    return step_fn


def run_brownian_dynamics(
    state: IntegratorState,
    topology,
    params,
    n_steps: int,
    dt: float = 0.01,
    kT: float = 1.0,
    gamma_base: float = 1.0,
    cutoff: float = 10.0,
    skin: float = 2.0,
    max_neighbors: int = 128,
    nl_rebuild_every: int = 10,
    save_every: int = 100,
) -> tuple:
    """Run a Brownian dynamics simulation with periodic neighbor list rebuilds.

    Uses an inner jax.lax.scan loop for JIT-compiled stepping. The outer
    Python loop handles neighbor list rebuilds and trajectory saving.

    Args:
        state: initial IntegratorState
        topology: Topology NamedTuple
        params: InteractionParams NamedTuple
        n_steps: total number of steps
        dt: timestep
        kT: thermal energy
        gamma_base: base friction
        cutoff: non-bonded cutoff (nm)
        skin: neighbor list skin (nm)
        max_neighbors: max neighbors per particle
        nl_rebuild_every: steps between neighbor list checks
        save_every: steps between trajectory saves

    Returns:
        (final_state, trajectory_positions) where trajectory_positions
        is a list of (N, 3) arrays saved every save_every steps.
    """
    step_fn = make_step_fn(topology, params, dt, kT, gamma_base, cutoff)

    # Inner scan: run nl_rebuild_every steps at a time
    def scan_body(state, _):
        return step_fn(state), None

    trajectory = [state.positions]
    current_step = 0

    while current_step < n_steps:
        # How many steps in this chunk
        chunk = min(nl_rebuild_every, n_steps - current_step)

        # Run chunk steps via scan
        state, _ = jax.lax.scan(scan_body, state, None, length=chunk)
        current_step += chunk

        # Check if neighbor list needs rebuild
        if needs_rebuild(state.positions, state.nl_reference_positions,
                         state.box_size, skin):
            nl = build_neighbor_list(state.positions, state.box_size, cutoff, skin, max_neighbors)
            if bool(nl.overflow):
                import warnings
                warnings.warn(
                    f"Neighbor list overflow: some particles have more than "
                    f"{max_neighbors} neighbors. Increase max_neighbors to "
                    f"avoid missing interactions.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            pair_i, pair_j, pair_mask = neighbor_pairs(nl)
            state = state._replace(
                nl_pair_i=pair_i,
                nl_pair_j=pair_j,
                nl_pair_mask=pair_mask,
                nl_reference_positions=state.positions,
            )

        # Save trajectory
        if current_step % save_every == 0 or current_step >= n_steps:
            trajectory.append(state.positions)

    return state, trajectory


# ---------------------------------------------------------------------------
# Full simulation: BD + binding MC + conformational switching
# ---------------------------------------------------------------------------

def run_full_simulation(
    system: dict,
    n_steps: int = 10000,
    dt: float = 0.01,
    kT: float = 1.0,
    gamma_base: float = 1.0,
    cutoff: float = 10.0,
    skin: float = 2.0,
    max_neighbors: int = 128,
    nl_rebuild_every: int = 10,
    save_every: int = 100,
    binding_interval: int = 5,
    n_binding_attempts: int = 10,
    conformational_interval: int = 5,
    gamma_phi: float = 1.0,
    rng_key=None,
    verbose: bool = True,
) -> dict:
    """Run a complete simulation with BD + binding MC + conformational switching.

    Combines all physics modules:
    1. Brownian dynamics (overdamped Langevin) for positions
    2. MC binding/unbinding for discrete saturable interactions
    3. Overdamped Langevin in phi-space for G3BP1 conformational switching

    Args:
        system: dict from build_system() with positions, topology, etc.
        n_steps: total simulation steps
        dt: BD timestep
        kT: thermal energy
        gamma_base: base friction (gamma_i = gamma_base * radius_i)
        cutoff: non-bonded interaction cutoff (nm)
        skin: neighbor list skin (nm)
        max_neighbors: max neighbors per particle
        nl_rebuild_every: steps between NL rebuild checks
        save_every: steps between trajectory snapshots
        binding_interval: steps between binding MC rounds
        n_binding_attempts: MC attempts per binding round
        conformational_interval: steps between phi updates
        gamma_phi: friction in phi-space
        rng_key: JAX PRNG key (auto-generated if None)
        verbose: print progress

    Returns:
        dict with final_positions, trajectory, binding_state,
        conformational_state, and history arrays.
    """
    from .binding import init_binding_state, mc_binding_step
    from .conformational import (
        init_conformational_state, update_conformational_state,
        count_rna_bound_per_molecule,
    )
    from .analysis import detect_clusters, cluster_statistics
    from .types import PARTICLE_TYPES as PT, BINDING_SITE_TYPES as BST
    from .parameters import default_params

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    topology = system["topology"]
    params = default_params()
    positions = system["positions"]
    box_size = system["box_size"]
    particle_types = system["particle_types"]
    particle_charges = system["particle_charges"]
    particle_radii = system["particle_radii"]
    molecule_ids = system["molecule_ids"]

    # --- Initialize integrator state ---
    key, init_key = jax.random.split(rng_key)
    state = init_fn(
        positions, particle_types, particle_charges, particle_radii,
        box_size, topology, params, init_key, cutoff, skin, max_neighbors,
    )

    # --- Initialize binding state ---
    n_sites = topology.n_sites
    max_bonds = max(n_sites, 100)
    binding_state = init_binding_state(n_sites, max_bonds)

    # --- Initialize conformational state for G3BP1 dimers ---
    # Find molecules containing both acidic IDR and RG IDR beads
    n_particles = positions.shape[0]
    acidic_type = PT["ACIDIC_IDR"]
    rg_type = PT["RG_IDR"]

    has_acidic = {}
    has_rg = {}
    ptypes_np = jnp.asarray(particle_types)
    mol_np = jnp.asarray(molecule_ids)

    switchable_mols = []
    acidic_indices = []
    rg_indices = []

    for i in range(n_particles):
        mid = int(mol_np[i])
        pt = int(ptypes_np[i])
        if pt == acidic_type and mid not in has_acidic:
            has_acidic[mid] = i
        if pt == rg_type and mid not in has_rg:
            has_rg[mid] = i

    for mid in sorted(has_acidic.keys()):
        if mid in has_rg:
            switchable_mols.append(mid)
            acidic_indices.append(has_acidic[mid])
            rg_indices.append(has_rg[mid])

    n_switchable = len(switchable_mols)
    conf_state = None

    if n_switchable > 0:
        conf_state = init_conformational_state(
            n_switchable,
            jnp.array(switchable_mols, dtype=jnp.int32),
            jnp.array(acidic_indices, dtype=jnp.int32),
            jnp.array(rg_indices, dtype=jnp.int32),
            initial_openness=0.0,
        )

    # --- Create step function ---
    step_fn = make_step_fn(topology, params, dt, kT, gamma_base, cutoff)

    # --- Histories ---
    trajectory = [positions]
    binding_history = [0]
    openness_history = [0.0 if n_switchable == 0 else float(jnp.mean(conf_state.openness))]
    cluster_history = []

    # --- Main loop ---
    current_step = 0

    while current_step < n_steps:
        # BD steps (chunk at a time)
        chunk = min(nl_rebuild_every, n_steps - current_step)

        def scan_body(s, _):
            return step_fn(s), None

        state, _ = jax.lax.scan(scan_body, state, None, length=chunk)
        current_step += chunk

        # Neighbor list rebuild check
        if needs_rebuild(state.positions, state.nl_reference_positions,
                         state.box_size, skin):
            nl = build_neighbor_list(state.positions, state.box_size, cutoff, skin, max_neighbors)
            if bool(nl.overflow):
                import warnings
                warnings.warn(
                    f"Neighbor list overflow: some particles have more than "
                    f"{max_neighbors} neighbors. Increase max_neighbors to "
                    f"avoid missing interactions.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            pair_i, pair_j, pair_mask = neighbor_pairs(nl)
            state = state._replace(
                nl_pair_i=pair_i,
                nl_pair_j=pair_j,
                nl_pair_mask=pair_mask,
                nl_reference_positions=state.positions,
            )

        # Binding MC moves
        if current_step % binding_interval == 0 and n_sites > 0:
            key, bind_key = jax.random.split(key)
            binding_state, bind_key = mc_binding_step(
                binding_state, state.positions, box_size,
                topology.binding_site_particle,
                topology.binding_site_type,
                topology.binding_site_molecule,
                params.binding_energy,
                params.binding_cutoff,
                params.binding_compatibility,
                kT, n_binding_attempts, bind_key,
            )
            key = bind_key

        # Conformational switching
        if current_step % conformational_interval == 0 and conf_state is not None:
            key, conf_key = jax.random.split(key)
            n_rna = count_rna_bound_per_molecule(
                binding_state,
                topology.binding_site_type,
                topology.binding_site_molecule,
                conf_state.molecule_indices,
                BST["RNA_BINDING_SITE"],
            )
            conf_state, conf_key = update_conformational_state(
                conf_state, state.positions, box_size,
                float(params.k_compact), float(params.r0_compact),
                float(params.eps_rna_expand),
                n_rna, gamma_phi, kT, dt * conformational_interval,
                conf_key,
            )
            key = conf_key

        # Save trajectory and stats
        if current_step % save_every == 0 or current_step >= n_steps:
            trajectory.append(state.positions)
            binding_history.append(int(binding_state.n_bound))

            if conf_state is not None:
                openness_history.append(float(jnp.mean(conf_state.openness)))
            else:
                openness_history.append(0.0)

            # Cluster detection
            labels = detect_clusters(state.positions, box_size, cutoff=5.0)
            stats = cluster_statistics(labels, state.positions, box_size, particle_types)
            cluster_history.append((
                int(stats["n_clusters"]),
                int(stats["largest_cluster_size"]),
            ))

            if verbose:
                n_cl, largest = cluster_history[-1]
                n_bound = binding_history[-1]
                phi_mean = openness_history[-1]
                frac = largest / n_particles
                print(
                    f"Step {current_step:>6d}/{n_steps} | "
                    f"Bonds: {n_bound:>3d} | "
                    f"Clusters: {n_cl:>3d} | "
                    f"Largest: {largest:>4d} ({frac:.1%}) | "
                    f"<phi>: {phi_mean:.3f}"
                )

    return {
        "final_positions": state.positions,
        "trajectory": trajectory,
        "binding_state": binding_state,
        "conformational_state": conf_state,
        "binding_history": binding_history,
        "openness_history": openness_history,
        "cluster_history": cluster_history,
        "box_size": box_size,
        "particle_types": particle_types,
        "molecule_ids": molecule_ids,
    }
