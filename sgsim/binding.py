"""Saturable competitive binding via hybrid MC+MD.

Binding is inherently discrete (site occupied or not). We use Metropolis
Monte Carlo moves interleaved with MD to correctly sample the binding
ensemble.

Algorithm (every binding_interval MD steps):
1. Pick a random site i
2. If occupied: propose unbinding
3. If unoccupied: find compatible, unoccupied, nearby sites → propose binding
4. Accept/reject via Metropolis: p = min(1, exp(-dE/kT))
5. Repeat for n_attempts using jax.lax.fori_loop

NTF2 pocket competition emerges naturally: once occupied (row-sum=1),
no other site can bind. Relative binding energies control preference.
"""

import jax
import jax.numpy as jnp
from functools import partial

from .types import BindingState
from .utils import safe_norm


def init_binding_state(n_sites: int, max_bonds: int = 1000) -> BindingState:
    """Initialize an empty binding state.

    Args:
        n_sites: total number of binding sites in the system
        max_bonds: maximum simultaneous binding events

    Returns:
        BindingState with no active bonds
    """
    return BindingState(
        bound_pairs=jnp.zeros((max_bonds, 2), dtype=jnp.int32),
        bound_mask=jnp.zeros(max_bonds, dtype=bool),
        n_bound=jnp.int32(0),
        site_occupied=jnp.zeros(n_sites, dtype=jnp.int32),
        site_partner=jnp.full(n_sites, -1, dtype=jnp.int32),
    )


def compute_binding_energy(
    positions: jnp.ndarray,
    box_size: jnp.ndarray,
    binding_state: BindingState,
    site_particle: jnp.ndarray,
    binding_energy_matrix: jnp.ndarray,
    site_types: jnp.ndarray,
    binding_cutoff_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """Compute total binding energy from all active bound pairs.

    Each bound pair (i, j) contributes:
      E_bind = binding_energy[type_i, type_j] * smooth_factor(r)
    where r is the distance between the parent particles.

    Args:
        positions: (N, 3) particle positions
        box_size: (3,)
        binding_state: current BindingState
        site_particle: (N_sites,) int32 — particle owning each site
        binding_energy_matrix: (N_site_types, N_site_types) — binding free energies (kT)
        site_types: (N_sites,) int32
        binding_cutoff_matrix: (N_site_types, N_site_types) — cutoff distances

    Returns:
        Scalar binding energy
    """
    # Get bound pair site indices
    pair_site_i = binding_state.bound_pairs[:, 0]  # (max_bonds,)
    pair_site_j = binding_state.bound_pairs[:, 1]  # (max_bonds,)

    # Parent particles
    part_i = site_particle[pair_site_i]
    part_j = site_particle[pair_site_j]

    # Positions of parent particles
    pos_i = positions[part_i]  # (max_bonds, 3)
    pos_j = positions[part_j]  # (max_bonds, 3)

    # PBC displacement
    dr = pos_i - pos_j
    dr = dr - box_size * jnp.round(dr / box_size)
    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-10)  # (max_bonds,)

    # Binding energy lookup
    type_i = site_types[pair_site_i]
    type_j = site_types[pair_site_j]
    e_bind = binding_energy_matrix[type_i, type_j]  # (max_bonds,)

    # Smooth distance weighting: 1 at close range, 0 beyond cutoff
    r_cut = binding_cutoff_matrix[type_i, type_j]
    # Simple smooth cutoff
    weight = jnp.where(dist < r_cut, 1.0, 0.0)

    # Total: sum over active bonds
    energies = e_bind * weight * binding_state.bound_mask
    return jnp.sum(energies)


def _compute_pair_energy(
    site_i: int,
    site_j: int,
    positions: jnp.ndarray,
    box_size: jnp.ndarray,
    site_particle: jnp.ndarray,
    site_types: jnp.ndarray,
    binding_energy_matrix: jnp.ndarray,
    binding_cutoff_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """Compute binding energy contribution from a single site pair."""
    part_i = site_particle[site_i]
    part_j = site_particle[site_j]

    dr = positions[part_i] - positions[part_j]
    dr = dr - box_size * jnp.round(dr / box_size)
    dist = jnp.sqrt(jnp.sum(dr ** 2) + 1e-10)

    type_i = site_types[site_i]
    type_j = site_types[site_j]

    e_bind = binding_energy_matrix[type_i, type_j]
    r_cut = binding_cutoff_matrix[type_i, type_j]

    return jnp.where(dist < r_cut, e_bind, 0.0)


def mc_binding_step(
    binding_state: BindingState,
    positions: jnp.ndarray,
    box_size: jnp.ndarray,
    site_particle: jnp.ndarray,
    site_types: jnp.ndarray,
    site_molecule: jnp.ndarray,
    binding_energy_matrix: jnp.ndarray,
    binding_cutoff_matrix: jnp.ndarray,
    binding_compatibility: jnp.ndarray,
    kT: float,
    n_attempts: int,
    rng_key: jnp.ndarray,
) -> tuple:
    """Perform multiple MC binding/unbinding attempts.

    Args:
        binding_state: current BindingState
        positions: (N, 3) particle positions
        box_size: (3,)
        site_particle: (N_sites,) — owning particle for each site
        site_types: (N_sites,) — type of each site
        site_molecule: (N_sites,) — owning molecule for each site
        binding_energy_matrix: (N_site_types, N_site_types) — energies
        binding_cutoff_matrix: (N_site_types, N_site_types) — cutoffs
        binding_compatibility: (N_site_types, N_site_types) — 1 if can bind
        kT: thermal energy
        n_attempts: number of MC attempts
        rng_key: JAX PRNG key

    Returns:
        (new_binding_state, new_rng_key)
    """
    n_sites = site_particle.shape[0]

    def attempt_body(carry, _):
        state, key = carry
        key, site_key, cand_key, accept_key = jax.random.split(key, 4)

        # Pick a random site
        site_i = jax.random.randint(site_key, (), 0, n_sites)

        # Is it currently bound?
        is_occupied = state.site_occupied[site_i] > 0

        # --- Unbinding proposal ---
        partner = state.site_partner[site_i]
        unbind_energy = _compute_pair_energy(
            site_i, jnp.maximum(partner, 0),  # clamp for safety
            positions, box_size, site_particle, site_types,
            binding_energy_matrix, binding_cutoff_matrix,
        )
        # dE for unbinding = -E_bind (removing the bond)
        dE_unbind = -unbind_energy

        # --- Binding proposal ---
        # Find compatible, unoccupied, nearby, different-molecule sites
        type_i = site_types[site_i]
        mol_i = site_molecule[site_i]
        part_i = site_particle[site_i]

        # Compatibility mask
        compatible = binding_compatibility[type_i, site_types] > 0.5  # (N_sites,) bool
        unoccupied = (state.site_occupied == 0)  # (N_sites,)
        different_mol = (site_molecule != mol_i)  # (N_sites,)
        not_self = (jnp.arange(n_sites) != site_i)  # (N_sites,)

        # Distance check
        pos_i = positions[part_i]
        all_part_pos = positions[site_particle]  # (N_sites, 3)
        dr = pos_i - all_part_pos
        dr = dr - box_size * jnp.round(dr / box_size)
        dists = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-10)
        cutoffs = binding_cutoff_matrix[type_i, site_types]
        within_range = dists < cutoffs

        # Combined candidate mask
        candidates = compatible & unoccupied & different_mol & not_self & within_range
        n_candidates = jnp.sum(candidates)

        # Select a random candidate (if any exist)
        # Use Gumbel trick for differentiable sampling
        gumbel_noise = -jnp.log(-jnp.log(
            jax.random.uniform(cand_key, (n_sites,)) + 1e-10
        ) + 1e-10)
        # Set non-candidates to -inf
        scores = jnp.where(candidates, gumbel_noise, -1e10)
        site_j = jnp.argmax(scores)

        bind_energy = _compute_pair_energy(
            site_i, site_j, positions, box_size, site_particle, site_types,
            binding_energy_matrix, binding_cutoff_matrix,
        )
        # dE for binding = E_bind (adding the bond)
        dE_bind = bind_energy

        # Choose proposal type
        dE = jnp.where(is_occupied, dE_unbind, dE_bind)
        has_candidate = jnp.where(is_occupied, True, n_candidates > 0)

        # Metropolis acceptance
        accept_prob = jnp.minimum(1.0, jnp.exp(-dE / kT))
        rand_val = jax.random.uniform(accept_key)
        accept = (rand_val < accept_prob) & has_candidate

        # Apply move
        new_state = jax.lax.cond(
            accept & is_occupied,
            lambda s: _apply_unbind(s, site_i),
            lambda s: s,
            state,
        )
        new_state = jax.lax.cond(
            accept & (~is_occupied) & (n_candidates > 0),
            lambda s: _apply_bind(s, site_i, site_j),
            lambda s: s,
            new_state,
        )

        return (new_state, key), None

    (new_state, new_key), _ = jax.lax.scan(
        attempt_body, (binding_state, rng_key), None, length=n_attempts,
    )

    return new_state, new_key


def _apply_bind(state: BindingState, site_i: int, site_j: int) -> BindingState:
    """Apply a binding event: mark both sites as occupied and record the pair."""
    n_bound = state.n_bound
    # Add to bound_pairs at position n_bound
    new_pairs = state.bound_pairs.at[n_bound, 0].set(site_i)
    new_pairs = new_pairs.at[n_bound, 1].set(site_j)
    new_mask = state.bound_mask.at[n_bound].set(True)

    new_occupied = state.site_occupied.at[site_i].set(1)
    new_occupied = new_occupied.at[site_j].set(1)
    new_partner = state.site_partner.at[site_i].set(site_j)
    new_partner = new_partner.at[site_j].set(site_i)

    return BindingState(
        bound_pairs=new_pairs,
        bound_mask=new_mask,
        n_bound=n_bound + 1,
        site_occupied=new_occupied,
        site_partner=new_partner,
    )


def _apply_unbind(state: BindingState, site_i: int) -> BindingState:
    """Apply an unbinding event: free both sites and remove the pair."""
    partner = state.site_partner[site_i]

    # Find which bound_pair entry corresponds to this bond
    # Check both orderings
    is_match = (
        ((state.bound_pairs[:, 0] == site_i) & (state.bound_pairs[:, 1] == partner)) |
        ((state.bound_pairs[:, 0] == partner) & (state.bound_pairs[:, 1] == site_i))
    ) & state.bound_mask

    new_mask = jnp.where(is_match, False, state.bound_mask)

    new_occupied = state.site_occupied.at[site_i].set(0)
    new_occupied = new_occupied.at[partner].set(0)
    new_partner = state.site_partner.at[site_i].set(-1)
    new_partner = new_partner.at[partner].set(-1)

    return BindingState(
        bound_pairs=state.bound_pairs,
        bound_mask=new_mask,
        n_bound=state.n_bound - 1,
        site_occupied=new_occupied,
        site_partner=new_partner,
    )
