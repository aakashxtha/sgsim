#!/usr/bin/env python
"""Lean production simulation — BD + binding + conformational switching only.

No cluster detection, no analysis plots. Just runs the physics and saves
trajectory to .xyz and .zarr as fast as possible.

Output: results/sim_only/trajectory.xyz  (51 frames, ~50-80 MB)
        results/sim_only/trajectory.zarr (~10-20 MB compressed)
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgsim.setup import build_system
from sgsim.simulate import init_fn, make_step_fn
from sgsim.partition import build_neighbor_list, neighbor_pairs, needs_rebuild
from sgsim.binding import init_binding_state, mc_binding_step
from sgsim.conformational import (
    init_conformational_state, update_conformational_state,
    count_rna_bound_per_molecule,
)
from sgsim.parameters import default_params
from sgsim.io import save_trajectory_zarr, export_xyz
from sgsim.types import PARTICLE_TYPES as PT, BINDING_SITE_TYPES as BST


# ── Configuration ─────────────────────────────────────────────────────────────

COMPOSITION = {
    "g3bp1_dimer": 250,
    "rna":         500,
}
BOX_SIZE        = 150.0   # nm
N_STEPS         = 100_000
SAVE_EVERY      = 2_000   # frames: 100k / 2k = 50 frames
DT              = 0.005
KT              = 1.0
GAMMA_BASE      = 1.0
CUTOFF          = 8.0     # nm
SKIN            = 2.0     # nm
MAX_NEIGHBORS   = 128
NL_REBUILD_EVERY    = 10
BINDING_INTERVAL    = 2
N_BINDING_ATTEMPTS  = 200
CONF_INTERVAL       = 5
GAMMA_PHI           = 1.0
OUTDIR = "results/sim_only"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SGSIM — Lean production run (no analysis)")
    print("=" * 70)
    print(f"Backend : {jax.default_backend().upper()}  {jax.devices()[0]}")
    print(f"System  : {COMPOSITION}")
    print(f"Box     : {BOX_SIZE} nm  |  Steps: {N_STEPS:,}  |  Frames: {N_STEPS // SAVE_EVERY}")
    print("-" * 70)

    # ── Build system ──────────────────────────────────────────────────────────
    t0 = time.time()
    key = jax.random.PRNGKey(42)
    system = build_system(COMPOSITION, BOX_SIZE, key, rna_beads=15, rna_exposure=1.0)

    positions       = system["positions"]
    particle_types  = system["particle_types"]
    particle_charges= system["particle_charges"]
    particle_radii  = system["particle_radii"]
    molecule_ids    = system["molecule_ids"]
    topology        = system["topology"]
    box_size        = system["box_size"]
    params          = default_params()
    n_particles     = positions.shape[0]

    print(f"Built {n_particles:,} beads, {topology.n_bonds:,} bonds, "
          f"{topology.n_sites:,} binding sites  [{time.time()-t0:.1f}s]")

    # ── Init integrator ───────────────────────────────────────────────────────
    key, init_key = jax.random.split(key)
    state = init_fn(
        positions, particle_types, particle_charges, particle_radii,
        box_size, topology, params, init_key, CUTOFF, SKIN, MAX_NEIGHBORS,
    )

    # ── Init binding ──────────────────────────────────────────────────────────
    n_sites = topology.n_sites
    binding_state = init_binding_state(n_sites, max_bonds=max(n_sites, 100))

    # ── Init conformational (G3BP1 molecules with acidic + RG IDR) ───────────
    acidic_type, rg_type = PT["ACIDIC_IDR"], PT["RG_IDR"]
    has_acidic, has_rg = {}, {}
    ptypes_np = np.asarray(particle_types)
    mol_np    = np.asarray(molecule_ids)

    for i in range(n_particles):
        mid, pt = int(mol_np[i]), int(ptypes_np[i])
        if pt == acidic_type and mid not in has_acidic:
            has_acidic[mid] = i
        if pt == rg_type and mid not in has_rg:
            has_rg[mid] = i

    switchable_mols, acidic_idx, rg_idx = [], [], []
    for mid in sorted(has_acidic):
        if mid in has_rg:
            switchable_mols.append(mid)
            acidic_idx.append(has_acidic[mid])
            rg_idx.append(has_rg[mid])

    conf_state = None
    if switchable_mols:
        conf_state = init_conformational_state(
            len(switchable_mols),
            jnp.array(switchable_mols, dtype=jnp.int32),
            jnp.array(acidic_idx,      dtype=jnp.int32),
            jnp.array(rg_idx,          dtype=jnp.int32),
            initial_openness=0.0,
        )

    # ── Simulation loop ───────────────────────────────────────────────────────
    step_fn = make_step_fn(topology, params, DT, KT, GAMMA_BASE, CUTOFF)

    def scan_body(s, _):
        return step_fn(s), None

    trajectory   = [np.asarray(positions)]
    current_step = 0
    t_sim = time.time()

    print("\nRunning (first step includes JIT compilation) ...\n")

    while current_step < N_STEPS:
        chunk = min(NL_REBUILD_EVERY, N_STEPS - current_step)
        state, _ = jax.lax.scan(scan_body, state, None, length=chunk)
        current_step += chunk

        # Neighbor list rebuild
        if needs_rebuild(state.positions, state.nl_reference_positions, box_size, SKIN):
            nl = build_neighbor_list(state.positions, box_size, CUTOFF, SKIN, MAX_NEIGHBORS)
            pi, pj, pm = neighbor_pairs(nl)
            state = state._replace(
                nl_pair_i=pi, nl_pair_j=pj, nl_pair_mask=pm,
                nl_reference_positions=state.positions,
            )

        # Binding MC
        if current_step % BINDING_INTERVAL == 0 and n_sites > 0:
            key, bk = jax.random.split(key)
            binding_state, key = mc_binding_step(
                binding_state, state.positions, box_size,
                topology.binding_site_particle, topology.binding_site_type,
                topology.binding_site_molecule,
                params.binding_energy, params.binding_cutoff,
                params.binding_compatibility,
                KT, N_BINDING_ATTEMPTS, bk,
            )

        # Conformational switching
        if current_step % CONF_INTERVAL == 0 and conf_state is not None:
            key, ck = jax.random.split(key)
            n_rna = count_rna_bound_per_molecule(
                binding_state, topology.binding_site_type,
                topology.binding_site_molecule,
                conf_state.molecule_indices, BST["RNA_BINDING_SITE"],
            )
            conf_state, key = update_conformational_state(
                conf_state, state.positions, box_size,
                float(params.k_compact), float(params.r0_compact),
                float(params.eps_rna_expand),
                n_rna, GAMMA_PHI, KT, DT * CONF_INTERVAL, ck,
            )

        # Save frame + print progress
        if current_step % SAVE_EVERY == 0 or current_step >= N_STEPS:
            trajectory.append(np.asarray(state.positions))
            n_bound  = int(binding_state.n_bound)
            phi_mean = float(jnp.mean(conf_state.openness)) if conf_state is not None else 0.0
            elapsed  = time.time() - t_sim
            sps      = current_step / max(elapsed, 1e-6)
            print(f"Step {current_step:>7,}/{N_STEPS:,} | "
                  f"Bonds: {n_bound:>4d} | "
                  f"<phi>: {phi_mean:.3f} | "
                  f"{sps:.1f} steps/s")

    elapsed = time.time() - t_sim
    print(f"\nDone in {elapsed:.1f}s  ({N_STEPS/elapsed:.1f} steps/s)")

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(OUTDIR, exist_ok=True)

    xyz_path  = f"{OUTDIR}/trajectory.xyz"
    zarr_path = f"{OUTDIR}/trajectory.zarr"

    print(f"\nSaving {len(trajectory)} frames ...")
    export_xyz(xyz_path, np.stack(trajectory), particle_types, box_size=box_size)
    save_trajectory_zarr(
        zarr_path, trajectory, particle_types, box_size,
        metadata={"composition": str(COMPOSITION), "n_steps": N_STEPS,
                  "dt": DT, "box_size": BOX_SIZE},
    )
    print(f"  trajectory.xyz  → {xyz_path}")
    print(f"  trajectory.zarr → {zarr_path}")
    print("\nOpen trajectory.xyz in OVITO to visualize the droplet.")


if __name__ == "__main__":
    main()
