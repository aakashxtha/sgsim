#!/usr/bin/env python
"""Diagnostic: check bond distances in the dense simulation final state."""

import jax.numpy as jnp
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgsim.io import load_trajectory_zarr
from sgsim.setup import build_system
from sgsim.space import periodic_displacement
import jax

# Rebuild the system to get topology (bond pairs + types)
key = jax.random.PRNGKey(42)
composition = {"g3bp1_dimer": 20, "rna": 50}
system = build_system(composition, 50.0, key, rna_beads=10, rna_exposure=1.0)
topo = system["topology"]

# Load trajectory
traj = load_trajectory_zarr("results/dense_phase_sep/trajectory.zarr")
positions = traj["positions"]  # (n_frames, N, 3)
box = traj["box_size"]

print(f"Frames: {positions.shape[0]}, Particles: {positions.shape[1]}")
print(f"Bonds: {topo.n_bonds}")
print(f"Box: {np.asarray(box)}")

bond_pairs = np.asarray(topo.bond_pairs)  # (n_bonds, 2)
bond_types = np.asarray(topo.bond_types)

# Bond type names
bond_type_names = ["DOMAIN_LINKER_STIFF", "IDR_LINKER_FLEX", "DIMER_BOND", "RNA_BACKBONE"]

# Check bond distances for first, middle, and last frames
disp_fn = periodic_displacement(box)

for frame_idx in [0, positions.shape[0] // 2, -1]:
    pos = positions[frame_idx]
    label = {0: "Initial", positions.shape[0]//2: "Middle", -1: "Final"}[frame_idx]

    # Compute minimum-image distances for all bonds
    r_i = pos[bond_pairs[:, 0]]  # (n_bonds, 3)
    r_j = pos[bond_pairs[:, 1]]
    dr = r_i - r_j
    dr = dr - box * jnp.round(dr / box)
    dists = jnp.sqrt(jnp.sum(dr**2, axis=-1))

    dists_np = np.asarray(dists)

    print(f"\n{'='*50}")
    print(f"{label} frame (index {frame_idx}):")
    print(f"  All bonds: mean={dists_np.mean():.3f}, max={dists_np.max():.3f}, min={dists_np.min():.3f} nm")

    for bt in range(4):
        mask = bond_types == bt
        if mask.sum() > 0:
            d = dists_np[mask]
            print(f"  {bond_type_names[bt]:25s}: n={mask.sum():4d}, "
                  f"mean={d.mean():.3f}, max={d.max():.3f}, min={d.min():.3f} nm")

    # Flag any suspiciously long bonds
    long_bonds = dists_np > 10.0
    if long_bonds.any():
        print(f"  WARNING: {long_bonds.sum()} bonds longer than 10 nm!")
        for idx in np.where(long_bonds)[0][:5]:
            i, j = bond_pairs[idx]
            print(f"    Bond {i}-{j} (type {bond_type_names[bond_types[idx]]}): {dists_np[idx]:.2f} nm")
