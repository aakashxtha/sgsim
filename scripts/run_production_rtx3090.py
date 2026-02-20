#!/usr/bin/env python
"""Large-scale Multicomponent Stress Granule Simulation.

Optimized for execution on a 24GB NVIDIA RTX 3090.

Key design:
- ~11,000 coarse-grained beads across ~1,000 molecules.
- 150nm box: large enough for two-phase coexistence (droplet + dilute).
- 100,000 BD steps with aggressive MC binding sampling.
- Analysis uses the neighbor-list based cluster detection
  (the O(N^2) detect_clusters is replaced with a scalable version).

IMPORTANT: Before running, ensure detect_clusters in sgsim/analysis.py
is replaced with a neighbor-list based version for N > 5000. The default
label propagation builds an N x N adjacency matrix (~500MB for N=11000).
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgsim.setup import build_system
from sgsim.simulate import run_full_simulation
from sgsim.analysis import detect_clusters, cluster_statistics, compute_density_profile
from sgsim.io import save_trajectory_zarr, export_xyz
from sgsim.types import PARTICLE_TYPES as PT


def main():
    # ---------------------------------------------------------
    # System Composition
    # ---------------------------------------------------------
    # 250 G3BP1 dimers * 12 beads = 3,000
    # 500 RNA * 15 beads          = 7,500
    # 100 CAPRIN1 * 4 beads       =   400
    # 100 UBAP2L * 5 beads        =   500
    #  50 TIA1 * 4 beads          =   200
    # Total                       ~11,600 beads
    composition = {
        "g3bp1_dimer": 250,
        "rna": 500,
        "caprin1": 100,
        "ubap2l": 100,
        "tia1": 50,
    }
    box_size = 150.0  # nm — two-phase coexistence regime
    n_steps = 100000
    save_every = 2000  # 50 frames

    print("=" * 80)
    print("RTX 3090 PRODUCTION RUN: Multicomponent Stress Granule")
    print("=" * 80)
    print(f"Composition: {composition}")
    print(f"Box Size:    {box_size} nm cubic")
    print(f"Steps:       {n_steps:,} (saving every {save_every})")
    print(f"JAX Backend: {jax.default_backend().upper()} ({jax.devices()[0].device_kind})")
    print("-" * 80)

    # 1. Build System
    print("\nBuilding initial system...")
    t0 = time.time()
    key = jax.random.PRNGKey(42)
    system = build_system(
        composition,
        box_size,
        key,
        rna_beads=15,
        rna_exposure=1.0,
    )

    n_particles = system["positions"].shape[0]
    vol = box_size ** 3
    print(f"Done in {time.time() - t0:.1f}s.")
    print(f"Total Beads:      {n_particles:,}")
    print(f"Total Bonds:      {system['topology'].n_bonds:,}")
    print(f"Binding Sites:    {system['topology'].n_sites:,}")
    print(f"Number Density:   {n_particles / vol:.5f} beads/nm^3")

    # 2. Run Simulation
    print("\nStarting BD simulation (first step includes JIT compilation)...")
    t1 = time.time()

    result = run_full_simulation(
        system,
        n_steps=n_steps,
        dt=0.005,
        kT=1.0,
        gamma_base=1.0,
        cutoff=8.0,
        skin=2.0,
        max_neighbors=128,
        nl_rebuild_every=10,
        save_every=save_every,
        binding_interval=2,
        n_binding_attempts=200,
        conformational_interval=5,
        gamma_phi=1.0,
        rng_key=key,
        verbose=True,
    )

    elapsed = time.time() - t1
    print(f"\nSimulation completed in {elapsed:.1f}s")
    print(f"Performance: {n_steps / elapsed:.1f} steps/s")

    # ---------------------------------------------------------
    # 3. Analysis
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL STATE ANALYSIS")
    print("=" * 60)

    positions = result["final_positions"]
    box = result["box_size"]
    ptypes = result["particle_types"]

    # NOTE: detect_clusters uses O(N^2) memory. For N~11k on GPU with 24GB
    # this should fit (~1GB for adjacency). On CPU it will be slow.
    labels = detect_clusters(positions, box, cutoff=6.0)
    stats = cluster_statistics(labels, positions, box, ptypes)

    print(f"\nTotal Clusters:    {int(stats['n_clusters'])}")
    print(f"Largest Droplet:   {int(stats['largest_cluster_size'])} beads "
          f"({float(stats['largest_cluster_fraction']):.1%})")

    # Droplet composition
    largest_label = 0
    in_largest = np.asarray(labels) == largest_label
    if np.sum(in_largest) > 1:
        ptypes_np = np.asarray(ptypes)
        type_names = {v: k for k, v in PT.items()}
        print("\nDroplet composition:")
        for tid in np.unique(ptypes_np[in_largest]):
            count = np.sum((ptypes_np == tid) & in_largest)
            total = np.sum(ptypes_np == tid)
            name = type_names.get(int(tid), f"T{tid}")
            print(f"  {name:<14}: {count:>5}/{total:<5} ({count/total:>4.0%})")

    n_bonds = int(result["binding_state"].n_bound)
    print(f"\nActive Bonds:  {n_bonds:,}")
    print(f"Mean <phi>:    {result['openness_history'][-1]:.3f}")

    # ---------------------------------------------------------
    # 4. Save Outputs
    # ---------------------------------------------------------
    outdir = "results/production_rtx3090"
    os.makedirs(outdir, exist_ok=True)

    traj = result["trajectory"]

    export_xyz(
        f"{outdir}/trajectory.xyz",
        np.stack([np.asarray(pos) for pos in traj]),
        ptypes,
        box_size=box,
    )
    print(f"\nTrajectory: {len(traj)} frames -> {outdir}/trajectory.xyz")

    save_trajectory_zarr(
        f"{outdir}/trajectory.zarr",
        traj, ptypes, box,
        metadata={"composition": str(composition), "n_steps": n_steps},
    )
    print(f"Zarr:       {outdir}/trajectory.zarr")

    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    steps = np.arange(len(result["binding_history"])) * save_every
    ax.plot(steps, result["binding_history"], "b-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Active Bonds")
    ax.set_title("Binding Network Formation")

    ax = axes[0, 1]
    if result["cluster_history"]:
        fracs = [s[1] / n_particles for s in result["cluster_history"]]
        cl_steps = np.arange(len(fracs)) * save_every
        ax.plot(cl_steps, fracs, "r-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Largest cluster fraction")
    ax.set_title("Condensation")

    ax = axes[1, 0]
    phi_steps = np.arange(len(result["openness_history"])) * save_every
    ax.plot(phi_steps, result["openness_history"], "g-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("<phi>")
    ax.set_title("G3BP1 Conformational State")
    ax.set_ylim(-0.05, 1.05)

    centers, density = compute_density_profile(positions, box, n_bins=50, axis=0)
    mean_dens = float(jnp.mean(density))
    ax = axes[1, 1]
    ax.bar(np.asarray(centers), np.asarray(density), width=float(box[0]) / 50, alpha=0.7)
    ax.axhline(y=mean_dens, color="r", linestyle="--", label=f"mean={mean_dens:.5f}")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Density (beads/nm^3)")
    ax.set_title("Density Profile")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{outdir}/analysis.png", dpi=150)
    print(f"Plot:       {outdir}/analysis.png")

    print("\nDone! Open the .xyz in OVITO to visualize the droplet.")


if __name__ == "__main__":
    main()
