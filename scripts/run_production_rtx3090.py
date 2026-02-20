#!/usr/bin/env python
"""Large-scale Multicomponent Stress Granule Simulation.

Optimized for execution on a 24GB NVIDIA RTX 3090.
This setup represents a true macroscopic droplet formation, easily 
filling a large periodic box.

Key changes from examples:
- Large particle count (1,000+ proteins/RNAs, ~10,000 total coarse-grained beads).
  (Will easily fit in 24GB vRAM given JAX's efficient memory management.)
- Extended simulation time (100,000 steps).
- Explicit pre-allocation for high neighbor counts.
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
    """Run production-scale stress granule simulation."""
    
    # ---------------------------------------------------------
    # System Composition: Macromolecular Condensate Scale
    # ---------------------------------------------------------
    # ~250 G3BP1 dimers (scaffold node)
    # ~500 RNA chains  (scaffold polymer)
    # ~100 CAPRIN1     (bridge that lowers concentration threshold)
    # ~100 UBAP2L      (node that boosts network connectivity)
    # ~50 TIA1         (accessory RNA binding protein)
    # Total molecules ~1000. Total discrete beads ~11,000.
    composition = {
        "g3bp1_dimer": 250,  
        "rna": 500,           
        "caprin1": 100,        
        "ubap2l": 100,         
        "tia1": 50,           
    }
    box_size = 200.0  # nm (large cubic box to allow distinct dilute + dense phases)
    n_steps = 100000  # 100k steps for equilibration
    save_every = 1000 # Save 100 trajectory frames

    print("=" * 80)
    print("RTX 3090 PRODUCTION RUN: Macroscopic Multicomponent Stress Granule")
    print("=" * 80)
    print(f"Composition: {composition}")
    print(f"Box Size:    {box_size} nm cubic")
    print(f"Steps:       {n_steps:,} (Saving every {save_every})")
    print(f"JAX Backend: {jax.default_backend().upper()} ({jax.devices()[0].device_kind})")
    print("-" * 80)

    # 1. Build System
    print("\nBuilding initial system architecture...")
    t0 = time.time()
    key = jax.random.PRNGKey(42)
    system = build_system(
        composition, 
        box_size, 
        key, 
        rna_beads=15,       # Med-Long RNA 
        rna_exposure=1.0    # Unfolded RNA easily binds G3BP1
    )

    n_particles = system["positions"].shape[0]
    print(f"Done in {time.time() - t0:.1f}s.")
    print(f"Total Beads:    {n_particles:,}")
    print(f"Total Bonds:    {system['topology'].n_bonds:,}")
    print(f"Binding Sites:  {system['topology'].n_sites:,}")
    
    # 2. Run Simulation
    print("\nStarting JIT compilation and Brownian Dynamics integration...")
    t1 = time.time()
    
    # Note: On the first step of run_full_simulation, JAX will JIT compile 
    # the entire inner scan loop. This may take 30-90 seconds. 
    # Afterwards, it will fly.
    result = run_full_simulation(
        system,
        n_steps=n_steps,
        dt=0.005,
        kT=1.0,
        gamma_base=1.0,
        cutoff=8.0,
        skin=2.0,
        # Bump up max_neighbors for safety in dense condensates
        max_neighbors=128,  
        nl_rebuild_every=10,
        save_every=save_every,
        binding_interval=5,
        n_binding_attempts=50,
        conformational_interval=5,
        gamma_phi=1.0,
        rng_key=key,
        verbose=True,  # Will print every `save_every` steps
    )

    elapsed = time.time() - t1
    print(f"\nSimulation completed in {elapsed:.1f} seconds ")
    print(f"Performance: {(n_steps / elapsed):.1f} steps/second")

    # ---------------------------------------------------------
    # 3. Post-Simulation Analysis
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL STATE ANALYSIS")
    print("=" * 60)

    positions = result["final_positions"]
    box = result["box_size"]
    ptypes = result["particle_types"]

    # Cluster analysis (using cutoff clustering)
    labels = detect_clusters(positions, box, cutoff=6.0)
    stats = cluster_statistics(labels, positions, box, ptypes)

    print(f"\nTotal Disconnected Clusters: {int(stats['n_clusters'])}")
    print(f"Largest Condensate Droplet:  {int(stats['largest_cluster_size'])} beads "
          f"({float(stats['largest_cluster_fraction']):.1%} of system)")

    # Print composition inside the largest droplet
    largest_label = 0 
    in_largest = np.asarray(labels) == largest_label
    if np.sum(in_largest) > 1:
        ptypes_np = np.asarray(ptypes)
        type_names = {v: k for k, v in PT.items()}
        print("\nComposition of largest condensate droplet:")
        for tid in np.unique(ptypes_np[in_largest]):
            count = np.sum((ptypes_np == tid) & in_largest)
            total = np.sum(ptypes_np == tid)
            name = type_names.get(int(tid), f"T{tid}")
            print(f"  {name:<14}: {count:>5}/{total:<5} ({count/total:>4.0%})")

    n_bonds = int(result["binding_state"].n_bound)
    print(f"\nTotal Active Binding Events: {n_bonds:,}")
    print(f"Mean G3BP1 Openness <phi>:   {result['openness_history'][-1]:.3f} (1.0 = Fully active/expanded)")

    # ---------------------------------------------------------
    # 4. Save Outputs
    # ---------------------------------------------------------
    outdir = "results/production_rtx3090"
    os.makedirs(outdir, exist_ok=True)

    # Export complete multi-frame XYZ for VMD / OVITO visualizer
    traj = result["trajectory"]
    export_xyz(
        f"{outdir}/production_trajectory.xyz",
        np.stack([np.asarray(pos) for pos in traj]), 
        ptypes, 
        box_size=box,
    )
    print(f"\nVisualizer: Saved full {len(traj)}-frame trajectory to {outdir}/production_trajectory.xyz")

    # Save complete compressed trajectory (zarr)
    save_trajectory_zarr(
        f"{outdir}/production_traj.zarr",
        result["trajectory"], ptypes, box,
        metadata={"composition": str(composition), "n_steps": n_steps},
    )
    print(f"Trajectory: Saved {len(result['trajectory'])} frames to {outdir}/production_traj.zarr")

    # Generate charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Binding Network Growth
    ax = axes[0, 0]
    steps = np.arange(len(result["binding_history"])) * save_every
    ax.plot(steps, result["binding_history"], "b-", linewidth=2)
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Total Active Bonds")
    ax.set_title("Binding Network Formation")

    # Plot 2: Droplet Growth (Phase Separation)
    ax = axes[0, 1]
    if result["cluster_history"]:
        fracs = [s[1] / n_particles for s in result["cluster_history"]]
        cl_steps = np.arange(len(fracs)) * save_every
        ax.plot(cl_steps, fracs, "r-", linewidth=2)
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Fraction of system in largest droplet")
    ax.set_title("Macroscopic Condensation")

    # Plot 3: Conformational switch
    ax = axes[1, 0]
    phi_steps = np.arange(len(result["openness_history"])) * save_every
    ax.plot(phi_steps, result["openness_history"], "g-", linewidth=2)
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("G3BP1 <phi>")
    ax.set_title("RNA-driven G3BP1 Unfolding")
    ax.set_ylim(-0.05, 1.05)

    # Plot 4: Absolute density profile
    centers, density = compute_density_profile(positions, box, n_bins=50, axis=0)
    mean_dens = float(jnp.mean(density))
    ax = axes[1, 1]
    ax.bar(np.asarray(centers), np.asarray(density), width=float(box[0]) / 50, alpha=0.7)
    ax.axhline(y=mean_dens, color="r", linestyle="--", label=f"Average = {mean_dens:.3f}")
    ax.set_xlabel("x Coordinate (nm)")
    ax.set_ylabel("Density (beads / nm^3)")
    ax.set_title("Droplet Density Heterogeneity")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{outdir}/rtx3090_analysis.png", dpi=150)
    print(f"Analytics:  Saved plot to {outdir}/rtx3090_analysis.png")
    
    print("\nRun complete! You can open the .xyz file in OVITO to render the droplet.")


if __name__ == "__main__":
    main()
