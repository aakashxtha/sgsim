#!/usr/bin/env python
"""Dense G3BP1 + RNA phase separation with trajectory saving.

Higher concentration (smaller box) and aggressive binding sampling
to observe clear condensate formation. Saves zarr trajectory and
multi-frame XYZ for visualization in OVITO.
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
    """Run dense G3BP1+RNA simulation with trajectory output."""

    # Dense system: 20 G3BP1 dimers + 50 RNA in a 50 nm box
    # 20 dimers * 12 beads = 240 protein beads
    # 50 RNA * 10 beads = 500 RNA beads
    # Total ~740 beads in 50^3 = 125,000 nm^3
    # Effective [G3BP1] ~ 20 dimers / (50nm)^3 = 0.16 / nm^3 ~ 265 uM (well above Csat)
    composition = {
        "g3bp1_dimer": 20,
        "rna": 50,
    }
    box_size = 50.0  # nm — much denser than 80 nm
    n_steps = 10000
    save_every = 200  # save 50 frames

    print("=" * 60)
    print("Dense G3BP1 + RNA Phase Separation")
    print("=" * 60)
    print(f"Composition: {composition}")
    print(f"Box: {box_size} nm cubic")
    print(f"Steps: {n_steps}, save every {save_every}")
    print()

    key = jax.random.PRNGKey(42)
    system = build_system(composition, box_size, key, rna_beads=10, rna_exposure=1.0)

    n_particles = system["positions"].shape[0]
    print(f"Total particles: {n_particles}")
    print(f"Bonds: {system['topology'].n_bonds}")
    print(f"Binding sites: {system['topology'].n_sites}")
    conc_nm3 = n_particles / box_size**3
    print(f"Number density: {conc_nm3:.4f} particles/nm^3")
    print()

    t0 = time.time()

    result = run_full_simulation(
        system,
        n_steps=n_steps,
        dt=0.005,
        kT=1.0,
        gamma_base=1.0,
        cutoff=8.0,
        skin=2.0,
        max_neighbors=64,
        nl_rebuild_every=10,
        save_every=save_every,
        binding_interval=2,          # bind every 2 steps (aggressive)
        n_binding_attempts=50,       # 50 MC attempts per call
        conformational_interval=5,
        gamma_phi=1.0,
        rng_key=key,
        verbose=True,
    )

    elapsed = time.time() - t0
    print(f"\nSimulation took {elapsed:.1f} s")

    # --- Analysis ---
    print("\n" + "=" * 60)
    print("FINAL STATE ANALYSIS")
    print("=" * 60)

    positions = result["final_positions"]
    box = result["box_size"]
    ptypes = result["particle_types"]

    labels = detect_clusters(positions, box, cutoff=5.0)
    stats = cluster_statistics(labels, positions, box, ptypes)

    print(f"\nClusters: {int(stats['n_clusters'])}")
    print(f"Largest cluster: {int(stats['largest_cluster_size'])} particles "
          f"({float(stats['largest_cluster_fraction']):.1%})")

    # Composition of largest cluster
    largest_label = 0
    in_largest = np.asarray(labels) == largest_label
    if np.sum(in_largest) > 1:
        ptypes_np = np.asarray(ptypes)
        type_names = {v: k for k, v in PT.items()}
        print("\nLargest cluster composition:")
        for tid in np.unique(ptypes_np[in_largest]):
            count = np.sum((ptypes_np == tid) & in_largest)
            total = np.sum(ptypes_np == tid)
            name = type_names.get(int(tid), f"T{tid}")
            print(f"  {name}: {count}/{total} ({count/total:.0%})")

    n_bonds = int(result["binding_state"].n_bound)
    print(f"\nActive bonds: {n_bonds}")
    if result["openness_history"]:
        print(f"Mean G3BP1 openness: {result['openness_history'][-1]:.3f}")

    # Density profile
    centers, density = compute_density_profile(positions, box, n_bins=20, axis=0)
    mean_dens = float(jnp.mean(density))
    max_dens = float(jnp.max(density))
    print(f"\nDensity: mean={mean_dens:.4f}, max={max_dens:.4f} particles/nm^3")
    if max_dens > 2 * mean_dens:
        print("  -> Strong density heterogeneity (condensate likely!)")

    # --- Save outputs ---
    outdir = "results/dense_phase_sep"
    os.makedirs(outdir, exist_ok=True)

    # Save multi-frame XYZ trajectory for OVITO
    traj = result["trajectory"]
    print(f"\nTrajectory frames: {len(traj)}")

    export_xyz(
        f"{outdir}/trajectory.xyz",
        np.stack([np.asarray(pos) for pos in traj]),
        ptypes, box_size=box,
    )
    print(f"XYZ trajectory saved to {outdir}/trajectory.xyz")

    # Save zarr trajectory
    save_trajectory_zarr(
        f"{outdir}/trajectory.zarr",
        traj, ptypes, box,
        metadata={
            "composition": str(composition),
            "n_steps": n_steps,
            "box_size": float(box_size),
            "dt": 0.005,
            "binding_interval": 2,
            "n_binding_attempts": 50,
        },
    )
    print(f"Zarr trajectory saved to {outdir}/trajectory.zarr")

    # Final frame XYZ
    export_xyz(f"{outdir}/final.xyz", positions, ptypes, box_size=box)
    print(f"Final frame saved to {outdir}/final.xyz")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Binding history
    ax = axes[0, 0]
    bind_hist = result["binding_history"]
    steps_b = np.arange(len(bind_hist)) * save_every
    ax.plot(steps_b, bind_hist, "b-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Active bonds")
    ax.set_title("Binding events over time")

    # Cluster growth
    ax = axes[0, 1]
    if result["cluster_history"]:
        fracs = [s[1] / n_particles for s in result["cluster_history"]]
        cl_steps = np.arange(len(fracs)) * save_every
        ax.plot(cl_steps, fracs, "r-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Largest cluster fraction")
    ax.set_title("Condensate growth")

    # Openness
    ax = axes[1, 0]
    if result["openness_history"]:
        phi_steps = np.arange(len(result["openness_history"])) * save_every
        ax.plot(phi_steps, result["openness_history"], "g-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("<phi>")
    ax.set_title("G3BP1 conformational state")
    ax.set_ylim(-0.05, 1.05)

    # Density profile
    ax = axes[1, 1]
    ax.bar(np.asarray(centers), np.asarray(density),
           width=float(box[0]) / 20, alpha=0.7)
    ax.axhline(y=mean_dens, color="r", linestyle="--",
               label=f"mean={mean_dens:.4f}")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Density (particles/nm^3)")
    ax.set_title("Density profile along x")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{outdir}/analysis.png", dpi=150)
    print(f"Plot saved to {outdir}/analysis.png")
    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
