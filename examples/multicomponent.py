#!/usr/bin/env python
"""Multicomponent stress granule simulation.

Full system with all key stress granule components:
- G3BP1 dimers (scaffold)
- RNA chains (scaffold)
- CAPRIN1 (bridge: lowers Csat)
- USP10 (cap: inhibits)
- UBAP2L (node: boosts connectivity)
- TIA1 (RNA-binding accessory)
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgsim.setup import build_system
from sgsim.simulate import run_full_simulation
from sgsim.analysis import (
    detect_clusters, cluster_statistics, compute_density_profile,
)
from sgsim.io import save_trajectory_zarr, export_xyz
from sgsim.types import PARTICLE_TYPES as PT


def main():
    composition = {
        "g3bp1_dimer": 15,
        "rna": 30,
        "caprin1": 8,
        "ubap2l": 8,
        "tia1": 8,
    }
    box_size = 60.0
    n_steps = 8000
    save_every = 500

    print("Multicomponent Stress Granule Simulation")
    print(f"Composition: {composition}")
    print(f"Box: {box_size} nm cubic")
    print()

    key = jax.random.PRNGKey(123)
    system = build_system(composition, box_size, key, rna_beads=10, rna_exposure=1.0)

    n_particles = system["positions"].shape[0]
    print(f"Total particles: {n_particles}")
    print(f"Bonds: {system['topology'].n_bonds}")
    print(f"Binding sites: {system['topology'].n_sites}")
    print()

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
        binding_interval=2,
        n_binding_attempts=50,
        conformational_interval=5,
        gamma_phi=1.0,
        rng_key=key,
        verbose=True,
    )

    # Analysis
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
            print(f"  {name:<14}: {count:>4}/{total:<4} ({count/total:>4.0%})")

    n_bonds = int(result["binding_state"].n_bound)
    print(f"\nActive bonds: {n_bonds}")
    print(f"Mean G3BP1 openness: {result['openness_history'][-1]:.3f}")

    # Density profile
    centers, density = compute_density_profile(positions, box, n_bins=20, axis=0)
    mean_dens = float(jnp.mean(density))
    max_dens = float(jnp.max(density))
    print(f"\nDensity: mean={mean_dens:.4f}, max={max_dens:.4f} particles/nm^3")

    # Save outputs
    outdir = "results/multicomponent"
    os.makedirs(outdir, exist_ok=True)

    export_xyz(f"{outdir}/final.xyz", positions, ptypes, box_size=box)
    print(f"\nXYZ saved to {outdir}/final.xyz")

    traj = result["trajectory"]
    export_xyz(
        f"{outdir}/trajectory.xyz",
        np.stack([np.asarray(pos) for pos in traj]),
        ptypes, box_size=box,
    )
    print(f"Trajectory XYZ saved to {outdir}/trajectory.xyz ({len(traj)} frames)")

    save_trajectory_zarr(
        f"{outdir}/trajectory.zarr",
        traj, ptypes, box,
        metadata={"composition": str(composition), "n_steps": n_steps},
    )
    print(f"Zarr saved to {outdir}/trajectory.zarr")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    steps = np.arange(len(result["binding_history"])) * save_every
    ax.plot(steps, result["binding_history"], "b-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Active bonds")
    ax.set_title("Binding events")

    ax = axes[0, 1]
    if result["cluster_history"]:
        fracs = [s[1] / n_particles for s in result["cluster_history"]]
        cl_steps = np.arange(len(fracs)) * save_every
        ax.plot(cl_steps, fracs, "r-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Largest cluster fraction")
    ax.set_title("Condensate growth")

    ax = axes[1, 0]
    phi_steps = np.arange(len(result["openness_history"])) * save_every
    ax.plot(phi_steps, result["openness_history"], "g-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("<phi>")
    ax.set_title("G3BP1 conformational state")
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1, 1]
    ax.bar(np.asarray(centers), np.asarray(density), width=float(box[0]) / 20, alpha=0.7)
    ax.axhline(y=mean_dens, color="r", linestyle="--", label=f"mean={mean_dens:.4f}")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Density (particles/nm^3)")
    ax.set_title("Density profile along x")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{outdir}/analysis.png", dpi=150)
    print(f"Plot saved to {outdir}/analysis.png")
    plt.close()


if __name__ == "__main__":
    main()
