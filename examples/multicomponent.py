#!/usr/bin/env python
"""Multicomponent stress granule simulation.

Full system with all key stress granule components:
- G3BP1 dimers (scaffold)
- RNA chains (scaffold)
- CAPRIN1 (bridge: lowers Csat)
- USP10 (cap: inhibits)
- UBAP2L (node: boosts connectivity)
- TIA1 (RNA-binding accessory)

This example demonstrates the full complexity of the simulation engine
with competitive binding, conformational switching, and multicomponent
phase separation.
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
    detect_clusters, cluster_statistics, compute_rdf, compute_density_profile,
)
from sgsim.io import save_trajectory_zarr, export_xyz
from sgsim.types import PARTICLE_TYPES as PT


def main():
    """Run a multicomponent stress granule simulation."""
    # System composition (moderate size for CPU testing)
    composition = {
        "g3bp1_dimer": 10,  # scaffold
        "rna": 25,           # scaffold
        "caprin1": 5,        # bridge
        "ubap2l": 5,         # node
        "tia1": 5,           # accessory RBP
    }
    box_size = 80.0  # nm
    n_steps = 5000

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

    # Run full simulation
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
        save_every=500,
        binding_interval=5,
        n_binding_attempts=30,
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
    mol_ids = result["molecule_ids"]

    # Cluster analysis
    labels = detect_clusters(positions, box, cutoff=5.0)
    stats = cluster_statistics(labels, positions, box, ptypes)

    print(f"\nClusters: {int(stats['n_clusters'])}")
    print(f"Largest cluster: {int(stats['largest_cluster_size'])} particles "
          f"({float(stats['largest_cluster_fraction']):.1%})")

    # Composition of largest cluster
    largest_label = 0  # label propagation gives smallest index to largest cluster
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

    # Binding stats
    n_bonds = int(result["binding_state"].n_bound)
    print(f"\nActive bonds: {n_bonds}")
    print(f"Mean G3BP1 openness: {result['openness_history'][-1]:.3f}")

    # Density profile
    centers, density = compute_density_profile(positions, box, n_bins=20, axis=0)
    mean_dens = float(jnp.mean(density))
    max_dens = float(jnp.max(density))
    print(f"\nDensity: mean={mean_dens:.4f}, max={max_dens:.4f} particles/nm^3")
    if max_dens > 2 * mean_dens:
        print("  -> Density heterogeneity detected (possible condensate)")
    else:
        print("  -> Relatively uniform (no strong condensate)")

    # Save outputs
    os.makedirs("results", exist_ok=True)

    # Export XYZ for OVITO
    export_xyz(
        "results/multicomponent_final.xyz",
        positions, ptypes, box_size=box,
    )
    print("\nXYZ saved to results/multicomponent_final.xyz")

    # Save trajectory
    save_trajectory_zarr(
        "results/multicomponent_traj.zarr",
        result["trajectory"], ptypes, box,
        metadata={"composition": str(composition), "n_steps": n_steps},
    )
    print("Trajectory saved to results/multicomponent_traj.zarr")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Binding history
    ax = axes[0, 0]
    steps = np.arange(len(result["binding_history"])) * 500
    ax.plot(steps, result["binding_history"], "b-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Active bonds")
    ax.set_title("Binding events")

    # Cluster growth
    ax = axes[0, 1]
    if result["cluster_history"]:
        fracs = [s[1] / n_particles for s in result["cluster_history"]]
        cl_steps = np.arange(len(fracs)) * 500
        ax.plot(cl_steps, fracs, "r-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Largest cluster fraction")
    ax.set_title("Condensate growth")

    # Openness
    ax = axes[1, 0]
    phi_steps = np.arange(len(result["openness_history"])) * 500
    ax.plot(phi_steps, result["openness_history"], "g-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("<phi>")
    ax.set_title("G3BP1 conformational state")
    ax.set_ylim(-0.05, 1.05)

    # Density profile
    ax = axes[1, 1]
    ax.bar(np.asarray(centers), np.asarray(density), width=float(box[0]) / 20, alpha=0.7)
    ax.axhline(y=mean_dens, color="r", linestyle="--", label=f"mean={mean_dens:.4f}")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Density (particles/nm^3)")
    ax.set_title("Density profile along x")
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/multicomponent_analysis.png", dpi=150)
    print("Plot saved to results/multicomponent_analysis.png")
    plt.close()


if __name__ == "__main__":
    main()
