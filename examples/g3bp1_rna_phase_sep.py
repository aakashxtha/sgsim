#!/usr/bin/env python
"""G3BP1 + RNA phase separation validation.

This is the primary validation: does the system phase separate when
G3BP1 dimers are mixed with long unfolded RNA?

Expected behavior (Guillen-Boixet 2020):
- G3BP1 + RNA → phase separation at ~2 uM Csat
- Condensate total density ~65 mg/mL
- G3BP1 ~7x enriched inside condensate
- Long unfolded RNA drives assembly; short/folded RNA does not

This script runs three conditions:
1. G3BP1 dimers + long RNA (should phase separate)
2. G3BP1 dimers only (should NOT phase separate)
3. G3BP1 dimers + short RNA (reduced/no phase separation)
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgsim.setup import build_system
from sgsim.simulate import run_full_simulation
from sgsim.analysis import detect_clusters, cluster_statistics, compute_rdf
from sgsim.io import save_trajectory_zarr, export_xyz
from sgsim.types import PARTICLE_TYPES as PT


def run_condition(name, composition, box_size, rna_beads=10, rna_exposure=1.0,
                  n_steps=5000, seed=42):
    """Run a single simulation condition."""
    print(f"\n{'='*60}")
    print(f"Condition: {name}")
    print(f"Composition: {composition}")
    print(f"Box size: {box_size} nm, RNA beads: {rna_beads}, exposure: {rna_exposure}")
    print(f"{'='*60}\n")

    key = jax.random.PRNGKey(seed)

    # Build system
    system = build_system(composition, box_size, key,
                          rna_beads=rna_beads, rna_exposure=rna_exposure)

    n_particles = system["positions"].shape[0]
    print(f"System: {n_particles} particles, "
          f"{system['topology'].n_bonds} bonds, "
          f"{system['topology'].n_sites} binding sites")

    # Run simulation
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
        n_binding_attempts=20,
        conformational_interval=5,
        gamma_phi=1.0,
        rng_key=key,
        verbose=True,
    )

    return result


def plot_results(results, output_dir="results"):
    """Plot comparison of conditions."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Binding events over time
    ax = axes[0, 0]
    for name, res in results.items():
        steps = np.arange(len(res["binding_history"])) * 500
        ax.plot(steps[:len(res["binding_history"])], res["binding_history"], label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of bonds")
    ax.set_title("Binding events")
    ax.legend()

    # 2. Largest cluster fraction over time
    ax = axes[0, 1]
    for name, res in results.items():
        if res["cluster_history"]:
            n_particles = res["final_positions"].shape[0]
            fracs = [s[1] / n_particles for s in res["cluster_history"]]
            steps = np.arange(len(fracs)) * 500
            ax.plot(steps[:len(fracs)], fracs, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Largest cluster fraction")
    ax.set_title("Cluster growth")
    ax.legend()

    # 3. Mean openness (phi) over time
    ax = axes[1, 0]
    for name, res in results.items():
        if res["openness_history"]:
            steps = np.arange(len(res["openness_history"])) * 500
            ax.plot(steps[:len(res["openness_history"])],
                    res["openness_history"], label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean openness <phi>")
    ax.set_title("G3BP1 conformational state")
    ax.legend()

    # 4. Final cluster size distributions
    ax = axes[1, 1]
    for name, res in results.items():
        labels = detect_clusters(res["final_positions"], res["box_size"], cutoff=5.0)
        labels_np = np.asarray(labels)
        unique, counts = np.unique(labels_np, return_counts=True)
        sizes = sorted(counts, reverse=True)
        ax.bar(np.arange(min(len(sizes), 20)) + list(results.keys()).index(name) * 0.25,
               sizes[:20], width=0.25, label=name, alpha=0.7)
    ax.set_xlabel("Cluster rank")
    ax.set_ylabel("Cluster size (particles)")
    ax.set_title("Final cluster size distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase_separation_comparison.png"), dpi=150)
    print(f"\nPlot saved to {output_dir}/phase_separation_comparison.png")
    plt.close()


def main():
    """Run the G3BP1 + RNA phase separation validation."""
    # Simulation parameters
    # Small system for quick validation on CPU
    # Scale up for production runs (100+ dimers, 200+ RNA)
    box_size = 80.0  # nm (cubic box)
    n_steps = 5000   # increase for production (50000+)

    results = {}

    # Condition 1: G3BP1 dimers + long RNA (should phase separate)
    results["G3BP1+RNA"] = run_condition(
        "G3BP1 dimers + long RNA",
        {"g3bp1_dimer": 15, "rna": 30},
        box_size,
        rna_beads=10,
        rna_exposure=1.0,
        n_steps=n_steps,
        seed=42,
    )

    # Condition 2: G3BP1 dimers only (should NOT phase separate)
    results["G3BP1 only"] = run_condition(
        "G3BP1 dimers only (no RNA)",
        {"g3bp1_dimer": 15},
        box_size,
        n_steps=n_steps,
        seed=42,
    )

    # Condition 3: G3BP1 dimers + short/folded RNA (reduced phase separation)
    results["G3BP1+short RNA"] = run_condition(
        "G3BP1 dimers + short folded RNA",
        {"g3bp1_dimer": 15, "rna": 30},
        box_size,
        rna_beads=3,        # short RNA
        rna_exposure=0.3,   # mostly folded
        n_steps=n_steps,
        seed=42,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, res in results.items():
        n_particles = res["final_positions"].shape[0]
        n_bonds = res["binding_history"][-1]
        if res["cluster_history"]:
            n_cl, largest = res["cluster_history"][-1]
            frac = largest / n_particles
        else:
            n_cl, largest, frac = 0, 0, 0.0
        phi = res["openness_history"][-1] if res["openness_history"] else 0.0

        print(f"\n{name}:")
        print(f"  Particles: {n_particles}")
        print(f"  Final bonds: {n_bonds}")
        print(f"  Clusters: {n_cl}, largest: {largest} ({frac:.1%})")
        print(f"  Mean openness: {phi:.3f}")

    # Plot
    plot_results(results)

    # Export final frame of G3BP1+RNA for visualization
    res = results["G3BP1+RNA"]
    export_xyz(
        "results/g3bp1_rna_final.xyz",
        res["final_positions"],
        res["particle_types"],
        box_size=res["box_size"],
    )
    print("\nXYZ exported to results/g3bp1_rna_final.xyz (open in OVITO)")


if __name__ == "__main__":
    main()
