#!/usr/bin/env python
"""G3BP1 + RNA phase separation validation.

Primary validation: does the system phase separate when
G3BP1 dimers are mixed with long unfolded RNA?

Expected behavior (Guillen-Boixet 2020):
- G3BP1 + RNA -> phase separation at ~2 uM Csat
- Long unfolded RNA drives assembly; short/folded RNA does not

Three conditions:
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgsim.setup import build_system
from sgsim.simulate import run_full_simulation
from sgsim.analysis import detect_clusters, cluster_statistics
from sgsim.io import export_xyz


def run_condition(name, composition, box_size, rna_beads=10, rna_exposure=1.0,
                  n_steps=5000, seed=42):
    """Run a single simulation condition."""
    print(f"\n{'='*60}")
    print(f"Condition: {name}")
    print(f"Composition: {composition}")
    print(f"Box: {box_size} nm, RNA beads: {rna_beads}, exposure: {rna_exposure}")
    print(f"{'='*60}\n")

    key = jax.random.PRNGKey(seed)
    system = build_system(composition, box_size, key,
                          rna_beads=rna_beads, rna_exposure=rna_exposure)

    n_particles = system["positions"].shape[0]
    print(f"System: {n_particles} particles, "
          f"{system['topology'].n_bonds} bonds, "
          f"{system['topology'].n_sites} binding sites")

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
        binding_interval=2,
        n_binding_attempts=50,
        conformational_interval=5,
        gamma_phi=1.0,
        rng_key=key,
        verbose=True,
    )
    return result


def main():
    box_size = 60.0  # nm — balanced density
    n_steps = 8000

    results = {}

    # Condition 1: G3BP1 dimers + long RNA (should phase separate)
    results["G3BP1+RNA"] = run_condition(
        "G3BP1 dimers + long RNA",
        {"g3bp1_dimer": 20, "rna": 40},
        box_size,
        rna_beads=10,
        rna_exposure=1.0,
        n_steps=n_steps,
        seed=42,
    )

    # Condition 2: G3BP1 dimers only (should NOT phase separate)
    results["G3BP1 only"] = run_condition(
        "G3BP1 dimers only (no RNA)",
        {"g3bp1_dimer": 20},
        box_size,
        n_steps=n_steps,
        seed=42,
    )

    # Condition 3: G3BP1 dimers + short/folded RNA (reduced phase separation)
    results["G3BP1+short RNA"] = run_condition(
        "G3BP1 dimers + short folded RNA",
        {"g3bp1_dimer": 20, "rna": 40},
        box_size,
        rna_beads=3,
        rna_exposure=0.3,
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
    outdir = "results/phase_separation"
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Binding events over time
    ax = axes[0, 0]
    for name, res in results.items():
        steps = np.arange(len(res["binding_history"])) * 500
        ax.plot(steps, res["binding_history"], label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of bonds")
    ax.set_title("Binding events")
    ax.legend()

    # Largest cluster fraction
    ax = axes[0, 1]
    for name, res in results.items():
        if res["cluster_history"]:
            n_p = res["final_positions"].shape[0]
            fracs = [s[1] / n_p for s in res["cluster_history"]]
            steps = np.arange(len(fracs)) * 500
            ax.plot(steps, fracs, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Largest cluster fraction")
    ax.set_title("Cluster growth")
    ax.legend()

    # Mean openness
    ax = axes[1, 0]
    for name, res in results.items():
        if res["openness_history"]:
            steps = np.arange(len(res["openness_history"])) * 500
            ax.plot(steps, res["openness_history"], label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean openness <phi>")
    ax.set_title("G3BP1 conformational state")
    ax.legend()

    # Final cluster size distributions
    ax = axes[1, 1]
    for idx, (name, res) in enumerate(results.items()):
        labels = detect_clusters(res["final_positions"], res["box_size"], cutoff=5.0)
        labels_np = np.asarray(labels)
        unique, counts = np.unique(labels_np, return_counts=True)
        sizes = sorted(counts, reverse=True)
        ax.bar(np.arange(min(len(sizes), 15)) + idx * 0.25,
               sizes[:15], width=0.25, label=name, alpha=0.7)
    ax.set_xlabel("Cluster rank")
    ax.set_ylabel("Cluster size (particles)")
    ax.set_title("Final cluster size distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{outdir}/comparison.png", dpi=150)
    print(f"\nPlot saved to {outdir}/comparison.png")
    plt.close()

    # Export XYZ of G3BP1+RNA for visualization
    res = results["G3BP1+RNA"]
    export_xyz(f"{outdir}/g3bp1_rna_final.xyz",
               res["final_positions"], res["particle_types"], box_size=res["box_size"])
    print(f"XYZ saved to {outdir}/g3bp1_rna_final.xyz")


if __name__ == "__main__":
    main()
