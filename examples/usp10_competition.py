#!/usr/bin/env python
"""USP10 competition / valence capping validation.

USP10 binds G3BP1's NTF2 pocket with high affinity (-8 kT)
but has NO RNA-binding domain. This reduces G3BP1's effective
valence, inhibiting stress granule formation stoichiometrically.

Expected (Sanders 2020):
- USP10 at 1:1 ratio with NTF2 pockets abolishes phase separation
- Inhibition is stoichiometric (slope ~1), not catalytic
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


def run_titration_point(n_g3bp1, n_rna, n_usp10, box_size, n_steps, seed=42):
    """Run one titration point and return final cluster stats."""
    composition = {"g3bp1_dimer": n_g3bp1, "rna": n_rna}
    if n_usp10 > 0:
        composition["usp10"] = n_usp10

    key = jax.random.PRNGKey(seed)
    system = build_system(composition, box_size, key, rna_beads=10, rna_exposure=1.0)

    n_particles = system["positions"].shape[0]
    print(f"  USP10: {n_usp10}, particles: {n_particles}, sites: {system['topology'].n_sites}")

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
        save_every=n_steps,
        binding_interval=2,
        n_binding_attempts=50,
        conformational_interval=5,
        rng_key=key,
        verbose=False,
    )

    labels = detect_clusters(result["final_positions"], result["box_size"], cutoff=5.0)
    stats = cluster_statistics(labels, result["final_positions"],
                               result["box_size"], result["particle_types"])

    return {
        "n_usp10": n_usp10,
        "ratio": n_usp10 / (2 * n_g3bp1),
        "largest_cluster_fraction": float(stats["largest_cluster_fraction"]),
        "largest_cluster_size": int(stats["largest_cluster_size"]),
        "n_bonds": int(result["binding_state"].n_bound),
        "n_particles": n_particles,
    }


def main():
    n_g3bp1 = 20
    n_rna = 40
    box_size = 60.0
    n_steps = 6000

    # USP10 titration: 0 to 40 (each dimer has 2 NTF2 pockets -> 40 pockets)
    usp10_counts = [0, 5, 10, 20, 30, 40]

    print("USP10 Competition Titration")
    print(f"G3BP1 dimers: {n_g3bp1}, RNA: {n_rna}, box: {box_size} nm")
    print(f"NTF2 pockets: {2 * n_g3bp1}")
    print()

    results = []
    for n_usp10 in usp10_counts:
        print(f"\nRunning USP10 = {n_usp10} (ratio = {n_usp10/(2*n_g3bp1):.2f})...")
        res = run_titration_point(n_g3bp1, n_rna, n_usp10, box_size, n_steps)
        results.append(res)
        print(f"  -> Largest cluster: {res['largest_cluster_size']} "
              f"({res['largest_cluster_fraction']:.1%}), bonds: {res['n_bonds']}")

    # Summary
    print("\n" + "=" * 70)
    print(f"{'USP10':>6} | {'Ratio':>6} | {'Largest':>8} | {'Fraction':>9} | {'Bonds':>5}")
    print("-" * 70)
    for r in results:
        print(f"{r['n_usp10']:>6d} | {r['ratio']:>6.2f} | "
              f"{r['largest_cluster_size']:>8d} | {r['largest_cluster_fraction']:>8.1%} | "
              f"{r['n_bonds']:>5d}")

    # Plot
    outdir = "results/usp10_titration"
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ratios = [r["ratio"] for r in results]
    fracs = [r["largest_cluster_fraction"] for r in results]
    bonds = [r["n_bonds"] for r in results]

    ax = axes[0]
    ax.plot(ratios, fracs, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.set_xlabel("USP10 / NTF2 pocket ratio")
    ax.set_ylabel("Largest cluster fraction")
    ax.set_title("USP10 inhibition of phase separation")
    ax.axhline(y=fracs[0], color="gray", linestyle="--", alpha=0.5, label="No USP10")
    ax.legend()

    ax = axes[1]
    ax.plot(ratios, bonds, "s-", color="coral", linewidth=2, markersize=8)
    ax.set_xlabel("USP10 / NTF2 pocket ratio")
    ax.set_ylabel("Number of bonds")
    ax.set_title("Binding events vs USP10")

    plt.tight_layout()
    plt.savefig(f"{outdir}/titration.png", dpi=150)
    print(f"\nPlot saved to {outdir}/titration.png")
    plt.close()


if __name__ == "__main__":
    main()
