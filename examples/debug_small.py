#!/usr/bin/env python
"""Small system debug: 2 G3BP1 dimers + 4 RNA in a tight box.

Visual check that bonds stay intact and molecules hold together.
Exports per-frame XYZ with bond connectivity info.
"""

import jax
import jax.numpy as jnp
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgsim.setup import build_system
from sgsim.simulate import run_full_simulation
from sgsim.io import export_xyz
from sgsim.types import PARTICLE_TYPES as PT


def main():
    composition = {"g3bp1_dimer": 2, "rna": 4}
    box_size = 25.0  # tight
    n_steps = 2000

    print(f"Composition: {composition}")
    print(f"Box: {box_size} nm")

    key = jax.random.PRNGKey(99)
    system = build_system(composition, box_size, key, rna_beads=10, rna_exposure=1.0)
    topo = system["topology"]

    n = system["positions"].shape[0]
    print(f"Particles: {n}")
    print(f"Bonds: {topo.n_bonds}")

    # Print molecule layout
    ptypes = np.asarray(system["particle_types"])
    mol_ids = np.asarray(system["molecule_ids"])
    type_names = {v: k for k, v in PT.items()}
    print("\nParticle layout:")
    for i in range(n):
        print(f"  [{i:3d}] mol={mol_ids[i]:2d}  type={type_names.get(int(ptypes[i]), '?'):15s}")

    print("\nBonds:")
    bond_pairs = np.asarray(topo.bond_pairs)
    bond_types = np.asarray(topo.bond_types)
    bt_names = ["DOMAIN_STIFF", "IDR_FLEX", "DIMER", "RNA_BACKBONE"]
    for b in range(topo.n_bonds):
        i, j = bond_pairs[b]
        print(f"  {i:3d} -- {j:3d}  ({bt_names[bond_types[b]]})")

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
        save_every=100,
        binding_interval=2,
        n_binding_attempts=10,
        conformational_interval=5,
        gamma_phi=1.0,
        rng_key=key,
        verbose=True,
    )

    # Check bond distances at each saved frame
    traj = result["trajectory"]
    box = result["box_size"]
    print(f"\n{'='*60}")
    print("Bond distance check across trajectory")
    print(f"{'='*60}")

    for fi, pos in enumerate(traj):
        pos = jnp.array(pos)
        r_i = pos[bond_pairs[:, 0]]
        r_j = pos[bond_pairs[:, 1]]
        dr = r_i - r_j
        dr = dr - box * jnp.round(dr / box)
        dists = jnp.sqrt(jnp.sum(dr**2, axis=-1))
        d = np.asarray(dists)
        print(f"  Frame {fi:3d} (step {fi*100:5d}): "
              f"mean={d.mean():.3f}, max={d.max():.3f}, min={d.min():.3f} nm")

    # Save trajectory
    outdir = "results/debug_small"
    os.makedirs(outdir, exist_ok=True)

    export_xyz(
        f"{outdir}/trajectory.xyz",
        np.stack([np.asarray(pos) for pos in traj]),
        result["particle_types"], box_size=box,
    )
    print(f"\nXYZ saved to {outdir}/trajectory.xyz")

    # Also save a LAMMPS-style data file with bonds for OVITO
    final_pos = np.asarray(result["final_positions"])
    with open(f"{outdir}/final_with_bonds.data", "w") as f:
        f.write("LAMMPS data file\n\n")
        f.write(f"{n} atoms\n")
        f.write(f"{topo.n_bonds} bonds\n")
        f.write(f"{len(set(ptypes))} atom types\n")
        f.write(f"{len(set(bond_types))} bond types\n\n")
        bx = np.asarray(box)
        f.write(f"0.0 {bx[0]:.4f} xlo xhi\n")
        f.write(f"0.0 {bx[1]:.4f} ylo yhi\n")
        f.write(f"0.0 {bx[2]:.4f} zlo zhi\n\n")
        f.write("Atoms\n\n")
        for i in range(n):
            f.write(f"{i+1} {mol_ids[i]+1} {ptypes[i]+1} "
                    f"{final_pos[i,0]:.4f} {final_pos[i,1]:.4f} {final_pos[i,2]:.4f}\n")
        f.write("\nBonds\n\n")
        for b in range(topo.n_bonds):
            i, j = bond_pairs[b]
            f.write(f"{b+1} {bond_types[b]+1} {i+1} {j+1}\n")

    print(f"LAMMPS data file saved to {outdir}/final_with_bonds.data")
    print("  (Load in OVITO: File > Load, then bonds will be visible)")


if __name__ == "__main__":
    main()
