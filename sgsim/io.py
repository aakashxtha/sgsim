"""Trajectory I/O, checkpointing, and export.

Supports:
- Zarr-based trajectory storage (chunked, compressed, appendable)
- Full state checkpointing via numpy + JSON metadata
- XYZ export for OVITO/VMD visualization
"""

import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path


def save_trajectory_zarr(
    path: str,
    trajectory: list,
    particle_types: jnp.ndarray = None,
    box_size: jnp.ndarray = None,
    metadata: dict = None,
):
    """Save trajectory to zarr format.

    Args:
        path: directory path for zarr store
        trajectory: list of (N, 3) position arrays
        particle_types: (N,) int32 (saved once)
        box_size: (3,) float32 (saved once)
        metadata: optional dict of simulation parameters
    """
    import zarr

    store = zarr.open(path, mode="w")

    # Stack trajectory: (n_frames, N, 3)
    positions = np.stack([np.asarray(pos) for pos in trajectory])
    store.create_array("positions", data=positions, chunks=(1, positions.shape[1], 3))

    if particle_types is not None:
        store.create_array("particle_types", data=np.asarray(particle_types))
    if box_size is not None:
        store.create_array("box_size", data=np.asarray(box_size))
    if metadata:
        store.attrs.update(metadata)


def load_trajectory_zarr(path: str) -> dict:
    """Load trajectory from zarr format.

    Args:
        path: directory path to zarr store

    Returns:
        dict with keys: positions (n_frames, N, 3), particle_types, box_size, metadata
    """
    import zarr

    store = zarr.open(path, mode="r")

    result = {
        "positions": jnp.array(store["positions"][:]),
    }

    if "particle_types" in store:
        result["particle_types"] = jnp.array(store["particle_types"][:])
    if "box_size" in store:
        result["box_size"] = jnp.array(store["box_size"][:])

    result["metadata"] = dict(store.attrs)
    return result


def save_checkpoint(
    path: str,
    positions: jnp.ndarray,
    particle_types: jnp.ndarray,
    particle_charges: jnp.ndarray,
    particle_radii: jnp.ndarray,
    molecule_ids: jnp.ndarray,
    box_size: jnp.ndarray,
    step: int,
    metadata: dict = None,
):
    """Save a full checkpoint as numpy arrays + JSON metadata.

    Args:
        path: directory for checkpoint files
        positions, particle_types, ...: simulation state arrays
        step: current step number
        metadata: optional dict of parameters
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    np.save(p / "positions.npy", np.asarray(positions))
    np.save(p / "particle_types.npy", np.asarray(particle_types))
    np.save(p / "particle_charges.npy", np.asarray(particle_charges))
    np.save(p / "particle_radii.npy", np.asarray(particle_radii))
    np.save(p / "molecule_ids.npy", np.asarray(molecule_ids))
    np.save(p / "box_size.npy", np.asarray(box_size))

    meta = {"step": int(step)}
    if metadata:
        meta.update(metadata)
    with open(p / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint.

    Returns:
        dict with positions, particle_types, charges, radii, molecule_ids,
        box_size, step, metadata
    """
    p = Path(path)

    result = {
        "positions": jnp.array(np.load(p / "positions.npy")),
        "particle_types": jnp.array(np.load(p / "particle_types.npy")),
        "particle_charges": jnp.array(np.load(p / "particle_charges.npy")),
        "particle_radii": jnp.array(np.load(p / "particle_radii.npy")),
        "molecule_ids": jnp.array(np.load(p / "molecule_ids.npy")),
        "box_size": jnp.array(np.load(p / "box_size.npy")),
    }

    with open(p / "metadata.json") as f:
        meta = json.load(f)

    result["step"] = meta.pop("step")
    result["metadata"] = meta
    return result


def export_xyz(
    path: str,
    positions: jnp.ndarray,
    particle_types: jnp.ndarray = None,
    type_names: dict = None,
    box_size: jnp.ndarray = None,
    append: bool = False,
):
    """Export positions to XYZ format for visualization.

    Args:
        path: output file path (.xyz)
        positions: (N, 3) or (n_frames, N, 3) positions
        particle_types: (N,) int32 for atom names
        type_names: dict mapping type_id -> string name
        box_size: (3,) for comment line
        append: if True, append to existing file
    """
    if type_names is None:
        from .types import PARTICLE_TYPE_NAMES
        type_names = PARTICLE_TYPE_NAMES

    pos = np.asarray(positions)
    if pos.ndim == 2:
        pos = pos[np.newaxis]  # single frame

    ptypes = np.asarray(particle_types) if particle_types is not None else None

    mode = "a" if append else "w"
    with open(path, mode) as f:
        for frame in pos:
            n = frame.shape[0]
            comment = ""
            if box_size is not None:
                bx = np.asarray(box_size)
                comment = f"box={bx[0]:.3f},{bx[1]:.3f},{bx[2]:.3f}"

            f.write(f"{n}\n")
            f.write(f"{comment}\n")
            for i in range(n):
                name = "X"
                if ptypes is not None:
                    name = type_names.get(int(ptypes[i]), f"T{ptypes[i]}")
                f.write(f"{name} {frame[i, 0]:.4f} {frame[i, 1]:.4f} {frame[i, 2]:.4f}\n")
