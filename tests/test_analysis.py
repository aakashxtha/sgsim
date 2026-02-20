"""Tests for analysis tools and I/O.

Validates:
- RDF normalization and peak detection
- MSD computation
- Cluster detection via label propagation
- Zarr trajectory I/O
- Checkpoint save/load roundtrip
- XYZ export
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from sgsim.analysis import (
    compute_rdf, compute_msd, detect_clusters,
    cluster_statistics, compute_density_profile,
)
from sgsim.io import (
    save_trajectory_zarr, load_trajectory_zarr,
    save_checkpoint, load_checkpoint,
    export_xyz,
)


class TestRDF:
    """Tests for radial distribution function."""

    def test_rdf_peak_at_spacing(self):
        """RDF of a regular grid should show a peak at the grid spacing."""
        # 4 particles in a line with spacing 5 nm
        positions = jnp.array([
            [5.0, 25.0, 25.0],
            [10.0, 25.0, 25.0],
            [15.0, 25.0, 25.0],
            [20.0, 25.0, 25.0],
        ])
        box_size = jnp.array([50.0, 50.0, 50.0])
        particle_types = jnp.zeros(4, dtype=jnp.int32)

        r, gr = compute_rdf(positions, box_size, jnp.array([0]), jnp.array([0]),
                            particle_types, n_bins=50, r_max=25.0)

        # g(r) should have a peak near r = 5.0
        peak_idx = jnp.argmax(gr)
        assert jnp.abs(r[peak_idx] - 5.0) < 1.5, \
            f"RDF peak at {r[peak_idx]}, expected ~5.0"

    def test_rdf_zero_at_short_range(self):
        """g(r) should be ~0 at distances shorter than particle separation."""
        positions = jnp.array([
            [10.0, 25.0, 25.0],
            [20.0, 25.0, 25.0],
        ])
        box_size = jnp.array([50.0, 50.0, 50.0])
        particle_types = jnp.zeros(2, dtype=jnp.int32)

        r, gr = compute_rdf(positions, box_size, jnp.array([0]), jnp.array([0]),
                            particle_types, n_bins=50, r_max=25.0)

        # g(r) at r < 5 should be zero (no pairs at that distance)
        short_range = gr[r < 5.0]
        assert jnp.all(short_range < 0.1), "g(r) should be ~0 at short range"


class TestMSD:
    """Tests for mean squared displacement."""

    def test_msd_zero_at_t0(self):
        """MSD at t=0 should be zero."""
        traj = [
            jnp.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]),
            jnp.array([[1.0, 0.0, 0.0], [5.0, 6.0, 5.0]]),
        ]
        box_size = jnp.array([50.0, 50.0, 50.0])
        msd = compute_msd(traj, box_size)
        assert jnp.abs(msd[0]) < 1e-6, "MSD at t=0 should be 0"

    def test_msd_positive_later(self):
        """MSD should be positive at later times."""
        traj = [
            jnp.array([[0.0, 0.0, 0.0]]),
            jnp.array([[3.0, 4.0, 0.0]]),  # displacement = 5
        ]
        box_size = jnp.array([50.0, 50.0, 50.0])
        msd = compute_msd(traj, box_size)
        assert msd[1] > 0, "MSD should be positive at t>0"
        assert jnp.abs(msd[1] - 25.0) < 0.1, f"MSD should be 25 (5^2), got {msd[1]}"


class TestClusterDetection:
    """Tests for cluster detection via label propagation."""

    def test_two_separate_clusters(self):
        """Two groups of particles far apart should form two clusters."""
        positions = jnp.array([
            [5.0, 5.0, 5.0],
            [7.0, 5.0, 5.0],   # cluster 1
            [40.0, 40.0, 40.0],
            [42.0, 40.0, 40.0],  # cluster 2
        ])
        box_size = jnp.array([80.0, 80.0, 80.0])
        labels = detect_clusters(positions, box_size, cutoff=5.0)

        # Particles 0,1 should share a label; 2,3 should share a different label
        assert labels[0] == labels[1], "Particles 0,1 should be in same cluster"
        assert labels[2] == labels[3], "Particles 2,3 should be in same cluster"
        assert labels[0] != labels[2], "The two clusters should have different labels"

    def test_single_cluster(self):
        """All close particles should form one cluster."""
        positions = jnp.array([
            [5.0, 5.0, 5.0],
            [7.0, 5.0, 5.0],
            [9.0, 5.0, 5.0],
        ])
        box_size = jnp.array([50.0, 50.0, 50.0])
        labels = detect_clusters(positions, box_size, cutoff=5.0)

        assert labels[0] == labels[1] == labels[2], "All should be in one cluster"

    def test_cluster_statistics(self):
        """Cluster statistics should report correct sizes."""
        positions = jnp.array([
            [5.0, 5.0, 5.0],
            [7.0, 5.0, 5.0],
            [9.0, 5.0, 5.0],
            [40.0, 40.0, 40.0],
        ])
        box_size = jnp.array([80.0, 80.0, 80.0])
        labels = detect_clusters(positions, box_size, cutoff=5.0)
        ptypes = jnp.zeros(4, dtype=jnp.int32)

        stats = cluster_statistics(labels, positions, box_size, ptypes)
        assert stats["largest_cluster_size"] == 3


class TestDensityProfile:
    """Tests for 1D density profiles."""

    def test_uniform_density(self):
        """Uniformly distributed particles should give ~flat density."""
        key = jax.random.PRNGKey(0)
        n = 500
        box_size = jnp.array([50.0, 50.0, 50.0])
        positions = jax.random.uniform(key, (n, 3)) * box_size

        centers, density = compute_density_profile(positions, box_size, n_bins=10)

        # Expected density = n / volume
        expected = n / jnp.prod(box_size)
        # Should be roughly uniform (within 50% for 500 particles)
        assert jnp.all(density > 0), "All bins should have particles"
        mean_density = jnp.mean(density)
        assert jnp.abs(mean_density - expected) / expected < 0.2, \
            f"Mean density {mean_density} should be ~{expected}"


class TestZarrIO:
    """Tests for zarr trajectory I/O."""

    def test_save_load_roundtrip(self):
        """Saved trajectory should match loaded trajectory."""
        traj = [
            jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            jnp.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]),
        ]
        ptypes = jnp.array([0, 1], dtype=jnp.int32)
        box = jnp.array([50.0, 50.0, 50.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.zarr")
            save_trajectory_zarr(path, traj, ptypes, box, {"n_steps": 100})
            loaded = load_trajectory_zarr(path)

            assert loaded["positions"].shape == (2, 2, 3)
            assert jnp.allclose(loaded["positions"][0], traj[0], atol=1e-5)
            assert jnp.allclose(loaded["particle_types"], ptypes)
            assert loaded["metadata"]["n_steps"] == 100


class TestCheckpoint:
    """Tests for checkpoint save/load."""

    def test_checkpoint_roundtrip(self):
        """Checkpoint save+load should preserve state."""
        positions = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ptypes = jnp.array([0, 1], dtype=jnp.int32)
        charges = jnp.array([-1.0, 2.0])
        radii = jnp.array([1.5, 2.0])
        mol_ids = jnp.array([0, 0], dtype=jnp.int32)
        box = jnp.array([50.0, 50.0, 50.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt")
            save_checkpoint(path, positions, ptypes, charges, radii, mol_ids, box, step=42)
            loaded = load_checkpoint(path)

            assert jnp.allclose(loaded["positions"], positions)
            assert loaded["step"] == 42
            assert jnp.allclose(loaded["box_size"], box)


class TestXYZExport:
    """Tests for XYZ file export."""

    def test_xyz_write(self):
        """XYZ export should produce a valid file."""
        positions = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ptypes = jnp.array([0, 4], dtype=jnp.int32)
        box = jnp.array([50.0, 50.0, 50.0])

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False, mode="r") as f:
            path = f.name

        try:
            export_xyz(path, positions, ptypes, box_size=box)

            with open(path) as f:
                lines = f.readlines()

            assert lines[0].strip() == "2", "First line should be atom count"
            assert "box=" in lines[1], "Comment should contain box info"
            assert len(lines) == 4, "Should have 4 lines (count + comment + 2 atoms)"
        finally:
            os.unlink(path)
