# Stress Granule Simulator (`sgsim`)

A coarse-grained molecular dynamics engine built in JAX to simulate the assembly, phase separation, and material properties of stress granules. 

It natively supports saturable competitive binding pockets and continuous conformational switching to model G3BP1-RNA physics.

## Installation

### CPU only (macOS / Linux)

```bash
git clone https://github.com/aakashxtha/sgsim.git
cd sgsim
conda create -n sgsim python=3.11 -y
conda activate sgsim
pip install -e ".[dev]"
```

### GPU (NVIDIA, CUDA 12)

> **⚠️ WINDOWS USERS**: Google JAX *does not* support native Windows for GPUs. If you are on a Windows gaming PC, you **must** install and run this inside **WSL2 (Windows Subsystem for Linux)**. If you run these commands in standard Windows PowerShell or Anaconda Prompt, pip will fail to find the `cuda12` packages and silently fall back to CPU.

```bash
git clone https://github.com/aakashxtha/sgsim.git
cd sgsim
conda create -n sgsim python=3.11 -y
conda activate sgsim

# Install JAX with CUDA 12 support FIRST
# --no-cache-dir prevents pip from reusing a cached CPU jaxlib
pip install --no-cache-dir -U "jax[cuda12]"

# Then install sgsim (jax is already satisfied, won't be reinstalled)
pip install --no-deps -e .
pip install "numpy>=1.24" "scipy>=1.11" "matplotlib>=3.8" "zarr>=2.16" pytest pytest-xdist
```

To verify GPU detection:
```bash
python -c "import jax; print(jax.devices())"
# Should show: [CudaDevice(id=0)]
```

**Troubleshooting:** If you see `[CpuDevice(id=0)]` instead, pip cached a CPU `jaxlib`. Fix with:
```bash
pip uninstall jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt -y
pip install --no-cache-dir -U "jax[cuda12]"
```

## Usage

We provide several pre-configured simulation scripts to demonstrate the capabilities.

### Production Run (Macroscopic Scale)
To run a massive-scale, multicomponent GPU-optimized droplet simulation (takes ~3-5 minutes on an RTX 3090):
```bash
python scripts/run_production_rtx3090.py
```

### Examples
There are also smaller examples available for rapid testing:
- **Phase separation base case**: `python examples/g3bp1_rna_phase_sep.py`
- **Dense droplet scaling**: `python examples/dense_phase_sep.py`
- **Valence capping / competitive bounding**: `python examples/usp10_competition.py`
- **Full complex multicomponent system**: `python examples/multicomponent.py`

### Visualizing Results
All scripts automatically generate `.png` graphical analytics dashboards and export standard mult-frame `.xyz` trajectory files into the `results/` directory.

To watch your droplets form dynamically, open the `.xyz` files using a molecular visualizer like **[OVITO](https://www.ovito.org/)** or **VMD**.
