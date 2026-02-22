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

**Requirements:**
- CUDA **12.1 or later**
- cuDNN **9.x** (cuDNN 8.x is not supported by JAX 0.9+)
- NVIDIA driver **535 or later**
- Linux (or WSL2 on Windows) — macOS has no CUDA support

```bash
git clone https://github.com/aakashxtha/sgsim.git
cd sgsim
conda create -n sgsim python=3.11 -y
conda activate sgsim

# Install JAX with CUDA 12 support, then sgsim
pip install "jax[cuda12]>=0.9.0"
pip install -e ".[dev]"
```

To verify GPU detection:
```bash
python -c "import jax; print(jax.devices())"
# Should show: [CudaDevice(id=0)]
```

**Troubleshooting:**

If you see `[CpuDevice(id=0)]` — JAX picked up a CPU build. Fix:
```bash
pip uninstall jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt -y
pip install --no-cache-dir "jax[cuda12]>=0.9.0"
```

If you get a CUDA initialization error — check your cuDNN version:
```bash
python -c "import jaxlib; print(jaxlib.__version__)"
nvidia-smi   # driver version must be ≥535
# cuDNN 9.x must be installed (not 8.x)
```

**Note:** JAX GPU requires Linux (or WSL2 on Windows). Native Windows and macOS do not have CUDA GPU support.

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
