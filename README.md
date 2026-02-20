# Stress Granule Simulator (`sgsim`)

A coarse-grained molecular dynamics engine built in JAX to simulate the assembly, phase separation, and material properties of stress granules. 

It natively supports saturable competitive binding pockets and continuous conformational switching to model G3BP1-RNA physics.

## Installation

We recommend using a conda environment to manage dependencies:

```bash
# 1. Clone the repository
git clone https://github.com/aakashxtha/sgsim.git
cd sgsim

# 2. Create and activate a new environment
conda create -n sgsim python=3.10
conda activate sgsim

# 3. Install the package in editable mode with CPU dependencies
pip install -e "."

# 4. If you have an NVIDIA GPU (e.g., RTX 3090), force-install the CUDA 12 bindings
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

*Note: JAX will automatically detect and utilize the GPU. The extra GPU installation step ensures the correct pre-compiled CUDA binaries are downloaded directly from Google and avoids Numpy 2.0 extension breakages.*

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
