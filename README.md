# Stress Granule Simulator (`sgsim`)

A coarse-grained molecular dynamics engine built in JAX to simulate the assembly, phase separation, and material properties of stress granules. 

It natively supports saturable competitive binding pockets and continuous conformational switching to model G3BP1-RNA physics.

## Installation

We recommend using a conda environment to manage dependencies:

```bash
# 1. Create and activate a new environment
conda create -n sgsim python=3.10
conda activate sgsim

# 2. Install the package in editable mode with dependencies
pip install -e "."
```

*Note: The engine uses JAX. For GPU acceleration (e.g., NVIDIA RTX 3090), ensure you have the appropriate CUDA drivers installed. JAX will automatically detect and utilize the GPU.*

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
