# Installation Guide

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ for training)
- **Memory**: 16GB+ RAM
- **Storage**: 10GB+ free space
- **OS**: Ubuntu 22.04+ (recommended) or macOS

### Software Requirements

- Python 3.11+
- CUDA 11.8+ (for GPU acceleration)
- Git

## Installation

### 1. Clone Repository

```bash
git clone --recurse-submodules https://github.com/l1-ca0/VLA-pi-demo.git
cd VLA-pi-demo
```

### 2. Create Environment

```bash
# Create virtual environment (replace python3.11 with your Python version)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install demo requirements
pip install -r requirements.txt

# Install OpenPI from submodule
cd openpi
pip install -e .
cd ..
```

## Verification

Test your installation:

```bash
# Test OpenPI import
python -c "import openpi; print('OpenPI ready!')"

# Test basic functionality
python examples/basic_inference.py --help
python examples/aloha_sim_demo.py --help

# Test JAX/GPU (if available)
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# Test core dependencies
python -c "import numpy, cv2, datasets, h5py, optax; print('All dependencies working!')"
```

## Quick Start

After installation, try running a demo:

```bash
# Basic inference demo
python examples/basic_inference.py --model pi0_aloha_towel --prompt "fold the towel"

# Simulation demo
python examples/aloha_sim_demo.py --model pi0_aloha_towel --task towel_folding --steps 10

# Explore available datasets
python scripts/explore_datasets.py
```

## Troubleshooting

### Common Issues

**OpenPI import errors:**
```bash
# Reinstall OpenPI
cd openpi && pip install -e . && cd ..
```

**Missing dependencies:**
```bash
# Reinstall requirements
pip install -r requirements.txt
```

**GPU not detected:**
```bash
# Check CUDA
nvidia-smi
# Test JAX GPU support
python -c "import jax; print('JAX devices:', jax.devices())"
```

**Package conflicts:**
```bash
# Clean reinstall
rm -rf venv
python3 -m venv venv  # Use your Python version
source venv/bin/activate
pip install -r requirements.txt
cd openpi && pip install -e . && cd ..
```

**JAX/Training issues:**
```bash
# Verify JAX installation
python -c "import jax, optax; print('JAX training ready!')"
```

### Environment Notes

- **Conda users**: The installation works with conda environments
- **Apple Silicon**: JAX works but may use CPU-only mode
- **Linux**: Recommended for full GPU acceleration with CUDA
- **Windows**: Use WSL2 for best compatibility

### Version Compatibility

- **Python**: 3.11+ (tested with 3.11, 3.12)
- **CUDA**: 11.8+ or 12.x for GPU support
- **JAX**: Automatically installed with OpenPI
- **PyTorch**: Automatically installed with OpenPI
