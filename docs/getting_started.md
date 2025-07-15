# Getting Started with VLA-π Demo

This tutorial walks you through VLA-π (OpenPI) demonstration.

## Quick Start Checklist

- [ ] Complete installation following [`docs/installation.md`](installation.md)
- [ ] Activate your Python environment
- [ ] Run your first demo
- [ ] Explore datasets and models
- [ ] Try training (optional)

## Prerequisites

Before starting this tutorial:

1. **Complete Installation**: Follow [`installation.md`](installation.md) for full setup
2. **GPU Access**: Recommended for inference (8GB+ VRAM), required for training (16GB+ VRAM)
3. **Environment**: Python 3.11+ with OpenPI installed

**Hardware Note**: ALOHA robot hardware is optional - all demos work in simulation mode.

## Step 1: Verify Installation

First, ensure everything is working:

```bash
# Activate your environment
source venv/bin/activate  # or: conda activate openpi

# Test OpenPI installation
python -c "import openpi; print('OpenPI ready!')"

# Check available models
python -c "from openpi.training import config; print('Models available')"
```

## Step 2: Run Your First Demo

### Basic Inference Demo

Start with a simple AI inference demonstration:

```bash
# Navigate to demo directory
cd VLA-pi-demo

# Run basic inference
python examples/basic_inference.py \
    --model pi0_aloha_towel \
    --prompt "fold the towel neatly"
```

**Expected Output:**
```
OpenPI ALOHA Inference Demo
Model pi0_aloha_towel loaded successfully!
Task: fold the towel neatly
Generated 50 actions for 50 timesteps
```

### Simulation Demo

Try a more complete simulation:

```bash
# Run ALOHA simulation demo
python examples/aloha_sim_demo.py \
    --model pi0_aloha_towel \
    --task towel_folding \
    --steps 20
```

This runs a complete task simulation with synthetic observations.

## Step 3: Explore Available Data

Discover what datasets and models are available:

```bash
# Explore datasets
python scripts/explore_datasets.py

# This will show:
# - Available datasets (ALOHA, etc.)
# - Dataset statistics
# - Sample data structure
```

## Step 4: Try Model Training (Optional)

If you have a compatible GPU (8GB+ VRAM):

```bash
# Quick training demo with LoRA fine-tuning
python scripts/train.py \
    --model pi0_aloha \
    --dataset aloha_towel \
    --method lora \
    --steps 100 \
    --learning-rate 1e-4
```

## Step 5: Evaluate Performance

Test model performance with task-specific evaluation:

```bash
# ALOHA evaluation with task-specific metrics
python examples/aloha_sim_demo.py \
    --model pi0_aloha_towel \
    --task towel_folding \
    --evaluation-mode task_specific \
    --episodes 5

# Compare with simple random evaluation
python examples/aloha_sim_demo.py \
    --model pi0_aloha_towel \
    --task towel_folding \
    --evaluation-mode random \
    --episodes 5
```

## Available Models

- **`pi0_aloha_towel`**: Towel folding specialist
- **`pi0_aloha_tupperware`**: Container manipulation
- **`pi0_aloha_sim`**: General simulation model
- **`pi0_aloha`**: General ALOHA foundation model
- **`pi0_libero`**: Foundation model for fine-tuning on custom datasets

## Available Tasks

- **`towel_folding`**: Fold towels neatly
- **`food_manipulation`**: Pick up food and place in container
- **`tupperware`**: Put lid on tupperware container  
- **`object_transfer`**: Pick up and transfer objects

## Evaluation Features

The repository includes comprehensive evaluation capabilities:

- **Task-Specific Mode**: Progress tracking and task-aware reward calculation
- **Random Mode**: Simple simulation mode for basic testing
- **Automatic Result Saving**: All evaluations save detailed JSON results
- **Progress Metrics**: Track task completion progress, coordination, and stability

## API Reference

If you want to use OpenPI directly in your own code:

### Basic Model Loading and Inference
```python
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.shared import download

# Load model configuration
config = _config.get_config("pi0_aloha_towel")

# Download and load pre-trained checkpoint
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_aloha_towel")
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Prepare observation (example structure)
observation = {
    "observation/exterior_image_1_left": image_array,  # Shape: (224, 224, 3)
    "observation/wrist_image_left": wrist_image,
    "observation/wrist_image_right": wrist_image,
    "observation.state": robot_state,  # Shape: (14,) for bimanual ALOHA
    "prompt": "fold the towel neatly"
}

# Run inference
result = policy.infer(observation)
actions = result["actions"]  # Shape: [50, 14] (timesteps × DOF)
```

### Available Model Configs
- `pi0_aloha_towel`: Towel folding specialist
- `pi0_aloha_tupperware`: Container manipulation
- `pi0_aloha_sim`: General simulation model
- `pi0_aloha`: General ALOHA foundation model
- `pi0_libero`: Foundation model for fine-tuning

For more advanced usage, training, and custom model development, see the examples in the `examples/` and `scripts/` directories.

