# VLA-π0 Demo

A demonstration repository for **Vision-Language-Action (VLA)** models using **OpenPI** (pi0) with **ALOHA** robots for manipulation tasks like towel folding and object handling. This repository includes training, evaluation, and inference capabilities.

## Quick Start

```bash
# Basic inference demo
python examples/basic_inference.py --model pi0_aloha_towel --prompt "fold the towel neatly"

# Full robot simulation
python examples/aloha_sim_demo.py --model pi0_aloha_towel --task towel_folding

# Explore available datasets
python scripts/explore_datasets.py

# Train custom model (requires --exp-name)
python scripts/train.py --config pi0_aloha --dataset aloha_towel --exp-name my_experiment

# Evaluate performance with task-specific evaluation
python scripts/evaluate.py --model pi0_aloha_towel --task towel_folding --use-pretrained --evaluation-mode task_specific --episodes 10
```

## Available Models

- **`pi0_aloha_towel`**: Towel folding specialist
- **`pi0_aloha_tupperware`**: Food container manipulation  
- **`pi0_aloha_sim`**: Simulation-optimized model
- **`pi0_aloha`**: General ALOHA foundation model
- **`pi0_libero`**: Foundation model for fine-tuning on custom datasets

## Available Tasks

- **`towel_folding`**: Bimanual towel manipulation
- **`food_manipulation`**: Pick up food and place in container  
- **`tupperware`**: Put lid on tupperware container
- **`object_transfer`**: Pick up and transfer objects

## Evaluation Modes

- **Task-Specific Mode**: Realistic evaluation with progress tracking and task-specific metrics
- **Random Mode**: Simple simulation mode for basic testing

## Requirements

**Hardware**: NVIDIA GPU (8GB+ VRAM), Optional: ALOHA robot system  
**Software**: Python 3.11+, CUDA 11.8+, OpenPI, JAX


## File Structure

```
VLA-pi-demo/
├── examples/           # Demo scripts
├── scripts/           # Training, evaluation, data exploration  
├── docs/              # Documentation and tutorials
├── data/              # Datasets for training and evaluation
└── openpi/            # OpenPI library 
```

## Data

This repository includes pre-downloaded datasets for immediate use:

### Datasets (`data/datasets/`)
- **4 ALOHA datasets** (145K+ samples total): `aloha_towel`, `aloha_static_coffee`, `aloha_static_vinh_cup`, `aloha_sim_transfer_cube_human`
- **Ready for training**: HuggingFace format, 14 DOF bimanual data
- **Exploration tool**: Run `python scripts/explore_datasets.py` to analyze


## Evaluation

### Detailed Evaluation (`scripts/evaluate.py`)
Task-specific evaluation with custom checkpoint support and reference dataset comparison.

```bash
# With pre-trained model
python scripts/evaluate.py --model pi0_aloha_towel --task towel_folding --use-pretrained --episodes 10
```

### Quick Demo (`examples/aloha_sim_demo.py`)
Fast prototyping with interactive feedback.

```bash
# Demo evaluation
python examples/aloha_sim_demo.py --model pi0_aloha_towel --task towel_folding --episodes 5
```

## Documentation

- **[Beginner's Guide](docs/beginner_guide.md)** - Overview and file explanations
- **[Installation Guide](docs/installation.md)** - Setup requirements and troubleshooting  
- **[Getting Started Tutorial](docs/getting_started.md)** - Step-by-step walkthrough
- **[Dataset Documentation](data/datasets/README.md)** - Dataset specifications and usage examples

## License

This project is provided for educational purposes. OpenPI is released under its own license terms.

## Acknowledgments

- **Physical Intelligence** for developing and open-sourcing OpenPI
- **ALOHA** team for the bimanual manipulation platform
- **LeRobot** project for dataset formats and tools 