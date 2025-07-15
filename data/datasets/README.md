# ALOHA Datasets

This directory contains high-quality ALOHA datasets for robotic manipulation tasks. All datasets are from the official **LeRobot** project and are ready for immediate use with OpenPI models.

## Available Datasets

### 1. `aloha_towel` (25,000 samples, 50 episodes)
- **Source**: `lerobot/aloha_sim_insertion_human`
- **Task**: Simulation-based insertion tasks (suitable for towel folding)
- **Features**: `observation.state`, `action`, episode metadata
- **Dimensions**: 14 DOF (7 per arm for bimanual ALOHA)
- **Use Case**: Basic towel folding and manipulation tasks

### 2. `aloha_static_coffee` (55,000 samples, 50 episodes)
- **Source**: `lerobot/aloha_static_coffee`
- **Task**: Coffee-related manipulation tasks
- **Features**: `observation.state`, `observation.effort`, `action`, episode metadata
- **Dimensions**: 14 DOF (7 per arm for bimanual ALOHA)
- **Use Case**: Complex manipulation tasks with effort feedback

### 3. `aloha_static_vinh_cup` (45,500 samples, 101 episodes)
- **Source**: `lerobot/aloha_static_vinh_cup`
- **Task**: Cup manipulation tasks
- **Features**: `observation.state`, `observation.effort`, `action`, episode metadata
- **Dimensions**: 14 DOF (7 per arm for bimanual ALOHA)
- **Use Case**: Object manipulation and handling

### 4. `aloha_sim_transfer_cube_human` (20,000 samples, 50 episodes)
- **Source**: `lerobot/aloha_sim_transfer_cube_human`
- **Task**: Cube transfer tasks in simulation
- **Features**: `observation.state`, `action`, episode metadata
- **Dimensions**: 14 DOF (7 per arm for bimanual ALOHA)
- **Use Case**: Simulation-based training and testing

## Data Format

All datasets follow the **LeRobot standard format** with these common fields:

- `observation.state`: Robot joint positions (14 dimensions)
- `action`: Target joint positions (14 dimensions)
- `episode_index`: Episode identifier
- `frame_index`: Frame within episode
- `timestamp`: Time within episode
- `next.done`: Episode termination flag
- `index`: Global sample index
- `task_index`: Task identifier

Some datasets also include:
- `observation.effort`: Joint effort/torque information

## Quick Start

### Explore Datasets
```bash
# Get overview of all datasets
python scripts/explore_datasets.py
```

### Load a Dataset
```python
from datasets import load_from_disk

# Load a dataset
dataset = load_from_disk('./data/datasets/aloha_towel')
print(f"Total samples: {len(dataset)}")

# Access a sample
sample = dataset[0]
print(f"Observation state: {sample['observation.state']}")
print(f"Action: {sample['action']}")
```

### Use with Training Scripts
```bash
# Train with a dataset
python scripts/train.py --model pi0_libero --dataset aloha_towel --method lora

# Evaluate performance with task-specific evaluation
python examples/aloha_sim_demo.py --model pi0_aloha_towel --task towel_folding --evaluation-mode task_specific --episodes 10
```

## Dataset Selection Guide

- **Towel folding**: Use `aloha_towel` (simulation-based, good for initial training)
- **Complex manipulation**: Use `aloha_static_coffee` (includes effort feedback)
- **Object handling**: Use `aloha_static_vinh_cup` (good variety of episodes)  
- **Simulation testing**: Use `aloha_sim_transfer_cube_human` (clean simulation data)

## Dataset Statistics

| Dataset | Samples | Episodes | Avg Length | Features | Size |
|---------|---------|----------|------------|----------|------|
| aloha_towel | 25,000 | 50 | 500 | state, action | 3.7MB |
| aloha_static_coffee | 55,000 | 50 | 1,100 | state, effort, action | 11MB |
| aloha_static_vinh_cup | 45,500 | 101 | 450 | state, effort, action | 9.4MB |
| aloha_sim_transfer_cube_human | 20,000 | 50 | 400 | state, action | 3.0MB |
| **Total** | **145,500** | **251** | **587** | - | **~28MB** |


## Repository Integration

These datasets integrate seamlessly with the demo repository:

- **Training**: Use with `scripts/train.py` for custom model fine-tuning
- **Evaluation**: Test with `scripts/evaluate.py` for performance benchmarking  
- **Exploration**: Analyze with `scripts/explore_datasets.py` for data understanding
- **Examples**: Referenced in `examples/` for inference demonstrations

## Troubleshooting

**Loading Issues:**
```bash
pip install datasets>=2.14.0 h5py>=3.7.0
```

**Memory Issues:**
```python
# Load subset for testing
dataset = load_from_disk('./data/datasets/aloha_towel')
subset = dataset.select(range(1000))  # First 1000 samples
```

## Source Attribution

All datasets are from the **LeRobot** project:
- **Website**: https://github.com/huggingface/lerobot
- **Papers**: Citations available in original dataset repositories
- **License**: Follow LeRobot project licensing terms

For additional datasets, visit the [LeRobot Datasets](https://huggingface.co/lerobot) collection on HuggingFace. 