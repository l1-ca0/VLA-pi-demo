# VLA-π Demo Repository - Complete Beginner's Guide

## What is this repository?

This is a **demonstration repository** for Vision-Language-Action (VLA) models using **OpenPI** (π₀) with **ALOHA** robots. It shows how to:
- Load pre-trained AI models that understand vision, language, and robot actions
- Control ALOHA bimanual robots for tasks like towel folding and object manipulation
- Train and fine-tune models for custom robotic tasks
- Evaluate model performance on real hardware

---

## Essential Files & Directory Structure

### Demo Files (examples/)

| File | What it does | When to use |
|------|-------------|-------------|
| `basic_inference.py` | Shows how to load a pre-trained model and run inference on robot tasks | First demo to understand how AI models work |
| `aloha_sim_demo.py` | Complete robot demonstration with episode running and action execution | See full robot control workflow |

**See README.md for basic command examples**

### Workflow Scripts (scripts/)

| File | What it does | When to use |
|------|-------------|-------------|
| `explore_datasets.py` | Analyzes available training datasets, shows statistics and data structure | Before training - understand your data |
| `train.py` | Trains/fine-tunes models on datasets using LoRA or full fine-tuning | Create custom models for your tasks |
| `evaluate.py` | Tests model performance, calculates success rates and metrics | After training - see how well your model works |

**See README.md for complete command examples**

### Documentation (docs/)

| File | What it contains | When to read |
|------|-----------------|-------------|
| `getting_started.md` | Complete setup tutorial with step-by-step instructions | Before starting - detailed setup guide |
| `installation.md` | Installation requirements and troubleshooting | When having setup issues |
| `beginner_guide.md` | This file - overview and explanations for newcomers | Start here for repository overview |

### Data (data/)

| Directory | What it contains | Purpose |
|-----------|-----------------|---------|
| `datasets/` | Training data for robot tasks (4 ALOHA datasets, ~28MB total) | Used for training custom models |

**Note**: The repository stores training outputs in `output/` (gitignored) and model cache in `cache/` (gitignored).

---

## Technical Requirements

### Hardware
- **GPU**: NVIDIA with 8GB+ VRAM (16GB+ for training)
- **RAM**: 16GB+ system memory
- **ALOHA Robot**: Optional but recommended for full functionality

### Software
- **Python**: 3.11+ 
- **CUDA**: 11.8+ or 12.x (for GPU acceleration)
- **ROS 2**: For hardware integration (optional)
- **OpenPI**: Vision-language-action model library
- **Dependencies**: JAX, Optax, NumPy, OpenCV, Datasets, H5PY (auto-installed)

---

## Available Datasets

The repository includes 4 pre-downloaded ALOHA datasets:

| Dataset | Size | Episodes | Task Type |
|---------|------|----------|-----------|
| `aloha_towel` | 3.7MB | 50 | Towel folding simulation |
| `aloha_static_coffee` | 11MB | 50 | Coffee manipulation |
| `aloha_static_vinh_cup` | 9.4MB | 101 | Cup handling |
| `aloha_sim_transfer_cube_human` | 3.0MB | 50 | Cube transfer simulation |

**Total**: 145K+ samples across 251 episodes

---

## Quick Start Checklist

- [ ] Read `README.md` for overview
- [ ] Follow `docs/installation.md` for setup
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Test basic functionality: `python examples/basic_inference.py --help`
- [ ] Explore data: `python scripts/explore_datasets.py`
- [ ] Run your first demo: `python examples/aloha_sim_demo.py --model pi0_aloha_towel --task towel_folding`

## Next Steps

1. **Try Examples**: Start with `basic_inference.py` for simple AI model interaction
2. **Explore Data**: Run `explore_datasets.py` to understand available training data
3. **Run Simulation**: Use `aloha_sim_demo.py` for complete robot workflow
4. **Training**: Use `train.py` to create custom models (requires sufficient GPU)
5. **Evaluation**: Use `evaluate.py` to test model performance with metrics

