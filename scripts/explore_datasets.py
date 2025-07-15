#!/usr/bin/env python3
"""
Script to explore the downloaded ALOHA datasets.
This script helps you understand the structure and content of the datasets.
"""

import json
import random
from pathlib import Path
from datasets import load_from_disk
import numpy as np

def explore_dataset(dataset_path):
    """Explore a single dataset and print its properties."""
    print(f"\n{'='*60}")
    print(f"EXPLORING: {dataset_path}")
    print(f"{'='*60}")
    
    try:
        # Load dataset
        dataset = load_from_disk(dataset_path)
        print("Successfully loaded dataset")
        print(f"  - Total samples: {len(dataset)}")
        
        # Show dataset info
        info_path = Path(dataset_path) / "dataset_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
            print(f"  - Dataset description: {info.get('description', 'N/A')}")
            print(f"  - Dataset features: {list(info.get('features', {}).keys())}")
        
        # Analyze sample data
        sample = dataset[0]
        print("\nSample Data Structure:")
        for key, value in sample.items():
            if isinstance(value, list):
                print(f"  - {key}: list of length {len(value)}")
                if len(value) > 0:
                    print(f"    └─ First element type: {type(value[0])}")
            else:
                print(f"  - {key}: {type(value).__name__} = {value}")
        
        # Analyze episodes
        episodes = set(sample['episode_index'] for sample in dataset)
        print("\nDataset Statistics:")
        print(f"  - Number of episodes: {len(episodes)}")
        avg_samples = len(dataset) / len(episodes) if len(episodes) > 0 else 0.0
        print(f"  - Average samples per episode: {avg_samples:.1f}")
        
        # Analyze observation and action dimensions
        obs_state = sample['observation.state']
        action = sample['action']
        print(f"  - Observation state dimensions: {len(obs_state)}")
        print(f"  - Action dimensions: {len(action)}")
        
        # Efficiently analyze ranges using sampling for large datasets
        print("\nData Ranges (sampled analysis):")
        sample_size = min(1000, len(dataset))  # Sample max 1000 items for efficiency
        if len(dataset) > sample_size:
            print(f"  (Analyzing {sample_size} samples out of {len(dataset)} for efficiency)")
            # Use random sampling for better representation
            sample_indices = random.sample(range(len(dataset)), sample_size)
            sampled_data = [dataset[i] for i in sample_indices]
        else:
            sampled_data = dataset
        
        # Collect data in single iteration
        obs_values = []
        action_values = []
        episode_samples = {}
        
        for s in sampled_data:
            obs_values.extend(s['observation.state'])
            action_values.extend(s['action'])
            ep_idx = s['episode_index']
            if ep_idx not in episode_samples:
                episode_samples[ep_idx] = 0
            episode_samples[ep_idx] += 1
        
        obs_array = np.array(obs_values)
        action_array = np.array(action_values)
        
        print(f"  - Observation state range: [{obs_array.min():.3f}, {obs_array.max():.3f}]")
        print(f"  - Action range: [{action_array.min():.3f}, {action_array.max():.3f}]")
        
        # Show some sample episodes (from sampled data)
        print("\nSample Episodes (from analysis sample):")
        for ep_idx in sorted(list(episode_samples.keys())[:3]):
            print(f"  - Episode {ep_idx}: ~{episode_samples[ep_idx]} samples (in analysis sample)")
            
        return True
        
    except Exception as e:
        print(f"ERROR: Error loading dataset: {e}")
        return False

def main():
    """Main function to explore all downloaded datasets."""
    print("ALOHA Dataset Explorer")
    print("=" * 60)
    
    # Get datasets directory
    datasets_dir = Path(__file__).parent.parent / "data" / "datasets"
    
    if not datasets_dir.exists():
        print(f"ERROR: Datasets directory not found: {datasets_dir}")
        return
    
    # Find all dataset directories
    dataset_dirs = [d for d in datasets_dir.iterdir() if d.is_dir()]
    
    if not dataset_dirs:
        print("ERROR: No datasets found in datasets directory")
        return
    
    print(f"Found {len(dataset_dirs)} datasets:")
    for i, dataset_dir in enumerate(dataset_dirs, 1):
        print(f"  {i}. {dataset_dir.name}")
    
    # Explore each dataset
    successful_datasets = []
    for dataset_dir in dataset_dirs:
        if explore_dataset(dataset_dir):
            successful_datasets.append(dataset_dir.name)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully loaded {len(successful_datasets)} datasets:")
    for dataset in successful_datasets:
        print(f"  - {dataset}")
    
    print("\nUsage Tips:")
    print("  - Use 'aloha_towel' for basic towel folding tasks")
    print("  - Use 'aloha_static_coffee' for manipulation tasks")
    print("  - Use 'aloha_sim_transfer_cube_human' for simulation")
    print("  - All datasets are in HuggingFace format")

    print("\nNext Steps:")
    print("  1. Update your config files to point to these datasets")
    print("  2. Use the datasets in your training scripts")
    print("  3. Try different datasets for different tasks")

if __name__ == "__main__":
    main() 