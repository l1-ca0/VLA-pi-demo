#!/usr/bin/env python3
"""
OpenPI ALOHA Basic Inference

This script demonstrates how to load a pre-trained OpenPI model and run inference
on ALOHA manipulation tasks.

Prerequisites:
- OpenPI installed and configured
- ALOHA hardware connected (for real robot mode)
- Model checkpoints downloaded from Google Cloud Storage

Usage:
    python examples/basic_inference.py --model pi0_aloha_towel --prompt "fold the towel neatly"
    python examples/basic_inference.py --model pi0_aloha_towel --prompt "fold the towel" --quiet  # Basic output only
    python examples/basic_inference.py --model pi0_aloha_towel --prompt "fold the towel" --output-file my_results.json  # Custom output file
"""

import argparse
import numpy as np
import cv2
import time
import json
from typing import Dict, Any
from pathlib import Path

# OpenPI imports 
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.shared import download

# ROS integration support 
def _import_ros_dependencies():
    """Import ROS dependencies dynamically to avoid static analysis errors."""
    try:
        
        rospy = __import__('rospy')
        sensor_msgs = __import__('sensor_msgs.msg', fromlist=['Image', 'JointState'])
        cv_bridge = __import__('cv_bridge', fromlist=['CvBridge'])
        
        Image = getattr(sensor_msgs, 'Image')
        JointState = getattr(sensor_msgs, 'JointState')
        CvBridge = getattr(cv_bridge, 'CvBridge')
        
        return True, rospy, Image, JointState, CvBridge
    except ImportError:
        return False, None, None, None, None

ROS_AVAILABLE, rospy, Image, JointState, CvBridge = _import_ros_dependencies()


class ALOHAInference:
    """OpenPI ALOHA inference wrapper."""
    
    def __init__(self, model_name: str, checkpoint_path: str = None):
        """
        Initialize OpenPI inference for ALOHA.
        
        Args:
            model_name: OpenPI model name (e.g., pi0_aloha_towel)
            checkpoint_path: Path to model checkpoint (optional)
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.policy = None
        self.config = None
        
        # Load model configuration
        self._load_model_config()
        
    def _load_model_config(self):
        """Load OpenPI model configuration."""
        # Get model configuration from OpenPI
        self.config = _config.get_config(self.model_name)
        
        # Download model if needed
        if self.checkpoint_path is None:
            checkpoint_url = f"gs://openpi-assets/checkpoints/{self.model_name}/"
            self.checkpoint_path = download.maybe_download(checkpoint_url)
            
    def load_policy(self):
        """Load the OpenPI policy from checkpoint."""
        self.policy = _policy_config.create_trained_policy(
            self.config,
            self.checkpoint_path
        )
        
    def infer(self, observation: Dict[str, Any], prompt: str) -> np.ndarray:
        """
        Run OpenPI inference.
        
        Args:
            observation: Observation dictionary with images and state
            prompt: Text prompt for the task
            
        Returns:
            Predicted actions
        """
        # Add prompt to observation
        observation["prompt"] = prompt
        
        # Run inference using correct OpenPI API
        result = self.policy.infer(observation)
        
        return result["actions"]


def create_observation_from_aloha() -> Dict[str, Any]:
    """
    Create observation from ALOHA hardware in OpenPI model input format.
    """
    if not ROS_AVAILABLE:
        raise ImportError("ROS dependencies not available. Install ros_workspace for hardware integration.")
    
    # Initialize ROS if not already done
    if not rospy.get_node_uri():
        rospy.init_node('openpi_aloha_inference', anonymous=True)
    
    bridge = CvBridge()
    images = {}
    image_masks = {}
    camera_map = {
        "base_0_rgb": '/cam_high/color/image_raw',
        "left_wrist_0_rgb": '/cam_left_wrist/color/image_raw',
        "right_wrist_0_rgb": '/cam_right_wrist/color/image_raw',
    }
    try:
        for key, topic in camera_map.items():
            msg = rospy.wait_for_message(topic, Image, timeout=5.0)
            img = bridge.imgmsg_to_cv2(msg, "rgb8")
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            images[key] = img
            image_masks[key] = True
    except rospy.ROSException as e:
        raise RuntimeError(f"Failed to get camera images from ALOHA: {e}")
    
    # Get robot state
    try:
        joint_msg = rospy.wait_for_message('/joint_states', JointState, timeout=5.0)
        robot_state = np.array(joint_msg.position)
    except rospy.ROSException as e:
        raise RuntimeError(f"Failed to get robot state from ALOHA: {e}")
    
    return {
        "image": images,
        "image_mask": image_masks,
        "state": robot_state,
    }


def create_synthetic_observation() -> Dict[str, Any]:
    """Create synthetic observation for simulation without hardware, matching ALOHA policy input format."""
    images = {}
    camera_names = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    for camera_name in camera_names:
        # Random RGB image as uint8 in CHW format 
        synthetic_image = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
        images[camera_name] = synthetic_image

    # Create synthetic robot state (14 DOF for bimanual ALOHA)
    synthetic_state = np.random.uniform(-1.0, 1.0, (14,)).astype(np.float32)

    return {
        "images": images,
        "state": synthetic_state,
    }


def convert_paths(obj):
    if isinstance(obj, dict):
        return {k: convert_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths(i) for i in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description="OpenPI ALOHA Basic Inference")
    parser.add_argument("--model", type=str, default="pi0_aloha_towel",
                      help="OpenPI model name")
    parser.add_argument("--checkpoint", type=str, default=None,
                      help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="fold the towel neatly",
                      help="Task prompt")
    parser.add_argument("--use-hardware", action="store_true",
                      help="Use real ALOHA hardware")
    parser.add_argument("--steps", type=int, default=50,
                      help="Number of inference steps")
    parser.add_argument("--verbose", action="store_true", default=True,
                      help="Show detailed output analysis (default: True)")
    parser.add_argument("--quiet", action="store_true",
                      help="Show only basic output (opposite of --verbose)")
    parser.add_argument("--output-file", type=str, default="inference_results.json",
                      help="Output file to save inference results (default: inference_results.json)")
    
    args = parser.parse_args()
    
    print("OpenPI ALOHA Inference Demo")
    print("=" * 40)
    print(f"Loading OpenPI model: {args.model}")
    
    try:
        # Initialize inference
        inference = ALOHAInference(args.model, args.checkpoint)
        inference.load_policy()
        
        print(f"Model {args.model} loaded successfully!")
        print(f"Task: {args.prompt}")
        
        if args.use_hardware:
            print("Using real ALOHA hardware")
            
            for step in range(args.steps):
                try:
                    # Get observation from ALOHA hardware
                    observation = create_observation_from_aloha()
                    
                    # Add prompt to observation
                    observation["prompt"] = args.prompt
                    
                    # Run inference
                    result = inference.policy.infer(observation)
                    actions = result['actions']
                    
                    print(f"Step {step+1}: Generated {len(actions)} actions")
                    if len(actions) > 0 and len(actions[0]) > 3:
                        print(f"Action preview: {actions[0][:3]}")  # Show first 3 actions of first timestep
                    elif len(actions) > 0:
                        print(f"Action preview: {actions[0]}")  # Show all actions if less than 3
                    
                    # Here you would send actions to ALOHA hardware
                    # This requires ALOHA control interface
                    
                    time.sleep(0.1)  # Control frequency
                    
                except Exception as e:
                    print(f"Error at step {step+1}: {e}")
                    break
        else:
            print("Using synthetic observations (hardware not available)")
            print("Use --use-hardware flag for real ALOHA integration")
            
            # Run with synthetic observation
            observation = create_synthetic_observation()
            observation["prompt"] = args.prompt
            
            # Get the full result (not just actions)
            print("\nRunning inference...")
            result = inference.policy.infer(observation)
            
            print("Inference completed!")
            print(f"Action shape: {result['actions'].shape}")
            print(f"Generated {len(result['actions'])} action timesteps for task: '{args.prompt}'")
            
            # Set verbose based on flags (default True, unless --quiet is used)
            verbose_mode = args.verbose and not args.quiet
            
            if verbose_mode:
                # Show timing information
                if 'policy_timing' in result:
                    timing = result['policy_timing']
                    print(f"\nTiming Information:")
                    print(f"  - Inference time: {timing.get('infer_ms', 0):.1f} ms")
                    print(f"  - Inference time: {timing.get('infer_ms', 0)/1000:.2f} seconds")
                
                # Show action statistics
                actions = result['actions']
                print(f"\nAction Analysis:")
                print(f"  - Action dimensions: {actions.shape[1]} DOF (14 for bimanual ALOHA)")
                print(f"  - Action horizon: {actions.shape[0]} timesteps")
                print(f"  - Action range: [{actions.min():.3f}, {actions.max():.3f}]")
                print(f"  - Action mean: {actions.mean():.3f}")
                print(f"  - Action std: {actions.std():.3f}")
                
                # Show first few actions
                print(f"\nFirst 3 Action Timesteps:")
                for i in range(min(3, len(actions))):
                    action_preview = actions[i][:6]  # Show first 6 DOF
                    print(f"  t={i}: [{', '.join(f'{x:.3f}' for x in action_preview)}, ...]")
                
                # Show action breakdown for ALOHA (14 DOF = 7 left arm + 7 right arm)
                if actions.shape[1] == 14:
                    print(f"\nALOHA Action Breakdown (14 DOF):")
                    left_arm = actions[:, :7]
                    right_arm = actions[:, 7:14]
                    
                    print(f"  - Left arm (DOF 0-6):")
                    print(f"    Range: [{left_arm.min():.3f}, {left_arm.max():.3f}]")
                    print(f"    Mean: {left_arm.mean():.3f}")
                    
                    print(f"  - Right arm (DOF 7-13):")
                    print(f"    Range: [{right_arm.min():.3f}, {right_arm.max():.3f}]")
                    print(f"    Mean: {right_arm.mean():.3f}")
                
                # Additional outputs information
                print(f"\nAvailable Outputs:")
                for key, value in result.items():
                    if key != 'actions':
                        if hasattr(value, 'shape'):
                            print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            print(f"  - {key}: type={type(value).__name__}")
                
                # Model information
                print(f"\nModel Information:")
                print(f"  - Model name: {args.model}")
                print(f"  - Checkpoint: {inference.checkpoint_path}")
                print(f"  - Model type: {inference.config.model.model_type}")
                print(f"  - Action dimension: {inference.config.model.action_dim}")
                print(f"  - Action horizon: {inference.config.model.action_horizon}")
            
            # Save results (always enabled)
            print(f"\nSaving results to: {args.output_file}")
            
            # Prepare data for JSON serialization
            save_data = {
                "model_name": args.model,
                "prompt": args.prompt,
                "checkpoint_path": str(inference.checkpoint_path) if inference.checkpoint_path is not None else None,
                "model_info": {
                    "model_type": str(inference.config.model.model_type),
                    "action_dim": inference.config.model.action_dim,
                    "action_horizon": inference.config.model.action_horizon
                },
                "inference_results": {
                    "actions": result['actions'].tolist(),  # Convert to list for JSON
                    "actions_shape": list(result['actions'].shape),
                    "actions_stats": {
                        "min": float(result['actions'].min()),
                        "max": float(result['actions'].max()),
                        "mean": float(result['actions'].mean()),
                        "std": float(result['actions'].std())
                    }
                },
                "additional_outputs": {}
            }
            
            # Add other outputs from inference result
            for key, value in result.items():
                if key != 'actions':
                    if isinstance(value, np.ndarray):
                        save_data["additional_outputs"][key] = {
                            "data": value.tolist(),
                            "shape": list(value.shape),
                            "dtype": str(value.dtype)
                        }
                    else:
                        save_data["additional_outputs"][key] = value
            
            # Add timestamp
            save_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Convert all Path objects to strings
            save_data = convert_paths(save_data)

            try:
                with open(args.output_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                print("Results saved successfully!")
            except Exception as e:
                print(f"Warning: Failed to save results to {args.output_file}: {e}")
            
            if args.steps > 1:
                print(f"\nNote: Multiple steps ({args.steps}) requested, but using single inference.")
                print("For multi-step execution, use --use-hardware flag with real robot.")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure OpenPI is properly installed")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure model checkpoints are available and OpenPI is configured")


if __name__ == "__main__":
    main() 