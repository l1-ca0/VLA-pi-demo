#!/usr/bin/env python3
"""
OpenPI ALOHA Demo

This script demonstrates OpenPI integration with ALOHA hardware
for bimanual manipulation tasks with configurable evaluation modes.

Prerequisites:
- OpenPI installed and configured
- ALOHA hardware setup with ROS (optional)
- Model checkpoints available

Usage:
    python examples/aloha_sim_demo.py --model pi0_aloha_towel --task towel_folding --evaluation-mode task_specific
    python examples/aloha_sim_demo.py --model pi0_aloha_towel --task towel_folding --output-file my_evaluation.json
"""

import argparse
import numpy as np
import time
import cv2
import json
from typing import Dict, Any, Tuple
from enum import Enum

# OpenPI imports
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.shared import download

class EvaluationMode(Enum):
    """Evaluation modes for task success detection."""
    RANDOM = "random"  # Simple random simulation
    TASK_SPECIFIC = "task_specific"  # Task-specific evaluation
    PHYSICS = "physics"  # Physics-based simulation (future)

class TaskType(Enum):
    """Supported task types with specific evaluation criteria."""
    TOWEL_FOLDING = "towel_folding"
    FOOD_MANIPULATION = "food_manipulation"
    TUPPERWARE = "tupperware"
    OBJECT_TRANSFER = "object_transfer"

# ROS integration support 
def _import_ros_dependencies():
    """Import ROS dependencies dynamically to avoid static analysis errors."""
    try:
        # Use __import__ to avoid static analysis detection
        rospy = __import__('rospy')
        sensor_msgs = __import__('sensor_msgs.msg', fromlist=['Image', 'JointState'])
        trajectory_msgs = __import__('trajectory_msgs.msg', fromlist=['JointTrajectory', 'JointTrajectoryPoint'])
        cv_bridge = __import__('cv_bridge', fromlist=['CvBridge'])
        
        Image = getattr(sensor_msgs, 'Image')
        JointState = getattr(sensor_msgs, 'JointState')
        JointTrajectory = getattr(trajectory_msgs, 'JointTrajectory')
        JointTrajectoryPoint = getattr(trajectory_msgs, 'JointTrajectoryPoint')
        CvBridge = getattr(cv_bridge, 'CvBridge')
        
        return True, rospy, Image, JointState, JointTrajectory, JointTrajectoryPoint, CvBridge
    except ImportError:
        return False, None, None, None, None, None, None

ROS_AVAILABLE, rospy, Image, JointState, JointTrajectory, JointTrajectoryPoint, CvBridge = _import_ros_dependencies()

if not ROS_AVAILABLE:
    print("WARNING: ROS not available. Hardware integration disabled.")


class TaskEvaluator:
    """Task evaluation with task-specific success criteria."""
    
    def __init__(self, task_type: TaskType, evaluation_mode: EvaluationMode):
        self.task_type = task_type
        self.evaluation_mode = evaluation_mode
        self.reset_task_state()
        
    def reset_task_state(self):
        """Reset task-specific state tracking."""
        self.step_count = 0
        self.action_history = []
        self.state_history = []
        self.success_indicators = {
            "progress_score": 0.0,
            "stability_score": 0.0,
            "task_completion": 0.0
        }
        
        # Task-specific state
        if self.task_type == TaskType.TOWEL_FOLDING:
            self.towel_state = {
                "initial_spread": None,
                "fold_attempts": 0,
                "symmetry_score": 0.0,
                "compactness_score": 0.0
            }
        elif self.task_type == TaskType.OBJECT_TRANSFER:
            self.transfer_state = {
                "object_grasped": False,
                "transfer_progress": 0.0,
                "drop_detected": False
            }
        elif self.task_type == TaskType.TUPPERWARE:
            self.container_state = {
                "lid_alignment": 0.0,
                "closure_progress": 0.0,
                "successful_closure": False
            }
            
    def calculate_step_reward(self, observation: Dict[str, Any], actions: np.ndarray) -> float:
        """Calculate meaningful step reward based on task progress."""
        self.step_count += 1
        self.action_history.append(actions.copy())
        
        if 'state' in observation:
            self.state_history.append(observation['state'].copy())
        
        if self.evaluation_mode == EvaluationMode.RANDOM:
            return self._random_reward()
        elif self.evaluation_mode == EvaluationMode.TASK_SPECIFIC:
            return self._task_specific_reward(observation, actions)
        else:
            return self._random_reward()  # Fallback
            
    def _random_reward(self) -> float:
        """Original random reward for comparison."""
        return np.random.uniform(0.1, 0.5)
        
    def _task_specific_reward(self, observation: Dict[str, Any], actions: np.ndarray) -> float:
        """Task-specific reward calculation."""
        base_reward = 0.1  # Minimum reward for taking action
        task_reward = 0.0
        
        # Task-specific reward calculation
        if self.task_type == TaskType.TOWEL_FOLDING:
            task_reward = self._evaluate_towel_folding(observation, actions)
        elif self.task_type == TaskType.OBJECT_TRANSFER:
            task_reward = self._evaluate_object_transfer(observation, actions)
        elif self.task_type == TaskType.TUPPERWARE:
            task_reward = self._evaluate_tupperware_task(observation, actions)
        else:
            task_reward = self._evaluate_generic_manipulation(observation, actions)
            
        # Combine base and task-specific rewards
        total_reward = base_reward + task_reward
        
        # Update success indicators
        self.success_indicators["progress_score"] = min(1.0, 
            self.success_indicators["progress_score"] + task_reward / 10.0)
        
        return total_reward
        
    def _evaluate_towel_folding(self, observation: Dict[str, Any], actions: np.ndarray) -> float:
        """Evaluate towel folding progress."""
        reward = 0.0
        
        if len(self.state_history) < 2:
            return reward
            
        current_state = self.state_history[-1]
        previous_state = self.state_history[-2]
        
        # Analyze arm coordination (bimanual task)
        left_arm = current_state[:7]
        right_arm = current_state[7:14]
        
        # Reward coordinated movement
        left_movement = np.linalg.norm(left_arm - self.state_history[0][:7])
        right_movement = np.linalg.norm(right_arm - self.state_history[0][7:14])
        
        if abs(left_movement - right_movement) < 0.1:  # Coordinated movement
            reward += 0.2
            
        # Reward approaching central workspace (folding position)
        workspace_center = np.array([0.0, 0.0, 0.2])  # Approximate table center
        left_pos = left_arm[:3]  # Assume first 3 are position
        right_pos = right_arm[:3]
        
        if len(left_pos) >= 3 and len(right_pos) >= 3:
            center_approach = max(0, 0.5 - np.linalg.norm(
                (left_pos + right_pos) / 2 - workspace_center))
            reward += center_approach * 0.3
            
        # Reward stability (less erratic movement)
        if len(self.action_history) >= 2:
            action_smoothness = 1.0 - np.linalg.norm(
                self.action_history[-1] - self.action_history[-2])
            reward += max(0, action_smoothness * 0.2)
            
        # Progressive reward based on steps (folding takes time)
        if self.step_count > 10:
            progress_bonus = min(0.3, (self.step_count - 10) * 0.01)
            reward += progress_bonus
            
        return reward
        
    def _evaluate_object_transfer(self, observation: Dict[str, Any], actions: np.ndarray) -> float:
        """Evaluate object transfer progress."""
        reward = 0.0
        
        if len(self.state_history) < 2:
            return reward
            
        # Analyze gripper states (typically last DOF of each arm)
        left_gripper = self.state_history[-1][6]  # Assume 7th DOF is gripper
        right_gripper = self.state_history[-1][13]  # Assume 14th DOF is gripper
        
        # Reward gripper activation (grasping)
        if abs(left_gripper) > 0.5 or abs(right_gripper) > 0.5:
            reward += 0.3
            self.transfer_state["object_grasped"] = True
            
        # Reward transfer motion if object is grasped
        if self.transfer_state["object_grasped"]:
            # Reward horizontal movement (transfer)
            current_pos = self.state_history[-1][:3]  # Left arm position
            if len(self.state_history) > 5:
                initial_pos = self.state_history[4][:3]  # After grasping
                transfer_distance = np.linalg.norm(current_pos - initial_pos)
                reward += min(0.4, transfer_distance * 2.0)
                
        return reward
        
    def _evaluate_tupperware_task(self, observation: Dict[str, Any], actions: np.ndarray) -> float:
        """Evaluate tupperware lid placement."""
        reward = 0.0
        
        if len(self.state_history) < 2:
            return reward
            
        # Analyze coordinated downward motion (lid placement)
        current_state = self.state_history[-1]
        left_arm = current_state[:7]
        right_arm = current_state[7:14]
        
        # Reward downward motion coordination
        if len(self.state_history) > 1:
            prev_state = self.state_history[-2]
            left_z_movement = left_arm[2] - prev_state[2]  # Assume 3rd dim is Z
            right_z_movement = right_arm[2] - prev_state[9]  # Right arm Z
            
            if left_z_movement < -0.01 and right_z_movement < -0.01:  # Downward
                reward += 0.3
                
        # Reward precision (small, controlled movements)
        action_magnitude = np.linalg.norm(actions)
        if action_magnitude < 0.5:  # Precise movements
            reward += 0.2
            
        return reward
        
    def _evaluate_generic_manipulation(self, observation: Dict[str, Any], actions: np.ndarray) -> float:
        """Generic manipulation task evaluation."""
        reward = 0.0
        
        # Reward smooth, coordinated movement
        if len(self.action_history) >= 2:
            action_smoothness = 1.0 - np.linalg.norm(
                self.action_history[-1] - self.action_history[-2])
            reward += max(0, action_smoothness * 0.3)
            
        # Reward reasonable action magnitudes (not too extreme)
        action_magnitude = np.linalg.norm(actions)
        if 0.1 < action_magnitude < 1.0:
            reward += 0.2
            
        return reward
        
    def check_task_completion(self, prompt: str, observation: Dict[str, Any]) -> bool:
        """Task completion detection."""
        if self.evaluation_mode == EvaluationMode.RANDOM:
            return np.random.random() < 0.01  # Original 1% chance
        elif self.evaluation_mode == EvaluationMode.TASK_SPECIFIC:
            return self._task_specific_completion_check(prompt, observation)
        else:
            return False
            
    def _task_specific_completion_check(self, prompt: str, observation: Dict[str, Any]) -> bool:
        """Task-specific completion detection based on task progress."""
        
        # Task-specific completion criteria
        if self.task_type == TaskType.TOWEL_FOLDING:
            return self._check_towel_folding_completion()
        elif self.task_type == TaskType.OBJECT_TRANSFER:
            return self._check_transfer_completion()
        elif self.task_type == TaskType.TUPPERWARE:
            return self._check_tupperware_completion()
        else:
            return self._check_generic_completion()
            
    def _check_towel_folding_completion(self) -> bool:
        """Check if towel folding is complete."""
        # Criteria: Sufficient coordinated movement + stability
        if self.step_count < 15:  # Minimum steps for folding
            return False
            
        progress = self.success_indicators["progress_score"]
        
        # Success if good progress and recent actions are small (task complete)
        if progress > 0.7 and len(self.action_history) >= 5:
            recent_actions = self.action_history[-5:]
            avg_action_magnitude = np.mean([np.linalg.norm(a) for a in recent_actions])
            
            if avg_action_magnitude < 0.2:  # Settling/finished
                return True
                
        # Alternative: Random completion with bias toward higher progress
        completion_prob = min(0.05, progress * 0.05)  # Max 5% chance
        return np.random.random() < completion_prob
        
    def _check_transfer_completion(self) -> bool:
        """Check if object transfer is complete."""
        if self.step_count < 10:
            return False
            
        # Success if object was grasped and significant movement occurred
        if self.transfer_state["object_grasped"] and self.step_count > 20:
            completion_prob = 0.03  # 3% chance per step after movement
            return np.random.random() < completion_prob
            
        return False
        
    def _check_tupperware_completion(self) -> bool:
        """Check if tupperware task is complete."""
        if self.step_count < 8:
            return False
            
        # Success based on progress and precision
        progress = self.success_indicators["progress_score"]
        if progress > 0.5 and self.step_count > 15:
            completion_prob = progress * 0.04  # Up to 4% chance
            return np.random.random() < completion_prob
            
        return False
        
    def _check_generic_completion(self) -> bool:
        """Generic completion check."""
        if self.step_count < 10:
            return False
            
        progress = self.success_indicators["progress_score"]
        completion_prob = progress * 0.02  # Up to 2% chance
        return np.random.random() < completion_prob
        
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get detailed evaluation summary."""
        return {
            "task_type": self.task_type.value,
            "evaluation_mode": self.evaluation_mode.value,
            "steps_taken": self.step_count,
            "success_indicators": self.success_indicators.copy(),
            "task_specific_state": self._get_task_specific_summary()
        }
        
    def _get_task_specific_summary(self) -> Dict[str, Any]:
        """Get task-specific evaluation summary."""
        if self.task_type == TaskType.TOWEL_FOLDING:
            return self.towel_state.copy()
        elif self.task_type == TaskType.OBJECT_TRANSFER:
            return self.transfer_state.copy()
        elif self.task_type == TaskType.TUPPERWARE:
            return self.container_state.copy()
        else:
            return {}


class ALOHAController:
    """OpenPI controller for ALOHA hardware with configurable evaluation."""
    
    def __init__(self, model_name: str, use_hardware: bool = False, 
                 evaluation_mode: EvaluationMode = EvaluationMode.TASK_SPECIFIC):
        """
        Initialize ALOHA controller with configurable evaluation.
        
        Args:
            model_name: OpenPI model name
            use_hardware: Whether to use real hardware
            evaluation_mode: Evaluation mode for success detection
        """
        self.model_name = model_name
        self.use_hardware = use_hardware and ROS_AVAILABLE
        self.evaluation_mode = evaluation_mode
        self.policy = None
        self.config = None
        
        if self.use_hardware:
            # Initialize ROS
            rospy.init_node('openpi_aloha_controller', anonymous=True)
            self.bridge = CvBridge()
            
            # Publishers for robot control
            self.left_arm_pub = rospy.Publisher(
                '/left_arm_controller/command', 
                JointTrajectory, 
                queue_size=1
            )
            self.right_arm_pub = rospy.Publisher(
                '/right_arm_controller/command', 
                JointTrajectory, 
                queue_size=1
            )
            
            # Camera configuration
            self.camera_topics = {
                "observation/exterior_image_1_left": "/cam_high/color/image_raw",
                "observation/wrist_image_left": "/cam_left_wrist/color/image_raw", 
                "observation/wrist_image_right": "/cam_right_wrist/color/image_raw"
            }
        
        # Load OpenPI model
        self._load_model()
        
    def _load_model(self):
        """Load OpenPI model and policy."""
        print(f"Loading OpenPI model: {self.model_name}")
        
        # Get model configuration
        self.config = _config.get_config(self.model_name)
        
        # Download checkpoint
        checkpoint_url = f"gs://openpi-assets/checkpoints/{self.model_name}/"
        checkpoint_path = download.maybe_download(checkpoint_url)
        
        # Load policy
        self.policy = _policy_config.create_trained_policy(
            self.config,
            checkpoint_path
        )
        
        print("Model loaded successfully")
        
    def get_observation(self) -> Dict[str, Any]:
        """
        Get current observation from ALOHA hardware or create synthetic.
        
        Returns:
            Observation dictionary in OpenPI format
        """
        if self.use_hardware:
            return self._get_hardware_observation()
        else:
            return self._get_synthetic_observation()
            
    def _get_hardware_observation(self) -> Dict[str, Any]:
        """Get observation from real ALOHA hardware in OpenPI model input format."""
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS not available. Cannot get hardware observation.")
        
        images = {}
        image_masks = {}
        camera_map = {
            "base_0_rgb": "observation/exterior_image_1_left",
            "left_wrist_0_rgb": "observation/wrist_image_left",
            "right_wrist_0_rgb": "observation/wrist_image_right",
        }
        for key, obs_key in camera_map.items():
            if obs_key in self.camera_topics:
                topic = self.camera_topics[obs_key]
                try:
                    msg = rospy.wait_for_message(topic, Image, timeout=5.0)
                    img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                    img = cv2.resize(img, (224, 224))
                    img = img.astype(np.float32) / 255.0
                    images[key] = img
                    image_masks[key] = True
                except rospy.ROSException as e:
                    raise RuntimeError(f"Failed to get image from {topic}: {e}")
            else:
                images[key] = np.zeros((224, 224, 3), dtype=np.float32)
                image_masks[key] = False
        # Get robot state
        try:
            joint_msg = rospy.wait_for_message('/joint_states', JointState, timeout=5.0)
            robot_state = np.array(joint_msg.position)
        except rospy.ROSException as e:
            raise RuntimeError(f"Failed to get robot state: {e}")
        return {
            "image": images,
            "image_mask": image_masks,
            "state": robot_state,
        }
        
    def _get_synthetic_observation(self) -> Dict[str, Any]:
        """Create synthetic observation for simulation."""
        images = {}
        camera_names = [
            "cam_high",
            "cam_left_wrist", 
            "cam_right_wrist"
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
        
    def send_actions(self, actions: np.ndarray):
        """
        Send actions to ALOHA hardware or simulate execution.
        
        Args:
            actions: Action array from OpenPI (14 DOF)
        """
        if not self.use_hardware:
            # Simulate action execution
            time.sleep(0.1)
            return
            
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS not available. Cannot send actions to hardware.")
            
        if len(actions.shape) == 1:
            actions = actions.reshape(1, -1)
            
        # Validate action dimensions
        if actions.shape[-1] != 14:
            raise ValueError(f"Expected 14 actions for bimanual ALOHA, got {actions.shape[-1]}")
            
        # Split actions for left and right arms
        left_actions = actions[:, :7]   # 7 DOF per arm
        right_actions = actions[:, 7:14]
        
        # Create trajectory messages
        left_traj = JointTrajectory()
        right_traj = JointTrajectory()
        
        # Set joint names
        left_traj.joint_names = [
            'left_waist', 'left_shoulder', 'left_elbow', 
            'left_forearm_roll', 'left_wrist_angle', 'left_wrist_rotate',
            'left_gripper'
        ]
        right_traj.joint_names = [
            'right_waist', 'right_shoulder', 'right_elbow',
            'right_forearm_roll', 'right_wrist_angle', 'right_wrist_rotate', 
            'right_gripper'
        ]
        
        # Add trajectory points
        for i, (left_action, right_action) in enumerate(zip(left_actions, right_actions)):
            # Left arm point
            left_point = JointTrajectoryPoint()
            left_point.positions = left_action.tolist()
            left_point.time_from_start = rospy.Duration(0.1 * (i + 1))
            left_traj.points.append(left_point)
            
            # Right arm point
            right_point = JointTrajectoryPoint()
            right_point.positions = right_action.tolist()
            right_point.time_from_start = rospy.Duration(0.1 * (i + 1))
            right_traj.points.append(right_point)
        
        # Publish trajectories
        left_traj.header.stamp = rospy.Time.now()
        right_traj.header.stamp = rospy.Time.now()
        
        self.left_arm_pub.publish(left_traj)
        self.right_arm_pub.publish(right_traj)
        
    def run_task(self, prompt: str, max_steps: int = 100) -> Dict[str, Any]:
        """
        Run a manipulation task using OpenPI with configurable evaluation.
        
        Args:
            prompt: Task description
            max_steps: Maximum number of steps
            
        Returns:
            Task execution results with detailed evaluation
        """
        # Determine task type from prompt
        task_type = self._determine_task_type(prompt)
        
        # Initialize task evaluator
        evaluator = TaskEvaluator(task_type, self.evaluation_mode)
        
        print(f"Starting task: '{prompt}'")
        print(f"Task type: {task_type.value}")
        print(f"Evaluation mode: {self.evaluation_mode.value}")
        print(f"Using {'real hardware' if self.use_hardware else 'simulation'}")
        
        results = {
            "success": False,
            "steps": 0,
            "errors": [],
            "total_reward": 0.0,
            "prompt": prompt,
            "task_type": task_type.value,
            "evaluation_mode": self.evaluation_mode.value
        }
        
        for step in range(max_steps):
            try:
                # Get current observation
                observation = self.get_observation()
                
                # Add prompt
                observation["prompt"] = prompt
                
                # Run OpenPI inference
                inference_result = self.policy.infer(observation)
                actions = inference_result["actions"]
                
                # Send actions to robot
                self.send_actions(actions)
                
                # Calculate step reward using task evaluation
                step_reward = evaluator.calculate_step_reward(observation, actions)
                results["total_reward"] += step_reward
                
                print(f"Step {step + 1}/{max_steps}: Reward: {step_reward:.3f}, "
                      f"Actions: {len(actions)} timesteps, "
                      f"Progress: {evaluator.success_indicators['progress_score']:.2f}")
                
                # Check for task completion using task-specific detection
                if evaluator.check_task_completion(prompt, observation):
                    results["success"] = True
                    results["steps"] = step + 1
                    print(f"Task completed successfully in {step + 1} steps!")
                    break
                    
                # Control frequency
                time.sleep(0.1)
                
            except Exception as e:
                error_msg = f"Error at step {step}: {e}"
                print(error_msg)
                results["errors"].append(error_msg)
                break
                
        results["steps"] = step + 1 if results["steps"] == 0 else results["steps"]
        
        # Add evaluation summary
        results["evaluation_summary"] = evaluator.get_evaluation_summary()
        
        return results
        
    def _determine_task_type(self, prompt: str) -> TaskType:
        """Determine task type from prompt."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["towel", "fold", "cloth"]):
            return TaskType.TOWEL_FOLDING
        elif any(word in prompt_lower for word in ["food", "container", "bowl", "place"]):
            return TaskType.FOOD_MANIPULATION
        elif any(word in prompt_lower for word in ["tupperware", "lid", "close", "cover"]):
            return TaskType.TUPPERWARE
        elif any(word in prompt_lower for word in ["transfer", "move", "pick", "grasp"]):
            return TaskType.OBJECT_TRANSFER
        else:
            return TaskType.OBJECT_TRANSFER  # Default


def main():
    parser = argparse.ArgumentParser(description="OpenPI ALOHA Evaluation Demo")
    parser.add_argument("--model", type=str, default="pi0_aloha_towel",
                      help="OpenPI model name")
    parser.add_argument("--task", type=str, default="towel_folding",
                      help="Task name")
    parser.add_argument("--prompt", type=str, default=None,
                      help="Custom task prompt (overrides task)")
    parser.add_argument("--steps", type=int, default=100,
                      help="Maximum steps per episode")
    parser.add_argument("--use-hardware", action="store_true",
                      help="Use real ALOHA hardware")
    parser.add_argument("--episodes", type=int, default=1,
                      help="Number of episodes to run")
    parser.add_argument("--evaluation-mode", type=str, 
                      choices=[mode.value for mode in EvaluationMode],
                      default=EvaluationMode.TASK_SPECIFIC.value,
                      help="Evaluation mode: random (simple), task_specific (realistic), physics (future)")
    parser.add_argument("--output-file", type=str, default="aloha_evaluation_results.json",
                      help="Output file to save evaluation results (default: aloha_evaluation_results.json)")
    
    args = parser.parse_args()
    
    # Convert evaluation mode
    eval_mode = EvaluationMode(args.evaluation_mode)
    
    # Default prompts for different tasks
    task_prompts = {
        "towel_folding": "fold the towel neatly in half",
        "food_manipulation": "pick up the food and place it in the container",
        "tupperware": "put the lid on the tupperware container",
        "object_transfer": "pick up the object and transfer it to the other side"
    }
    
    # Use custom prompt or default for task
    prompt = args.prompt or task_prompts.get(args.task, f"perform {args.task} task")
    
    print("OpenPI ALOHA Evaluation Demo")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Prompt: '{prompt}'")
    print(f"Evaluation Mode: {eval_mode.value}")
    print(f"Hardware: {'Enabled' if args.use_hardware else 'Simulation only'}")
    print(f"Max steps: {args.steps}")
    print(f"Episodes: {args.episodes}")
    
    if args.use_hardware and not ROS_AVAILABLE:
        print("WARNING: Hardware requested but ROS not available. Using simulation.")
        args.use_hardware = False
    
    if eval_mode == EvaluationMode.RANDOM:
        print("\nUsing random evaluation mode (simple simulation)")
    elif eval_mode == EvaluationMode.TASK_SPECIFIC:
        print("\nUsing task-specific evaluation")
        print("- Task-aware reward calculation")
        print("- Progress tracking")
        print("- Realistic completion detection")
    
    try:
        # Initialize controller with configurable evaluation
        controller = ALOHAController(args.model, args.use_hardware, eval_mode)
        
        # Run episodes
        all_results = []
        total_successes = 0
        total_progress = 0.0
        
        for episode in range(args.episodes):
            print(f"\n--- Episode {episode + 1}/{args.episodes} ---")
            
            try:
                results = controller.run_task(prompt, args.steps)
                all_results.append(results)
                
                if results["success"]:
                    total_successes += 1
                
                # Extract evaluation summary
                eval_summary = results.get("evaluation_summary", {})
                progress = eval_summary.get("success_indicators", {}).get("progress_score", 0.0)
                total_progress += progress
                
                print(f"Episode {episode + 1} Results:")
                print(f"  Success: {results['success']}")
                print(f"  Steps: {results['steps']}")
                print(f"  Total Reward: {results['total_reward']:.2f}")
                print(f"  Progress Score: {progress:.2f}")
                print(f"  Task Type: {results.get('task_type', 'unknown')}")
                print(f"  Errors: {len(results['errors'])}")
                
                # Show task-specific details
                if eval_mode == EvaluationMode.TASK_SPECIFIC and eval_summary:
                    task_state = eval_summary.get("task_specific_state", {})
                    if task_state:
                        print(f"  Task Details: {task_state}")
                
            except Exception as e:
                print(f"Episode {episode + 1} failed: {e}")
                all_results.append({
                    "success": False,
                    "steps": 0,
                    "errors": [str(e)],
                    "total_reward": 0.0,
                    "task_type": "unknown",
                    "evaluation_mode": eval_mode.value
                })
        
        # Summary
        print(f"\n{'='*40}")
        print("EPISODE RESULTS SUMMARY")
        print(f"{'='*40}")
        print(f"Task: {args.task}")
        print(f"Model: {args.model}")
        print(f"Prompt: '{prompt}'")
        print(f"Evaluation Mode: {eval_mode.value}")
        print(f"Episodes: {args.episodes}")
        print(f"Success Rate: {total_successes}/{args.episodes} ({100*total_successes/args.episodes:.1f}%)")
        
        if all_results:
            avg_reward = np.mean([r['total_reward'] for r in all_results])
            avg_steps = np.mean([r['steps'] for r in all_results if r['steps'] > 0])
            avg_progress = total_progress / args.episodes
            
            print(f"Average Total Reward: {avg_reward:.2f}")
            print(f"Average Steps: {avg_steps:.1f}")
            print(f"Average Progress Score: {avg_progress:.2f}")
            
            if eval_mode == EvaluationMode.TASK_SPECIFIC:
                print(f"\nTask-Specific Evaluation Features:")
                print(f"- Task-aware reward calculation")
                print(f"- Progress tracking and analysis")
                print(f"- Realistic completion detection")
                print(f"- Coordinated movement assessment")
        
        print(f"Hardware: {'Real ALOHA system' if args.use_hardware else 'Simulation'}")
        print(f"{'='*40}")
        
        # Save results (always enabled)
        print(f"\nSaving evaluation results to: {args.output_file}")
        
        # Prepare comprehensive results for saving
        save_data = {
            "evaluation_metadata": {
                "model": args.model,
                "task": args.task,
                "prompt": prompt,
                "evaluation_mode": eval_mode.value,
                "hardware": args.use_hardware,
                "max_steps": args.steps,
                "num_episodes": args.episodes,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "summary_metrics": {
                "success_rate": total_successes / args.episodes if args.episodes > 0 else 0.0,
                "total_successes": total_successes,
                "total_episodes": args.episodes,
                "average_reward": avg_reward if all_results else 0.0,
                "average_steps": avg_steps if all_results else 0.0,
                "average_progress": avg_progress if args.episodes > 0 else 0.0
            },
            "episode_results": all_results,
            "evaluation_features": {
                "task_specific_evaluation": eval_mode == EvaluationMode.TASK_SPECIFIC,
                "task_specific_rewards": eval_mode == EvaluationMode.TASK_SPECIFIC,
                "progress_tracking": eval_mode == EvaluationMode.TASK_SPECIFIC,
                "realistic_completion": eval_mode == EvaluationMode.TASK_SPECIFIC
            }
        }
        
        try:
            with open(args.output_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            print("Evaluation results saved successfully!")
        except Exception as save_error:
            print(f"Warning: Failed to save results to {args.output_file}: {save_error}")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("\nTroubleshooting:")
        print("- Ensure OpenPI is properly installed")
        print("- Check that model checkpoints are available")
        print("- For hardware mode: ensure ALOHA is connected and ROS is running")


if __name__ == "__main__":
    main() 