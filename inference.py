import os
import json
import torch
import numpy as np
import time
from environment import HospitalBedEnv
from q_network import DQNAgent

class BedAllocationInference:
    """
    Class to run inference using a trained DQN model for hospital bed allocation.
    """
    def __init__(self, model_path, config_path='config.json'):
        """
        Initialize the inference engine.
        
        Args:
            model_path (str): Path to the saved model weights
            config_path (str): Path to the configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize environment
        self.env = HospitalBedEnv(self.config)
        env_params = self.config.get('environment_parameters')
        
        # Get dimensions from environment
        state = self.env.reset()
        self.patient_feature_dim = len(state['patient'])
        self.bed_feature_dim = len(state['beds'][0]) if state['beds'] else 1
        self.max_beds = env_params.get('max_beds_in_state')
        self.action_dim = len(self.env.beds) + 1  # +1 for wait action
        
        # Initialize agent
        learning_params = self.config.get('learning_parameters')
        self.agent = DQNAgent(
            patient_feature_dim=self.patient_feature_dim,
            bed_feature_dim=self.bed_feature_dim,
            max_beds=self.max_beds,
            action_dim=self.action_dim,
            learning_rate=learning_params.get('learning_rate', 0.001),
            gamma=learning_params.get('gamma', 0.99),
            epsilon_start=0,  # No exploration during inference
            epsilon_end=0,
            epsilon_decay=1,
            target_update=learning_params.get('target_update', 100),
            replay_buffer_size=learning_params.get('replay_buffer_size')
        )
        
        # Load trained weights
        self.agent.load_checkpoint(model_path)
        self.agent.policy_net.eval()  # Set to evaluation mode
        
    def visualize_beds(self):
        """Visualizes the current bed occupancy."""
        bed_status = []
        occupied_count = 0
        
        for i, bed in enumerate(self.env.beds):
            is_occupied = bed.is_occupied()
            if is_occupied:
                occupied_count += 1
                patient = bed.current_patient
                wait_time = patient.wait_time if patient else "N/A"
                time_in_bed = bed.time_occupied
                bed_status.append(f"Bed {i}: OCCUPIED (Patient wait: {wait_time}, Time in bed: {time_in_bed}/{bed.occupancy_delta})")
            else:
                bed_status.append(f"Bed {i}: EMPTY")
        
        print(f"\n==== BED STATUS: {occupied_count}/{len(self.env.beds)} occupied ====")
        for status in bed_status:
            print(status)
        print("="*50)
    
    def get_action_probabilities(self, state):
        """
        Get Q-values for all actions in the current state.
        
        Args:
            state: Current environment state
            
        Returns:
            List of (action, q_value) pairs sorted by q_value
        """
        with torch.no_grad():
            q_values = self.agent.policy_net(state).cpu().numpy()
        
        # Create action-value pairs
        action_values = []
        for i, q in enumerate(q_values):
            if i == 0:
                action_desc = "WAIT"
            else:
                action_desc = f"ASSIGN to bed {i-1}"
            action_values.append((i, action_desc, q))
        
        # Sort by Q-value in descending order
        return sorted(action_values, key=lambda x: x[2], reverse=True)
    
    def run_episode(self, max_steps=100, visualize=True, delay=0.5):
        """
        Run a full episode using the trained model.
        
        Args:
            max_steps (int): Maximum steps for the episode
            visualize (bool): Whether to print visualization
            delay (float): Delay between steps for visualization
            
        Returns:
            dict: Episode statistics
        """
        state = self.env.reset()
        episode_reward = 0
        step_count = 0
        
        if visualize:
            print("\n===== STARTING NEW EPISODE =====")
            print(f"Queue length: {len(self.env.patient_queue)}")
            self.visualize_beds()
            
            # Show current patient
            if self.env.current_patient:
                print(f"Current patient: Wait time = {self.env.current_patient.wait_time}")
            else:
                print("No patient waiting")
        
        for step in range(max_steps):
            # Get action from policy
            action = self.agent.select_action(state)
            
            # Get action probabilities for visualization
            if visualize:
                action_probs = self.get_action_probabilities(state)
                print("\nAction probabilities:")
                for i, desc, q_val in action_probs[:3]:  # Show top 3 actions
                    if i == action:
                        print(f"* {desc}: {q_val:.4f} (SELECTED)")
                    else:
                        print(f"  {desc}: {q_val:.4f}")
            
            # Step environment
            next_state, reward, done, info = self.env.step(action)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Visualize if requested
            if visualize:
                print(f"\n--- Step {step+1} ---")
                if action == 0:
                    print("Action: WAIT")
                else:
                    print(f"Action: ASSIGN to bed {action-1}")
                print(f"Reward: {reward:.2f}")
                print(f"Cumulative reward: {episode_reward:.2f}")
                print(f"Queue length: {len(self.env.patient_queue)}")
                self.visualize_beds()
                
                # Show current patient
                if self.env.current_patient:
                    print(f"Current patient: Wait time = {self.env.current_patient.wait_time}")
                else:
                    print("No patient waiting")
                
                # Add delay for better visualization
                time.sleep(delay)
            
            if done:
                break
        
        # Episode summary
        if visualize:
            print("\n===== EPISODE COMPLETE =====")
            print(f"Steps: {step_count}")
            print(f"Total reward: {episode_reward:.2f}")
            print(f"Final queue length: {len(self.env.patient_queue)}")
            print("="*30)
        
        return {
            "steps": step_count,
            "total_reward": episode_reward,
            "final_queue_length": len(self.env.patient_queue),
            "occupied_beds": sum(1 for bed in self.env.beds if bed.is_occupied())
        }
    
    def run_multiple_episodes(self, num_episodes=10, max_steps=100, visualize=False):
        """
        Run multiple episodes and collect statistics.
        
        Args:
            num_episodes (int): Number of episodes to run
            max_steps (int): Maximum steps per episode
            visualize (bool): Whether to print visualization
            
        Returns:
            dict: Aggregated statistics
        """
        stats = []
        
        for episode in range(num_episodes):
            if visualize:
                print(f"\nRunning episode {episode+1}/{num_episodes}")
            
            episode_stats = self.run_episode(max_steps, visualize)
            stats.append(episode_stats)
        
        # Calculate aggregate statistics
        avg_reward = sum(s["total_reward"] for s in stats) / num_episodes
        avg_steps = sum(s["steps"] for s in stats) / num_episodes
        avg_queue = sum(s["final_queue_length"] for s in stats) / num_episodes
        avg_occupancy = sum(s["occupied_beds"] for s in stats) / num_episodes
        
        print("\n===== OVERALL STATISTICS =====")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Average steps: {avg_steps:.2f}")
        print(f"Average final queue length: {avg_queue:.2f}")
        print(f"Average bed occupancy: {avg_occupancy:.2f}/{len(self.env.beds)}")
        
        return {
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "avg_queue": avg_queue,
            "avg_occupancy": avg_occupancy
        }


def main():
    """
    Main function to run inference with a trained model.
    """
    # Ask for model path
    default_model_path = "./weights/dqn_steps_1000.pt"  # Default path
    model_path = input(f"Enter model path (default: {default_model_path}): ") or default_model_path
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    # Create inference engine
    inference = BedAllocationInference(model_path)
    
    # Choose mode
    print("\nSelect run mode:")
    print("1. Run a single episode with visualization")
    print("2. Run multiple episodes and get statistics")
    mode = input("Enter choice (1/2): ")
    
    if mode == "1":
        # Run single episode with visualization
        max_steps = int(input("Enter max steps (default: 100): ") or 100)
        delay = float(input("Enter delay between steps in seconds (default: 1.0): ") or 1.0)
        
        inference.run_episode(max_steps=max_steps, visualize=True, delay=delay)
    else:
        # Run multiple episodes
        num_episodes = int(input("Enter number of episodes (default: 10): ") or 10)
        max_steps = int(input("Enter max steps per episode (default: 100): ") or 100)
        visualize = input("Visualize episodes? (y/n, default: n): ").lower() == 'y'
        
        inference.run_multiple_episodes(num_episodes, max_steps, visualize)


if __name__ == "__main__":
    main()