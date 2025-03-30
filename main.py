import os
import sys
import time
import json
from environment import HospitalBedEnv
from q_network import DQNAgent, train_dqn

def visualize_beds(env):
    #Visualizing bed occupancy
    bed_status = []
    occupied_count = 0
    
    for i, bed in enumerate(env.beds):
        is_occupied = bed.is_occupied()
        if is_occupied:
            occupied_count += 1
        status = 1 if is_occupied else 0
        bed_status.append(status)

    print(f"DEBUG: {occupied_count} out of {len(env.beds)} beds are occupied")
    
    # Check if current_patient is None in any occupied bed (should never happen)
    for i, bed in enumerate(env.beds):
        if bed.is_occupied() and bed.current_patient is None:
            print(f"ERROR: Bed {i} shows as occupied but has no patient!")
            
    return "Beds: " + str(bed_status)

def visualize_state(state):
    """
    Visualize the state information.
    Returns a string representation of the state.
    """
    patient_info = f"Patient features: {state['patient']}"
    return f"{patient_info}\n"

def print_environment_info(env):
    """
    Print general information about the environment.
    """
    print("\n=== Environment Information ===")
    print(f"Total number of beds: {len(env.beds)}")
    print(f"Current queue length: {len(env.patient_queue)}")
    print(f"Current patients in beds: {sum(1 for bed in env.beds if bed.is_occupied())}")
    print(f"Current time step: {env.curr_time}")
    print("===============================\n")

def run_training_with_verbose(env, agent, num_episodes=10, max_steps=100, print_interval=1):
    """
    Run training with detailed output of states and actions.
    
    Args:
        env: Hospital bed environment
        agent: DQN agent
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        print_interval: How often to print detailed step information
    """
    print("\n=== Starting Training with Verbose Output ===\n")
    
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        
        print(f"\n=== Episode {episode+1}/{num_episodes} ===")
        print_environment_info(env)
        print("Initial state:")
        print(visualize_state(state))
        print(visualize_beds(env))
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Interpret action
            if action == 0:
                action_description = "WAIT"
            else:
                action_description = f"ASSIGN to bed {action-1}"
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Print step information at intervals
            if step % print_interval == 0:
                print(f"\nStep {step+1}:")
                print(f"Action: {action_description}")
                print(f"Reward: {reward:.2f}")
                print(f"Cumulative reward: {episode_reward:.2f}")
                print(f"Epsilon: {agent.epsilon:.4f}")
                print(visualize_state(state))
                print(visualize_beds(env))
                print(f"Queue length: {len(env.patient_queue)}")
            
            # Update agent
            agent.steps_done += 1
            agent.update_target_network()
            agent.update_epsilon()
            
            if done:
                print("\nEpisode terminated early.")
                break
        
        # Episode summary
        print(f"\n=== Episode {episode+1} Summary ===")
        print(f"Total steps: {step+1}")
        print(f"Total reward: {episode_reward:.2f}")
        print(f"Final epsilon: {agent.epsilon:.4f}")
        print_environment_info(env)
        
        # Save checkpoint every episode
        agent.save_checkpoint(folder=f"./weights")

def main():
    """
    Main function to set up and run the hospital bed allocation simulation.
    """
    # Load configuration
    config_path = 'config.json'
    # if not os.path.exists(config_path):
    #     print(f"Configuration file {config_path} not found. Creating default config.")
    #     create_default_config(config_path)
    
    with open(config_path, 'r') as f:
        env_config = json.load(f)
    
    print(f"Loaded configuration from {config_path}")
    
    # Initialize environment
    env = HospitalBedEnv(env_config)
    learning_param_env = env_config.get('learning_parameters')
    # Get dimensions from environment
    state = env.reset()
    patient_feature_dim = len(state['patient'])
    bed_feature_dim = len(state['beds'][0]) if state['beds'] else 1
    max_beds = env_config.get('environment_parameters').get('max_beds_in_state')
    print(f"max_bed: {max_beds}")
    
    action_dim = len(env.beds) + 1  # +1 for wait action
    
    print(f"State dimensions: Patient features: {patient_feature_dim}, Bed features: {bed_feature_dim}")
    print(f"Max beds: {max_beds}, Action dim: {action_dim}")
    
    # Initialize agent
    agent = DQNAgent(
        patient_feature_dim=patient_feature_dim,
        bed_feature_dim=bed_feature_dim,
        max_beds=max_beds,
        action_dim=action_dim,
        learning_rate=learning_param_env.get('learning_rate', 0.001),
        gamma=learning_param_env.get('gamma', 0.99),
        epsilon_start=learning_param_env.get('epsilon_start'),
        epsilon_end=learning_param_env.get('epsilon_end'),
        epsilon_decay=learning_param_env.get('epsilon_decay'),
        target_update=learning_param_env.get('target_update', 100),
        replay_buffer_size=learning_param_env.get('replay_buffer_size')
    )
    
    # Choose mode (train or verbose)
    mode = input("Select mode (1=Train, 2=Verbose): ")
    
    if mode == "1":
        # Normal training
        print("\nStarting normal training...")
        train_dqn(
            env, 
            agent, 
            num_episodes=learning_param_env.get('num_episodes'), 
            max_steps=learning_param_env.get('max_steps'),
            batch_size=learning_param_env.get('batch_size')
        )
    else:
        # Verbose mode
        print("\nStarting verbose training...")
        num_episodes = int(input("Enter number of episodes (default=5): ") or 5)
        max_steps = int(input("Enter max steps per episode (default=50): ") or 50)
        print_interval = int(input("Print every N steps (default=1): ") or 1)
        
        run_training_with_verbose(
            env, 
            agent, 
            num_episodes=num_episodes,
            max_steps=max_steps,
            print_interval=print_interval
        )

if __name__ == "__main__":
    main()