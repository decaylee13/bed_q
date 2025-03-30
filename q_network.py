import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from environment import HospitalBedEnv
import os
from torch.cuda.amp import autocast, GradScaler
from collections import deque

# Check if CUDA is available
device = torch.device("cpu")

class ReplayBuffer:
    def __init__(self, capacity=50000):
        """
        Initialize a replay buffer with fixed capacity.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.buffer = []
        self.capacity = capacity
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions randomly.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNetwork(nn.Module):
    def __init__(self, patient_feature_dim, bed_feature_dim, max_beds, action_dim):
        """
        Initialize the DQN model.
        
        Args:
            patient_feature_dim (int): Dimension of patient feature vector
            bed_feature_dim (int): Dimension of single bed feature vector
            max_beds (int): Maximum number of beds in state
            action_dim (int): Number of possible actions (wait + assign beds)
        """
        super(DQNetwork, self).__init__()
        
        # Calculate total input dimension
        # Patient features + flattened bed features
        self.total_input_dim = patient_feature_dim + (bed_feature_dim * max_beds)
        self.patient_feature_dim = patient_feature_dim
        self.bed_feature_dim = bed_feature_dim
        self.max_beds = max_beds
        
        # 2-layer network as specified
        self.network = nn.Sequential(
            nn.Linear(self.total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # action_dim = num_beds + 1 (for wait)
        )
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Either a dictionary with 'patient' and 'beds' keys,
                  or a pre-processed tensor
                  
        Returns:
            torch.Tensor: Q-values for each action
        """
        # If state is already a tensor (batched data), pass directly
        if isinstance(state, torch.Tensor):
            return self.network(state)
            
        # Process the state dictionary (single state)
        if isinstance(state, dict):
            # Extract patient and bed features
            patient_features = torch.tensor(state['patient'], dtype=torch.float32, device=device)
            
            # Process bed features - flatten the list of lists
            bed_features_flat = []
            for bed in state['beds']:
                bed_features_flat.extend(bed)
            bed_features_tensor = torch.tensor(bed_features_flat, dtype=torch.float32, device=device)
            
            # Combine features
            combined = torch.cat([patient_features, bed_features_tensor])
            
            # Pass through network
            return self.network(combined)

    def process_batch_states(self, states):
        """
        Process a batch of states from the replay buffer for training.
        
        Args:
            states: List of state dictionaries
            
        Returns:
            torch.Tensor: Batch of processed states
        """
        batch_size = len(states)
        processed_states = torch.zeros(batch_size, self.total_input_dim, device=device)
        
        for i, state in enumerate(states):
            # Extract patient features
            patient_features = torch.tensor(state['patient'], dtype=torch.float32, device=device)
            
            # Flatten bed features
            bed_features_flat = []
            for bed in state['beds']:
                bed_features_flat.extend(bed)
            bed_features_tensor = torch.tensor(bed_features_flat, dtype=torch.float32, device=device)
            
            # Combine patient and bed features
            processed_states[i] = torch.cat([patient_features, bed_features_tensor])
        
        return processed_states

class DQNAgent:
    def __init__(self, 
                 patient_feature_dim, 
                 bed_feature_dim, 
                 max_beds, 
                 action_dim,
                 learning_rate, 
                 gamma, 
                 epsilon_start, 
                 epsilon_end, 
                 epsilon_decay, 
                 target_update,
                 replay_buffer_size):
        """
        Initialize the DQN agent.
        
        Args:
            patient_feature_dim (int): Dimension of patient feature vector
            bed_feature_dim (int): Dimension of single bed feature vector
            max_beds (int): Maximum number of beds in state
            action_dim (int): Number of possible actions
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor
            epsilon_start (float): Starting exploration rate
            epsilon_end (float): Minimum exploration rate
            epsilon_decay (float): Decay rate for exploration
            target_update (int): Steps between target network updates
            replay_buffer_size (int): Size of replay buffer
        """
        # Initialize networks
        self.policy_net = DQNetwork(patient_feature_dim, bed_feature_dim, max_beds, action_dim).to(device)
        self.target_net = DQNetwork(patient_feature_dim, bed_feature_dim, max_beds, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained directly
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize AMP scaler
        self.scaler = GradScaler()
        
        # Setup parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.action_dim = action_dim
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        # Training steps counter
        self.steps_done = 0
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            int: Selected action
        """
        if random.random() > self.epsilon:
            with torch.no_grad():
                with autocast():
                    # Use policy net to select best action
                    return self.policy_net(state).argmax().item()
        else:
            # Random action
            return random.randrange(self.action_dim)
    
    def update_epsilon(self):
        """Decay epsilon value"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update_target_network(self):
        """Update target network if needed"""
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_checkpoint(self, folder='./weights'):
        """
        Save model checkpoint.
        
        Args:
            folder (str): Directory to save checkpoint
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        filename = f"{folder}/dqn_steps_{self.steps_done}.pt"
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon
        }, filename)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filepath):
        """
        Load model checkpoint.
        
        Args:
            filepath (str): Path to checkpoint file
        """
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps_done = checkpoint['steps_done']
            self.epsilon = checkpoint['epsilon']
            print(f"Checkpoint loaded: {filepath}")
        else:
            print(f"No checkpoint found at {filepath}")
    
    def train(self, batch_size=32):
        """
        Train the model with a batch from replay buffer.
        
        Args:
            batch_size (int): Batch size for training
        """
        # Increment steps
        self.steps_done += 1
        
        # Update target network if needed
        self.update_target_network()
        
        # Check if enough samples in replay buffer
        if len(self.replay_buffer) < batch_size:
            return
        
        # This is where you would implement the replay buffer training
        # Code is commented out as requested, to be manually implemented later
        
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Process states
        state_batch = self.policy_net.process_batch_states(states)
        next_state_batch = self.target_net.process_batch_states(next_states)
        
        # Convert other data to tensors
        action_batch = torch.tensor(actions, dtype=torch.long, device=device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=device)
        done_batch = torch.tensor(dones, dtype=torch.bool, device=device)
        
        # Use AMP for forward pass
        with autocast():
            # Get Q values for chosen actions
            q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
            
            # Compute target Q values (no grad needed for target network)
            with torch.no_grad():
                next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * next_q_values * (~done_batch)
            
            # Compute loss
            loss = nn.functional.smooth_l1_loss(q_values, target_q_values)
        
        # Optimize with AMP
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        
        # Update epsilon
        self.update_epsilon()
        
        # Save checkpoint periodically
        if self.steps_done % 1000 == 0:
            self.save_checkpoint()

def train_dqn(env, agent, num_episodes, max_steps, batch_size=32, checkpoint_interval=5):
    """
    Train the DQN agent.
    
    Args:
        env: Hospital bed environment
        agent: DQN agent
        num_episodes (int): Number of episodes to train
        max_steps (int): Maximum steps per episode
        batch_size (int): Batch size for training
        checkpoint_interval (int): Episodes between checkpoints
    """
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Train agent
            agent.train(batch_size)  # This would be implemented later
            
            # For now, manually update steps
            agent.steps_done += 1
            
            # Update target network if needed
            agent.update_target_network()
            
            # Save checkpoint periodically
            if agent.steps_done % 1000 == 0:
                agent.save_checkpoint()
            
            if done:
                break
        
        # Update epsilon after each episode
        agent.update_epsilon()
        
        # Print episode results
        print(f"Episode {episode}/{num_episodes}, Steps: {step}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.4f}")
        
        # Save checkpoint at intervals
        if episode % checkpoint_interval == 0:
            agent.save_checkpoint(folder=f"./weights/episode_{episode}")

