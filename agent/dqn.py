"""
DQN Agent with Double DQN and soft target updates.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional

from networks import DQNNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent with:
    - Double DQN (use online network to select actions, target to evaluate)
    - Soft target updates (Polyak averaging)
    - Epsilon-greedy exploration with linear decay
    """
    
    def __init__(
        self,
        num_actions: int = 23,
        input_channels: int = 4,
        num_scalars: int = 3,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100000,
        buffer_capacity: int = 100000,
        buffer_min_size: int = 1000,
        batch_size: int = 32,
        device: str = None
    ):
        """
        Args:
            num_actions: Number of discrete actions
            input_channels: Number of stacked frames
            num_scalars: Number of scalar observations
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            tau: Soft update coefficient (1.0 = hard update)
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon
            buffer_capacity: Replay buffer capacity
            buffer_min_size: Min experiences before training
            batch_size: Batch size for training
            device: 'cuda', 'mps', 'cpu', or None for auto-detect
        """
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)
        print(f"DQNAgent using device: {self.device}")
        
        # Networks
        self.q_network = DQNNetwork(num_actions, input_channels, num_scalars).to(self.device)
        self.target_network = DQNNetwork(num_actions, input_channels, num_scalars).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is not trained
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, min_size=buffer_min_size)
        
        # Hyperparameters
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        
        # Counters
        self.step_count = 0
        self.train_count = 0
    
    def get_epsilon(self) -> float:
        """Get current epsilon value based on step count."""
        progress = min(1.0, self.step_count / self.epsilon_decay_steps)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress
    
    def select_action(self, state: Dict, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Observation dict with 'pov', 'time', 'yaw', 'pitch'
            explore: Whether to use exploration (False for evaluation)
        
        Returns:
            action: Selected action index (0-22)
        """
        epsilon = self.get_epsilon() if explore else 0.0
        
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_experience(self, state: Dict, action: int, reward: float, 
                        next_state: Dict, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.step_count += 1
    
    def train_step(self) -> Optional[Dict]:
        """
        Perform one training step if buffer is ready.
        
        Returns:
            dict with training metrics, or None if buffer not ready
        """
        if not self.replay_buffer.is_ready():
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = self._batch_states_to_tensor(states)
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_state_batch = self._batch_states_to_tensor(next_states)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Current Q-values
        q_values = self.q_network(state_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select best action, target to evaluate
        with torch.no_grad():
            next_q_online = self.q_network(next_state_batch)
            best_actions = next_q_online.argmax(dim=1)
            
            next_q_target = self.target_network(next_state_batch)
            next_q_values = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            
            # Target Q-values
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        # Loss (Huber loss is more stable than MSE)
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update()
        
        self.train_count += 1
        
        return {
            'loss': loss.item(),
            'q_mean': q_values.mean().item(),
            'q_max': q_values.max().item(),
            'epsilon': self.get_epsilon()
        }
    
    def _soft_update(self):
        """Soft update target network weights: θ_target = τ*θ_local + (1-τ)*θ_target"""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                              self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def _state_to_tensor(self, state: Dict) -> Dict:
        """Convert single state dict to batched tensor dict."""
        return {
            'pov': torch.tensor(state['pov'], dtype=torch.float32, device=self.device).unsqueeze(0),
            'time': torch.tensor([state.get('time', 0.0)], dtype=torch.float32, device=self.device),
            'yaw': torch.tensor([state.get('yaw', 0.0)], dtype=torch.float32, device=self.device),
            'pitch': torch.tensor([state.get('pitch', 0.0)], dtype=torch.float32, device=self.device)
        }
    
    def _batch_states_to_tensor(self, states: list) -> Dict:
        """Convert list of state dicts to batched tensor dict."""
        povs = np.stack([s['pov'] for s in states])
        times = np.array([s.get('time', 0.0) for s in states])
        yaws = np.array([s.get('yaw', 0.0) for s in states])
        pitches = np.array([s.get('pitch', 0.0) for s in states])
        
        return {
            'pov': torch.tensor(povs, dtype=torch.float32, device=self.device),
            'time': torch.tensor(times, dtype=torch.float32, device=self.device),
            'yaw': torch.tensor(yaws, dtype=torch.float32, device=self.device),
            'pitch': torch.tensor(pitches, dtype=torch.float32, device=self.device)
        }
    
    def save(self, path: str):
        """Save agent state to file."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'train_count': self.train_count
        }, path)
        print(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent state from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.train_count = checkpoint['train_count']
        print(f"Agent loaded from {path}")


if __name__ == "__main__":
    # Quick test
    print("Testing DQNAgent...")
    
    agent = DQNAgent(
        num_actions=23,
        buffer_capacity=100,
        buffer_min_size=10,
        batch_size=4,
        device='cpu'
    )
    
    # Generate fake experiences
    print("\n  Filling replay buffer...")
    for i in range(20):
        state = {
            'pov': np.random.randint(0, 256, (4, 64, 64), dtype=np.uint8),
            'time': float(i) / 20,
            'yaw': 0.0,
            'pitch': 0.0
        }
        action = agent.select_action(state, explore=True)
        next_state = {
            'pov': np.random.randint(0, 256, (4, 64, 64), dtype=np.uint8),
            'time': float(i + 1) / 20,
            'yaw': 0.0,
            'pitch': 0.0
        }
        agent.store_experience(state, action, reward=-0.001, next_state=next_state, done=(i == 19))
    
    print(f"  Buffer size: {len(agent.replay_buffer)}")
    print(f"  Buffer ready: {agent.replay_buffer.is_ready()}")
    
    # Train a few steps
    print("\n  Training steps...")
    for i in range(5):
        metrics = agent.train_step()
        if metrics:
            print(f"    Step {i+1}: loss={metrics['loss']:.4f}, q_mean={metrics['q_mean']:.4f}, epsilon={metrics['epsilon']:.3f}")
    
    # Test action selection
    print("\n  Testing action selection...")
    state = {
        'pov': np.random.randint(0, 256, (4, 64, 64), dtype=np.uint8),
        'time': 0.5,
        'yaw': 0.0,
        'pitch': 0.0
    }
    action_explore = agent.select_action(state, explore=True)
    action_greedy = agent.select_action(state, explore=False)
    print(f"    Action (explore): {action_explore}")
    print(f"    Action (greedy): {action_greedy}")
    
    print("\n✅ DQNAgent test passed!")

