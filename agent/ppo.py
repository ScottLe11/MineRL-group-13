"""
PPO (Proximal Policy Optimization) Agent.

Implements PPO-Clip algorithm with:
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Value function clipping
- Entropy bonus for exploration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional

from networks.policy_network import ActorCriticNetwork


class RolloutBuffer:
    """
    Buffer for storing rollout experiences for PPO.
    
    Unlike replay buffer, this stores complete trajectories
    and computes advantages using GAE.
    """
    
    def __init__(self, buffer_size: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Args:
            buffer_size: Maximum number of steps to store
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()
    
    def reset(self):
        """Clear the buffer."""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.ptr = 0
    
    def add(self, obs: dict, action: int, log_prob: float, reward: float, 
            value: float, done: bool):
        """Add a single transition to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.ptr += 1
    
    def compute_returns_and_advantages(self, last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and advantages using GAE.
        
        Args:
            last_value: Value estimate for the state after the last step
            
        Returns:
            (returns, advantages): Arrays of shape (buffer_size,)
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)
        
        # GAE computation
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + np.array(self.values)
        
        return returns, advantages
    
    def get_batches(self, batch_size: int, last_value: float):
        """
        Generate minibatches for PPO update.
        
        Args:
            batch_size: Size of each minibatch
            last_value: Value estimate for computing advantages
            
        Yields:
            Batches of (obs, actions, old_log_probs, returns, advantages)
        """
        returns, advantages = self.compute_returns_and_advantages(last_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create random indices
        indices = np.random.permutation(len(self.rewards))
        
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            # Batch observations
            batch_obs = {
                'pov': np.stack([self.observations[i]['pov'] for i in batch_indices]),
                'time': np.array([self.observations[i]['time'] for i in batch_indices]),
                'yaw': np.array([self.observations[i]['yaw'] for i in batch_indices]),
                'pitch': np.array([self.observations[i]['pitch'] for i in batch_indices]),
            }
            
            yield (
                batch_obs,
                np.array([self.actions[i] for i in batch_indices]),
                np.array([self.log_probs[i] for i in batch_indices]),
                returns[batch_indices],
                advantages[batch_indices]
            )
    
    def __len__(self):
        return len(self.rewards)


class PPOAgent:
    """
    PPO Agent with clipped objective and GAE.
    
    Key hyperparameters:
    - clip_epsilon: PPO clipping parameter (typically 0.1-0.2)
    - entropy_coef: Entropy bonus coefficient
    - value_coef: Value loss coefficient
    - max_grad_norm: Gradient clipping
    """
    
    def __init__(
        self,
        num_actions: int = 23,
        input_channels: int = 4,
        num_scalars: int = 3,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = None
    ):
        """
        Args:
            num_actions: Number of discrete actions
            input_channels: Number of stacked frames
            num_scalars: Number of scalar observations
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_steps: Number of steps to collect before update
            n_epochs: Number of epochs per update
            batch_size: Minibatch size
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
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
        print(f"PPOAgent using device: {self.device}")
        
        # Network
        self.policy = ActorCriticNetwork(
            num_actions=num_actions,
            input_channels=input_channels,
            num_scalars=num_scalars
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Rollout buffer
        self.buffer = RolloutBuffer(buffer_size=n_steps, gamma=gamma, gae_lambda=gae_lambda)
        
        # Hyperparameters
        self.num_actions = num_actions
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Counters
        self.step_count = 0
        self.update_count = 0
    
    def select_action(self, state: dict) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        
        Args:
            state: Observation dict
            
        Returns:
            (action, log_prob, value): Action, its log probability, and state value
        """
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            action, log_prob, _, value = self.policy.get_action_and_value(state_tensor)
            
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state: dict, action: int, log_prob: float,
                        reward: float, value: float, done: bool):
        """Store transition in rollout buffer."""
        self.buffer.add(state, action, log_prob, reward, value, done)
        self.step_count += 1
    
    def update(self, last_state: dict) -> Dict:
        """
        Perform PPO update using collected rollout.
        
        Args:
            last_state: State after the last step (for bootstrap value)
            
        Returns:
            Dict of training metrics
        """
        # Get bootstrap value
        with torch.no_grad():
            last_state_tensor = self._state_to_tensor(last_state)
            last_value = self.policy.get_value(last_state_tensor).item()
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        # Multiple epochs over the data
        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size, last_value):
                obs, actions, old_log_probs, returns, advantages = batch
                
                # Convert to tensors
                obs_tensor = self._batch_to_tensor(obs)
                actions = torch.tensor(actions, dtype=torch.long, device=self.device)
                old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
                returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
                advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
                
                # Get current policy outputs
                _, new_log_probs, entropy, values = self.policy.get_action_and_value(obs_tensor, actions)
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values, returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1
        
        # Clear buffer
        self.buffer.reset()
        self.update_count += 1
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }
    
    def _state_to_tensor(self, state: dict) -> dict:
        """Convert single state to tensor dict."""
        return {
            'pov': torch.tensor(state['pov'], dtype=torch.float32, device=self.device).unsqueeze(0),
            'time': torch.tensor([state.get('time', 0.0)], dtype=torch.float32, device=self.device),
            'yaw': torch.tensor([state.get('yaw', 0.0)], dtype=torch.float32, device=self.device),
            'pitch': torch.tensor([state.get('pitch', 0.0)], dtype=torch.float32, device=self.device),
        }
    
    def _batch_to_tensor(self, obs: dict) -> dict:
        """Convert batch dict to tensor dict."""
        return {
            'pov': torch.tensor(obs['pov'], dtype=torch.float32, device=self.device),
            'time': torch.tensor(obs['time'], dtype=torch.float32, device=self.device),
            'yaw': torch.tensor(obs['yaw'], dtype=torch.float32, device=self.device),
            'pitch': torch.tensor(obs['pitch'], dtype=torch.float32, device=self.device),
        }
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'update_count': self.update_count,
        }, path)
        print(f"PPO agent saved to {path}")
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.update_count = checkpoint['update_count']
        print(f"PPO agent loaded from {path}")


if __name__ == "__main__":
    print("✅ PPOAgent Test")
    
    agent = PPOAgent(
        num_actions=23,
        n_steps=64,
        n_epochs=2,
        batch_size=16,
        device='cpu'
    )
    
    # Generate fake rollout
    print("\n  Collecting rollout...")
    state = {
        'pov': np.random.randint(0, 256, (4, 64, 64), dtype=np.uint8),
        'time': 1.0,
        'yaw': 0.0,
        'pitch': 0.0
    }
    
    for i in range(64):
        action, log_prob, value = agent.select_action(state)
        reward = -0.001
        done = (i == 63)
        
        agent.store_transition(state, action, log_prob, reward, value, done)
        
        state = {
            'pov': np.random.randint(0, 256, (4, 64, 64), dtype=np.uint8),
            'time': 1.0 - i / 64,
            'yaw': 0.0,
            'pitch': 0.0
        }
    
    print(f"  Buffer size: {len(agent.buffer)}")
    
    # Perform update
    print("\n  Performing PPO update...")
    metrics = agent.update(state)
    print(f"  Policy loss: {metrics['policy_loss']:.4f}")
    print(f"  Value loss: {metrics['value_loss']:.4f}")
    print(f"  Entropy: {metrics['entropy']:.4f}")
    
    print("\n✅ PPOAgent validated!")

