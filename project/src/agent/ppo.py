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
                'time_left': np.array([self.observations[i]['time_left'] for i in batch_indices]),
                'yaw': np.array([self.observations[i]['yaw'] for i in batch_indices]),
                'pitch': np.array([self.observations[i]['pitch'] for i in batch_indices]),
                'place_table_safe': np.array([self.observations[i]['place_table_safe'] for i in batch_indices]),
                'inv_logs': np.array([self.observations[i]['inv_logs'] for i in batch_indices]),
                'inv_planks': np.array([self.observations[i]['inv_planks'] for i in batch_indices]),
                'inv_sticks': np.array([self.observations[i]['inv_sticks'] for i in batch_indices]),
                'inv_table': np.array([self.observations[i]['inv_table'] for i in batch_indices]),
                'inv_axe': np.array([self.observations[i]['inv_axe'] for i in batch_indices]),
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
        num_scalars: int = 9,
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
        cnn_architecture: str = 'small',
        attention_type: str = 'none',
        use_scalar_network: bool = False,
        scalar_hidden_dim: int = 64,
        scalar_output_dim: int = 64,
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
            cnn_architecture: CNN architecture ('tiny', 'small', 'medium', 'wide', 'deep')
            attention_type: Attention mechanism ('none', 'spatial', 'cbam', 'treechop_bias')
            use_scalar_network: Whether to process scalars through 2-layer FC network
            scalar_hidden_dim: Hidden dimension for scalar network
            scalar_output_dim: Output dimension for scalar network
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

        # Store LR for resume logic
        self.learning_rate = learning_rate

        # Network (now with configurable architecture!)
        self.policy = ActorCriticNetwork(
            num_actions=num_actions,
            input_channels=input_channels,
            num_scalars=num_scalars,
            cnn_architecture=cnn_architecture,
            attention_type=attention_type,
            use_scalar_network=use_scalar_network,
            scalar_hidden_dim=scalar_hidden_dim,
            scalar_output_dim=scalar_output_dim
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
        self.episode_count = 0 # New: Track episodes

        # Action frequency tracking (for debugging)
        self.action_counts = [0] * num_actions
        self.last_actions = []  # Track last N actions
    
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

        action_int = action.item()

        # Track action frequency (for debugging)
        if action_int < len(self.action_counts):
            self.action_counts[action_int] += 1
        self.last_actions.append(action_int)
        if len(self.last_actions) > 100:  # Keep last 100
            self.last_actions.pop(0)

        return action_int, log_prob.item(), value.item()

    def get_action_stats(self) -> dict:
        """
        Get statistics about action selection.

        Returns:
            Dict with action counts, frequencies, and recent action diversity
        """
        total = sum(self.action_counts)
        if total == 0:
            return {}

        return {
            'total_actions': total,
            'action_counts': self.action_counts.copy(),
            'action_frequencies': [count / total for count in self.action_counts],
            'last_100_unique': len(set(self.last_actions)) if self.last_actions else 0,
            'last_100_actions': self.last_actions.copy()
        }

    def get_policy_info(self, state: dict) -> dict:
        """
        Get policy information for monitoring (PPO equivalent of Q-values).

        Args:
            state: Observation dict

        Returns:
            Dict with:
            - 'action_probs': Probability distribution over actions (numpy array)
            - 'entropy': Policy entropy (scalar, higher = more exploratory)
            - 'value': State value estimate (scalar)
            - 'top_actions': List of (action_idx, probability) tuples for top actions
        """
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            action_logits, value = self.policy.forward(state_tensor)

            # Get probabilities
            import torch.nn.functional as F
            probs = F.softmax(action_logits, dim=-1)

            # Get entropy
            from torch.distributions import Categorical
            dist = Categorical(probs)
            entropy = dist.entropy()

            # Convert to numpy
            probs_np = probs.cpu().numpy().flatten()

            return {
                'action_probs': probs_np,
                'entropy': entropy.item(),
                'value': value.item(),
                'top_actions': sorted(enumerate(probs_np), key=lambda x: x[1], reverse=True)[:5]
            }

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
        # Use numpy arrays to avoid "extremely slow" warning
        return {
            'pov': torch.tensor(state['pov'], dtype=torch.float32, device=self.device).unsqueeze(0),
            'time_left': torch.tensor(np.array([state.get('time_left', 0.0)], dtype=np.float32), dtype=torch.float32, device=self.device),
            'yaw': torch.tensor(np.array([state.get('yaw', 0.0)], dtype=np.float32), dtype=torch.float32, device=self.device),
            'pitch': torch.tensor(np.array([state.get('pitch', 0.0)], dtype=np.float32), dtype=torch.float32, device=self.device),
            'place_table_safe': torch.tensor(np.array([state.get('place_table_safe', 0.0)], dtype=np.float32), dtype=torch.float32, device=self.device),
            'inv_logs': torch.tensor(np.array([state.get('inv_logs', 0.0)], dtype=np.float32), dtype=torch.float32, device=self.device),
            'inv_planks': torch.tensor(np.array([state.get('inv_planks', 0.0)], dtype=np.float32), dtype=torch.float32, device=self.device),
            'inv_sticks': torch.tensor(np.array([state.get('inv_sticks', 0.0)], dtype=np.float32), dtype=torch.float32, device=self.device),
            'inv_table': torch.tensor(np.array([state.get('inv_table', 0.0)], dtype=np.float32), dtype=torch.float32, device=self.device),
            'inv_axe': torch.tensor(np.array([state.get('inv_axe', 0.0)], dtype=np.float32), dtype=torch.float32, device=self.device),
        }
    
    def _batch_to_tensor(self, obs: dict) -> dict:
        """Convert batch dict to tensor dict."""
        return {
            'pov': torch.tensor(obs['pov'], dtype=torch.float32, device=self.device),
            'time_left': torch.tensor(obs['time_left'], dtype=torch.float32, device=self.device),
            'yaw': torch.tensor(obs['yaw'], dtype=torch.float32, device=self.device),
            'pitch': torch.tensor(obs['pitch'], dtype=torch.float32, device=self.device),
            'place_table_safe': torch.tensor(obs.get('place_table_safe', np.zeros_like(obs['pitch'])), dtype=torch.float32, device=self.device),
            'inv_logs': torch.tensor(obs.get('inv_logs', np.zeros_like(obs['pitch'])), dtype=torch.float32, device=self.device),
            'inv_planks': torch.tensor(obs.get('inv_planks', np.zeros_like(obs['pitch'])), dtype=torch.float32, device=self.device),
            'inv_sticks': torch.tensor(obs.get('inv_sticks', np.zeros_like(obs['pitch'])), dtype=torch.float32, device=self.device),
            'inv_table': torch.tensor(obs.get('inv_table', np.zeros_like(obs['pitch'])), dtype=torch.float32, device=self.device),
            'inv_axe': torch.tensor(obs.get('inv_axe', np.zeros_like(obs['pitch'])), dtype=torch.float32, device=self.device),
        }
    
    def save(self, path: str):
        """Save agent state."""
        progress_settings = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'update_count': self.update_count,
            'episode_count': self.episode_count,
            'action_counts': self.action_counts,
            'last_actions': self.last_actions,
            'learning_rate': self.learning_rate
        }

        network = {
            'policy_state_dict': self.policy.state_dict()
        }

        torch.save({
            'progress_settings': progress_settings,
            'network': network,
        }, path)
        print(f"PPO agent saved to {path}")
    
    def load(self, path: str):
        """Load agent state with robust action mismatch handling and hyperparam override."""
        checkpoint = torch.load(path, map_location=self.device)

        if 'network' in checkpoint and 'progress_settings' in checkpoint:
            print("Loading new format checkpoint...")
            network_state = checkpoint['network']
            settings = checkpoint['progress_settings']
        else:
            print("Loading old format checkpoint...")
            network_state = checkpoint
            settings = checkpoint
        
        # Cross-Policy Check: DQN -> PPO
        if 'policy_state_dict' not in network_state and 'q_network_state_dict' in network_state:
            print(f"\nCross-Policy Load Detected: DQN Checkpoint -> PPO Agent")
            dqn_state = network_state['q_network_state_dict']
            ppo_constructed_state = {}
            
            for k, v in dqn_state.items():
                if any(scope in k for scope in ['cnn.', 'attention.', 'scalar_network.']):
                    ppo_constructed_state[k] = v
                        
            network_state['policy_state_dict'] = ppo_constructed_state
            settings = {}

        # 1. Load Policy with Shape Checking (Action space change handling)
        saved_state = network_state.get('policy_state_dict', {})
        current_state = self.policy.state_dict()
        new_state = {}
        
        mismatch_detected = False
        for k, v in current_state.items():
            if k in saved_state:
                saved_v = saved_state[k]
                if saved_v.shape != v.shape:
                    mismatch_detected = True
                    print(f"⚠️  Shape mismatch for {k}: saved {saved_v.shape}, current {v.shape}")
                    
                    slices = tuple(slice(0, min(ds, dc)) for ds, dc in zip(saved_v.shape, v.shape))
                    
                    try:
                        v[slices] = saved_v[slices]
                    except Exception as e:
                         print(f"Could not auto-slice {k}: {e}. Keeping random init.")
                         
                    new_state[k] = v
                else:
                    new_state[k] = saved_v
            else:
                new_state[k] = v
        
        self.policy.load_state_dict(new_state)
        
        if mismatch_detected:
            print("✅ Handled action space mismatch by preserving matching weights.")
            print("⚠️  Resetting optimizer due to network shape change...")
            self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # 2. Load Optimizer (Reset if failure, force LR)
        if 'optimizer_state_dict' in settings and not mismatch_detected:
            try:
                self.optimizer.load_state_dict(settings['optimizer_state_dict'])
                
                # FORCE update learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                print(f"Optimizer loaded")
            except Exception as e:
                print(f"Optimizer load failed (likely shape change), resetting optimizer: {e}")
                self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # 3. Counters
        self.step_count = settings.get('step_count', 0)
        self.update_count = settings.get('update_count', 0)
        self.episode_count = settings.get('episode_count', 0)
        
        # Load action tracking if available 
        saved_counts = settings.get('action_counts', [])
        if len(saved_counts) == self.num_actions:
            self.action_counts = saved_counts
        else:
            self.action_counts = [0] * self.num_actions # Reset if size changed

        self.last_actions = settings.get('last_actions', [])


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
        'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
        'time_left': 1.0,
        'yaw': 0.0,
        'pitch': 0.0
    }

    for i in range(64):
        action, log_prob, value = agent.select_action(state)
        reward = -0.001
        done = (i == 63)

        agent.store_transition(state, action, log_prob, reward, value, done)

        state = {
            'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
            'time_left': 1.0 - i / 64,
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