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
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class EpsilonSchedule:
    """
    Linear epsilon decay schedule for exploration.
    
    Decays linearly from start to end over decay_steps,
    then stays at end value.
    """
    
    def __init__(self, start: float = 1.0, end: float = 0.05, decay_steps: int = 100000):
        """
        Args:
            start: Initial epsilon value (typically 1.0)
            end: Final epsilon value (e.g., 0.05 or 0.1)
            decay_steps: Number of steps to decay from start to end
        """
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
    
    def get_epsilon(self, step: int) -> float:
        """
        Get epsilon value for a given step.
        
        Args:
            step: Current step count
            
        Returns:
            Epsilon value between start and end
        """
        if step >= self.decay_steps:
            return self.end
        progress = step / self.decay_steps
        return self.start + (self.end - self.start) * progress


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
        target_update_method: str = "soft",
        soft_update_freq: int = 1,
        hard_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100000,
        buffer_capacity: int = 100000,
        buffer_min_size: int = 1000,
        batch_size: int = 32,
        max_grad_norm: float = 10.0,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
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
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            tau: Soft update coefficient (used when target_update_method="soft")
            target_update_method: "soft" (Polyak averaging) or "hard" (periodic copy)
            soft_update_freq: Train steps between soft updates (used when target_update_method="soft")
            hard_update_freq: Train steps between hard updates (used when target_update_method="hard")
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon
            buffer_capacity: Replay buffer capacity
            buffer_min_size: Min experiences before training
            batch_size: Batch size for training
            max_grad_norm: Maximum gradient norm for clipping
            use_per: Whether to use Prioritized Experience Replay
            per_alpha: Priority exponent for PER
            per_beta_start: Initial importance sampling exponent
            per_beta_end: Final importance sampling exponent
            cnn_architecture: CNN architecture name ('tiny', 'small', 'medium', 'wide', 'deep')
            attention_type: Attention mechanism ('none', 'spatial', 'cbam', 'treechop_bias')
            use_scalar_network: Whether to process scalars through 2-layer FC network
            scalar_hidden_dim: Hidden dimension for scalar network
            scalar_output_dim: Output dimension for scalar network
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

        # Store critical hyperparams for resume
        self.learning_rate = learning_rate
        self.num_actions = num_actions

        # Networks (with configurable architecture)
        self.q_network = DQNNetwork(
            num_actions,
            input_channels,
            num_scalars,
            cnn_architecture=cnn_architecture,
            attention_type=attention_type,
            use_scalar_network=use_scalar_network,
            scalar_hidden_dim=scalar_hidden_dim,
            scalar_output_dim=scalar_output_dim
        ).to(self.device)
        self.target_network = DQNNetwork(
            num_actions,
            input_channels,
            num_scalars,
            cnn_architecture=cnn_architecture,
            attention_type=attention_type,
            use_scalar_network=use_scalar_network,
            scalar_hidden_dim=scalar_hidden_dim,
            scalar_output_dim=scalar_output_dim
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is not trained

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer (PER or uniform)
        self.use_per = use_per
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity,
                min_size=buffer_min_size,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_end=per_beta_end,
                beta_anneal_steps=epsilon_decay_steps  # Anneal beta with epsilon
            )
            print("Using Prioritized Experience Replay")
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, min_size=buffer_min_size)
            print("Using uniform replay buffer")

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        # Target update settings
        self.target_update_method = target_update_method.lower()
        self.soft_update_freq = soft_update_freq
        self.hard_update_freq = hard_update_freq
        if self.target_update_method not in ("soft", "hard"):
            raise ValueError(f"target_update_method must be 'soft' or 'hard', got '{target_update_method}'")
        print(f"Target updates: {self.target_update_method}" +
              (f" (tau={tau}, every {soft_update_freq} steps)" if self.target_update_method == "soft" 
               else f" (every {hard_update_freq} steps)"))

        # Exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # PER settings (for beta annealing)
        self.per_beta_start = per_beta_start
        self.per_beta_end = per_beta_end

        # Counters
        self.step_count = 0
        self.train_count = 0
        self.episode_count = 0 

        # Action frequency tracking (for debugging)
        self.action_counts = [0] * num_actions
        self.last_actions = []  # Track last N actions
    
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
            action = random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()

        # Track action frequency (for debugging)
        if action < len(self.action_counts):
            self.action_counts[action] += 1
        self.last_actions.append(action)
        if len(self.last_actions) > 100:  # Keep last 100
            self.last_actions.pop(0)

        return action

    def get_q_values(self, state: Dict) -> np.ndarray:
        """
        Get Q-values for all actions for a given state.

        Args:
            state: Observation dict with 'pov', 'time', 'yaw', 'pitch'

        Returns:
            Array of Q-values for each action (shape: [num_actions])
        """
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy()[0]  # [0] to remove batch dimension

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

        # Sample batch (PER returns weights and indices)
        if self.use_per:
            states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(self.batch_size)
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size, device=self.device)
            indices = None

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

        # TD errors (for PER priority updates)
        td_errors = torch.abs(q_values - target_q_values)

        # Weighted loss (PER uses importance sampling weights)
        elementwise_loss = nn.SmoothL1Loss(reduction='none')(q_values, target_q_values)
        loss = (elementwise_loss * weights).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (use configured value)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        # Update priorities in PER buffer
        if self.use_per and indices is not None:
            priorities = td_errors.detach().cpu().numpy() + 1e-6  # Small epsilon to avoid zero priority
            self.replay_buffer.update_priorities(indices, priorities)

        # Update target network (soft or hard)
        if self.target_update_method == "soft" and self.train_count % self.soft_update_freq == 0:
            self._soft_update()
        elif self.target_update_method == "hard" and self.train_count % self.hard_update_freq == 0:
            self._hard_update()

        self.train_count += 1

        # Build metrics dict
        metrics = {
            'loss': loss.item(),
            'q_mean': q_values.mean().item(),
            'td_error_mean': td_errors.mean().item(),
            'epsilon': self.get_epsilon()
        }

        # Add PER beta if using PER
        if self.use_per:
            metrics['per_beta'] = self.replay_buffer.beta

        return metrics
    
    def _soft_update(self):
        """Soft update target network weights: θ_target = τ*θ_local + (1-τ)*θ_target"""
        for target_param, local_param in zip(self.target_network.parameters(),
                                             self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def _hard_update(self):
        """Hard update target network: copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _state_to_tensor(self, state: Dict) -> Dict:
        """Convert single state dict to batched tensor dict."""
        # Use numpy arrays to avoid "extremely slow" warning
        return {
            'pov': torch.tensor(state['pov'], dtype=torch.float32, device=self.device).unsqueeze(0),
            'time': torch.tensor(np.array([state.get('time', 0.0)], dtype=np.float32), dtype=torch.float32, device=self.device),
            'yaw': torch.tensor(np.array([state.get('yaw', 0.0)], dtype=np.float32), dtype=torch.float32, device=self.device),
            'pitch': torch.tensor(np.array([state.get('pitch', 0.0)], dtype=np.float32), dtype=torch.float32, device=self.device),
            'place_table_safe': torch.tensor(np.array([state.get('place_table_safe', 0.0)], dtype=np.float32), dtype=torch.float32, device=self.device)
        }
    
    def _batch_states_to_tensor(self, states: list) -> Dict:
        """Convert list of state dicts to batched tensor dict."""
        povs = np.stack([s['pov'] for s in states])
        times = np.array([s.get('time', 0.0) for s in states])
        yaws = np.array([s.get('yaw', 0.0) for s in states])
        pitches = np.array([s.get('pitch', 0.0) for s in states])
        place_table_safes = np.array([s.get('place_table_safe', 0.0) for s in states])
        
        return {
            'pov': torch.tensor(povs, dtype=torch.float32, device=self.device),
            'time': torch.tensor(times, dtype=torch.float32, device=self.device),
            'yaw': torch.tensor(yaws, dtype=torch.float32, device=self.device),
            'pitch': torch.tensor(pitches, dtype=torch.float32, device=self.device),
            'place_table_safe': torch.tensor(place_table_safes, dtype=torch.float32, device=self.device)
        }
    
    def save(self, path: str):
        """Save agent state to file."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'train_count': self.train_count,
            'episode_count': self.episode_count
        }, path)
        print(f"DQN agent saved to {path}")
    
    def load(self, path: str):
        """Load agent state from file with shape handling and hyperparam override."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 1. Load Q Network with Shape Mismatch Handling (Action Space Change)
        saved_state = checkpoint['q_network_state_dict']
        current_state = self.q_network.state_dict()
        new_state = {}
        
        mismatch_detected = False
        for k, v in current_state.items():
            if k in saved_state:
                saved_v = saved_state[k]
                if saved_v.shape != v.shape:
                    mismatch_detected = True
                    print(f"⚠️  Shape mismatch for {k}: saved {saved_v.shape}, current {v.shape}")
                    
                    # Create buffer matching current shape
                    # Copy matching parts, effectively slicing if saved is larger, or padding if smaller
                    min_dim = min(saved_v.shape[0], v.shape[0])
                    
                    # Assume dim 0 is output (actions)
                    if len(v.shape) == 1: 
                        v[:min_dim] = saved_v[:min_dim]
                    elif len(v.shape) >= 2: 
                         v[:min_dim, ...] = saved_v[:min_dim, ...]
                         
                    new_state[k] = v
                else:
                    new_state[k] = saved_v
            else:
                new_state[k] = v 

        self.q_network.load_state_dict(new_state)
        self.target_network.load_state_dict(new_state) 
        
        if mismatch_detected:
            print("Handled action space mismatch by preserving matching weights.")

        # 2. Load Optimizer but FORCE current Learning Rate
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Force update learning rate from current config
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                    
                print(f"Optimizer loaded")
            except Exception as e:
                print(f"Optimizer load failed (likely shape change), resetting optimizer: {e}")
                self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # 3. Restore Counters
        self.step_count = checkpoint['step_count']
        self.train_count = checkpoint['train_count']
        self.episode_count = checkpoint.get('episode_count', 0)

        # 4. Restore replay buffer if it exists
        if 'replay_buffer' in checkpoint:
            print("🔄 Restoring replay buffer from checkpoint...")
            try:
                buffer_data = checkpoint['replay_buffer']

                if hasattr(self.replay_buffer, 'buffer'):
                    # Regular ReplayBuffer
                    self.replay_buffer.buffer.extend(buffer_data)
                    self.replay_buffer.position = len(self.replay_buffer.buffer) % self.replay_buffer.capacity
                    print(f"✅ Restored {len(buffer_data)} experiences to replay buffer")
                elif hasattr(self.replay_buffer, 'add_experience'):
                    # PrioritizedReplayBuffer
                    for exp in buffer_data:
                        self.replay_buffer.add_experience(*exp)
                    print(f"✅ Restored {len(buffer_data)} experiences to prioritized replay buffer")
            except Exception as e:
                print(f"Could not restore buffer (format might differ): {e}")
        else:
            print("ℹ️  No replay buffer in checkpoint")

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
            'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
            'time': float(i) / 20,
            'yaw': 0.0,
            'pitch': 0.0
        }
        action = agent.select_action(state, explore=True)
        next_state = {
            'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
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
        'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
        'time': 0.5,
        'yaw': 0.0,
        'pitch': 0.0
    }
    action_explore = agent.select_action(state, explore=True)
    action_greedy = agent.select_action(state, explore=False)
    print(f"    Action (explore): {action_explore}")
    print(f"    Action (greedy): {action_greedy}")
    
    print("\n✅ DQNAgent test passed!")