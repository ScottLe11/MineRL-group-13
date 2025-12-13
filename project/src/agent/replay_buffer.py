"""
Experience replay buffer for DQN training.
"""

import random
import numpy as np
from collections import deque
from typing import Tuple, List, Dict, Any


class ReplayBuffer:
    """
    Simple uniform replay buffer for storing and sampling experiences.
    
    Each experience is a tuple: (state, action, reward, next_state, done)
    Where state/next_state are observation dicts with 'pov', 'time', 'yaw', 'pitch'.
    """
    
    def __init__(self, capacity: int = 100000, min_size: int = 1000):
        """
        Args:
            capacity: Maximum number of experiences to store
            min_size: Minimum experiences required before sampling is allowed
        """
        self.capacity = capacity
        self.min_size = min_size
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current observation dict
            action: Action taken (0-22)
            reward: Reward received
            next_state: Next observation dict
            done: Whether episode ended
        """
        # Store copies to avoid reference issues
        experience = (
            self._copy_state(state),
            action,
            reward,
            self._copy_state(next_state),
            done
        )
        self.buffer.append(experience)
    
    def _copy_state(self, state: Dict) -> Dict:
        """Create a copy of the state dict with numpy arrays."""
        if state is None:
            return None
        
        copied = {}
        for key, value in state.items():
            if hasattr(value, 'copy'):
                copied[key] = value.copy()
            elif hasattr(value, 'numpy'):
                copied[key] = value.numpy().copy()
            else:
                copied[key] = value
        return copied
    
    def sample(self, batch_size: int) -> Tuple[List, List, List, List, List]:
        """
        Sample a batch of experiences uniformly.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            Each is a list of batch_size items
        
        Raises:
            ValueError: If buffer is not ready (hasn't reached min_size)
        """
        if not self.is_ready():
            raise ValueError(f"Replay buffer not ready: {len(self.buffer)} < min_size {self.min_size}")
        
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer size {len(self.buffer)} < batch_size {batch_size}")
        
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return list(states), list(actions), list(rewards), list(next_states), list(dones)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough experiences for training."""
        return len(self.buffer) >= self.min_size
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear all experiences from the buffer."""
        self.buffer.clear()


class SumTree:
    """
    Binary sum tree for efficient priority-based sampling.
    
    Each leaf stores a priority value, and internal nodes store sums.
    Allows O(log n) sampling and priority updates.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree stored in array
        self.data = [None] * capacity  # Circular buffer for experiences
        self.write_ptr = 0  # Where to write next
        self.n_entries = 0  # Current number of entries
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find the leaf index for a given cumulative sum s."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Return total priority (root of tree)."""
        return self.tree[0]
    
    def add(self, priority: float, data):
        """Add experience with given priority."""
        tree_idx = self.write_ptr + self.capacity - 1
        
        self.data[self.write_ptr] = data
        self.update(tree_idx, priority)
        
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, tree_idx: int, priority: float):
        """Update priority at tree index."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        Get experience for cumulative sum s.
        
        Returns:
            (tree_index, priority, data)
        """
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], self.data[data_idx]
    
    def __len__(self) -> int:
        return self.n_entries


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using proportional prioritization.
    
    Experiences with higher TD-error are sampled more frequently.
    Uses importance sampling weights to correct for bias.
    
    Reference: Schaul et al., "Prioritized Experience Replay" (2015)
    """
    
    def __init__(
        self, 
        capacity: int = 100000, 
        min_size: int = 1000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 100000,
        epsilon: float = 1e-6
    ):
        """
        Args:
            capacity: Maximum buffer size
            min_size: Minimum experiences before sampling allowed
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling exponent
            beta_end: Final importance sampling exponent
            beta_anneal_steps: Steps to anneal beta from start to end
            epsilon: Small constant added to priorities to avoid zero probability
        """
        self.capacity = capacity
        self.min_size = min_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0  # Track max for new experiences
        self.step_count = 0
        self.beta = beta_start  # Current beta value for importance sampling
    
    def _get_beta(self) -> float:
        """Get current beta value based on annealing schedule."""
        progress = min(1.0, self.step_count / self.beta_anneal_steps)
        return self.beta_start + (self.beta_end - self.beta_start) * progress
    
    def add(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool, priority: float = None):
        """
        Add experience to buffer.
        
        Args:
            state: Current observation dict
            action: Action taken
            reward: Reward received
            next_state: Next observation dict
            done: Whether episode ended
            priority: Optional explicit priority (uses max_priority if None)
        """
        if priority is None:
            priority = self.max_priority
        
        experience = (
            self._copy_state(state),
            action,
            reward,
            self._copy_state(next_state),
            done
        )
        
        # Priority^alpha for proportional prioritization
        self.tree.add(priority ** self.alpha, experience)
    
    def _copy_state(self, state: Dict) -> Dict:
        """Create a copy of the state dict."""
        if state is None:
            return None
        
        copied = {}
        for key, value in state.items():
            if hasattr(value, 'copy'):
                copied[key] = value.copy()
            elif hasattr(value, 'numpy'):
                copied[key] = value.numpy().copy()
            else:
                copied[key] = value
        return copied
    
    def sample(self, batch_size: int) -> Tuple[List, List, List, List, List, np.ndarray, List[int]]:
        """
        Sample batch with prioritized sampling.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            (states, actions, rewards, next_states, dones, weights, indices)
            - weights: Importance sampling weights for loss correction
            - indices: Tree indices for priority updates
        """
        if not self.is_ready():
            raise ValueError(f"Buffer not ready: {len(self.tree)} < min_size {self.min_size}")
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        indices = []
        priorities = []
        
        # Divide total priority into batch_size segments
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            # Sample uniformly from segment
            low = segment * i
            high = segment * (i + 1)
            s = np.random.uniform(low, high)
            
            tree_idx, priority, experience = self.tree.get(s)
            
            if experience is not None:
                state, action, reward, next_state, done = experience
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                indices.append(tree_idx)
                priorities.append(priority)
        
        self.step_count += 1
        
        # Update beta based on annealing schedule
        self.beta = self._get_beta()
        
        # Compute importance sampling weights
        n = len(self.tree)
        
        # P(i) = priority_i / sum(priorities)
        probs = np.array(priorities) / self.tree.total()
        
        # w_i = (N * P(i))^(-beta)
        weights = (n * probs) ** (-self.beta)
        
        # Normalize weights by max weight for stability
        weights = weights / weights.max()
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Tree indices from sample()
            td_errors: Absolute TD errors for each experience
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (np.abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough experiences."""
        return len(self.tree) >= self.min_size
    
    def __len__(self) -> int:
        return len(self.tree)
    
    def get_all_experiences(self) -> List:
        """
        Get all experiences in the buffer (for checkpointing).
        
        Returns:
            List of experience tuples: (state, action, reward, next_state, done)
        """
        experiences = []
        n = len(self.tree)
        if n == 0:
            return experiences
        
        # Extract all valid experiences from the circular buffer
        for i in range(min(n, self.capacity)):
            experience = self.tree.data[i]
            if experience is not None:
                experiences.append(experience)
        return experiences
    
    def clear(self):
        """Clear all experiences."""
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        self.step_count = 0
        self.beta = self.beta_start


if __name__ == "__main__":
    # Quick test
    print("Testing ReplayBuffer...")
    
    buffer = ReplayBuffer(capacity=100, min_size=10)
    
    # Add some experiences
    for i in range(20):
        state = {
            'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
            'time': float(i) / 20,
            'yaw': 0.0,
            'pitch': 0.0
        }
        next_state = {
            'pov': np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8),
            'time': float(i + 1) / 20,
            'yaw': 0.0,
            'pitch': 0.0
        }
        buffer.add(state, action=i % 23, reward=-0.001, next_state=next_state, done=(i == 19))
    
    print(f"  Buffer size: {len(buffer)}")
    print(f"  Is ready: {buffer.is_ready()}")
    
    # Sample a batch
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=5)
    print(f"  Sampled batch size: {len(states)}")
    print(f"  Sample state pov shape: {states[0]['pov'].shape}")
    print(f"  Sample actions: {actions}")
    print(f"  Sample rewards: {rewards}")
    print(f"  Sample dones: {dones}")
    
    # Verify original buffer not modified
    assert len(buffer) == 20, "Buffer should still have 20 items"
    
    print("✅ ReplayBuffer test passed!")

