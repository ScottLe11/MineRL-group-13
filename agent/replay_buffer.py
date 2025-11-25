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
            ValueError: If buffer doesn't have enough experiences
        """
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


if __name__ == "__main__":
    # Quick test
    print("Testing ReplayBuffer...")
    
    buffer = ReplayBuffer(capacity=100, min_size=10)
    
    # Add some experiences
    for i in range(20):
        state = {
            'pov': np.random.randint(0, 256, (4, 64, 64), dtype=np.uint8),
            'time': float(i) / 20,
            'yaw': 0.0,
            'pitch': 0.0
        }
        next_state = {
            'pov': np.random.randint(0, 256, (4, 64, 64), dtype=np.uint8),
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

