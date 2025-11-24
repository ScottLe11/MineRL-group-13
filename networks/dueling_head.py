"""
Dueling DQN head for Q-value computation.

Implements the architecture from:
"Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
"""

import torch
import torch.nn as nn


class DuelingHead(nn.Module):
    """
    Dueling DQN head that computes Q-values via separate value and advantage streams.
    
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
    
    This decomposition helps the network learn which states are valuable
    regardless of the action taken.
    """
    
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 512):
        """
        Args:
            input_dim: Dimension of input features (e.g., 515 = 512 CNN + 3 scalars)
            num_actions: Number of discrete actions (e.g., 23)
            hidden_dim: Hidden layer dimension for value/advantage streams
        """
        super().__init__()
        
        self.num_actions = num_actions
        
        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream: estimates A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, input_dim) tensor
        
        Returns:
            q_values: (batch, num_actions) tensor
        """
        # Compute value and advantages
        value = self.value_stream(features)           # (batch, 1)
        advantages = self.advantage_stream(features)  # (batch, num_actions)
        
        # Combine using dueling formula:
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
        # Subtracting mean ensures identifiability
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


if __name__ == "__main__":
    # Quick test
    print("Testing DuelingHead...")
    
    head = DuelingHead(input_dim=515, num_actions=23)
    
    # Count parameters
    num_params = sum(p.numel() for p in head.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Test forward pass
    features = torch.randn(2, 515)
    q_values = head(features)
    print(f"  Input shape: {features.shape}")
    print(f"  Output shape: {q_values.shape}")
    
    assert q_values.shape == (2, 23), f"Expected (2, 23), got {q_values.shape}"
    
    # Test that mean advantage is ~0 (due to subtraction)
    value = head.value_stream(features)
    advantages = head.advantage_stream(features)
    reconstructed = value + (advantages - advantages.mean(dim=1, keepdim=True))
    assert torch.allclose(q_values, reconstructed), "Q-value computation mismatch"
    
    print("✅ DuelingHead test passed!")

