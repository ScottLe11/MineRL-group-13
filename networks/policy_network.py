"""
Policy Network for PPO (and other policy gradient methods).

Outputs both action probabilities (policy) and state value (critic).
Uses shared CNN backbone with separate heads for actor and critic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .cnn import SmallCNN


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    
    Architecture:
    - Shared CNN backbone for visual features
    - Scalar features concatenated after CNN
    - Separate heads for policy (actor) and value (critic)
    
    This is more parameter-efficient than separate networks
    and allows feature sharing between actor and critic.
    """
    
    def __init__(
        self, 
        num_actions: int = 23,
        input_channels: int = 4,
        num_scalars: int = 3,
        hidden_size: int = 512
    ):
        """
        Args:
            num_actions: Number of discrete actions
            input_channels: Number of stacked frames (default 4)
            num_scalars: Number of scalar observations (time, yaw, pitch)
            hidden_size: Size of hidden layers
        """
        super().__init__()
        
        self.num_actions = num_actions
        
        # Shared CNN backbone
        self.cnn = SmallCNN(input_channels=input_channels)
        
        # Feature dimension after CNN + scalars
        self.feature_dim = 512 + num_scalars  # CNN outputs 512
        
        # Actor head (policy): outputs action logits
        self.actor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        
        # Critic head (value): outputs state value
        self.critic = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize actor and critic heads with orthogonal initialization."""
        for module in [self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, obs: dict) -> tuple:
        """
        Forward pass returning policy logits and value.
        
        Args:
            obs: Dict with 'pov', 'time', 'yaw', 'pitch' tensors
            
        Returns:
            (action_logits, value): Tuple of policy logits and state value
        """
        # Process visual observation
        pov = obs['pov'].float() / 255.0  # Normalize to [0, 1]
        cnn_features = self.cnn(pov)  # (batch, 512)
        
        # Concatenate scalar features
        scalars = torch.stack([
            obs['time'].view(-1),
            obs['yaw'].view(-1),
            obs['pitch'].view(-1)
        ], dim=1)  # (batch, 3)
        
        features = torch.cat([cnn_features, scalars], dim=1)  # (batch, 515)
        
        # Actor and critic outputs
        action_logits = self.actor(features)  # (batch, num_actions)
        value = self.critic(features).squeeze(-1)  # (batch,)
        
        return action_logits, value
    
    def get_action_and_value(self, obs: dict, action: torch.Tensor = None):
        """
        Get action, log probability, entropy, and value for PPO.
        
        Args:
            obs: Observation dict
            action: Optional action to evaluate (for training). If None, samples action.
            
        Returns:
            (action, log_prob, entropy, value): Tuple for PPO training
        """
        action_logits, value = self.forward(obs)
        
        # Create categorical distribution
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value
    
    def get_value(self, obs: dict) -> torch.Tensor:
        """Get only the value estimate (for GAE computation)."""
        _, value = self.forward(obs)
        return value


if __name__ == "__main__":
    print("✅ ActorCriticNetwork Test")
    
    network = ActorCriticNetwork(num_actions=23)
    print(f"  Network created with {sum(p.numel() for p in network.parameters()):,} parameters")
    
    # Test forward pass
    obs = {
        'pov': torch.randint(0, 256, (2, 4, 64, 64), dtype=torch.uint8),
        'time': torch.tensor([0.8, 0.5]),
        'yaw': torch.tensor([0.0, 0.5]),
        'pitch': torch.tensor([0.0, -0.2]),
    }
    
    logits, value = network(obs)
    print(f"\n  Forward pass:")
    print(f"    Logits shape: {logits.shape} (expected: [2, 23])")
    print(f"    Value shape: {value.shape} (expected: [2])")
    
    # Test get_action_and_value
    action, log_prob, entropy, value = network.get_action_and_value(obs)
    print(f"\n  get_action_and_value:")
    print(f"    Action: {action.tolist()}")
    print(f"    Log prob: {log_prob.tolist()}")
    print(f"    Entropy: {entropy.tolist()}")
    print(f"    Value: {value.tolist()}")
    
    # Test with provided action
    fixed_action = torch.tensor([0, 5])
    _, log_prob2, _, _ = network.get_action_and_value(obs, action=fixed_action)
    print(f"\n  With fixed action [0, 5]:")
    print(f"    Log prob: {log_prob2.tolist()}")
    
    print("\n✅ ActorCriticNetwork validated!")

