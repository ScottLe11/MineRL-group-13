"""
Policy Network for PPO (and other policy gradient methods).

Outputs both action probabilities (policy) and state value (critic).
Uses shared CNN backbone with separate heads for actor and critic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .cnn import create_cnn
from .attention import create_attention
from .scalar_network import ScalarNetwork


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network for PPO.

    Architecture:
    - Shared CNN backbone for visual features (configurable)
    - Optional attention mechanism
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
        hidden_size: int = 512,
        cnn_architecture: str = 'small',
        attention_type: str = 'none',
        use_scalar_network: bool = False,
        scalar_hidden_dim: int = 64,
        scalar_output_dim: int = 64
    ):
        """
        Args:
            num_actions: Number of discrete actions
            input_channels: Number of stacked frames (default 4)
            num_scalars: Number of scalar observations (time_left, yaw, pitch)
            hidden_size: Size of hidden layers
            cnn_architecture: CNN architecture ('tiny', 'small', 'medium', 'wide', 'deep')
            attention_type: Attention mechanism ('none', 'spatial', 'cbam', 'treechop_bias')
            use_scalar_network: Whether to process scalars through 2-layer FC network (default: False)
            scalar_hidden_dim: Hidden dimension for scalar network (default: 64)
            scalar_output_dim: Output dimension for scalar network (default: 64)
        """
        super().__init__()

        self.num_actions = num_actions
        self.num_scalars = num_scalars
        self.cnn_architecture = cnn_architecture
        self.attention_type = attention_type
        self.use_scalar_network = use_scalar_network

        # Shared CNN backbone (configurable architecture!)
        self.cnn = create_cnn(cnn_architecture, input_channels=input_channels)
        cnn_output_dim = self.cnn.get_output_dim()  # Architecture-dependent (256 or 512)

        # Optional attention mechanism (same as DQN)
        self.use_attention = (attention_type != 'none')
        if self.use_attention:
            if hasattr(self.cnn, 'conv'):
                # Get number of channels from last conv layer
                last_conv = None
                for module in reversed(list(self.cnn.conv.modules())):
                    if isinstance(module, nn.Conv2d):
                        last_conv = module
                        break
                if last_conv is not None:
                    conv_channels = last_conv.out_channels
                    if attention_type in ['cbam', 'channel']:
                        self.attention = create_attention(attention_type, channels=conv_channels)
                    elif attention_type == 'spatial':
                        self.attention = create_attention(attention_type, kernel_size=7)
                    elif attention_type == 'treechop_bias':
                        self.attention = create_attention(attention_type, height=7, width=7)
                    else:
                        self.attention = nn.Identity()
                        self.use_attention = False
                else:
                    print(f"Warning: Could not find conv layer for attention, disabling attention")
                    self.attention = nn.Identity()
                    self.use_attention = False
            else:
                print(f"Warning: CNN architecture {cnn_architecture} doesn't support attention, disabling")
                self.attention = nn.Identity()
                self.use_attention = False
        else:
            self.attention = nn.Identity()

        # Optional scalar network for processing non-visual features
        if use_scalar_network:
            self.scalar_network = ScalarNetwork(
                num_scalars=num_scalars,
                hidden_dim=scalar_hidden_dim,
                output_dim=scalar_output_dim
            )
            scalar_dim = scalar_output_dim
        else:
            self.scalar_network = None
            scalar_dim = num_scalars

        # Feature dimension after CNN + scalars
        self.feature_dim = cnn_output_dim + scalar_dim

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
            obs: Dict with 'pov', 'time_left', 'yaw', 'pitch' tensors

        Returns:
            (action_logits, value): Tuple of policy logits and state value
        """
        # Process visual observation
        pov = obs['pov']
        if pov.max() > 1.0:
            pov = pov.float() / 255.0  # Normalize to [0, 1]

        # Extract visual features (with attention if enabled)
        if self.use_attention and hasattr(self.cnn, 'conv') and hasattr(self.cnn, 'fc'):
            # Conv features
            conv_features = self.cnn.conv(pov)  # (batch, channels, H, W)

            # Apply attention
            attended_features, _ = self.attention(conv_features)  # (batch, channels, H, W)

            # Flatten and FC layer
            flattened = attended_features.view(attended_features.size(0), -1)
            cnn_features = self.cnn.fc(flattened)  # (batch, cnn_output_dim)
        else:
            # No attention, use CNN directly
            cnn_features = self.cnn(pov)  # (batch, cnn_output_dim)

        # Concatenate scalar features
        scalars = torch.stack([
            obs['time_left'].view(-1),
            obs['yaw'].view(-1),
            obs['pitch'].view(-1),
            obs['place_table_safe'].view(-1)
        ], dim=1)  # (batch, num_scalars)

        # Process scalars (optionally through scalar network)
        if self.scalar_network is not None:
            scalar_features = self.scalar_network(scalars)  # (batch, scalar_output_dim)
        else:
            scalar_features = scalars  # (batch, num_scalars)

        # Concatenate visual and scalar features
        features = torch.cat([cnn_features, scalar_features], dim=1)  # (batch, feature_dim)

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
        'pov': torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8),
        'time_left': torch.tensor([0.8, 0.5]),
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

