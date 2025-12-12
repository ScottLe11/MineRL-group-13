"""
Complete DQN network combining CNN feature extractor and Dueling head.
"""

import torch
import torch.nn as nn

from .cnn import create_cnn
from .attention import create_attention
from .dueling_head import DuelingHead
from .scalar_network import ScalarNetwork


class DQNNetwork(nn.Module):
    """
    Complete DQN network for MineRL tree-chopping task.

    Architecture:
        Visual Input (4, 84, 84) -> CNN -> [Attention] -> (cnn_dim,)
        Scalar Input (9,) -> [ScalarNetwork] -> (scalar_dim,)
        Combined -> [FusionLayer] -> DuelingHead -> Q-values (num_actions,)

    Observation format expected:
        {
            'pov': (batch, 4, 84, 84) uint8 or float,
            'time_left': (batch,) or (batch, 1) float,
            'yaw': (batch,) or (batch, 1) float,
            'pitch': (batch,) or (batch, 1) float,
            'place_table_safe': (batch,) or (batch, 1) float,
            'inv_logs': (batch,) or (batch, 1) float,
            'inv_planks': (batch,) or (batch, 1) float,
            'inv_sticks': (batch,) or (batch, 1) float,
            'inv_table': (batch,) or (batch, 1) float,
            'inv_axe': (batch,) or (batch, 1) float
        }
    """

    def __init__(
        self,
        num_actions: int = 23,
        input_channels: int = 4,
        num_scalars: int = 9,
        cnn_architecture: str = 'small',
        attention_type: str = 'none',
        use_scalar_network: bool = False,
        scalar_hidden_dim: int = 64,
        scalar_output_dim: int = 64,
        use_fusion_layer: bool = False,
        fusion_hidden_dim: int = 128
    ):
        """
        Args:
            num_actions: Number of discrete actions (default: 23)
            input_channels: Number of stacked frames (default: 4)
            num_scalars: Number of scalar observations (default: 9 - time_left, yaw, pitch, place_table_safe, inv_logs, inv_planks, inv_sticks, inv_table, inv_axe)
            cnn_architecture: CNN architecture ('tiny', 'small', 'medium', 'wide', 'deep')
            attention_type: Attention mechanism ('none', 'spatial', 'cbam', 'treechop_bias')
            use_scalar_network: Whether to process scalars through 2-layer FC network (default: False)
            scalar_hidden_dim: Hidden dimension for scalar network (default: 64)
            scalar_output_dim: Output dimension for scalar network (default: 64)
            use_fusion_layer: Whether to add FC layer after concatenation for feature fusion (default: False)
            fusion_hidden_dim: Hidden dimension for fusion layer (default: 128)
        """
        super().__init__()

        self.num_actions = num_actions
        self.num_scalars = num_scalars
        self.cnn_architecture = cnn_architecture
        self.attention_type = attention_type
        self.use_scalar_network = use_scalar_network

        # CNN for visual features (configurable architecture)
        self.cnn = create_cnn(cnn_architecture, input_channels=input_channels)
        cnn_output_dim = self.cnn.get_output_dim()  # Architecture-dependent (256 or 512)

        # Optional attention mechanism
        # Note: Attention is applied to conv features, not final FC output
        # We'll apply it in forward() if needed
        self.use_attention = (attention_type != 'none')
        if self.use_attention:
            # For spatial/treechop_bias, we need the conv output shape (channels, H, W)
            # This requires accessing cnn internals
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

        # Combined feature dimension (before fusion)
        combined_dim = cnn_output_dim + scalar_dim

        # Optional fusion layer for learning interactions between visual and scalar features
        self.use_fusion_layer = use_fusion_layer
        if use_fusion_layer:
            self.fusion_layer = nn.Sequential(
                nn.Linear(combined_dim, fusion_hidden_dim),
                nn.ReLU()
            )
            final_dim = fusion_hidden_dim
        else:
            self.fusion_layer = None
            final_dim = combined_dim

        # Dueling head
        self.head = DuelingHead(input_dim=final_dim, num_actions=num_actions)
    
    def forward(self, obs: dict, return_attention: bool = False):
        """
        Forward pass through the network.

        Args:
            obs: Dictionary with keys 'pov', 'time_left', 'yaw', 'pitch', 'place_table_safe',
                 'inv_logs', 'inv_planks', 'inv_sticks', 'inv_table', 'inv_axe'
                 OR a tensor for pov only (for simple testing)
            return_attention: If True, return attention maps (only if attention is enabled)

        Returns:
            q_values: (batch, num_actions) tensor
            attention_map: (batch, 1, H, W) tensor (only if return_attention=True and attention enabled)
        """
        if isinstance(obs, torch.Tensor):
            # Simple case: just pov tensor, no scalars
            pov = obs
            batch_size = pov.size(0)
            scalars = torch.zeros(batch_size, self.num_scalars, device=pov.device)
        else:
            # Full observation dict
            pov = obs['pov']
            batch_size = pov.size(0)
            device = pov.device

            # Handle scalar dimensions
            time_left = obs.get('time_left', torch.zeros(batch_size, device=device))
            yaw = obs.get('yaw', torch.zeros(batch_size, device=device))
            pitch = obs.get('pitch', torch.zeros(batch_size, device=device))
            place_table_safe = obs.get('place_table_safe', torch.zeros(batch_size, device=device))
            inv_logs = obs.get('inv_logs', torch.zeros(batch_size, device=device))
            inv_planks = obs.get('inv_planks', torch.zeros(batch_size, device=device))
            inv_sticks = obs.get('inv_sticks', torch.zeros(batch_size, device=device))
            inv_table = obs.get('inv_table', torch.zeros(batch_size, device=device))
            inv_axe = obs.get('inv_axe', torch.zeros(batch_size, device=device))

            # Ensure scalars are (batch, 1)
            if time_left.dim() == 1:
                time_left = time_left.unsqueeze(1)
            if yaw.dim() == 1:
                yaw = yaw.unsqueeze(1)
            if pitch.dim() == 1:
                pitch = pitch.unsqueeze(1)
            if place_table_safe.dim() == 1:
                place_table_safe = place_table_safe.unsqueeze(1)
            if inv_logs.dim() == 1:
                inv_logs = inv_logs.unsqueeze(1)
            if inv_planks.dim() == 1:
                inv_planks = inv_planks.unsqueeze(1)
            if inv_sticks.dim() == 1:
                inv_sticks = inv_sticks.unsqueeze(1)
            if inv_table.dim() == 1:
                inv_table = inv_table.unsqueeze(1)
            if inv_axe.dim() == 1:
                inv_axe = inv_axe.unsqueeze(1)

            scalars = torch.cat([time_left, yaw, pitch, place_table_safe, inv_logs, inv_planks, inv_sticks, inv_table, inv_axe], dim=1)  # (batch, 9)

        # Extract visual features
        # If using attention, apply it to conv features before FC layer
        attention_map = None
        if self.use_attention and hasattr(self.cnn, 'conv') and hasattr(self.cnn, 'fc'):
            # Normalize input
            if pov.max() > 1.0:
                pov = pov.float() / 255.0

            # Conv features
            conv_features = self.cnn.conv(pov)  # (batch, channels, H, W)

            # Apply attention
            attended_features, attention_map = self.attention(conv_features)  # (batch, channels, H, W), (batch, 1, H, W)

            # Flatten and pass through FC
            attended_features = attended_features.view(attended_features.size(0), -1)
            visual_features = self.cnn.fc(attended_features)  # (batch, cnn_output_dim)
        else:
            # Standard forward pass (no attention)
            visual_features = self.cnn(pov)  # (batch, cnn_output_dim)

        # Process scalars (optionally through scalar network)
        if self.scalar_network is not None:
            scalar_features = self.scalar_network(scalars)  # (batch, scalar_output_dim)
        else:
            scalar_features = scalars  # (batch, num_scalars)

        # Concatenate visual and scalar features
        combined = torch.cat([visual_features, scalar_features], dim=1)  # (batch, cnn_dim + scalar_dim)

        # Optional fusion layer for feature interaction
        if self.fusion_layer is not None:
            combined = self.fusion_layer(combined)  # (batch, fusion_hidden_dim)

        # Compute Q-values
        q_values = self.head(combined)  # (batch, num_actions)

        if return_attention and attention_map is not None:
            return q_values, attention_map
        else:
            return q_values
    
    def get_action(self, obs: dict, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            obs: Single observation (will be batched internally)
            epsilon: Exploration rate
        
        Returns:
            action: Selected action index
        """
        import random
        
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        with torch.no_grad():
            # Add batch dimension if needed
            if isinstance(obs, dict):
                obs_batched = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else torch.tensor([v])
                              for k, v in obs.items()}
            else:
                obs_batched = obs.unsqueeze(0)
            
            q_values = self.forward(obs_batched)
            return q_values.argmax(dim=1).item()


if __name__ == "__main__":
    # Quick test
    print("Testing DQNNetwork...")

    network = DQNNetwork(num_actions=23, input_channels=4, num_scalars=9)
    
    # Count parameters
    num_params = sum(p.numel() for p in network.parameters())
    print(f"  Total parameters: {num_params:,} (~{num_params/1e6:.2f}M)")
    
    # Test with dict observation
    print("\n  Testing with dict observation...")
    obs = {
        'pov': torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.float32),
        'time_left': torch.tensor([0.8, 0.5]),
        'yaw': torch.tensor([45.0, -30.0]),
        'pitch': torch.tensor([0.0, 10.0])
    }
    q_values = network(obs)
    print(f"    Input pov shape: {obs['pov'].shape}")
    print(f"    Output Q-values shape: {q_values.shape}")
    assert q_values.shape == (2, 23), f"Expected (2, 23), got {q_values.shape}"
    
    # Test with tensor only
    print("\n  Testing with tensor only...")
    pov_only = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.float32)
    q_values = network(pov_only)
    print(f"    Input shape: {pov_only.shape}")
    print(f"    Output shape: {q_values.shape}")
    assert q_values.shape == (2, 23)
    
    # Test action selection
    print("\n  Testing action selection...")
    single_obs = {
        'pov': torch.randint(0, 256, (4, 84, 84), dtype=torch.float32),
        'time_left': torch.tensor(0.5),
        'yaw': torch.tensor(0.0),
        'pitch': torch.tensor(0.0)
    }
    action = network.get_action(single_obs, epsilon=0.0)
    print(f"    Selected action (greedy): {action}")
    assert 0 <= action < 23
    
    print("\n✅ DQNNetwork test passed!")

