"""
FiLM-CBAM DQN network architecture with dual fusion.

This network combines:
- Medium CNN backbone (conv1-3)
- FiLM for early scalar fusion
- CBAM attention
- Conv4 for additional spatial reasoning
- Scalar network for late fusion
- Multi-layer decision network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .film import FiLMGenerator, FiLMLayer
from .attention import CBAM
from .dueling_head import DuelingHead


class FiLMCBAMNetwork(nn.Module):
    """
    FiLM-CBAM DQN network with both early fusion (FiLM) and late fusion (concatenation).

    Architecture:
        Visual pathway:
            Input: (4, 84, 84)
            Conv1: 32 filters, 8x8, stride 4 -> (32, 20, 20)
            Conv2: 64 filters, 4x4, stride 2 -> (64, 9, 9)
            Conv3: 128 filters, 3x3, stride 1 -> (128, 7, 7)
            FiLM conditioning with scalars
            CBAM attention
            Conv4: 128 filters, 3x3, stride 2 -> (128, 3, 3)
            Flatten -> 1,152 features

        Scalar pathway:
            Input: 9 scalars
            FC1: 9 -> 32, ReLU
            FC2: 32 -> 32, ReLU

        Fusion:
            Concatenate: 1,152 + 32 = 1,184
            FC1: 1,184 -> 512, ReLU
            FC2: 512 -> 128, ReLU

        Dueling head:
            Value stream: 128 -> 128 -> 1
            Advantage stream: 128 -> 128 -> num_actions
            Q = V + (A - mean(A))
    """

    def __init__(
        self,
        num_actions: int = 23,
        input_channels: int = 4,
        num_scalars: int = 9
    ):
        """
        Args:
            num_actions: Number of discrete actions (default: 23)
            input_channels: Number of stacked frames (default: 4)
            num_scalars: Number of scalar observations (default: 9)
        """
        super().__init__()

        self.num_actions = num_actions
        self.num_scalars = num_scalars

        # === VISUAL PATHWAY ===
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

        # === FUSION POINT 1: FiLM ===
        self.film_gen = FiLMGenerator(scalar_dim=num_scalars, film_dim=128, hidden_dim=64)
        self.film_layer = FiLMLayer()

        # === ATTENTION ===
        self.cbam = CBAM(channels=128)

        # === SPATIAL REASONING ===
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)  # 7x7 -> 3x3

        # === SCALAR NETWORK ===
        self.scalar_network = nn.Sequential(
            nn.Linear(num_scalars, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # === DECISION LAYERS (after concatenation) ===
        # 128 × 3 × 3 = 1,152 visual features
        # + 32 scalar features
        # = 1,184 total
        self.fc1 = nn.Linear(1152 + 32, 512)
        self.fc2 = nn.Linear(512, 128)

        # === DUELING HEAD ===
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs: dict, return_attention: bool = False):
        """
        Forward pass through the network.

        Args:
            obs: Dictionary with keys 'pov', 'time_left', 'yaw', 'pitch', 'place_table_safe',
                 'inv_logs', 'inv_planks', 'inv_sticks', 'inv_table', 'inv_axe'
                 OR a tensor for pov only (for simple testing)
            return_attention: If True, return attention maps

        Returns:
            q_values: (batch, num_actions) tensor
            attention_info: dict with attention maps (only if return_attention=True)
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

            # Ensure scalars are (batch,)
            if time_left.dim() == 2:
                time_left = time_left.squeeze(1)
            if yaw.dim() == 2:
                yaw = yaw.squeeze(1)
            if pitch.dim() == 2:
                pitch = pitch.squeeze(1)
            if place_table_safe.dim() == 2:
                place_table_safe = place_table_safe.squeeze(1)
            if inv_logs.dim() == 2:
                inv_logs = inv_logs.squeeze(1)
            if inv_planks.dim() == 2:
                inv_planks = inv_planks.squeeze(1)
            if inv_sticks.dim() == 2:
                inv_sticks = inv_sticks.squeeze(1)
            if inv_table.dim() == 2:
                inv_table = inv_table.squeeze(1)
            if inv_axe.dim() == 2:
                inv_axe = inv_axe.squeeze(1)

            # Stack into single tensor
            scalars = torch.stack([
                time_left, yaw, pitch, place_table_safe,
                inv_logs, inv_planks, inv_sticks, inv_table, inv_axe
            ], dim=1)  # (batch, 9)

        # Normalize visual input
        if pov.max() > 1.0:
            pov = pov.float() / 255.0

        # === VISUAL PROCESSING ===
        x = F.relu(self.conv1(pov))    # (batch, 32, 20, 20)
        x = F.relu(self.conv2(x))      # (batch, 64, 9, 9)
        x = F.relu(self.conv3(x))      # (batch, 128, 7, 7)

        # === FiLM: Early fusion with scalars ===
        gamma, beta = self.film_gen(scalars)
        x = self.film_layer(x, gamma, beta)  # Context-aware features!

        # === ATTENTION on context-aware features ===
        x, attention_info = self.cbam(x)  # (batch, 128, 7, 7)

        # === ADDITIONAL SPATIAL PROCESSING ===
        x = F.relu(self.conv4(x))  # (batch, 128, 3, 3) - spatial reasoning!

        # === FLATTEN VISUAL FEATURES ===
        visual_features = x.view(x.size(0), -1)  # (batch, 1152)

        # === PROCESS SCALARS (late fusion pathway) ===
        scalar_features = self.scalar_network(scalars)  # (batch, 32)

        # === CONCATENATE VISUAL AND SCALAR FEATURES ===
        combined = torch.cat([visual_features, scalar_features], dim=1)  # (batch, 1184)

        # === DECISION LAYERS ===
        x = F.relu(self.fc1(combined))  # (batch, 512)
        x = F.relu(self.fc2(x))         # (batch, 128)

        # === DUELING HEAD ===
        value = self.value_stream(x)        # (batch, 1)
        advantage = self.advantage_stream(x)  # (batch, num_actions)

        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        if return_attention:
            return q_values, attention_info
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
                obs_batched = {}
                for k, v in obs.items():
                    if isinstance(v, torch.Tensor):
                        if v.dim() == 0:
                            obs_batched[k] = v.unsqueeze(0)
                        elif v.dim() == 3 and k == 'pov':
                            obs_batched[k] = v.unsqueeze(0)
                        else:
                            obs_batched[k] = v
                    else:
                        obs_batched[k] = torch.tensor([v])
            else:
                obs_batched = obs.unsqueeze(0)

            q_values = self.forward(obs_batched)
            return q_values.argmax(dim=1).item()


if __name__ == "__main__":
    print("Testing FiLMCBAMNetwork...")

    network = FiLMCBAMNetwork(num_actions=23, input_channels=4, num_scalars=9)

    # Count parameters
    num_params = sum(p.numel() for p in network.parameters())
    print(f"  Total parameters: {num_params:,} (~{num_params/1e6:.2f}M)")

    # Test with dict observation
    print("\n  Testing with dict observation...")
    obs = {
        'pov': torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.float32),
        'time_left': torch.tensor([0.8, 0.5]),
        'yaw': torch.tensor([45.0, -30.0]),
        'pitch': torch.tensor([0.0, 10.0]),
        'place_table_safe': torch.tensor([1.0, 0.0]),
        'inv_logs': torch.tensor([5.0, 2.0]),
        'inv_planks': torch.tensor([0.0, 4.0]),
        'inv_sticks': torch.tensor([0.0, 0.0]),
        'inv_table': torch.tensor([0.0, 1.0]),
        'inv_axe': torch.tensor([0.0, 0.0])
    }
    q_values = network(obs)
    print(f"    Input pov shape: {obs['pov'].shape}")
    print(f"    Output Q-values shape: {q_values.shape}")
    assert q_values.shape == (2, 23), f"Expected (2, 23), got {q_values.shape}"

    # Test with return_attention
    print("\n  Testing with return_attention=True...")
    q_values, attention_info = network(obs, return_attention=True)
    print(f"    Q-values shape: {q_values.shape}")
    print(f"    Attention info keys: {attention_info.keys()}")

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
        'pitch': torch.tensor(0.0),
        'place_table_safe': torch.tensor(1.0),
        'inv_logs': torch.tensor(3.0),
        'inv_planks': torch.tensor(0.0),
        'inv_sticks': torch.tensor(0.0),
        'inv_table': torch.tensor(0.0),
        'inv_axe': torch.tensor(0.0)
    }
    action = network.get_action(single_obs, epsilon=0.0)
    print(f"    Selected action (greedy): {action}")
    assert 0 <= action < 23

    print("\n✅ FiLMCBAMNetwork test passed!")
