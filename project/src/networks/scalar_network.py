"""
Scalar feature processing network for multimodal fusion.

Processes scalar observations (time_left, yaw, pitch, etc.) through
fully connected layers before concatenating with visual features.
This allows the network to learn better scalar representations.
"""

import torch
import torch.nn as nn


class ScalarNetwork(nn.Module):
    """
    2-layer FC network for processing scalar observations.

    This network processes non-visual observations (scalars) through
    fully connected layers before fusion with CNN visual features.

    Architecture:
        Input: (batch, num_scalars)
        FC1 -> ReLU -> FC2 -> ReLU
        Output: (batch, output_dim)
    """

    def __init__(self, num_scalars: int = 4, hidden_dim: int = 64, output_dim: int = 64):
        """
        Args:
            num_scalars: Number of scalar inputs (e.g., 4 for time_left, yaw, pitch, place_table_safe)
            hidden_dim: Hidden layer dimension (default: 64)
            output_dim: Output dimension (default: 64)
        """
        super().__init__()

        self.num_scalars = num_scalars
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Two-layer MLP
        self.network = nn.Sequential(
            nn.Linear(num_scalars, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through scalar network.

        Args:
            scalars: (batch, num_scalars) tensor of scalar observations

        Returns:
            features: (batch, output_dim) tensor of processed scalar features
        """
        return self.network(scalars)

    def get_output_dim(self) -> int:
        """Returns the output feature dimension."""
        return self.output_dim


if __name__ == "__main__":
    print("Testing ScalarNetwork...")

    # Test with 4 scalars (time_left, yaw, pitch, place_table_safe)
    network = ScalarNetwork(num_scalars=4, hidden_dim=64, output_dim=64)

    # Count parameters
    num_params = sum(p.numel() for p in network.parameters())
    print(f"  Parameters: {num_params:,}")

    # Test forward pass
    batch_size = 8
    scalars = torch.randn(batch_size, 4)
    output = network(scalars)

    print(f"  Input shape: {scalars.shape}")
    print(f"  Output shape: {output.shape}")

    assert output.shape == (batch_size, 64), f"Expected ({batch_size}, 64), got {output.shape}"

    # Test different configurations
    print("\n  Testing different configurations:")
    configs = [
        (3, 32, 32),   # 3 scalars, small
        (4, 64, 64),   # 4 scalars, default
        (4, 128, 128), # 4 scalars, large
    ]

    for num_scalars, hidden, output in configs:
        net = ScalarNetwork(num_scalars=num_scalars, hidden_dim=hidden, output_dim=output)
        params = sum(p.numel() for p in net.parameters())
        test_input = torch.randn(2, num_scalars)
        test_output = net(test_input)
        print(f"    ({num_scalars}→{hidden}→{output}): {params:,} params, output shape {test_output.shape}")
        assert test_output.shape == (2, output)

    print("\n✅ ScalarNetwork test passed!")
