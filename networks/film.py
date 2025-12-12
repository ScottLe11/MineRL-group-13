"""
Feature-wise Linear Modulation (FiLM) for conditional neural networks.

FiLM allows scalar features to modulate visual features by learning
affine transformations (gamma, beta) conditioned on the scalar input.
This provides early fusion of multimodal information.

Reference: "FiLM: Visual Reasoning with a General Conditioning Layer"
https://arxiv.org/abs/1709.07871
"""

import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """
    Generates FiLM parameters (gamma, beta) from scalar inputs.

    Takes scalar features and produces channel-wise affine transformation
    parameters to modulate convolutional feature maps.

    Architecture:
        Input: (batch, scalar_dim)
        FC -> ReLU -> FC
        Output: gamma (batch, film_dim), beta (batch, film_dim)
    """

    def __init__(self, scalar_dim: int, film_dim: int, hidden_dim: int = 64):
        """
        Args:
            scalar_dim: Dimension of input scalar features (e.g., 9 for all scalars)
            film_dim: Number of channels to modulate (e.g., 128 for conv3 output)
            hidden_dim: Hidden layer dimension (default: 64)
        """
        super().__init__()

        self.scalar_dim = scalar_dim
        self.film_dim = film_dim

        # Shared backbone for processing scalars
        self.backbone = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim),
            nn.ReLU()
        )

        # Separate heads for gamma and beta
        self.gamma_head = nn.Linear(hidden_dim, film_dim)
        self.beta_head = nn.Linear(hidden_dim, film_dim)

        # Initialize gamma to 1 and beta to 0 (identity transformation initially)
        nn.init.constant_(self.gamma_head.weight, 0)
        nn.init.constant_(self.gamma_head.bias, 1)
        nn.init.constant_(self.beta_head.weight, 0)
        nn.init.constant_(self.beta_head.bias, 0)

    def forward(self, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate FiLM parameters from scalar inputs.

        Args:
            scalars: (batch, scalar_dim) tensor of scalar features

        Returns:
            gamma: (batch, film_dim) scale parameters
            beta: (batch, film_dim) shift parameters
        """
        h = self.backbone(scalars)
        gamma = self.gamma_head(h)  # Scale
        beta = self.beta_head(h)    # Shift
        return gamma, beta


class FiLMLayer(nn.Module):
    """
    Applies FiLM modulation to feature maps.

    Applies channel-wise affine transformation:
        output = gamma * input + beta

    where gamma and beta are generated from scalar features.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation to feature maps.

        Args:
            x: (batch, channels, height, width) feature maps
            gamma: (batch, channels) scale parameters
            beta: (batch, channels) shift parameters

        Returns:
            modulated: (batch, channels, height, width) modulated feature maps
        """
        # Reshape gamma and beta for broadcasting: (batch, channels, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # Apply affine transformation
        return gamma * x + beta


if __name__ == "__main__":
    print("Testing FiLM modules...")

    # Test FiLMGenerator
    batch_size = 8
    scalar_dim = 9
    film_dim = 128

    film_gen = FiLMGenerator(scalar_dim=scalar_dim, film_dim=film_dim)
    scalars = torch.randn(batch_size, scalar_dim)
    gamma, beta = film_gen(scalars)

    print(f"\nFiLMGenerator:")
    print(f"  Input shape: {scalars.shape}")
    print(f"  Gamma shape: {gamma.shape}")
    print(f"  Beta shape: {beta.shape}")

    assert gamma.shape == (batch_size, film_dim)
    assert beta.shape == (batch_size, film_dim)

    # Test FiLMLayer
    film_layer = FiLMLayer()
    feature_maps = torch.randn(batch_size, film_dim, 7, 7)
    modulated = film_layer(feature_maps, gamma, beta)

    print(f"\nFiLMLayer:")
    print(f"  Input shape: {feature_maps.shape}")
    print(f"  Modulated shape: {modulated.shape}")

    assert modulated.shape == feature_maps.shape

    # Test that initial transformation is close to identity
    # (since gamma initialized to 1, beta to 0)
    film_gen_init = FiLMGenerator(scalar_dim=scalar_dim, film_dim=film_dim)
    gamma_init, beta_init = film_gen_init(torch.zeros(1, scalar_dim))

    print(f"\nInitial parameters (should be ~identity):")
    print(f"  Gamma mean: {gamma_init.mean().item():.3f} (expected ~1.0)")
    print(f"  Beta mean: {beta_init.mean().item():.3f} (expected ~0.0)")

    print("\n✅ FiLM test passed!")
