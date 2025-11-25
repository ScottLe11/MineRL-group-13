"""
CNN feature extractor for visual observations.
"""

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """
    Small CNN architecture (~400K params) similar to Atari DQN.
    
    Input: (batch, input_channels, 84, 84) - stacked grayscale frames
    Output: (batch, 512) - feature vector
    
    Architecture:
        Conv1: 32 filters, 8x8 kernel, stride 4 -> (batch, 32, 20, 20)
        Conv2: 64 filters, 4x4 kernel, stride 2 -> (batch, 64, 9, 9)
        Conv3: 64 filters, 3x3 kernel, stride 1 -> (batch, 64, 7, 7)
        Flatten -> (batch, 3136)
        FC -> (batch, 512)
    """
    
    def __init__(self, input_channels: int = 4):
        """
        Args:
            input_channels: Number of stacked frames (default: 4 for grayscale)
        """
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # For 84x84 input:
        # Conv1: (84-8)/4 + 1 = 20 -> (32, 20, 20)
        # Conv2: (20-4)/2 + 1 = 9  -> (64, 9, 9)
        # Conv3: (9-3)/1 + 1 = 7   -> (64, 7, 7)
        # Flatten: 64 * 7 * 7 = 3136
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_channels, 84, 84) tensor, values in [0, 255] or [0, 1]
        
        Returns:
            features: (batch, 512) tensor
        """
        # Normalize to [0, 1] if needed
        if x.max() > 1.0:
            x = x.float() / 255.0
        
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
    
    def get_output_dim(self) -> int:
        """Returns the output feature dimension."""
        return 512


if __name__ == "__main__":
    # Quick test
    print("Testing SmallCNN...")
    
    model = SmallCNN(input_channels=4)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} (~{num_params/1e6:.2f}M)")
    
    # Test forward pass with 84x84 input
    batch = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.float32)
    output = model(batch)
    print(f"  Input shape: {batch.shape}")
    print(f"  Output shape: {output.shape}")
    
    assert output.shape == (2, 512), f"Expected (2, 512), got {output.shape}"
    print("✅ SmallCNN test passed!")

