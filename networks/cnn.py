"""
CNN feature extractor architectures for visual observations.

Available architectures:
- TinyCNN:   ~150K params - Fastest, for quick experiments
- SmallCNN:  ~400K params - Default, similar to Atari DQN
- MediumCNN: ~600K params - More capacity
- WideCNN:   ~1M params   - More filters per layer
- DeepCNN:   ~500K params - More layers, deeper features
"""

import torch
import torch.nn as nn
from typing import Tuple


def _calculate_conv_output_size(input_size: int, kernel_size: int, stride: int, padding: int = 0) -> int:
    """Calculate output size of a conv layer."""
    return (input_size - kernel_size + 2 * padding) // stride + 1


def _initialize_weights(module: nn.Module):
    """Initialize weights using He initialization for ReLU."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class TinyCNN(nn.Module):
    """
    Tiny CNN architecture (~150K params) for quick experiments.
    
    Input: (batch, input_channels, 84, 84)
    Output: (batch, 256)
    
    Architecture:
        Conv1: 16 filters, 8x8, stride 4 -> (16, 20, 20)
        Conv2: 32 filters, 4x4, stride 2 -> (32, 9, 9)
        Flatten -> (2592)
        FC -> (256)
    """
    
    def __init__(self, input_channels: int = 4):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        
        # 84 -> 20 -> 9, 32 * 9 * 9 = 2592
        self.fc = nn.Sequential(
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU()
        )
        
        self._output_dim = 256
        _initialize_weights(self)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.max() > 1.0:
            x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def get_output_dim(self) -> int:
        return self._output_dim


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
        
        self._output_dim = 512
        _initialize_weights(self)
    
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
        return self._output_dim


class MediumCNN(nn.Module):
    """
    Medium CNN architecture (~600K params) with more capacity.
    
    Input: (batch, input_channels, 84, 84)
    Output: (batch, 512)
    
    Architecture:
        Conv1: 32 filters, 8x8, stride 4 -> (32, 20, 20)
        Conv2: 64 filters, 4x4, stride 2 -> (64, 9, 9)
        Conv3: 128 filters, 3x3, stride 1 -> (128, 7, 7)
        Flatten -> (6272)
        FC1 -> (512)
    """
    
    def __init__(self, input_channels: int = 4):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 128 * 7 * 7 = 6272
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU()
        )
        
        self._output_dim = 512
        _initialize_weights(self)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.max() > 1.0:
            x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def get_output_dim(self) -> int:
        return self._output_dim


class WideCNN(nn.Module):
    """
    Wide CNN architecture (~1M params) with more filters per layer.
    
    Input: (batch, input_channels, 84, 84)
    Output: (batch, 512)
    
    Architecture:
        Conv1: 64 filters, 8x8, stride 4 -> (64, 20, 20)
        Conv2: 128 filters, 4x4, stride 2 -> (128, 9, 9)
        Conv3: 128 filters, 3x3, stride 1 -> (128, 7, 7)
        Flatten -> (6272)
        FC -> (512)
    """
    
    def __init__(self, input_channels: int = 4):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 128 * 7 * 7 = 6272
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU()
        )
        
        self._output_dim = 512
        _initialize_weights(self)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.max() > 1.0:
            x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def get_output_dim(self) -> int:
        return self._output_dim


class DeepCNN(nn.Module):
    """
    Deep CNN architecture (~500K params) with more layers.
    
    Input: (batch, input_channels, 84, 84)
    Output: (batch, 512)
    
    Architecture:
        Conv1: 32 filters, 5x5, stride 2 -> (32, 40, 40)
        Conv2: 64 filters, 3x3, stride 2 -> (64, 19, 19)
        Conv3: 64 filters, 3x3, stride 2 -> (64, 9, 9)
        Conv4: 64 filters, 3x3, stride 1 -> (64, 7, 7)
        Flatten -> (3136)
        FC -> (512)
    """
    
    def __init__(self, input_channels: int = 4):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 84 -> 40 -> 19 -> 9 -> 7
        # 64 * 7 * 7 = 3136
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU()
        )
        
        self._output_dim = 512
        _initialize_weights(self)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.max() > 1.0:
            x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def get_output_dim(self) -> int:
        return self._output_dim


# Registry of available architectures
CNN_ARCHITECTURES = {
    'tiny': TinyCNN,
    'small': SmallCNN,
    'medium': MediumCNN,
    'wide': WideCNN,
    'deep': DeepCNN,
}


def create_cnn(arch_name: str, input_channels: int = 4) -> nn.Module:
    """
    Factory function to create CNN by architecture name.
    
    Args:
        arch_name: One of 'tiny', 'small', 'medium', 'wide', 'deep'
        input_channels: Number of input channels (stacked frames)
    
    Returns:
        CNN module instance
    
    Raises:
        ValueError: If arch_name is not recognized
    """
    arch_name = arch_name.lower()
    if arch_name not in CNN_ARCHITECTURES:
        available = ', '.join(CNN_ARCHITECTURES.keys())
        raise ValueError(f"Unknown architecture '{arch_name}'. Available: {available}")
    
    return CNN_ARCHITECTURES[arch_name](input_channels)


def get_architecture_info() -> dict:
    """
    Get information about available architectures.
    
    Returns:
        Dict with architecture names as keys, and info dict as values
    """
    info = {}
    for name, cls in CNN_ARCHITECTURES.items():
        model = cls(input_channels=4)
        num_params = sum(p.numel() for p in model.parameters())
        info[name] = {
            'class': cls.__name__,
            'params': num_params,
            'output_dim': model.get_output_dim(),
            'description': cls.__doc__.split('\n')[1].strip() if cls.__doc__ else ''
        }
    return info


if __name__ == "__main__":
    # Test all architectures
    print("Testing CNN Architectures...")
    print("=" * 60)
    
    # Get architecture info
    info = get_architecture_info()
    
    for name, arch_info in info.items():
        print(f"\n{name.upper()} ({arch_info['class']}):")
        print(f"  Parameters: {arch_info['params']:,}")
        print(f"  Output dim: {arch_info['output_dim']}")
        
        # Test forward pass
        model = create_cnn(name, input_channels=4)
        batch = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.float32)
        output = model(batch)
        
        expected_dim = arch_info['output_dim']
        assert output.shape == (2, expected_dim), f"Expected (2, {expected_dim}), got {output.shape}"
        print(f"  Forward pass: ✓ {batch.shape} -> {output.shape}")
    
    print("\n" + "=" * 60)
    print("✅ All CNN architecture tests passed!")

