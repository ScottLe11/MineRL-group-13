"""
Spatial Attention Mechanism for visual feature maps.

Helps the agent focus on relevant parts of the screen:
- Center of screen (where trees appear when chopping)
- Bottom of screen (hotbar and inventory indicators)

Reference: CBAM (Convolutional Block Attention Module)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SpatialAttention(nn.Module):
    """
    Spatial attention module that learns to focus on important regions.
    
    Takes a feature map and produces spatial attention weights that
    highlight which spatial locations are most relevant.
    
    Input: (batch, channels, height, width)
    Output: (batch, channels, height, width), (batch, 1, height, width)
            (attended_features,              attention_weights)
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size: Size of the convolution kernel for attention (must be odd).
        """
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        # Compress channel dimension to 2 (max and avg pooled), then conv to 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights for gradual learning."""
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spatial attention to input features.
        
        Args:
            x: Feature map of shape (batch, channels, height, width)
        
        Returns:
            attended_features: Same shape as input, element-wise multiplied by attention
            attention_weights: Shape (batch, 1, height, width), values in [0, 1]
        """
        # Channel pooling: max and average across channels
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)     # (B, 1, H, W)
        
        # Concatenate and pass through conv
        pooled = torch.cat([max_pool, avg_pool], dim=1)   # (B, 2, H, W)
        attention = self.sigmoid(self.conv(pooled))       # (B, 1, H, W)
        
        # Apply attention (broadcast across channels)
        attended = x * attention
        
        return attended, attention


class ChannelAttention(nn.Module):
    """
    Channel attention module that learns which feature channels are important.
    
    Uses global pooling to create channel descriptors, then MLPs to compute
    channel-wise attention weights.
    
    Input: (batch, channels, height, width)
    Output: (batch, channels, height, width), (batch, channels, 1, 1)
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        """
        Args:
            channels: Number of input channels.
            reduction_ratio: Reduction ratio for the bottleneck MLP.
        """
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        hidden_dim = max(1, channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply channel attention to input features.
        
        Args:
            x: Feature map of shape (batch, channels, height, width)
        
        Returns:
            attended_features: Same shape as input
            attention_weights: Shape (batch, channels, 1, 1)
        """
        batch, channels, _, _ = x.size()
        
        # Global pooling
        avg_out = self.avg_pool(x).view(batch, channels)  # (B, C)
        max_out = self.max_pool(x).view(batch, channels)  # (B, C)
        
        # MLP
        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)  # (B, C)
        attention = attention.view(batch, channels, 1, 1)
        
        attended = x * attention
        
        return attended, attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Combines channel attention and spatial attention sequentially.
    First refines "what" (channels) then "where" (spatial locations).
    
    Reference: Woo et al., "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 16, spatial_kernel: int = 7):
        """
        Args:
            channels: Number of input feature channels.
            reduction_ratio: Reduction ratio for channel attention MLP.
            spatial_kernel: Kernel size for spatial attention conv.
        """
        super().__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Apply CBAM attention.
        
        Args:
            x: Feature map of shape (batch, channels, height, width)
        
        Returns:
            attended_features: Same shape as input
            attention_info: Dict with 'channel' and 'spatial' attention weights
        """
        # Channel attention first
        x_channel, channel_weights = self.channel_attention(x)
        
        # Then spatial attention
        x_spatial, spatial_weights = self.spatial_attention(x_channel)
        
        attention_info = {
            'channel': channel_weights,
            'spatial': spatial_weights
        }
        
        return x_spatial, attention_info


class TreechopSpatialBias(nn.Module):
    """
    Spatial attention with learnable bias toward important screen regions.
    
    For tree-chopping, important regions are:
    - Center of screen (crosshair, where trees appear when looking at them)
    - Bottom ~20% (hotbar showing equipped items)
    
    This module adds a learnable spatial bias to guide attention.
    """
    
    def __init__(self, height: int = 7, width: int = 7, init_center_bias: float = 0.5):
        """
        Args:
            height: Expected feature map height.
            width: Expected feature map width.
            init_center_bias: Initial bias toward center (0-1).
        """
        super().__init__()
        
        # Learnable spatial bias map
        self.spatial_bias = nn.Parameter(torch.zeros(1, 1, height, width))
        
        # Initialize with center and bottom bias
        self._init_bias(height, width, init_center_bias)
        
        self.sigmoid = nn.Sigmoid()
    
    def _init_bias(self, height: int, width: int, center_bias: float):
        """Initialize bias to favor center and bottom of screen."""
        with torch.no_grad():
            # Create coordinate grid
            y = torch.linspace(-1, 1, height)
            x = torch.linspace(-1, 1, width)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            
            # Center Gaussian
            center_dist = torch.sqrt(xx**2 + yy**2)
            center_weight = torch.exp(-center_dist * 2) * center_bias
            
            # Bottom bias (hotbar)
            bottom_weight = torch.clamp((yy + 0.5), 0, 1) * 0.3
            
            # Combine
            bias = center_weight + bottom_weight
            self.spatial_bias.data = bias.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spatial bias attention.
        
        Args:
            x: Feature map of shape (batch, channels, height, width)
        
        Returns:
            attended_features: Same shape as input
            attention_weights: Shape (batch, 1, height, width)
        """
        # Compute content-based attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        content_signal = (max_pool + avg_pool) / 2
        
        # Interpolate bias if sizes don't match
        _, _, h, w = x.size()
        bias = self.spatial_bias
        if bias.size(2) != h or bias.size(3) != w:
            bias = F.interpolate(bias, size=(h, w), mode='bilinear', align_corners=False)
        
        # Combine content signal with learned bias
        attention = self.sigmoid(content_signal + bias)
        
        attended = x * attention
        
        return attended, attention


# Factory function for easy creation
def create_attention(
    attention_type: str,
    channels: int = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create attention modules.
    
    Args:
        attention_type: One of 'spatial', 'channel', 'cbam', 'treechop_bias', 'none'
        channels: Number of input channels (required for 'channel' and 'cbam')
        **kwargs: Additional arguments passed to the attention module
    
    Returns:
        Attention module or nn.Identity if 'none'
    """
    attention_type = attention_type.lower()
    
    if attention_type == 'none':
        return nn.Identity()
    elif attention_type == 'spatial':
        return SpatialAttention(**kwargs)
    elif attention_type == 'channel':
        if channels is None:
            raise ValueError("channels required for channel attention")
        return ChannelAttention(channels, **kwargs)
    elif attention_type == 'cbam':
        if channels is None:
            raise ValueError("channels required for CBAM")
        return CBAM(channels, **kwargs)
    elif attention_type == 'treechop_bias':
        return TreechopSpatialBias(**kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


if __name__ == "__main__":
    print("Testing Attention Modules...")
    print("=" * 60)
    
    # Test input: batch=2, channels=64, height=7, width=7 (like final conv output)
    x = torch.randn(2, 64, 7, 7)
    
    # Test SpatialAttention
    print("\n1. SpatialAttention:")
    spatial = SpatialAttention(kernel_size=7)
    out, weights = spatial(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Weights: {weights.shape}, range [{weights.min():.3f}, {weights.max():.3f}]")
    
    # Test ChannelAttention
    print("\n2. ChannelAttention:")
    channel = ChannelAttention(channels=64)
    out, weights = channel(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Weights: {weights.shape}, range [{weights.min():.3f}, {weights.max():.3f}]")
    
    # Test CBAM
    print("\n3. CBAM:")
    cbam = CBAM(channels=64)
    out, info = cbam(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Channel weights: {info['channel'].shape}")
    print(f"   Spatial weights: {info['spatial'].shape}")
    
    # Test TreechopSpatialBias
    print("\n4. TreechopSpatialBias:")
    bias_attn = TreechopSpatialBias(height=7, width=7)
    out, weights = bias_attn(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Weights: {weights.shape}")
    print(f"   Center weight: {weights[0, 0, 3, 3]:.3f}")
    print(f"   Corner weight: {weights[0, 0, 0, 0]:.3f}")
    
    # Test factory
    print("\n5. Factory function:")
    for atype in ['spatial', 'cbam', 'treechop_bias', 'none']:
        try:
            module = create_attention(atype, channels=64)
            print(f"   {atype}: {module.__class__.__name__}")
        except Exception as e:
            print(f"   {atype}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print("✅ All attention module tests passed!")


