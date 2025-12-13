"""
Neural network components for RL agents.

Supports both DQN and PPO architectures.

Available CNN architectures:
- TinyCNN:   ~150K params - Fastest, for quick experiments
- SmallCNN:  ~400K params - Default, similar to Atari DQN
- MediumCNN: ~600K params - More capacity
- WideCNN:   ~1M params   - More filters per layer
- DeepCNN:   ~500K params - More layers, deeper features

Available attention modules:
- SpatialAttention:     Learns where to focus spatially
- ChannelAttention:     Learns which channels are important
- CBAM:                 Combined channel + spatial attention
- TreechopSpatialBias:  Spatial attention with center/hotbar bias

Use create_cnn('name') and create_attention('type') factory functions.
"""

from .cnn import (
    TinyCNN,
    SmallCNN,
    MediumCNN,
    WideCNN,
    DeepCNN,
    CNN_ARCHITECTURES,
    create_cnn,
    get_architecture_info,
)
from .attention import (
    SpatialAttention,
    ChannelAttention,
    CBAM,
    TreechopSpatialBias,
    create_attention,
)
from .dueling_head import DuelingHead
from .scalar_network import ScalarNetwork
from .dqn_network import DQNNetwork
from .policy_network import ActorCriticNetwork

__all__ = [
    # CNN architectures
    'TinyCNN',
    'SmallCNN', 
    'MediumCNN',
    'WideCNN',
    'DeepCNN',
    'CNN_ARCHITECTURES',
    'create_cnn',
    'get_architecture_info',
    # Attention modules
    'SpatialAttention',
    'ChannelAttention',
    'CBAM',
    'TreechopSpatialBias',
    'create_attention',
    # Network heads
    'DuelingHead',
    'ScalarNetwork',
    'DQNNetwork',
    'ActorCriticNetwork',
]

