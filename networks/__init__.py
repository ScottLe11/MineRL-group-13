"""
Neural network components for RL agents.

Supports both DQN and PPO architectures.

Available CNN architectures:
- TinyCNN:   ~150K params - Fastest, for quick experiments
- SmallCNN:  ~400K params - Default, similar to Atari DQN
- MediumCNN: ~600K params - More capacity
- WideCNN:   ~1M params   - More filters per layer
- DeepCNN:   ~500K params - More layers, deeper features

Use create_cnn('name') factory function to create by name.
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
from .dueling_head import DuelingHead
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
    # Network heads
    'DuelingHead', 
    'DQNNetwork',
    'ActorCriticNetwork',
]

