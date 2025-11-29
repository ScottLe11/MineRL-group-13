# Scalar Network for Multimodal Fusion

## Overview

The `ScalarNetwork` is a 2-layer fully connected network that processes scalar observations (time_left, yaw, pitch, place_table_safe) before fusing them with visual CNN features.

## Why Use a Scalar Network?

**Without scalar network** (default):
```
Visual: CNN → 512-dim features ────┐
Scalars: [time, yaw, pitch, ...] ──┴─→ Concat(512+4) → Head → Output
```

**With scalar network**:
```
Visual: CNN → 512-dim features ──────┐
Scalars: [time, yaw, pitch, ...] → FC(64) → FC(64) ──┴─→ Concat(512+64) → Head → Output
```

**Benefits**:
- **Better representations**: Learns non-linear transformations of scalar features
- **Balanced fusion**: Scalar features get similar dimensionality to visual features (64 vs 512)
- **More capacity**: Additional ~65K parameters for processing scalars

## Architecture

```python
class ScalarNetwork(nn.Module):
    def __init__(self, num_scalars=4, hidden_dim=64, output_dim=64):
        self.network = nn.Sequential(
            nn.Linear(num_scalars, hidden_dim),  # 4 → 64
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),   # 64 → 64
            nn.ReLU()
        )
```

## Usage

### In Config (config.yaml)

Add these parameters to your network config:

```yaml
network:
  architecture: "wide"
  attention: "spatial"

  # Scalar network for multimodal fusion
  use_scalar_network: true        # Enable scalar processing
  scalar_hidden_dim: 64            # Hidden layer size
  scalar_output_dim: 64            # Output size (fused with CNN features)
```

### Programmatically

```python
from networks import DQNNetwork, ActorCriticNetwork

# DQN with scalar network
dqn = DQNNetwork(
    num_actions=8,
    num_scalars=4,
    use_scalar_network=True,
    scalar_hidden_dim=64,
    scalar_output_dim=64
)

# PPO with scalar network
ppo = ActorCriticNetwork(
    num_actions=8,
    num_scalars=4,
    use_scalar_network=True,
    scalar_hidden_dim=64,
    scalar_output_dim=64
)
```

## Parameter Count

- **Without scalar network**: ~2.2M params
- **With scalar network**: ~2.3M params (+65K)

The scalar network adds:
- Input layer: 4 × 64 + 64 (bias) = **320 params**
- Hidden layer: 64 × 64 + 64 (bias) = **4,160 params**
- **Total**: ~4,480 params per scalar network

## When to Use

✅ **Use scalar network when:**
- Scalar features are important (time pressure, spatial orientation)
- You want better multimodal fusion
- You have enough training data (~thousands of episodes)
- Computational cost is not a constraint

❌ **Don't use scalar network when:**
- Doing quick experiments / hyperparameter search
- Limited training data (risk of overfitting)
- Visual features are dominant (scalars are less important)

## Integration

The scalar network is integrated into both:
- **DQNNetwork** ([dqn_network.py](dqn_network.py))
- **ActorCriticNetwork** ([policy_network.py](policy_network.py))

Both networks automatically handle:
- Scalar processing when `use_scalar_network=True`
- Direct concatenation when `use_scalar_network=False` (default, backward compatible)

## Example: Training with Scalar Network

```bash
# 1. Update config.yaml
sed -i '' 's/use_scalar_network: false/use_scalar_network: true/' config/config.yaml

# 2. Train normally
python scripts/train.py
```

The network architecture will automatically use scalar processing!

## Testing

```python
# Test scalar network standalone
python networks/scalar_network.py

# Test DQN integration
python -c "from networks import DQNNetwork; net = DQNNetwork(use_scalar_network=True); print('✓')"

# Test PPO integration
python -c "from networks import ActorCriticNetwork; net = ActorCriticNetwork(use_scalar_network=True); print('✓')"
```

All tests should pass! ✅
