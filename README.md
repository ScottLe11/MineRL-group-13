# MineRL Tree-Chopping Deep RL Agent

**Project Status**: ✅ Core Implementation Complete - Ready for Testing
**Timeline**: 1.5 weeks
**Algorithms**: DQN (Double Dueling) and PPO

---

## 🎯 Project Goal

Build a Deep Reinforcement Learning agent that learns to **efficiently chop trees in Minecraft** within episodes.

**Success Criteria**: 
- Consistently collects wood (>80% success rate)
- Stretch goal: Learns to craft and use wooden axe (2× faster mining)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10
- Java JDK 8 (required for MineRL)

### Installation
```bash
# Create environment
conda create -n minerl-env python=3.10
conda activate minerl-env

# Install dependencies
pip install -r requirements.txt

# Install MineRL v1.0.2 from GitHub
pip install git+https://github.com/minerllabs/minerl@v1.0.2
```

### Run Training
```bash
# DQN training (default)
python scripts/train.py

# With custom config
python scripts/train.py --config config/config.yaml
```

### Run Tests
```bash
pytest tests/ -v
```

### Evaluate a Checkpoint
```bash
python scripts/evaluate.py --checkpoint checkpoints/final_model.pt --episodes 10
```

---

## 📁 Project Structure

```
MineRL-group-13/
├── config/
│   └── config.yaml              # All hyperparameters (DQN + PPO)
│
├── wrappers/                    # Environment wrappers
│   ├── vision.py                # Frame stacking (84x84 grayscale)
│   ├── hold_attack.py           # Attack duration handling
│   ├── reward.py                # Reward scaling
│   ├── observation.py           # Time/yaw/pitch scalars
│   └── actions.py               # 23 discrete actions
│
├── networks/                    # Neural network architectures
│   ├── cnn.py                   # SmallCNN (84x84 → 512 features)
│   ├── dueling_head.py          # Dueling Q-value head
│   ├── dqn_network.py           # Full DQN network
│   └── policy_network.py        # Actor-Critic for PPO
│
├── agent/                       # RL agents
│   ├── replay_buffer.py         # Experience replay (DQN)
│   ├── dqn.py                   # Double DQN agent
│   └── ppo.py                   # PPO agent with GAE
│
├── utils/                       # Utilities
│   ├── config.py                # Config loader
│   └── logger.py                # TensorBoard logging
│
├── scripts/                     # Entry points
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
│
├── crafting/                    # Tested crafting macros
│   ├── crafting_guide.py        # Craft planks/sticks/table/axe
│   └── gui_clicker.py           # GUI interaction helper
│
├── tests/                       # Unit tests (47 tests)
│   ├── test_networks.py
│   ├── test_agent.py
│   └── test_wrappers.py
│
└── main.py                      # Original demo/reference
```

---

## 🔧 Configuration

All settings in `config/config.yaml`:

```yaml
algorithm: "dqn"                  # "dqn" or "ppo"

environment:
  frame_shape: [84, 84]           # Grayscale frame size
  max_steps: 8000                 # Max steps per episode

dqn:
  num_actions: 23                 # Total discrete actions
  learning_rate: 0.0001
  gamma: 0.99
  batch_size: 32
  replay_buffer:
    capacity: 100000
    min_size: 10000

rewards:
  wood_value: 1.0                 # Points per log
  step_penalty: -0.001            # -0.001 per frame

device: "auto"                    # cpu, cuda, mps, or auto
```

---

## 🎮 Action Space (23 Actions)

| Index | Action | Frames | Description |
|-------|--------|--------|-------------|
| 0 | noop | 4 | Do nothing |
| 1-4 | movement | 4 | forward, back, right, left |
| 5 | jump | 4 | Jump |
| 6 | attack | 4 | Attack/mine |
| 7-10 | turn_left | 4 | 30°, 45°, 60°, 90° |
| 11-14 | turn_right | 4 | 30°, 45°, 60°, 90° |
| 15-16 | look_up | 4 | 12°, 20° |
| 17-18 | look_down | 4 | 12°, 20° |
| 19 | craft_planks | ~50 | Logs → Planks |
| 20 | make_table | ~100 | Craft + place table |
| 21 | craft_sticks | ~50 | Planks → Sticks |
| 22 | craft_axe | ~100 | Craft + equip axe |

---

## 🧠 Observation Space

| Component | Shape | Description |
|-----------|-------|-------------|
| pov | (4, 84, 84) | Stacked grayscale frames |
| time | (1,) | Normalized time remaining [0, 1] |
| yaw | (1,) | Horizontal rotation [-1, 1] |
| pitch | (1,) | Vertical rotation [-1, 1] |

---

## 💰 Reward Function

```
reward_per_frame = (logs × wood_value) + step_penalty
```

- **wood_value** points per log (default: 1.0)
- **step_penalty** per MineRL frame (default: -0.001, so -0.004 per decision)

**Example**: Mine 1 log over 4 frames = `(1 × 1.0) + (-0.001 × 4) = +0.996`

---

## 🔄 Wrapper Stack

```
MineRL Base Environment
    ↓
StackAndProcessWrapper (84x84 grayscale, 4 frames)
    ↓
HoldAttackWrapper (attack duration)
    ↓
RewardWrapper (add step penalty)
    ↓
ObservationWrapper (time, yaw, pitch)
    ↓
ExtendedActionWrapper (23 discrete actions)
    ↓
Agent
```

---

## 📊 Algorithms

### DQN (Default)
- Double DQN (reduces overestimation)
- Dueling architecture (value + advantage streams)
- Experience replay (100K buffer)
- Epsilon-greedy exploration (1.0 → 0.1)
- Soft target updates (τ = 0.005)

### PPO (Alternative)
- Clipped surrogate objective (ε = 0.2)
- GAE advantage estimation (λ = 0.95)
- Entropy bonus (0.01)
- Rollout buffer (2048 steps)

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_networks.py -v

# Run with coverage
pytest tests/ --cov=.
```

**Test coverage**:
- 47 tests across networks, agent, and wrappers
- Network dimensions, gradient flow, soft updates
- Replay buffer, epsilon schedule, training steps
- Observation wrapper, action wrapper, reward shaping

---

## 📈 Training Flow

1. **Initialize**: Load config, create environment, create agent
2. **Collect**: Agent selects action → env.step() → store experience
3. **Train**: Sample batch → compute loss → update network
4. **Log**: TensorBoard metrics (loss, Q-values, rewards)
5. **Save**: Periodic checkpoints

---

## 🔍 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Frame size | 84×84 | Atari standard, good balance |
| Frame stack | 4 | Motion/temporal information |
| Step penalty | Once per decision | Not 4× per frame |
| Macros | Always execute | Learn through experience |
| Inventory | Not observed | Learn from visual hotbar |

---

## 📚 Architecture Document

See `REVISED_ARCHITECTURE.md` for detailed implementation plan.

---

## 🛠️ Technology Stack

- **MineRL 1.0.2** - Minecraft environment
- **PyTorch** - Deep learning
- **OpenCV** - Image processing
- **TensorBoard** - Monitoring
- **pytest** - Testing

---

## 📝 License

[Your license here]

---

**Ready to train! 🚀🌳**
