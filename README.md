# MineRL Tree-Chopping Deep RL Agent

**Project Status**: ✅ Planning Complete - Ready for Implementation
**Timeline**: 1.5 weeks (12 days)
**Team**: 3 students, 3 machines (2 M1 Macs 8GB, 1 Windows 32GB GPU)

---

## 🎯 Project Goal

Build a Deep Reinforcement Learning agent that learns to **efficiently chop trees in Minecraft** within 60-second episodes.

**Core Challenge**: Agent must learn to:
1. Navigate to trees
2. Mine wood with fists (slow)
3. *Stretch goal*: Craft and use wooden axe (2× faster mining)

**Success Criteria**: 
- Consistently collects wood in 60-second episodes (>80% success rate)
- Stretch: Learns to craft axe for efficiency


## 🏗️ System Overview

### High-Level Architecture
```
User/Script → Config → Trainer → (Environment, Agent, Logger, Evaluator)
                                        ↓         ↓
                                   (Wrappers)  (Q-Network, Replay Buffer)
```

### Key Components

**Environment** (`environment/`):
- MineRL base + custom wrappers
- 23 discrete actions (primitives, camera, macros)
- Observations: visual (4×64×64) + scalars (time, yaw, pitch)
- Reward: wood collection (+1.0) + step penalty (-0.001)

**Actions** (`actions/`):
- 7 primitives (noop, movement, attack) - 4 frames each
- 12 camera (turn, look) - 4 frames each
- 4 macros (craft planks/sticks/table/axe) - 8-12 frames

**Networks** (`networks/`):
- 5 CNN architectures (tiny → deep)
- Dueling DQN head (value + advantage)
- Optional spatial attention

**Algorithm** (`algorithms/dqn/`):
- Double DQN (reduce overestimation)
- Replay buffer (100K capacity)
- Epsilon-greedy exploration
- Soft target network updates

**Training** (`training/`):
- Phase 1: Reconnaissance (40 configs, 200 episodes, 20s)
- Phase 2: Validation (3-5 configs, 1000 episodes, 60s)
- Phase 3: Final training (best config, 3000-5000 episodes, Gorila)

---

## 🔑 Key Design Decisions

> **See [`IMPLEMENTATION_DECISIONS.md`](IMPLEMENTATION_DECISIONS.md) for complete details**

### Critical Choices

1. **Temporal Abstraction**: 1 decision = 4 frames = 200ms
   - Agent decides at 5 Hz, MineRL runs at 20 Hz
   - ONE experience per decision (not per frame)

2. **Observation Space**: Visual + 3 Scalars
   - Visual: (4, 64, 64) grayscale stacked frames
   - Scalars: time_left, yaw, pitch
   - **NO inventory** (learns from visual hotbar)

3. **Reward Function**: Wood Only
   - +1.0 per wood collected
   - -0.001 per step
   - **NO crafting sub-rewards** (learns instrumental value)

4. **Macro Behavior**: Always Execute
   - Macros run full sequence even without materials
   - Agent learns through experience not to craft without items

5. **Action Space**: 23 Actions
   - 0-6: Primitives (movement, attack)
   - 7-18: Camera (turn/look)
   - 19-22: Crafting macros

---

## 📊 Three-Phase Training

### Phase 1: Reconnaissance (Days 2-3)
- **Goal**: Find 8 configs that get 1+ wood
- **Method**: 40 configs × 200 episodes × 20s
- **Execution**: 3 machines parallel (overnight)
- **Output**: Comparison CSV, select top 3-5

### Phase 2: Validation (Days 4-5)
- **Goal**: Confirm configs work on 60s episodes
- **Method**: 3-5 configs × 1000 episodes × 60s
- **Execution**: Windows machine sequential
- **Output**: Best config identified

### Phase 3: Final Training (Days 6-9)
- **Goal**: Train to convergence
- **Method**: Best config × 3000-5000 episodes × 60s
- **Execution**: Gorila distributed (4 actors, 2 learners)
- **Output**: Final model, videos, analysis

---

## 🎯 Success Metrics

### Phase 1 (Recon)
- ✅ At least 5 configs show learning
- ✅ Gets wood in >80% episodes

### Phase 2 (Validation)
- ✅ Best config gets wood consistently in all episodes
- ✅ Stable learning curve

### Phase 3 (Final)
- ✅ Final agent gets 20+ wood each episodes
- 🎁 Stretch: Agent learns to craft & use axe

### Comparison to Baseline
- Random agent: ~0.3% success rate (60s episodes)
- Expected: >80% success rate (267× better!)

---

## 🗓️ Timeline

| Days | Phase | Activities |
|------|-------|-----------|
| **1-2** | Infrastructure | Build modules, tests, configs |
| **2-3** | Recon | Run 40 configs on 3 machines |
| **4** | Analysis | Compare results, select top configs |
| **4-5** | Validation | Validate on 60s episodes |
| **6-9** | Final Training | Train best config with Gorila |
| **10-11** | Analysis | Generate visualizations, report |
| **12** | Buffer | Polish, fix issues |

**Current Status**: End of Day 1 → Day 2 (Infrastructure)

---

## 🛠️ Technology Stack

**Core**:
- MineRL 1.0 (Minecraft environment)
- PyTorch (deep learning)
- OpenCV (image processing)

**RL Algorithm**:
- Double DQN with Dueling Architecture
- Experience Replay (100K capacity)
- Epsilon-greedy exploration

**Infrastructure**:
- TensorBoard (monitoring)
- YAML (configuration)
- pytest (testing)

**Optional**:
- Gorila distributed training (Phase 3)
- Human demonstrations (behavior cloning)
- Curriculum learning

---

## 🔍 Key Features

### Temporal Abstraction
- Agent makes decisions every 200ms (not every 50ms frame)
- Reduces action space complexity
- Enables macro actions (crafting sequences)

### Dueling DQN Architecture
- Separates value (how good is state) from advantage (how good is action)
- More stable learning
- Better generalization

### Double DQN
- Reduces Q-value overestimation
- Selects action with online network
- Evaluates with target network

### Optional: Gorila Distributed Training
- 4 parallel actors generate experiences
- 2 learners compute gradients
- 1 parameter server coordinates updates
- 3-4× speedup over single agent

---

## 🎓 Learning Challenges

### Why This is Hard
1. **Sparse Rewards**: Only get reward when collecting wood
2. **Long Horizons**: 300 decisions per episode (60s)
3. **Visual Input**: Must learn from pixels
4. **Instrumental Learning**: Must discover crafting helps mining
5. **Exploration**: Vast action/state space

### How We Address It
1. **Step Penalty**: Encourages faster wood collection
2. **Frame Stacking**: Provides motion/temporal info
3. **CNN Features**: Learns visual representations
4. **Large Replay Buffer**: Learns from past experiences
5. **Epsilon Decay**: Gradual exploration→exploitation

---

## 🤝 Contributing

### For Team Members

**Implementing a module?**
1. Read relevant section in `PROJECT_ARCHITECTURE.md`
2. Check `IMPLEMENTATION_DECISIONS.md` for design choices
3. Use `MODULE_SPEC_TEMPLATE.md` for detailed design
4. Write tests in `tests/`
5. Update `MODULE_CHECKLIST.md` when complete

**Stuck?**
1. Check documentation (especially COMPONENT_INTERACTIONS.md)
2. Run relevant tests
3. Ask in team channel with details (error messages, what you tried)

---

## 📖 References

### Papers
- DQN: [Mnih et al. 2015](https://www.nature.com/articles/nature14236)
- Double DQN: [van Hasselt et al. 2015](https://arxiv.org/abs/1509.06461)
- Dueling DQN: [Wang et al. 2015](https://arxiv.org/abs/1511.06581)
- Prioritized Replay: [Schaul et al. 2015](https://arxiv.org/abs/1511.05952)
- Gorila: [Nair et al. 2015](https://arxiv.org/abs/1507.04296)

### Documentation
- MineRL: https://minerl.readthedocs.io/en/v1.0/
- PyTorch: https://pytorch.org/docs/
- OpenAI Gym: https://www.gymlibrary.dev/

---

## 📝 License

[Your license here]

---

## 🎉 Status

**✅ Planning Complete** (2025-11-24)
- All design decisions finalized
- Architecture documented
- Ready for implementation


---

**Let's build this! 🚀🌳**

