# Training Run Suggestions

**Date**: 2025-11-25  
**Purpose**: Soft recommendations for humans/agents running training experiments.

---

## Overview

Instead of rigid config generation, this document provides **flexible suggestions** for training runs. Operators should use judgment based on current results and available compute.

---

## Phase 1: Quick Sanity Checks

### Before Full Training

**Goal**: Verify the pipeline works end-to-end.

| Check | Command | Expected |
|-------|---------|----------|
| Unit tests pass | `pytest tests/ -v` | All green |
| Config loads | `python -c "from utils.config import load_config; print(load_config())"` | No errors |
| Mock training starts | `python scripts/train.py` | Starts without crash |

### First Real Run (5-10 minutes)

**Goal**: See if agent interacts with environment correctly.

```bash
# Suggested settings for first run:
# - Short episodes (100 steps = 20 seconds)
# - High epsilon (lots of exploration)
# - Small replay buffer (start training quickly)
# - Watch TensorBoard for sanity
```

**Watch for**:
- Q-values not exploding (stay < 100 initially)
- Loss decreasing or stable
- Epsilon decaying
- No NaN values

---

## Phase 2: Hyperparameter Exploration

### Key Parameters to Vary

**Ordered by impact**:

1. **Learning Rate**: `[1e-4, 5e-4, 1e-3]`
   - Start with 1e-4 (safe default)
   - Try 5e-4 if learning seems too slow
   - 1e-3 can diverge - use with caution

2. **Network Size**: `[small, medium]`
   - Small (~400K params): Faster, less overfitting risk
   - Medium (~600K params): More capacity, slower
   - Deep architectures: NOT recommended initially

3. **Batch Size**: `[32, 64]`
   - 32: More updates, more variance
   - 64: Smoother but slower

4. **Epsilon Decay**: `[50k, 100k, 200k steps]`
   - Short episodes (20s): Use 50-100k
   - Full episodes (60s): Use 100-200k

### Suggested First Explorations

**Run A**: Conservative baseline
```yaml
learning_rate: 1e-4
batch_size: 32
epsilon_decay_steps: 100000
target_update: soft (tau=0.005)
```

**Run B**: Faster learning
```yaml
learning_rate: 5e-4
batch_size: 64
epsilon_decay_steps: 50000
target_update: soft (tau=0.01)
```

**Run C**: Longer exploration
```yaml
learning_rate: 1e-4
batch_size: 32
epsilon_decay_steps: 200000
target_update: hard (every 1000 steps)
```

---

## Phase 3: Starting Position Experiments

### Order of Difficulty (Easiest → Hardest)

1. **with_axe**: Agent has axe, just needs to find and chop trees
   - Use for: Learning navigation + attack timing
   - Success metric: 80% episodes get ≥2 wood

2. **with_logs**: Agent has logs, can craft or chop more
   - Use for: Learning if crafting helps
   - Success metric: 70% episodes get ≥3 wood or craft axe

3. **with_planks**: Agent has planks, can make table → sticks → axe
   - Use for: Learning craft sequence
   - Success metric: 50% episodes craft axe

4. **random**: Full task from scratch
   - Use for: Final evaluation
   - Success metric: 30% episodes get wood, 10% craft axe

### Suggested Progression

```
Week 1, Day 1-2: Run with_axe until 80% success
Week 1, Day 3-4: Run with_logs until crafting emerges  
Week 1, Day 5+:  Run random and evaluate
```

---

## Machine-Specific Suggestions

### M1 Mac (8GB RAM)

**Constraints**: Limited memory, MPS GPU
**Suggestions**:
- Batch size: 32 max
- Replay buffer: 50K max
- Use `device: mps`
- Run one experiment at a time

### Windows (32GB RAM, GPU)

**Constraints**: None significant
**Suggestions**:
- Can use larger batch sizes (64-128)
- Replay buffer: 100K+
- Can run multiple experiments with different seeds

---

## What to Watch in TensorBoard

### Early Training (First 10K steps)

| Metric | Healthy | Concerning |
|--------|---------|------------|
| Loss | Decreasing or stable | Increasing, NaN |
| Q-mean | Small values (-1 to 10) | Exploding (>100) |
| Epsilon | Decreasing | Stuck at 1.0 |
| Episode reward | Negative (step penalty) | Extremely negative |

### Mid Training (10K-100K steps)

| Metric | Healthy | Concerning |
|--------|---------|------------|
| Loss | Fluctuating but bounded | Diverging |
| Q-mean | Slowly increasing | Stuck at same value |
| Episode reward | Occasionally positive | Always negative |
| Wood collected | Occasional 1-2 | Always 0 |

### Late Training (100K+ steps)

| Metric | Healthy | Concerning |
|--------|---------|------------|
| Success rate | > 20% | < 5% |
| Avg wood | > 0.5 | < 0.1 |
| Q-mean | Stable, positive | Oscillating wildly |

---

## When to Stop/Restart

### Signs to Restart

1. **Q-values exploding**: Reduce learning rate, check for bugs
2. **Loss NaN**: Gradient issue, check inputs
3. **Zero wood after 50K steps**: Check action space, exploration
4. **Epsilon not decaying**: Check step counter

### Signs to Continue

1. **Loss stable but high**: Normal for DQN, keep going
2. **Occasional success**: Learning signal present
3. **Q-values slowly growing**: Value estimates improving

### Signs to Declare Success

1. **>50% success rate for 100 consecutive episodes**
2. **Average wood ≥1.0 over last 50 episodes**
3. **Agent crafts axe at least once**

---

## Debugging Common Issues

### "Agent just spins in circles"

**Likely cause**: Camera actions dominating
**Try**: 
- Check action distribution (are camera actions overrepresented?)
- Increase exploration (higher epsilon longer)
- Check reward signal

### "Agent attacks empty air"

**Likely cause**: Not associating visual input with reward
**Try**:
- Verify frame stacking is correct
- Check POV observation shape
- Start with `with_axe` config (simplify task)

### "Crafting macros never produce items"

**Likely cause**: Calling macros without prerequisites
**Expected behavior**: This is fine! Agent learns through wasted frames.
**Don't**: Try to "fix" this - it's the learning signal

### "Training seems stuck"

**Likely cause**: Local optimum, insufficient exploration
**Try**:
- Reset with different seed
- Increase epsilon_decay_steps
- Try different starting position

---

## Recording Experiments

### What to Log

For each run, record:
```
Run ID: run_001
Date: 2025-11-25
Machine: M1 Mac
Config: {key hyperparameters}
Duration: X hours
Final metrics:
  - Success rate: XX%
  - Avg wood: X.X
  - Best episode wood: XX
Notes: {observations}
```

### When to Record Video

- First successful wood collection
- First axe craft
- End of training (evaluation run)
- Interesting failure modes

---

## Questions to Ask Before Each Run

1. **What am I trying to learn from this run?**
   - "Does higher LR help?"
   - "Can agent learn crafting?"
   - "Is my baseline working?"

2. **What would make me stop this run early?**
   - "If no wood after 50K steps"
   - "If loss diverges"

3. **What's my next step based on results?**
   - Success → Move to harder config
   - Failure → Try different hyperparameter
   - Unclear → Run longer

---

*Last Updated: 2025-11-25*

