# Revised Architecture - Building on Existing Code

**Date**: 2025-11-24
**Status**: Replaces previous architecture plans

This document describes how to build the DQN agent by **extending existing tested components** rather than creating parallel structures.

---

## Existing Components Analysis

### What Already Works (DO NOT REPLACE)

#### 1. `wrappers/vision.py` - StackAndProcessWrapper ✅
- Resizes frames to configurable size (default 84x84)
- Converts RGB to grayscale
- Stacks 4 frames
- Stores `last_full_obs` for rendering
- **Use as-is**

#### 2. `wrappers/hold_attack.py` - HoldAttackWrapper ✅
- Extends single attack into held attack sequence (35 ticks)
- Has `set_hold_suppressed()` for crafting macros
- GUI-aware (doesn't attack during inventory)
- **Use as-is**

#### 3. `wrappers/actions.py` - SimpleActionWrapper ✅
- 7 primitive actions: noop, forward, back, right, left, jump, attack
- 1 PIPELINE macro calling `craft_pipeline_make_and_equip_axe`
- Reward accumulation via `_Tracer` wrapper
- Suppresses HoldAttack during macros
- **EXTEND this** (don't replace)

#### 4. `crafting/crafting_guide.py` - Tested Macro Functions ✅
Available functions:
- `craft_planks_from_logs()` - converts logs to planks in 2x2 grid
- `craft_table_in_inventory()` - makes crafting table from planks
- `place_and_open_table()` - places and opens table
- `craft_sticks_in_table()` - makes sticks in 3x3 grid
- `craft_wooden_axe()` - crafts axe and equips it
- `craft_pipeline_make_and_equip_axe()` - full sequence
- **Use these directly** in extended action wrapper

#### 5. `crafting/gui_clicker.py` - GuiClicker ✅
- Mouse movement, clicks, hotbar selection, inventory toggle
- **Use as-is** (required by crafting functions)

---

## What Needs to Be Added

### 1. Extended Action Wrapper
**File**: `wrappers/actions.py` (modify existing)

**New Actions to Add**:
- 12 camera actions (turn left/right at 30°/45°/60°/90°, look up/down at 12°/20°)
- Optionally: individual crafting macros instead of just PIPELINE

**Final Action Space (23 actions)**:
```
0: noop
1: forward
2: back  
3: right
4: left
5: jump
6: attack
7-10: turn_left_30, turn_left_45, turn_left_60, turn_left_90
11-14: turn_right_30, turn_right_45, turn_right_60, turn_right_90
15-16: look_up_12, look_up_20
17-18: look_down_12, look_down_20
19: craft_planks (from logs)
20: make_table (craft + place + open table)
21: craft_sticks (in table)
22: craft_axe (in table)
```

### 2. Observation Wrapper (New)
**File**: `wrappers/observation.py` (new file in existing directory)

**Purpose**: Add scalar features to observation dict
- time_left: (max_steps - current_step) / max_steps
- yaw: agent facing direction (track from camera actions)
- pitch: head angle (track from camera actions)

### 3. DQN Network (New)
**Directory**: `networks/`

**Files**:
- `networks/__init__.py`
- `networks/cnn.py` - SmallCNN feature extractor
- `networks/dueling_head.py` - Dueling Q-value head
- `networks/dqn_network.py` - Combined network

### 4. DQN Agent (New)
**Directory**: `agent/`

**Files**:
- `agent/__init__.py`
- `agent/replay_buffer.py` - Experience storage
- `agent/dqn.py` - Agent with training logic

### 5. Training Infrastructure (New)
**Files**:
- `config/config.yaml` - Single config file
- `utils/config.py` - Simple loader
- `utils/logger.py` - TensorBoard logging
- `scripts/train.py` - Main training script

---

## Revised Directory Structure

```
MineRL-group-13/
├── config/
│   └── config.yaml                 # Single config file
│
├── wrappers/                       # EXTEND existing
│   ├── __init__.py                 # [EXISTS] - update exports
│   ├── vision.py                   # [EXISTS] - StackAndProcessWrapper
│   ├── hold_attack.py              # [EXISTS] - HoldAttackWrapper
│   ├── actions.py                  # [EXISTS] - EXTEND with camera + macros
│   ├── observation.py              # [NEW] - Add time/yaw/pitch scalars
│   └── recorder.py                 # [EXISTS]
│
├── crafting/                       # DO NOT MODIFY
│   ├── __init__.py                 # [EXISTS]
│   ├── crafting_guide.py           # [EXISTS] - tested macro functions
│   └── gui_clicker.py              # [EXISTS] - GuiClicker
│
├── networks/                       # NEW - DQN network
│   ├── __init__.py
│   ├── cnn.py                      # SmallCNN feature extractor
│   ├── dueling_head.py             # Dueling Q-value computation
│   └── dqn_network.py              # Combined network class
│
├── agent/                          # NEW - DQN agent
│   ├── __init__.py
│   ├── replay_buffer.py            # Experience replay
│   └── dqn.py                      # DQN agent class
│
├── utils/                          # NEW - utilities
│   ├── __init__.py
│   ├── config.py                   # Config loader
│   └── logger.py                   # TensorBoard logger
│
├── scripts/                        # NEW - entry points
│   ├── train.py                    # Training script
│   └── evaluate.py                 # Evaluation script
│
├── tests/                          # NEW - essential tests
│   ├── test_wrappers.py            # Test extended wrappers
│   └── test_agent.py               # Test DQN agent
│
├── main.py                         # [EXISTS] - reference/demo
├── treechop_spec.py                # [EXISTS] - env specification
├── recorder_gameplay.py            # [EXISTS] - human recording
├── requirements.txt                # [EXISTS/UPDATE]
└── README.md                       # [EXISTS/UPDATE]
```

**Total new files**: ~15 (vs 40+ in previous plan)

---

## Implementation Plan

### Step 1: Extend SimpleActionWrapper (2-3 hours)

Modify `wrappers/actions.py` to add:

1. **Camera Actions** (12 new actions):
   - Each camera action executes for 4 frames
   - Distribute angle evenly: turn_left_30 = 7.5° per frame × 4 frames
   
2. **Individual Crafting Macros** (4 new actions):
   - craft_planks: calls `craft_planks_from_logs()` 
   - craft_table: calls `craft_table_in_inventory()`
   - craft_sticks: calls `craft_sticks_in_table()` (requires table open)
   - craft_axe: calls full pipeline or just `craft_wooden_axe()`

3. **Use existing patterns**:
   - Use `_Tracer` pattern for reward accumulation
   - Use `set_hold_suppressed()` for macro execution
   - Use `GuiClicker` and crafting functions directly

### Step 2: Add Observation Wrapper (1 hour)

Create `wrappers/observation.py`:
- Wraps env after StackAndProcessWrapper
- Tracks step count for time_left
- Tracks yaw/pitch from camera actions (or defaults to 0)
- Returns dict with 'pov', 'time', 'yaw', 'pitch'

### Step 3: Build DQN Network (2-3 hours)

Create `networks/`:
- SmallCNN: 3 conv layers, output 512 features
- DuelingHead: Value and Advantage streams
- DQNNetwork: CNN + concat scalars + DuelingHead

### Step 4: Build DQN Agent (3-4 hours)

Create `agent/`:
- ReplayBuffer: Simple deque-based, uniform sampling
- DQNAgent: Network, target network, optimizer, epsilon schedule, training logic

### Step 5: Training Script (1-2 hours)

Create `scripts/train.py`:
- Load config
- Create wrapped environment (StackAndProcess → Observation → Action)
- Create agent
- Training loop with logging

---

## Key Changes from Previous Architecture

### What's Being DELETED

Remove these directories/files created in this session:
- `environment/` directory (duplicate of wrappers/)
- `actions/` directory (duplicates SimpleActionWrapper)
- Complex multi-file config system
- Distributed/Gorila planning (premature)
- Many documentation files (consolidate)

### What's Being KEPT

From this session:
- `networks/` concept (genuinely new code needed)
- `agent/` concept (genuinely new code needed)
- Core design decisions from IMPLEMENTATION_DECISIONS.md

From existing repo:
- ALL of `wrappers/` (extend, don't replace)
- ALL of `crafting/` (use directly)
- `main.py`, `treechop_spec.py`, etc.

---

## Detailed: Extending SimpleActionWrapper

The existing `SimpleActionWrapper` in `wrappers/actions.py` should be modified to:

**Current state** (8 actions):
- Actions 0-6: primitives (noop, forward, back, right, left, jump, attack)
- Action 7: PIPELINE (full craft sequence)

**Extended state** (23 actions):
- Actions 0-6: primitives (unchanged)
- Actions 7-18: camera movements (NEW)
- Actions 19-22: individual crafting macros (NEW, using existing crafting/ functions)

**Implementation approach**:
1. Add camera action definitions (list of 4-frame sequences)
2. Add macro handlers that call existing `crafting/crafting_guide.py` functions
3. Use the same `_Tracer` pattern for reward accumulation
4. Keep backward compatibility with existing code

---

## Detailed: Using Existing Crafting Functions

The `crafting/crafting_guide.py` functions are **already tested** and should be called directly:

**For craft_planks action (19)**:
- Call `craft_planks_from_logs(env, helper, logs_to_convert=3, ...)`
- Returns after crafting planks from logs

**For craft_table action (20)**:
- Call `craft_table_in_inventory(env, helper, ...)`
- Requires planks in inventory
- Creates crafting table

**For craft_sticks action (21)**:
- Call `craft_sticks_in_table(env, helper, ...)`
- Requires table to be open and planks available

**For craft_axe action (22)**:
- Call `craft_pipeline_make_and_equip_axe(env, helper, ...)` 
- OR call individual steps if table is already placed

**Key consideration**: The crafting functions assume certain state (logs available, table placed, etc.). If prerequisites aren't met, they still execute (wasting frames but not crashing) - this matches Q1 answer.

---

## Environment Wrapper Stack

Final wrapper order:
```
Base MineRL Env (MineRLcustom_treechop-v0)
    ↓
StackAndProcessWrapper (wrappers/vision.py)
    - Resizes to 64x64 grayscale
    - Stacks 4 frames
    ↓
HoldAttackWrapper (wrappers/hold_attack.py)  
    - Extends attack actions
    - Suppressed during crafting
    ↓
ObservationWrapper (wrappers/observation.py) [NEW]
    - Adds time_left, yaw, pitch scalars
    ↓
ExtendedActionWrapper (wrappers/actions.py) [EXTENDED]
    - 23 discrete actions
    - Executes multi-frame actions
    - Accumulates rewards
    ↓
Agent
```

---

## Next Steps

1. **Revert/Clean**: Delete `environment/`, `actions/` directories created in this session
2. **Extend**: Modify `wrappers/actions.py` to add camera + macro actions
3. **Add**: Create `wrappers/observation.py` 
4. **Build**: Create `networks/` and `agent/` directories
5. **Test**: Create minimal test to verify wrapper chain works
6. **Train**: Create training script and run first test

---

## Files to Delete

These files/directories were created in this session and should be deleted to avoid confusion:

```
DELETE:
├── environment/                    # Duplicate of wrappers/
│   ├── __init__.py
│   ├── treechop_env.py
│   └── wrappers/
│       ├── __init__.py
│       ├── observation_wrapper.py
│       └── action_wrapper.py
│
├── actions/                        # Duplicates SimpleActionWrapper
│   ├── __init__.py
│   ├── primitives.py
│   ├── camera.py
│   └── action_executor.py
│
├── utils/                          # Will recreate simpler version
│   ├── __init__.py
│   └── config_loader.py
```

---

## Summary

**Philosophy Change**: Build ON existing code, not BESIDE it.

The existing `wrappers/` and `crafting/` code is tested and working. The DQN agent needs to integrate with it, not replace it. The only genuinely new components are:
- Neural network (`networks/`)
- RL agent logic (`agent/`)
- Training orchestration (`scripts/train.py`)

Everything else should extend or use existing code.

