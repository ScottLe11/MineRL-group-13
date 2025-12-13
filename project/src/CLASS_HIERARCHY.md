# MineRL Treechop Environment Class Hierarchy

## Overview

This document explains the class hierarchy for the treechop environment, who created each class, and how data flows through them.

---

## Class Hierarchy Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  HumanControlEnvSpec                                         │
│  [MineRL Library - William H. Guss, Brandon Houghton]       │
│  ────────────────────────────────────────────────────────   │
│  Base class for all human-controllable MineRL environments  │
│  - Defines template methods (create_rewardables, etc.)      │
│  - Handles gym environment registration                      │
│  - Manages Minecraft server communication                    │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ inherits
                              │
┌─────────────────────────────────────────────────────────────┐
│  Treechop                                                    │
│  [MineRL Library - Original Authors]                        │
│  ────────────────────────────────────────────────────────   │
│  Standard MineRL treechop task                              │
│  - Hardcoded: Start with 3 logs                            │
│  - Hardcoded: Goal is 64 logs                              │
│  - Hardcoded: max_episode_steps = 8000                     │
│  - Defines forest biome world generation                    │
│  - Defines rewards, actions, termination conditions         │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ inherits
                              │
┌─────────────────────────────────────────────────────────────┐
│  ConfigurableTreechop                                        │
│  [Your Project - Custom Implementation]                     │
│  ────────────────────────────────────────────────────────   │
│  Curriculum-learning variant of Treechop                    │
│  - Configurable: with_logs (0-64)                          │
│  - Configurable: with_axe (True/False)                     │
│  - Configurable: spawn_type ("random" or "near_tree")      │
│  - Configurable: max_episode_steps                         │
│  - Overrides create_agent_start() for custom inventory     │
└─────────────────────────────────────────────────────────────┘
```

---

## Class Details

### 1. **HumanControlEnvSpec** (MineRL Library)

**Author**: William H. Guss, Brandon Houghton (MineRL authors)
**Location**: `minerl.herobraine.env_specs.human_controls`
**Purpose**: Base class for all human-controllable MineRL environments

**Key Responsibilities**:
- Template pattern for environment specification
- Defines abstract methods that subclasses must implement:
  - `create_rewardables()` - What rewards the agent gets
  - `create_agent_start()` - Agent's starting state (inventory, position)
  - `create_actionables()` - What actions are available
  - `create_server_world_generators()` - World generation settings
  - `create_server_quit_producers()` - Episode termination conditions
  - `create_server_decorators()` - Additional server modifications
  - `create_server_initial_conditions()` - Time, spawning, etc.

**Inheritance Chain**:
```
HumanControlEnvSpec
  ↓
EnvSpec (MineRL base)
  ↓
object
```

---

### 2. **Treechop** (MineRL Library)

**Author**: William H. Guss, Brandon Houghton (MineRL library)
**Location**: `treechop_spec.py:37` (but based on MineRL's original)
**Purpose**: Standard MineRL treechop task specification

**What it defines**:
```python
# Lines 48-53: Rewards
def create_rewardables(self):
    return [handlers.RewardForCollectingItems([
        dict(type="log", amount=1, reward=1.0),  # +1 per log
    ])]

# Lines 55-60: Starting inventory (HARDCODED)
def create_agent_start(self):
    return super().create_agent_start() + [
        handlers.SimpleInventoryAgentStart([
            dict(type="oak_log", quantity=3),  # Always 3 logs
        ])
    ]

# Lines 62-67: Termination condition
def create_agent_handlers(self):
    return [handlers.AgentQuitFromPossessingItem([
        dict(type="log", amount=64)  # Episode ends at 64 logs
    ])]

# Lines 75-80: World generation (Dark Forest biome)
def create_server_world_generators(self):
    return [handlers.DefaultWorldGenerator(
        force_reset="true",
        generator_options=TREECHOP_WORLD_GENERATOR_OPTIONS
    )]

# Lines 82-87: Episode timeout
def create_server_quit_producers(self):
    return [
        handlers.ServerQuitFromTimeUp((TREECHOP_LENGTH * MS_PER_STEP)),
        handlers.ServerQuitWhenAnyAgentFinishes()
    ]
```

**Hardcoded Values**:
- Starting inventory: 3 oak logs
- Goal: 64 logs
- Max episode steps: 8000
- World: Dark Forest biome (ID 11)

---

### 3. **ConfigurableTreechop** (Your Project)

**Author**: Your team (for curriculum learning)
**Location**: `treechop_spec.py:112`
**Purpose**: Flexible treechop variant for curriculum learning

**Constructor Parameters**:
```python
def __init__(
    self,
    spawn_type: str = "random",      # "random" or "near_tree"
    with_logs: int = 0,               # 0-64 starting logs
    with_axe: bool = False,           # Start with wooden axe?
    max_episode_steps: int = None,   # Override default 8000
    *args,
    **kwargs
):
```

**Key Override** (Lines 170-199):
```python
def create_agent_start(self):
    # Bypass Treechop's hardcoded 3 logs
    # Go directly to HumanControlEnvSpec's base implementation
    base_handlers = HumanControlEnvSpec.create_agent_start(self)

    # Build custom inventory based on constructor params
    inventory = []
    if self.with_logs > 0:
        inventory.append(dict(type="oak_log", quantity=self.with_logs))
    if self.with_axe:
        inventory.append(dict(type="wooden_axe", quantity=1))

    if inventory:
        base_handlers.append(
            handlers.SimpleInventoryAgentStart(inventory)
        )

    return base_handlers
```

**Why it bypasses Treechop's create_agent_start()**:
- `Treechop.create_agent_start()` always adds 3 logs
- `ConfigurableTreechop` needs variable starting inventory
- Solution: Call grandparent's method directly (`HumanControlEnvSpec.create_agent_start()`)
- Then add custom inventory based on parameters

---

## Data Flow: How Environment is Created

```
1. User Config (config.yaml)
   └─> curriculum:
         with_logs: 0
         with_axe: True
         spawn_type: "random"

2. env_factory.py
   └─> register_custom_env()
         └─> Creates ConfigurableTreechop instance:
               ConfigurableTreechop(
                   with_logs=0,
                   with_axe=True,
                   spawn_type="random",
                   max_episode_steps=8000
               )

3. ConfigurableTreechop.__init__()
   ├─> Stores params: self.with_logs = 0, self.with_axe = True
   ├─> Sets custom name: "MineRLConfigTreechop_axe-v0"
   └─> Calls super().__init__(**kwargs)
         └─> Treechop.__init__()
               ├─> Sets name if not provided
               ├─> Sets max_episode_steps if not provided
               └─> Calls super().__init__(**kwargs)
                     └─> HumanControlEnvSpec.__init__()
                           └─> Sets up MineRL environment spec

4. When env.reset() is called, MineRL calls:
   └─> create_agent_start()
         └─> ConfigurableTreechop.create_agent_start()
               ├─> Calls HumanControlEnvSpec.create_agent_start() [grandparent]
               ├─> Adds custom inventory:
               │     └─> wooden_axe (quantity=1) [because with_axe=True]
               │     └─> oak_log NOT added [because with_logs=0]
               └─> Returns handlers list to MineRL

5. MineRL starts Minecraft server with:
   - Forest biome world (from Treechop)
   - Agent with wooden axe, no logs (from ConfigurableTreechop)
   - Rewards for collecting logs (from Treechop)
   - Episode termination at 64 logs or timeout (from Treechop)
```

---

## Method Override Summary

| Method | HumanControlEnvSpec | Treechop | ConfigurableTreechop |
|--------|-------------------|----------|---------------------|
| `create_agent_start()` | Base (empty inventory) | Adds 3 logs | Custom inventory (0-64 logs, optional axe) |
| `create_rewardables()` | Abstract | +1 per log | Inherited from Treechop |
| `create_agent_handlers()` | Abstract | Quit at 64 logs | Inherited from Treechop |
| `create_server_world_generators()` | Abstract | Dark Forest | Inherited from Treechop |
| `create_actionables()` | Abstract | Keyboard + camera | Inherited from Treechop |
| `__init__()` | Base setup | Sets defaults | Configurable params |

---

## Why This Design?

### Problem:
- **Treechop** always gives 3 starting logs (hardcoded)
- Need variable starting conditions for curriculum learning

### Solution:
- Create **ConfigurableTreechop** that inherits from **Treechop**
- Override `create_agent_start()` to:
  1. Skip parent's hardcoded 3 logs
  2. Call grandparent directly (`HumanControlEnvSpec`)
  3. Add custom inventory based on constructor params

### Benefits:
- Reuses all of Treechop's other features (rewards, world gen, termination)
- Only changes starting inventory
- Enables curriculum learning: Easy → Medium → Hard

---

## Pre-defined Curriculum Stages

```python
# From treechop_spec.py:221-223

CURRICULUM_EASY = dict(
    spawn_type="random",
    with_logs=0,
    with_axe=True      # Just chop, already have axe
)

CURRICULUM_MEDIUM = dict(
    spawn_type="random",
    with_logs=6,       # Need to craft axe (2 logs → planks → table → 4 logs → sticks + axe)
    with_axe=False
)

CURRICULUM_HARD = dict(
    spawn_type="random",
    with_logs=0,       # Full task from scratch
    with_axe=False
)
```

---

## Summary

1. **HumanControlEnvSpec** (MineRL) - Base class, template pattern
2. **Treechop** (MineRL) - Standard task, hardcoded starting conditions
3. **ConfigurableTreechop** (Your team) - Curriculum variant, flexible starting conditions

**Key Insight**: ConfigurableTreechop carefully bypasses Treechop's hardcoded inventory while inheriting everything else (rewards, world, termination).
