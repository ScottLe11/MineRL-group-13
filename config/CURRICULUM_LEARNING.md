# Curriculum Learning Configuration Plan

**Date**: 2025-11-25  
**Purpose**: Define starting position configurations using independent switches that can be combined.

---

## Overview

Starting conditions are controlled by **3 independent switches** that can be combined:

| Switch | Options | Description |
|--------|---------|-------------|
| `spawn_type` | `random` \| `near_tree` | Where agent spawns |
| `with_logs` | `0-10` (integer) | Number of logs in starting inventory |
| `with_axe` | `true` \| `false` | Whether agent starts with wooden axe |

**All combinations are valid.** For example:
- `spawn_type=random, with_logs=0, with_axe=false` → Hardest (pure task)
- `spawn_type=near_tree, with_logs=3, with_axe=true` → Easiest
- `spawn_type=random, with_logs=6, with_axe=false` → Medium (has materials for axe)

---

## Switch Details

### 1. `spawn_type`: Where Agent Spawns

| Value | Description | Implementation Status |
|-------|-------------|----------------------|
| `random` | Default forest biome spawn | ✅ Default behavior |
| `near_tree` | Spawn 5-15 blocks from tree, facing it | ❌ Needs custom handler |

**Note**: `near_tree` requires a custom MineRL handler that finds nearest log block and teleports player. For now, use `random` - forest biome (fixedBiome=11) naturally has many trees.

### 2. `with_logs`: Starting Log Count

| Value | Description | Use Case |
|-------|-------------|----------|
| `0` | No logs (default) | Full task |
| `3` | 3 logs (12 planks worth) | Can craft table + sticks |
| `6` | 6 logs (24 planks worth) | Can craft full axe |
| `10` | 10 logs | Extra materials |

**Implementation**:
```python
handlers.SimpleInventoryAgentStart([
    dict(type="oak_log", quantity=n_logs),
])
```

### 3. `with_axe`: Starting with Axe

| Value | Description | Use Case |
|-------|-------------|----------|
| `false` | No axe (default) | Learn crafting |
| `true` | Wooden axe in slot 0 | Pure collection efficiency |

**Implementation**:
```python
handlers.SimpleInventoryAgentStart([
    dict(type="wooden_axe", quantity=1),
])
```

---

## Configuration Examples

### Config in YAML

```yaml
# config/curriculum/easy.yaml
curriculum:
  spawn_type: "random"      # or "near_tree"
  with_logs: 6              # 0-10
  with_axe: true            # true/false
```

### Difficulty Progression

| Stage | spawn_type | with_logs | with_axe | Goal |
|-------|------------|-----------|----------|------|
| 1 (Easiest) | random | 0 | true | Learn chopping |
| 2 | random | 6 | false | Learn crafting |
| 3 | random | 3 | false | Partial materials |
| 4 | random | 0 | false | Full task |
| 5 (Later) | near_tree | 0 | false | If near_tree implemented |

---

## Implementation

### Single Configurable Environment Class

```python
# In treechop_spec.py or new file

class ConfigurableTreechop(Treechop):
    """
    Treechop environment with configurable starting conditions.
    
    Args:
        spawn_type: "random" or "near_tree"
        with_logs: Number of starting logs (0-10)
        with_axe: Whether to start with wooden axe
    """
    
    def __init__(self, spawn_type="random", with_logs=0, with_axe=False, **kwargs):
        self.spawn_type = spawn_type
        self.with_logs = with_logs
        self.with_axe = with_axe
        super().__init__(**kwargs)
    
    def create_agent_start(self) -> list:
        base_handlers = super().create_agent_start()
        
        # Build inventory list
        inventory = []
        
        if self.with_logs > 0:
            inventory.append(dict(type="oak_log", quantity=self.with_logs))
        
        if self.with_axe:
            inventory.append(dict(type="wooden_axe", quantity=1))
        
        if inventory:
            base_handlers.append(
                handlers.SimpleInventoryAgentStart(inventory)
            )
        
        # Spawn type (near_tree needs custom handler)
        if self.spawn_type == "near_tree":
            # TODO: Custom handler - defer for now
            pass
        
        return base_handlers
```

### Usage

```python
# Create environment with specific config
env_easy = ConfigurableTreechop(
    spawn_type="random",
    with_logs=0,
    with_axe=True
).make()

env_medium = ConfigurableTreechop(
    spawn_type="random", 
    with_logs=6,
    with_axe=False
).make()

env_hard = ConfigurableTreechop(
    spawn_type="random",
    with_logs=0,
    with_axe=False
).make()
```

---

## All Valid Combinations

With `spawn_type` fixed at `random` (since `near_tree` is deferred):

| with_logs | with_axe | Difficulty | Description |
|-----------|----------|------------|-------------|
| 0 | false | Hard | Pure task from scratch |
| 0 | true | Easy | Just need to find trees and chop |
| 3 | false | Medium | Can craft table, sticks, need more logs for axe |
| 3 | true | Easy+ | Has axe + materials to craft another |
| 6 | false | Medium | Full materials for axe craft |
| 6 | true | Very Easy | Has everything |

---

## Questions Resolved

1. **Near-tree spawn**: DEFERRED - forest biome is dense enough
2. **Inventory slot placement**: Items go to first available slot, crafting macros handle this
3. **Combinations**: All 3 switches are independent, any combination valid

---

*Last Updated: 2025-11-25*

