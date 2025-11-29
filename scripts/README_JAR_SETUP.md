# MineRL Environment Setup Scripts

These scripts modify the MineRL JAR file to optimize the environment for tree-chopping training.

## Quick Start

```bash
# Apply all modifications at once
bash scripts/setup_minerl_environment.sh
```

## What This Does

### 1. Tall Birch Forest Biome (`setup_tall_birch_biome.sh`)
- Forces all spawns to occur in tall birch forest biome
- Ensures **dense tree coverage** everywhere (no plains, deserts, etc.)
- Effectively implements "near_tree" spawn without custom handler
- Trees are taller and closer together than normal forest

### 2. Remove Unwanted Drops (`remove_unwanted_drops.sh`)
- Clears loot tables for non-essential blocks
- **Removed drops:**
  - Grass/ferns → no seeds
  - All leaves → no saplings, no apples
  - Dirt variants → no dirt blocks
  - Flowers → no flower drops
  - Cocoa → no cocoa beans
- **Result:** Cleaner inventory, less visual clutter, agent focuses on wood

## Individual Scripts

If you want more control, run scripts separately:

```bash
# Just change biome
bash scripts/setup_tall_birch_biome.sh

# Just remove drops
bash scripts/remove_unwanted_drops.sh
```

## Restoring Original Environment

```bash
# Revert all changes
bash scripts/restore_original_jar.sh
```

## JAR File Location

Your JAR: `/opt/anaconda3/envs/minerl-env/lib/python3.9/site-packages/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar`

## Backup

- First run creates `mcprec-6.13.jar.backup_original`
- Subsequent runs preserve the original backup
- Safe to run scripts multiple times

## Why This Helps Training

1. **Consistent environment**: Every episode has trees nearby
2. **Faster learning**: Agent doesn't waste time searching for trees
3. **Cleaner state space**: Less inventory clutter, simpler observations
4. **Higher sample efficiency**: More relevant training data per episode

## Notes

- Changes persist until you run `restore_original_jar.sh`
- Safe to modify JAR while environment is not running
- **Do not modify** while MineRL is running
- Re-run setup after reinstalling MineRL

## Troubleshooting

**Q: Script says JAR not found**
- Check conda environment is correct: `conda activate minerl-env`
- Verify JAR path in script matches your installation

**Q: Changes not taking effect**
- Ensure you restart Python/MineRL after running scripts
- Check backup was created successfully
- Try running `restore_original_jar.sh` then setup again

**Q: Want to test without modifications**
- Run `restore_original_jar.sh` to revert
- Re-run setup scripts anytime to re-apply
