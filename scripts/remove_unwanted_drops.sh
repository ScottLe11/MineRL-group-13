#!/bin/bash
# Script to remove unwanted item drops from MineRL environment
# Removes: flowers, saplings, apples, dirt blocks, seeds, cocoa beans
# This keeps the environment clean and focused on tree-chopping

JAR="/opt/anaconda3/envs/minerl-env/lib/python3.9/site-packages/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar"

echo "=================================================="
echo "Removing unwanted item drops from MineRL"
echo "=================================================="
echo "JAR: $JAR"

# Check if JAR exists
if [ ! -f "$JAR" ]; then
    echo "ERROR: JAR file not found at $JAR"
    exit 1
fi

# Create backup
if [ ! -f "${JAR}.backup_original" ]; then
    echo "Creating original backup: ${JAR}.backup_original"
    cp "$JAR" "${JAR}.backup_original"
fi

# Create temporary directory
TMPDIR=$(mktemp -d)
echo "Extracting JAR to: $TMPDIR"

# Extract JAR
unzip -q "$JAR" -d "$TMPDIR"

# Navigate to loot tables directory
cd "$TMPDIR/data/minecraft/loot_tables/blocks"

echo "Clearing loot tables for unwanted drops..."

# List of blocks to clear
# - grass, tall_grass, fern, large_fern: Remove seeds
# - cocoa: Remove cocoa beans
# - all leaves: Remove saplings and apples
# - dirt variants: Remove dirt blocks
# - flowers: Remove flower drops
BLOCKS=(
    # Grass variants (drop seeds)
    "grass"
    "tall_grass"
    "fern"
    "large_fern"

    # Cocoa
    "cocoa"

    # Leaves (drop saplings and apples)
    "oak_leaves"
    "birch_leaves"
    "spruce_leaves"
    "jungle_leaves"
    "acacia_leaves"
    "dark_oak_leaves"

    # Dirt variants
    "dirt"
    "coarse_dirt"
    "podzol"
    "grass_block"

    # Flowers
    "dandelion"
    "poppy"
    "lily_of_the_valley"
    "lilac"
    "peony"
    "rose_bush"
)

# Clear loot table for each block
for block in "${BLOCKS[@]}"; do
    if [ -f "${block}.json" ]; then
        printf '{ "type": "minecraft:block", "pools": [] }\n' > "${block}.json"
        echo "  ✓ Cleared: ${block}.json"
    else
        echo "  ⚠ Not found: ${block}.json (may not exist in this version)"
    fi
done

# Return to temp directory root
cd "$TMPDIR"

# Repackage JAR
echo ""
echo "Repackaging JAR..."
zip -Xqr "$JAR" .

# Cleanup
cd -
rm -rf "$TMPDIR"

echo "=================================================="
echo "SUCCESS: Unwanted drops removed!"
echo "=================================================="
echo "Removed drops from:"
echo "  - Grass/ferns (no seeds)"
echo "  - Leaves (no saplings/apples)"
echo "  - Dirt (no dirt blocks)"
echo "  - Flowers (no flower drops)"
echo "  - Cocoa (no cocoa beans)"
echo ""
echo "To restore original JAR:"
echo "  cp ${JAR}.backup_original $JAR"
