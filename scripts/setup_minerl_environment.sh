#!/bin/bash
# Combined script to set up MineRL environment for tree-chopping training
# Applies both tall birch forest biome and removes unwanted drops

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo "MineRL Environment Setup for Tree-Chopping Training"
echo "=========================================================="
echo ""
echo "This will:"
echo "  1. Set spawn biome to tall birch forest (dense trees)"
echo "  2. Remove unwanted drops (flowers, seeds, saplings, etc.)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Step 1/2: Setting up tall birch forest biome..."
echo "--------------------------------------------------"
bash "$SCRIPT_DIR/setup_tall_birch_biome.sh"

echo ""
echo "Step 2/2: Removing unwanted drops..."
echo "--------------------------------------------------"
bash "$SCRIPT_DIR/remove_unwanted_drops.sh"

echo ""
echo "=========================================================="
echo "✓ Environment setup complete!"
echo "=========================================================="
echo ""
echo "Your MineRL environment is now optimized for tree-chopping:"
echo "  • All spawns in tall birch forest (dense trees everywhere)"
echo "  • Clean environment (no flower/seed/sapling clutter)"
echo ""
echo "To restore the original JAR, run:"
echo "  bash $SCRIPT_DIR/restore_original_jar.sh"
