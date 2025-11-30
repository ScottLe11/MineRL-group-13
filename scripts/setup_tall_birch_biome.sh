#!/bin/bash
# Script to modify MineRL JAR to spawn only in tall birch forest
# This ensures the agent always spawns in a dense tree environment

JAR="/opt/anaconda3/envs/minerl-env/lib/python3.9/site-packages/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar"

echo "=================================================="
echo "Setting up Tall Birch Forest spawn for MineRL"
echo "=================================================="
echo "JAR: $JAR"

# Check if JAR exists
if [ ! -f "$JAR" ]; then
    echo "ERROR: JAR file not found at $JAR"
    exit 1
fi

# Create backup
BACKUP="${JAR}.backup_$(date +%Y%m%d_%H%M%S)"
if [ ! -f "${JAR}.backup_original" ]; then
    echo "Creating original backup: ${JAR}.backup_original"
    cp "$JAR" "${JAR}.backup_original"
fi

# Create temporary directory
TMPDIR=$(mktemp -d)
echo "Extracting JAR to: $TMPDIR"

# Extract JAR
unzip -q "$JAR" -d "$TMPDIR"

# Navigate to temp directory
cd "$TMPDIR"

# Create dimension directory
mkdir -p data/minecraft/dimension

# Create overworld.json with tall birch forest biome
echo "Creating tall birch forest dimension configuration..."
cat << 'EOF' > data/minecraft/dimension/overworld.json
{
  "type": "minecraft:overworld",
  "generator": {
    "type": "minecraft:noise",
    "seed": 0,
    "settings": "minecraft:overworld",
    "biome_source": {
      "type": "minecraft:fixed",
      "biome": "minecraft:tall_birch_forest"
    }
  }
}
EOF

# Repackage JAR
echo "Repackaging JAR..."
zip -Xqr "$JAR" .

# Cleanup
cd -
rm -rf "$TMPDIR"

echo "=================================================="
echo "SUCCESS: Tall birch forest biome configured!"
echo "=================================================="
echo "Note: All spawns will now be in tall birch forest"
echo ""
echo "To restore original JAR:"
echo "  cp ${JAR}.backup_original $JAR"
