#!/bin/bash
# Restore the original MineRL JAR file

JAR="/opt/anaconda3/envs/minerl-env/lib/python3.9/site-packages/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar"
BACKUP="${JAR}.backup_original"

echo "=========================================================="
echo "Restoring Original MineRL JAR"
echo "=========================================================="

if [ ! -f "$BACKUP" ]; then
    echo "ERROR: Backup not found at $BACKUP"
    echo "Cannot restore original JAR."
    exit 1
fi

echo "JAR: $JAR"
echo "Backup: $BACKUP"
echo ""
read -p "Restore original JAR? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

cp "$BACKUP" "$JAR"

echo ""
echo "=========================================================="
echo "✓ Original JAR restored successfully!"
echo "=========================================================="
echo ""
echo "Your MineRL environment has been reset to default settings."
