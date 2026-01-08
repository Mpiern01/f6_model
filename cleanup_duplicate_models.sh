#!/bin/bash
# Cleanup script to remove duplicate model paths
# Standardizes on models/Jan-v2-VL-high/

set -e

echo "F6 StreamTrain - Model Path Cleanup"
echo "===================================="

MODEL_DIR="/Users/marcpierne/Desktop/f6_model/models"
STANDARD_PATH="$MODEL_DIR/Jan-v2-VL-high"
OLD_PATH="$MODEL_DIR/janhq_Jan-v2-VL-high"

if [ ! -d "$STANDARD_PATH" ]; then
    echo "ERROR: Standard path not found: $STANDARD_PATH"
    exit 1
fi

if [ ! -d "$OLD_PATH" ]; then
    echo "No duplicate found. System is clean."
    exit 0
fi

echo "Found duplicate model paths:"
echo "  Standard: $STANDARD_PATH"
echo "  Old:      $OLD_PATH"
echo ""

# Check sizes
STANDARD_SIZE=$(du -sh "$STANDARD_PATH" | cut -f1)
OLD_SIZE=$(du -sh "$OLD_PATH" | cut -f1)

echo "Sizes:"
echo "  Standard: $STANDARD_SIZE"
echo "  Old:      $OLD_SIZE"
echo ""

# Ask for confirmation
read -p "Remove old path ($OLD_PATH)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing $OLD_PATH..."
    rm -rf "$OLD_PATH"
    echo "✓ Cleanup complete. System now uses only: $STANDARD_PATH"
else
    echo "Cleanup cancelled."
fi

