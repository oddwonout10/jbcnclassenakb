#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="data/circulars"

if [ ! -d "$TARGET_DIR" ]; then
  echo "No $TARGET_DIR directory to clean."
  exit 0
fi

echo "Removing local copies from $TARGET_DIR …"
rm -rf "$TARGET_DIR"
echo "Done."
