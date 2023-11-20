#!/bin/bash

URL="https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip"
DEST_DIR="$(pwd)"

# Download the zip file with progress
echo "Downloading..."
wget --progress=bar "$URL" -P "$DEST_DIR"

# Unzip the file
echo "Extracting..."
unzip -q "$DEST_DIR/maestro-v2.0.0-midi.zip" -d "$DEST_DIR"

# Remove the compressed folder
rm -f "$DEST_DIR/maestro-v2.0.0-midi.zip"

# Move the desired folder out and delete the intermediate folder
mv "$DEST_DIR/maestro-v2.0.0-midi/maestro-v2.0.0" "$DEST_DIR"
rmdir "$DEST_DIR/maestro-v2.0.0-midi"
