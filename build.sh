#!/bin/bash

# -----------------------------
# CONFIGURATION
# -----------------------------
DOCKERFILE_PATH="./.devcontainer/Dockerfile"  # Dockerfile path
IMAGE_NAME="flash_attention_image"           # Docker image name
CONTAINER_NAME="flash_attention_container"   # Container name
HOST_DIR="$(pwd)"                             # Host folder mount
SCRIPT_PATH="Flash_Attention/program.py"          # Python script inside container
OUTPUT_FILE="flash_attention_output.txt"     # Output file inside host folder
# -----------------------------

echo "🔹 Building Docker image..."
docker build -f "$DOCKERFILE_PATH" -t "$IMAGE_NAME" .

echo "🔹 Running container in detached mode..."
docker run -d \
    --name "$CONTAINER_NAME" \
    -v "$HOST_DIR":/workspace \
    "$IMAGE_NAME" \
    bash -c "python $SCRIPT_PATH > $OUTPUT_FILE 2>&1"

echo "✅ Container is running in detached mode."
echo "You can now safely close the terminal."
