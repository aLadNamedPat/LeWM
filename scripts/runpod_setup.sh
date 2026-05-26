#!/bin/bash
# Setup script to run inside RunPod container on startup

set -e

echo "🚀 Le World Model - RunPod Setup"

# Set up environment
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export PYTHONPATH="/workspace/le-world-model:${PYTHONPATH}"

# Create necessary directories
mkdir -p /workspace/data
mkdir -p /workspace/checkpoints
mkdir -p /workspace/logs

# Check GPU availability
echo "📊 GPU Information:"
nvidia-smi

# Check if data volume is mounted
if [ -d "/runpod-volume" ]; then
    echo "✅ Persistent volume detected at /runpod-volume"
    ln -sfn /runpod-volume/data /workspace/data
    ln -sfn /runpod-volume/checkpoints /workspace/checkpoints
else
    echo "⚠️  No persistent volume found. Using local storage."
fi

# Login to wandb if API key is provided
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "🔐 Logging in to Weights & Biases..."
    wandb login "$WANDB_API_KEY"
else
    echo "⚠️  WANDB_API_KEY not set. Running without W&B logging."
fi

echo "✅ Setup complete!"
