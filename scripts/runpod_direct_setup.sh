#!/bin/bash
# Direct setup script for RunPod (no Docker needed!)
# Run this inside a RunPod pod terminal

set -e

echo "🚀 Le World Model - Direct RunPod Setup (No Docker)"

# Install UV
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:${PATH}"
fi

# Clone repo if needed
if [ ! -d "/workspace/le-world-model" ]; then
    echo "📥 Cloning repository..."
    cd /workspace
    git clone https://github.com/aLadNamedPat/LeWM.git le-world-model
    cd le-world-model
else
    echo "✅ Repository already cloned"
    cd /workspace/le-world-model
    git pull
fi

# Set up environment
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export PYTHONPATH="/workspace/le-world-model:${PYTHONPATH}"

# Create necessary directories
mkdir -p /workspace/data
mkdir -p /workspace/checkpoints
mkdir -p /workspace/logs

# Link to persistent volume if available
if [ -d "/runpod-volume" ]; then
    echo "✅ Persistent volume detected at /runpod-volume"
    ln -sfn /runpod-volume/data /workspace/data
    ln -sfn /runpod-volume/checkpoints /workspace/checkpoints
fi

# Install dependencies with UV (super fast!)
echo "📦 Installing dependencies with UV..."
uv pip install --system -e .

# Login to wandb if API key is provided
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "🔐 Logging in to Weights & Biases..."
    wandb login "$WANDB_API_KEY"
fi

# Download dataset if URL is provided
if [ ! -z "$DATASET_URL" ]; then
    echo "📦 Downloading dataset..."
    bash scripts/download_data.sh
else
    echo "⚠️  No DATASET_URL set. Skipping dataset download."
    echo "   Set DATASET_URL environment variable to download automatically."
    echo "   Or manually upload data to /workspace/data/ or /runpod-volume/data/"
fi

# Check GPU
echo "📊 GPU Information:"
nvidia-smi

echo "✅ Setup complete! Ready to train."
echo ""
echo "To start training, run:"
echo "  cd /workspace/le-world-model"
echo "  python train.py"
