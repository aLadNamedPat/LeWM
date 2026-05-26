#!/bin/bash
# Download and extract Le World Model dataset

set -e

DATA_DIR="${DATA_DIR:-/workspace/data}"
DATASET_URL="${DATASET_URL:-}"  # Set this to your download URL

echo "📦 Downloading Le World Model dataset..."

# Create data directory
mkdir -p "$DATA_DIR"

if [ -z "$DATASET_URL" ]; then
    echo "❌ Error: DATASET_URL environment variable not set"
    echo "Set it to your Google Drive/S3/Hugging Face URL"
    exit 1
fi

# Check if data already exists
if [ -d "$DATA_DIR/two_rooms" ] || [ -f "$DATA_DIR/.data_ready" ]; then
    echo "✅ Data already exists, skipping download"
    exit 0
fi

# Install dependencies
if ! command -v zstd &> /dev/null; then
    echo "📦 Installing zstd..."
    apt-get update && apt-get install -y zstd wget
fi

# Download dataset
echo "⬇️  Downloading from: $DATASET_URL"
cd /tmp

# For Google Drive (use gdown)
if [[ "$DATASET_URL" == *"drive.google.com"* ]]; then
    pip install -q gdown
    gdown "$DATASET_URL" -O dataset.tar.zst
# For direct URLs
else
    wget "$DATASET_URL" -O dataset.tar.zst
fi

# Extract
echo "📂 Extracting dataset..."
tar -I zstd -xvf dataset.tar.zst -C "$DATA_DIR"

# Mark as ready
touch "$DATA_DIR/.data_ready"

# Cleanup
rm -f /tmp/dataset.tar.zst

echo "✅ Dataset ready at: $DATA_DIR"
ls -lh "$DATA_DIR"
