#!/bin/bash
# Download and extract Le World Model dataset from Hugging Face

set -e

DATA_DIR="${DATA_DIR:-/workspace/data}"
HF_DATASET="${HF_DATASET:-quentinll/lewm-tworooms}"

echo "📦 Downloading Le World Model dataset from Hugging Face..."

# Create data directory
mkdir -p "$DATA_DIR"

# Check if data already exists
if [ -f "$DATA_DIR/.data_ready" ]; then
    echo "✅ Data already exists, skipping download"
    exit 0
fi

# Install Hugging Face Hub CLI
echo "📦 Installing Hugging Face Hub..."
pip install -q huggingface-hub[cli]

# Download dataset using hf CLI
echo "⬇️  Downloading from: $HF_DATASET"
cd "$DATA_DIR"

# Download all files from the dataset
hf download "$HF_DATASET" --repo-type dataset --local-dir .

# Check if tar.zst file exists and extract
if ls *.tar.zst 1> /dev/null 2>&1; then
    echo "📂 Found compressed archive, extracting..."

    # Install zstd if needed
    if ! command -v zstd &> /dev/null; then
        echo "📦 Installing zstd..."
        apt-get update && apt-get install -y zstd
    fi

    # Extract
    for file in *.tar.zst; do
        echo "Extracting $file..."
        tar -I zstd -xvf "$file"
    done
fi

# Mark as ready
touch "$DATA_DIR/.data_ready"

echo "✅ Dataset ready at: $DATA_DIR"
echo "Contents:"
ls -lh "$DATA_DIR"
