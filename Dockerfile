# Optimized Dockerfile for RunPod deployment with UV
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install UV (blazingly fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Set working directory
WORKDIR /workspace/le-world-model

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY model/ ./model/
COPY configs/ ./configs/

# Install Python dependencies using UV
# UV is much faster than pip and handles dependency resolution better
RUN uv pip install --system -e .

# Create directories for data and outputs
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/logs

# Set default command (can be overridden by RunPod)
CMD ["python3", "train.py"]
