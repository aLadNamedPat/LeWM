# Le World Model

Implementation of Le World Model - a latent dynamics prediction system using Vision Transformers.

## Architecture

- **Encoder**: ViT-Tiny (~5M params) maps observations to latent embeddings
- **Predictor**: Transformer (~10M params) with AdaLN for action-conditioned dynamics
- **Decoder**: Lightweight transformer for visualization (optional)

## Training Objective

L_LeWM = L_pred + λ * SIGReg(Z)

- **L_pred**: Prediction loss (teacher-forcing MSE)
- **SIGReg**: Sketched-Isotropic-Gaussian Regularizer (prevents collapse)

## Setup with UV

### Local Development

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project
uv pip install -e .

# Install dev dependencies
uv pip install -e ".[dev]"
```

## RunPod Deployment

### 1. Build and Push Docker Image

```bash
# Set your Docker Hub username
export DOCKER_USERNAME="your-dockerhub-username"

# Build and push
chmod +x scripts/build_and_push.sh
./scripts/build_and_push.sh
```

### 2. Create RunPod Template

1. Go to [RunPod Templates](https://www.runpod.io/console/templates)
2. Create new template with:
   - **Container Image**: `your-dockerhub-username/le-world-model:latest`
   - **Docker Command**: Leave empty (uses default from Dockerfile)
   - **Container Disk**: 20GB
   - **Volume Mount Path**: `/runpod-volume`

### 3. Launch Pod

```bash
# On RunPod pod terminal
bash /workspace/le-world-model/scripts/runpod_setup.sh

# Start training
cd /workspace/le-world-model
python train.py
```

### 4. Environment Variables (Set in RunPod)

- `WANDB_API_KEY`: Your Weights & Biases API key
- `WORLD_SIZE`: (Auto-set for multi-GPU)
- `RANK`: (Auto-set for multi-GPU)
- `LOCAL_RANK`: (Auto-set for multi-GPU)

## Configuration

Configurations are managed with Hydra. See `configs/` directory.

### Training Config

```yaml
# configs/training/default.yaml
batch_size: 32
learning_rate: 1e-4
num_epochs: 100
gradient_clip: 1.0
```

### Model Config

```yaml
# configs/model/default.yaml
encoder:
  z_dim: 192
  n_layers: 12
  n_heads: 3

predictor:
  n_layers: 6
  n_heads: 16
  dropout: 0.1
```

## Project Structure

```
le-world-model/
├── model/
│   ├── components/          # Model building blocks
│   │   ├── encoder.py       # ViT encoder
│   │   ├── predictor.py     # Forward dynamics predictor
│   │   ├── decoder.py       # Visualization decoder
│   │   ├── adaln.py         # Adaptive LayerNorm
│   │   └── model_blocks.py  # Shared components
│   ├── configs/             # Pydantic schemas
│   │   └── models.py
│   ├── LWM.py               # Main model class
│   └── loss.py              # Loss functions
├── configs/                 # Hydra configs
│   ├── config.yaml
│   ├── model/
│   ├── training/
│   └── loss/
├── scripts/                 # Deployment scripts
│   ├── build_and_push.sh
│   └── runpod_setup.sh
├── train.py                 # Training script
├── pyproject.toml           # UV dependencies
├── Dockerfile               # RunPod container
└── README.md
```

## Cost Optimization Tips

1. **Use Spot Instances**: 70-80% cheaper than on-demand
2. **Save checkpoints frequently**: Resume after preemption
3. **Use Network Volumes**: Persist data across pod restarts
4. **Multi-GPU**: Use DDP for faster training on multiple GPUs

## License

This is an implementation for research and educational purposes. Please refer to the original paper for licensing information.
