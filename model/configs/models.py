from pydantic import BaseModel, Field
from typing import Literal, Optional


class EncoderConfig(BaseModel):
    """Configuration for the encoder.

    The `type` discriminator selects between the in-house ViT and a pretrained
    DINOv3 backbone (loaded via timm). DINO-mode ignores the ViT-shape fields.
    """

    type: Literal["vit", "dino"] = Field(default="vit", description="Encoder backbone")

    # ViT-only fields (unused when type=="dino")
    img_size: int = Field(default=224, description="Input image size")
    patch_size: int = Field(default=14, description="Patch size for ViT")
    in_channels: int = Field(default=3, description="Number of input channels")
    embed_dim: int = Field(default=192, description="Embedding dimension (ViT-Tiny: 192)")
    n_layers: int = Field(default=12, description="Number of transformer layers")
    n_heads: int = Field(default=3, description="Number of attention heads")
    mlp_ratio: float = Field(default=4.0, description="MLP hidden dim ratio")
    dropout: float = Field(default=0.0, description="Dropout rate")

    # Shared output dim — predictor.z_dim / decoder.cls_dim must match.
    z_dim: int = Field(default=192, description="Observation embedding dimension")

    # DINO-only fields (unused when type=="vit")
    dino_model_id: str = Field(
        default="vit_small_patch16_dinov3.lvd1689m",
        description="timm model id for the DINO backbone",
    )

    class Config:
        frozen = True  # Make config immutable


class PredictorConfig(BaseModel):
    """Configuration for Predictor (forward dynamics model)."""

    z_dim: int = Field(default=192, description="Dimension of observation embeddings")
    action_dim: int = Field(default=4, description="Dimension of action space")
    embed_dim: int = Field(default=192, description="Internal embedding dimension")
    n_layers: int = Field(default=6, description="Number of transformer layers")
    n_heads: int = Field(default=16, description="Number of attention heads")
    mlp_ratio: float = Field(default=4.0, description="MLP hidden dimension ratio")
    dropout: float = Field(default=0.1, description="Dropout rate (10%)")
    max_seq_len: int = Field(default=128, description="Maximum sequence length")

    class Config:
        frozen = True


class DecoderConfig(BaseModel):
    """Configuration for Decoder (visualization only)."""

    cls_dim: int = Field(default=192, description="Dimension of [CLS] token from encoder")
    hidden_dim: int = Field(default=256, description="Hidden dimension for decoder")
    n_layers: int = Field(default=4, description="Number of decoder layers")
    n_heads: int = Field(default=8, description="Number of attention heads")
    mlp_ratio: float = Field(default=4.0, description="MLP hidden dimension ratio")
    dropout: float = Field(default=0.0, description="Dropout rate")
    img_size: int = Field(default=224, description="Output image size")
    patch_size: int = Field(default=16, description="Output patch size")
    out_channels: int = Field(default=3, description="Number of output channels (RGB)")

    class Config:
        frozen = True


class ModelConfig(BaseModel):
    """Main model configuration for Le World Model."""

    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    predictor: PredictorConfig = Field(default_factory=PredictorConfig)
    decoder: Optional[DecoderConfig] = Field(default=None, description="Decoder for visualization (optional)")
    device: str = Field(default="cuda", description="Device to run model on")

    class Config:
        frozen = True


class LossConfig(BaseModel):
    """Loss function configuration."""

    lambda_sigreg: float = Field(default=1.0, description="Weight for SIGReg regularization (λ)")
    num_projections: int = Field(default=100, description="Number of random projections M for SIGReg")

    class Config:
        frozen = True


class TrainingConfig(BaseModel):
    """Training configuration."""

    batch_size: int = Field(default=32, description="Training batch size")
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    num_epochs: int = Field(default=100, description="Number of training epochs")
    warmup_steps: int = Field(default=1000, description="Number of warmup steps")
    gradient_clip: float = Field(default=1.0, description="Gradient clipping value")
    seed: int = Field(default=42, description="Random seed")
    encoder_lr_mult: float = Field(
        default=0.1,
        description="Multiplier applied to learning_rate for encoder params. "
                    "Set to 1.0 to disable separate encoder LR.",
    )

    class Config:
        frozen = True


class LWMConfig(BaseModel):
    """Top-level configuration for Le World Model."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    experiment_name: str = Field(default="lwm_experiment", description="Experiment name")
    entity_name: str = Field(default="LeWM_Experiments", description="Entity name")
    project_name: str = Field(default="le-world-model", description="Project name")
    checkpoint_dir: str = Field(default="LeWM_Experiments", description="Checkpoint directory")
    log_dir: str = Field(default="logs", description="Logging directory")

    class Config:
        frozen = True
