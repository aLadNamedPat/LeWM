import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from .components.encoder import ViTEncoder
from .components.dino_encoder import DinoEncoder
from .components.predictor import Predictor
from .components.decoder import ViTDecoder
from .configs.models import ModelConfig


class LWM(nn.Module):
    """Le World Model main class.

    Components:
    - Encoder: Maps observations o_t to latent embeddings z_t
    - Predictor: Predicts next latent z_{t+1} given history and action
    - Decoder (optional): Reconstructs images from z_t for visualization

    Forward dynamics: z_{t+1} = predictor(z_{:t+1}, a_t)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Initialize encoder (ViT or DINOv2)
        if config.encoder.encoder_type == "dino":
            self.encoder = DinoEncoder(
                model_name=config.encoder.dino_model,
                z_dim=config.encoder.z_dim,
                freeze_backbone=config.encoder.freeze_backbone,
            )
        else:  # vit
            self.encoder = ViTEncoder(
                img_size=config.encoder.img_size,
                patch_size=config.encoder.patch_size,
                in_channels=config.encoder.in_channels,
                embed_dim=config.encoder.embed_dim,
                n_layers=config.encoder.n_layers,
                n_heads=config.encoder.n_heads,
                mlp_ratio=config.encoder.mlp_ratio,
                dropout=config.encoder.dropout,
                z_dim=config.encoder.z_dim,
            )

        # Initialize predictor (forward dynamics model)
        self.predictor = Predictor(
            z_dim=config.predictor.z_dim,
            action_dim=config.predictor.action_dim,
            embed_dim=config.predictor.embed_dim,
            n_layers=config.predictor.n_layers,
            n_heads=config.predictor.n_heads,
            mlp_ratio=config.predictor.mlp_ratio,
            dropout=config.predictor.dropout,
            max_seq_len=config.predictor.max_seq_len,
        )

        # Initialize decoder (optional, for visualization only)
        self.decoder = None
        if config.decoder is not None:
            self.decoder = ViTDecoder(
                cls_dim=config.decoder.cls_dim,
                hidden_dim=config.decoder.hidden_dim,
                n_layers=config.decoder.n_layers,
                n_heads=config.decoder.n_heads,
                mlp_ratio=config.decoder.mlp_ratio,
                dropout=config.decoder.dropout,
                img_size=config.decoder.img_size,
                patch_size=config.decoder.patch_size,
                out_channels=config.decoder.out_channels,
            )

    def encode(self, observations):
        """Encode observations into latent embeddings z_t.

        Args:
            observations: Batch of observations (B, C, H, W)

        Returns:
            z_t: Latent embeddings (B, z_dim)
        """
        return self.encoder(observations)

    def predict(self, z_history, actions):
        """Predict next latent states given history and actions.

        Args:
            z_history: History of latent embeddings (B, N, z_dim)
            actions: Actions taken (B, N, action_dim)

        Returns:
            z_pred: Predicted next latents (B, N, z_dim)
        """
        return self.predictor(z_history, actions)

    def predict_next(self, z_history, action):
        """Predict single next latent given history and action.

        Args:
            z_history: History of latents (B, N, z_dim)
            action: Single action (B, action_dim)

        Returns:
            z_next: Predicted next latent (B, z_dim)
        """
        return self.predictor.predict_next(z_history, action)

    def decode(self, z_embedding, detach: bool = True):
        """Reconstruct image from latent embedding (visualization only).

        Args:
            z_embedding: Latent embedding (B, z_dim)
            detach: If True, detach gradients to prevent backprop through encoder

        Returns:
            reconstructed_image: RGB image (B, 3, H, W)

        Note:
            Requires decoder to be initialized. Gradients are detached by default
            to prevent reconstruction loss from flowing to encoder/predictor.
        """
        if self.decoder is None:
            raise RuntimeError("Decoder not initialized. Set decoder config to use this method.")
        return self.decoder(z_embedding, detach=detach)

    def forward(self, observations, actions=None):
        """Forward pass through the model.

        Args:
            observations: Batch of observations (B, C, H, W) or sequence (B, N, C, H, W)
            actions: Optional actions (B, N, action_dim) for prediction

        Returns:
            If actions is None: returns z_t (B, z_dim)
            If actions provided: returns (z_t, z_pred) where z_pred is (B, N, z_dim)
        """
        # Handle single observation vs sequence
        if observations.dim() == 4:
            # Single observation: (B, C, H, W)
            z_t = self.encode(observations)
            return z_t
        elif observations.dim() == 5:
            # Sequence of observations: (B, N, C, H, W)
            B, N = observations.shape[:2]
            # Flatten and encode
            obs_flat = observations.view(B * N, *observations.shape[2:])
            z_flat = self.encode(obs_flat)
            z_sequence = z_flat.view(B, N, -1)

            if actions is not None:
                # Predict next states
                z_pred = self.predict(z_sequence, actions)
                return z_sequence, z_pred
            return z_sequence
        else:
            raise ValueError(f"Invalid observation shape: {observations.shape}")

    @classmethod
    def from_hydra_config(cls, cfg: DictConfig):
        """Create model from Hydra config."""
        # Convert OmegaConf to dict and validate with Pydantic
        config_dict = OmegaConf.to_container(cfg.model, resolve=True)
        model_config = ModelConfig(**config_dict)
        return cls(model_config)
