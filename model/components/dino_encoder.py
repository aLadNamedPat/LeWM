"""DINOv2-based encoder for Le World Model."""

import torch
import torch.nn as nn


class DinoEncoder(nn.Module):
    """DINOv2 Vision Transformer Encoder for Le World Model.

    Uses pre-trained DINOv2 (not frozen) and adds a projection head.
    The observation embedding z_t is constructed from the [CLS] token
    followed by a projection (1-layer MLP with batch norm).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",  # vits14, vitb14, vitl14, vitg14
        z_dim: int = 192,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            model_name: DINOv2 model variant (dinov2_vits14, dinov2_vitb14, etc.)
            z_dim: Output dimension for observation embeddings
            freeze_backbone: If True, freeze DINOv2 weights (original DinoWM)
        """
        super().__init__()

        # Load pre-trained DINOv2
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        self.embed_dim = self.backbone.embed_dim  # 384 for vits14, 768 for vitb14
        self.z_dim = z_dim

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head: embed_dim -> z_dim
        # Same as ViT encoder: Linear + BatchNorm + GELU
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x: Input images (B, C, H, W)

        Returns:
            z: Observation embeddings (B, z_dim) from [CLS] token
        """
        B = x.shape[0]

        # Get [CLS] token from DINOv2
        # DINOv2 forward returns dict with 'x_norm_clstoken' key
        with torch.set_grad_enabled(self.training):
            features = self.backbone.forward_features(x)
            cls_token = features['x_norm_clstoken']  # (B, embed_dim)

        # Project to z_dim
        z = self.projection(cls_token)  # (B, z_dim)

        return z
