import torch
import torch.nn as nn
import timm
from .model_blocks import PatchEmbedding, TransformerBlock


class ViTEncoder(nn.Module):
    """Vision Transformer Encoder for Le World Model.

    Uses ViT-Tiny configuration:
    - Patch size: 14
    - Layers: 12
    - Attention heads: 3
    - Hidden dimension: 192

    The observation embedding z_t is constructed from the [CLS] token
    of the last layer followed by a projection (1-layer MLP with batch norm).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 192,
        n_layers: int = 12,
        n_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        z_dim: int = 192,  # Dimension of observation embedding
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.z_dim = z_dim

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # [CLS] token - learnable parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Projection head: [CLS] token -> observation embedding z_t
        # 1-layer MLP with batch norm
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.GELU()
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize [CLS] token and position embeddings
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x: Input images of shape (B, C, H, W)

        Returns:
            z_t: Observation embeddings of shape (B, z_dim)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, embed_dim)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract [CLS] token from last layer
        cls_output = x[:, 0]  # (B, embed_dim)

        # Project to observation embedding space
        z_t = self.projection(cls_output)  # (B, z_dim)

        return z_t


# ImageNet statistics — DINOv3 ViT-S/16 was pretrained with these
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoEncoder(nn.Module):
    """Pretrained DINOv3 encoder, returning the [CLS] token as z_t.

    Loaded via timm (>=1.0.20). Default model is ViT-S/16 distilled from DINOv3
    ViT-7B on LVD-1689M. No projection head — z_t is the raw 384-d CLS token,
    so the predictor and decoder must be configured at the matching dim.

    Inputs are expected in [0, 1]; ImageNet mean/std normalization is applied
    inside forward() so the dataset/transform pipeline can stay shared with
    the ViT path.

    Note on img_size: DINOv3 was pretrained at 256x256 but we run at 224 to
    avoid touching the dataset. timm auto-interpolates the pretrained position
    embeddings. Revisit if features look weak or training is unstable.
    """

    def __init__(
        self,
        model_id: str = "vit_small_patch16_dinov3.lvd1689m",
        img_size: int = 224,
        z_dim: int = 384,
    ):
        super().__init__()
        self.z_dim = z_dim

        self.backbone = timm.create_model(
            model_id,
            pretrained=True,
            img_size=img_size,
            num_classes=0,  # drop the classifier head
        )

        backbone_dim = self.backbone.num_features
        if backbone_dim != z_dim:
            raise ValueError(
                f"DinoEncoder z_dim={z_dim} does not match backbone "
                f"output dim {backbone_dim} for model {model_id!r}"
            )

        self.register_buffer(
            "_norm_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_norm_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Input images of shape (B, C, H, W) in [0, 1]

        Returns:
            z_t: CLS-token embeddings of shape (B, z_dim)
        """
        x = (x - self._norm_mean) / self._norm_std
        return self.backbone(x)
