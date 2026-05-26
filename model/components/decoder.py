"""Lightweight transformer decoder for visualization only.

The decoder reconstructs images from [CLS] token embeddings to visualize
what information is retained. Gradients are detached to prevent backprop
through encoder/predictor.
"""

import torch
import torch.nn as nn
from .model_blocks import MLP


class CrossAttention(nn.Module):
    """Cross-attention layer for decoder."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == embed_dim

        # Query from learnable tokens, Key/Value from encoded representation
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        Args:
            query: Learnable query tokens (B, N_patches, embed_dim)
            key_value: Encoded representation (B, 1, embed_dim) - the [CLS] token

        Returns:
            output: (B, N_patches, embed_dim)
        """
        B, N_q, C = query.shape
        B, N_kv, C = key_value.shape

        # Project queries
        q = self.q(query).reshape(B, N_q, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Project keys and values
        kv = self.kv(key_value).reshape(B, N_kv, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class DecoderBlock(nn.Module):
    """Transformer decoder block with cross-attention."""

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, query, key_value):
        """
        Args:
            query: Learnable query tokens (B, N_patches, embed_dim)
            key_value: Encoded [CLS] representation (B, 1, embed_dim)
        """
        # Cross-attention with residual
        query = query + self.cross_attn(self.norm1(query), key_value)
        # MLP with residual
        query = query + self.mlp(self.norm2(query))
        return query


class ViTDecoder(nn.Module):
    """Lightweight transformer decoder for visualization.

    Reconstructs images from [CLS] token embeddings. This is used ONLY
    as a diagnostic tool to visualize what visual information is retained.

    Architecture:
    - Projects [CLS] token to hidden dimension (used as key/value)
    - Learnable query tokens (one per output patch)
    - Cross-attention layers with residual MLPs
    - Linear projection to pixel patches
    - Rearrange to RGB image

    IMPORTANT: Gradients are detached in forward pass to prevent backprop
    through encoder/predictor.
    """

    def __init__(
        self,
        cls_dim: int = 192,  # Dimension of [CLS] token from encoder
        hidden_dim: int = 256,  # Hidden dimension for decoder
        n_layers: int = 4,  # Number of decoder layers
        n_heads: int = 8,  # Number of attention heads
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        img_size: int = 224,  # Output image size
        patch_size: int = 16,  # Output patch size
        out_channels: int = 3,  # RGB
    ):
        super().__init__()
        self.cls_dim = cls_dim
        self.hidden_dim = hidden_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        # Number of patches: P = (img_size / patch_size)^2
        self.n_patches = (img_size // patch_size) ** 2

        # Project [CLS] token to hidden dimension (for key/value)
        self.cls_proj = nn.Linear(cls_dim, hidden_dim)

        # Learnable query tokens (one per output patch)
        self.query_tokens = nn.Parameter(torch.zeros(1, self.n_patches, hidden_dim))

        # Positional embeddings for query tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, hidden_dim))

        # Decoder blocks (cross-attention + MLP)
        self.blocks = nn.ModuleList([
            DecoderBlock(hidden_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Project to pixel patches: hidden_dim -> patch_size^2 * out_channels
        patch_pixels = patch_size * patch_size * out_channels
        self.to_pixels = nn.Linear(hidden_dim, patch_pixels)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, cls_embedding, detach: bool = True):
        """
        Reconstruct image from [CLS] token embedding.

        Args:
            cls_embedding: [CLS] token from encoder, shape (B, cls_dim)
            detach: If True, detach gradients to prevent backprop through encoder

        Returns:
            reconstructed_image: RGB image of shape (B, 3, H, W)
        """
        B = cls_embedding.shape[0]

        # IMPORTANT: Detach to prevent gradients flowing to encoder
        if detach:
            cls_embedding = cls_embedding.detach()

        # Project [CLS] to hidden dimension and add batch dimension
        cls_hidden = self.cls_proj(cls_embedding).unsqueeze(1)  # (B, 1, hidden_dim)

        # Initialize query tokens
        queries = self.query_tokens.expand(B, -1, -1)  # (B, n_patches, hidden_dim)
        queries = queries + self.pos_embed

        # Apply decoder blocks
        for block in self.blocks:
            queries = block(queries, cls_hidden)

        queries = self.norm(queries)

        # Project to pixel patches
        patches = self.to_pixels(queries)  # (B, n_patches, patch_size^2 * 3)

        # Reshape to image
        # (B, n_patches, patch_size^2 * 3) -> (B, n_patches, patch_size, patch_size, 3)
        patches = patches.reshape(
            B, self.n_patches, self.patch_size, self.patch_size, self.out_channels
        )

        # Rearrange patches to image
        # Number of patches per side
        patches_per_side = self.img_size // self.patch_size

        # Reshape: (B, n_patches, p, p, 3) -> (B, h, w, p, p, 3)
        patches = patches.reshape(
            B, patches_per_side, patches_per_side,
            self.patch_size, self.patch_size, self.out_channels
        )

        # Rearrange: (B, h, w, p, p, 3) -> (B, h, p, w, p, 3) -> (B, h*p, w*p, 3)
        image = patches.permute(0, 1, 3, 2, 4, 5).reshape(
            B, self.img_size, self.img_size, self.out_channels
        )

        # Convert to (B, C, H, W) format
        image = image.permute(0, 3, 1, 2)

        return image

    def reconstruct_from_z(self, z_embedding, detach: bool = True):
        """
        Convenience method for reconstructing from any embedding.
        Alias for forward().
        """
        return self.forward(z_embedding, detach=detach)
