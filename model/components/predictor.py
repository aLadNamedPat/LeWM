import torch
import torch.nn as nn
from .adaln import AdaptiveLayerNorm
from .model_blocks import MultiHeadAttention, MLP


class PredictorBlock(nn.Module):
    """Transformer block with AdaLN conditioning on actions."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        action_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        causal: bool = True
    ):
        super().__init__()
        self.adaln1 = AdaptiveLayerNorm(embed_dim, action_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout, causal=causal)
        self.adaln2 = AdaptiveLayerNorm(embed_dim, action_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x, action_embedding):
        """
        Args:
            x: Input sequence of shape (B, N, embed_dim)
            action_embedding: Action embedding of shape (B, action_dim)
        """
        # Apply AdaLN before attention
        x = x + self.attn(self.adaln1(x, action_embedding))
        # Apply AdaLN before MLP
        x = x + self.mlp(self.adaln2(x, action_embedding))
        return x


class Predictor(nn.Module):
    """Transformer-based predictor for forward dynamics in latent space.

    Configuration:
    - 6 layers
    - 16 attention heads
    - 10% dropout
    - AdaLN for action conditioning
    - Causal masking for autoregressive prediction
    - Projection head (1-layer MLP with BatchNorm)

    Takes history of N frame embeddings and predicts next frame embedding.
    """

    def __init__(
        self,
        z_dim: int = 192,
        action_dim: int = 4,
        embed_dim: int = 192,
        n_layers: int = 6,
        n_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        """
        Args:
            z_dim: Dimension of observation embeddings from encoder
            action_dim: Dimension of action space
            embed_dim: Internal embedding dimension for transformer
            n_layers: Number of transformer layers (default: 6)
            n_heads: Number of attention heads (default: 16)
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate (default: 0.1 = 10%)
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        # Input projection: map z_t to embed_dim if different
        self.input_proj = nn.Linear(z_dim, embed_dim) if z_dim != embed_dim else nn.Identity()

        # Action embedding network
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Positional embeddings for temporal information
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer blocks with AdaLN
        self.blocks = nn.ModuleList([
            PredictorBlock(embed_dim, n_heads, embed_dim, mlp_ratio, dropout, causal=True)
            for _ in range(n_layers)
        ])

        # Final normalization (standard LayerNorm, not AdaLN)
        self.norm = nn.LayerNorm(embed_dim)

        # Projection head: map back to z_dim with BatchNorm
        # (prevents LayerNorm from interfering with anti-collapse objective)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.GELU()
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, z_history, actions):
        """
        Args:
            z_history: History of frame embeddings of shape (B, N, z_dim)
                      where N is the sequence length (history length)
            actions: Actions corresponding to transitions, shape (B, N, action_dim)
                    actions[t] is the action taken at timestep t

        Returns:
            z_pred: Predicted next frame embeddings of shape (B, N, z_dim)
                   z_pred[:, t] is the prediction for z_{t+1} given z_{:t+1} and a_t
        """
        B, N, _ = z_history.shape

        # Project input embeddings
        x = self.input_proj(z_history)  # (B, N, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed[:, :N, :]
        x = self.pos_dropout(x)

        # Embed actions
        action_embeddings = self.action_embed(actions)  # (B, N, embed_dim)

        # Apply transformer blocks with action conditioning
        # Note: Each position is conditioned on its corresponding action
        for block in self.blocks:
            # For simplicity, we condition each position on its action
            # Alternative: could pool action_embeddings across sequence
            x_out = []
            for t in range(N):
                if t == 0:
                    x_t = block(x[:, :t+1], action_embeddings[:, t])
                else:
                    x_t = block(x[:, :t+1], action_embeddings[:, t])
                x_out.append(x_t[:, -1:])  # Take last position
            x = torch.cat(x_out, dim=1)  # (B, N, embed_dim)

        x = self.norm(x)

        # Reshape for BatchNorm: (B, N, embed_dim) -> (B*N, embed_dim)
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)

        # Apply projection head with BatchNorm
        z_pred_flat = self.projector(x_flat)  # (B*N, z_dim)

        # Reshape back: (B*N, z_dim) -> (B, N, z_dim)
        z_pred = z_pred_flat.reshape(B, N, self.z_dim)

        return z_pred

    def predict_next(self, z_history, action):
        """
        Predict single next frame given history and action.

        Args:
            z_history: History of shape (B, N, z_dim)
            action: Single action of shape (B, action_dim)

        Returns:
            z_next: Predicted next frame of shape (B, z_dim)
        """
        # Expand action to match sequence dimension
        actions = action.unsqueeze(1).expand(-1, z_history.shape[1], -1)

        # Get predictions for all positions
        z_pred = self.forward(z_history, actions)  # (B, N, z_dim)

        # Return prediction at last position
        return z_pred[:, -1, :]  # (B, z_dim)
