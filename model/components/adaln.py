import torch
import torch.nn as nn


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on actions.

    AdaLN modulates the scale and shift parameters of LayerNorm based on
    action input, allowing the model to condition on actions at each layer.

    Parameters are initialized to zero to stabilize training and ensure
    action conditioning impacts training progressively.
    """

    def __init__(self, normalized_shape: int, action_dim: int):
        """
        Args:
            normalized_shape: Dimension to normalize (typically embed_dim)
            action_dim: Dimension of action input
        """
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, elementwise_affine=False)

        # MLP to generate scale and shift from action
        # Initialized to zero for stable training
        self.action_proj = nn.Linear(action_dim, normalized_shape * 2)
        nn.init.zeros_(self.action_proj.weight)
        nn.init.zeros_(self.action_proj.bias)

        # Learnable default scale and shift when no action conditioning
        self.scale = nn.Parameter(torch.ones(normalized_shape))
        self.shift = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x, action_embedding):
        """
        Args:
            x: Input tensor of shape (B, N, D) where N is sequence length
            action_embedding: Action embedding of shape (B, action_dim)

        Returns:
            Modulated output of shape (B, N, D)
        """
        # Standard layer normalization
        x_norm = self.ln(x)

        # Generate scale and shift from action
        action_params = self.action_proj(action_embedding)  # (B, 2*D)
        action_scale, action_shift = action_params.chunk(2, dim=-1)  # Each (B, D)

        # Add 1 to scale (so initial zero weights give scale=1)
        # Reshape for broadcasting: (B, 1, D)
        scale = (1 + action_scale).unsqueeze(1) * self.scale
        shift = action_shift.unsqueeze(1) + self.shift

        # Apply adaptive scaling and shifting
        return scale * x_norm + shift
