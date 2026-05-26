"""Loss functions for Le World Model training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def epps_pulley_test(h):
    """
    Epps-Pulley univariate statistical test for normality.

    Tests whether a one-dimensional sample comes from a Gaussian distribution.

    Args:
        h: One-dimensional projections of shape (N*B,) where N is sequence length
           and B is batch size

    Returns:
        T: Test statistic value (scalar)
    """
    n = h.shape[0]
    h = h.unsqueeze(1)  # (n, 1)

    # Pairwise differences: h_i - h_j for all i, j
    diff = h - h.T  # (n, n)

    # Epps-Pulley test statistic computation
    # T(h) = (2/n²) * Σᵢⱼ exp(-|hᵢ - hⱼ|²/2) - 2*(1/n) * Σᵢ exp(-hᵢ²/4) + sqrt(2)

    # First term: (2/n²) * Σᵢⱼ exp(-|hᵢ - hⱼ|²/2)
    term1 = (2.0 / (n * n)) * torch.sum(torch.exp(-diff.pow(2) / 2.0))

    # Second term: 2*(1/n) * Σᵢ exp(-hᵢ²/4)
    term2 = (2.0 / n) * torch.sum(torch.exp(-h.pow(2) / 4.0))

    # Third term: constant
    term3 = math.sqrt(2.0)

    T = term1 - term2 + term3

    return T


class SIGRegLoss(nn.Module):
    """
    Sketched-Isotropic-Gaussian Regularizer (SIGReg) loss.

    Encourages latent embeddings to match an isotropic Gaussian target distribution
    by projecting embeddings onto M random unit-norm directions and optimizing
    the Epps-Pulley test statistic along the resulting 1D projections.

    Prevents representation collapse by promoting feature diversity.
    """

    def __init__(self, num_projections: int = 100, embedding_dim: int = 192):
        """
        Args:
            num_projections: Number of random projections M
            embedding_dim: Dimension of embeddings d
        """
        super().__init__()
        self.num_projections = num_projections
        self.embedding_dim = embedding_dim

        # Pre-generate random unit-norm directions u^(m) ∈ S^(d-1)
        # These are kept fixed during training
        self.register_buffer(
            'projections',
            F.normalize(torch.randn(num_projections, embedding_dim), dim=1)
        )

    def forward(self, Z):
        """
        Compute SIGReg loss.

        Args:
            Z: Latent embeddings of shape (N, B, d) where:
               N = sequence length (history length)
               B = batch size
               d = embedding dimension

        Returns:
            loss: SIGReg loss value (scalar)
        """
        N, B, d = Z.shape

        # Flatten N and B dimensions: (N, B, d) -> (N*B, d)
        Z_flat = Z.reshape(N * B, d)

        # Project embeddings onto M random directions
        # h^(m) = Z @ u^(m), shape: (N*B, M)
        projections = Z_flat @ self.projections.T  # (N*B, M)

        # Compute Epps-Pulley test statistic for each projection
        total_statistic = 0.0
        for m in range(self.num_projections):
            h_m = projections[:, m]  # (N*B,)
            T_m = epps_pulley_test(h_m)
            total_statistic += T_m

        # Average over all projections
        sigreg_loss = total_statistic / self.num_projections

        return sigreg_loss


class PredictionLoss(nn.Module):
    """
    Prediction loss (teacher-forcing) for Le World Model.

    Computes MSE between predicted and actual next-step embeddings:
    L_pred = ||ẑ_{t+1} - z_{t+1}||²
    """

    def __init__(self):
        super().__init__()

    def forward(self, z_pred, z_target):
        """
        Args:
            z_pred: Predicted embeddings ẑ_{t+1} of shape (B, N, d)
            z_target: Target embeddings z_{t+1} of shape (B, N, d)

        Returns:
            loss: Mean squared error (scalar)
        """
        return F.mse_loss(z_pred, z_target)


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for decoder (visualization only).

    Computes MSE between reconstructed and original images.
    IMPORTANT: Input embeddings are detached to prevent gradients
    from flowing to encoder/predictor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, reconstructed, target):
        """
        Args:
            reconstructed: Reconstructed images (B, C, H, W)
            target: Target images (B, C, H, W)

        Returns:
            loss: Mean squared error (scalar)
        """
        return F.mse_loss(reconstructed, target)


class LeWMLoss(nn.Module):
    """
    Complete Le World Model training objective.

    L_LeWM = L_pred + λ * SIGReg(Z)

    Combines prediction loss (teacher-forcing) with SIGReg regularization
    to learn predictable representations while preventing collapse.

    Optionally includes reconstruction loss for decoder (visualization only),
    which does NOT backprop through encoder/predictor.
    """

    def __init__(
        self,
        lambda_sigreg: float = 1.0,
        num_projections: int = 100,
        embedding_dim: int = 192,
        use_reconstruction: bool = False,
        lambda_recon: float = 0.1,
    ):
        """
        Args:
            lambda_sigreg: Weight λ for SIGReg regularization term
            num_projections: Number of random projections M for SIGReg
            embedding_dim: Dimension of embeddings d
            use_reconstruction: Whether to include reconstruction loss
            lambda_recon: Weight for reconstruction loss (only used if use_reconstruction=True)
        """
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.use_reconstruction = use_reconstruction
        self.lambda_recon = lambda_recon

        self.prediction_loss = PredictionLoss()
        self.sigreg_loss = SIGRegLoss(num_projections, embedding_dim)

        if use_reconstruction:
            self.reconstruction_loss = ReconstructionLoss()

    def forward(self, z_pred, z_target, Z_history, reconstructed=None, target_images=None):
        """
        Compute complete LeWM loss.

        Args:
            z_pred: Predicted next embeddings ẑ_{t+1} of shape (B, N, d)
            z_target: Target next embeddings z_{t+1} of shape (B, N, d)
            Z_history: All latent embeddings for regularization, shape (N, B, d)
            reconstructed: Optional reconstructed images (B, C, H, W) for visualization
            target_images: Optional target images (B, C, H, W) for reconstruction loss

        Returns:
            total_loss: Combined loss value
            decoder_loss: Reconstruction loss (or None if not used)
            loss_dict: Dictionary with individual loss components
        """
        # Prediction loss (teacher-forcing)
        L_pred = self.prediction_loss(z_pred, z_target)

        # SIGReg regularization
        L_sigreg = self.sigreg_loss(Z_history)

        # Combined loss (for encoder + predictor)
        total_loss = L_pred + self.lambda_sigreg * L_sigreg

        loss_dict = {
            'total': total_loss.item(),
            'prediction': L_pred.item(),
            'sigreg': L_sigreg.item(),
        }

        # Reconstruction loss (for decoder only, detached from encoder)
        if self.use_reconstruction and reconstructed is not None and target_images is not None:
            L_recon = self.reconstruction_loss(reconstructed, target_images)
            # Add to loss dict but NOT to total_loss (decoder trained separately)
            loss_dict['reconstruction'] = L_recon.item()
            # Return reconstruction loss separately for optional decoder training
            return total_loss, L_recon * self.lambda_recon, loss_dict

        return total_loss, None, loss_dict
