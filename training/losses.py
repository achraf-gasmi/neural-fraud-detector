"""
FraudShield — Loss Functions
=============================
Focal Loss: handles severe class imbalance without oversampling.
Combined Loss: focal + anomaly reconstruction (weighted sum).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - gamma > 0 reduces the loss for well-classified examples,
      focusing training on hard, misclassified cases.
    - alpha balances positive/negative samples.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1) or (B,) raw model output (before sigmoid)
            targets: (B,) binary labels {0, 1}
        """
        logits = logits.squeeze(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        p_t = torch.exp(-bce)  # probability of correct class
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = torch.where(targets == 1,
                              torch.tensor(self.alpha, device=logits.device),
                              torch.tensor(1 - self.alpha, device=logits.device))

        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedFraudLoss(nn.Module):
    """
    Combined loss = Focal(classification) + λ * MSE(reconstruction)

    The reconstruction term (anomaly head) acts as a regularizer:
    the model must simultaneously classify fraud AND learn what
    normal transactions look like — making it more robust.
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        anomaly_weight: float = 0.3,
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.anomaly_weight = anomaly_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reconstruction: torch.Tensor = None,
        original_features: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            logits: (B, 1) model logits
            targets: (B,) binary labels
            reconstruction: (B, n_features) anomaly head output
            original_features: (B, n_features) input features

        Returns:
            dict with total loss and component losses
        """
        cls_loss = self.focal(logits, targets)

        losses = {
            "focal_loss": cls_loss,
            "anomaly_loss": torch.tensor(0.0, device=logits.device),
            "total_loss": cls_loss,
        }

        if reconstruction is not None and original_features is not None:
            # Only reconstruct on legitimate transactions (index 0 = normal)
            # This way the model learns "normal" more cleanly
            normal_mask = targets == 0
            if normal_mask.sum() > 0:
                recon_loss = F.mse_loss(
                    reconstruction[normal_mask],
                    original_features[normal_mask]
                )
            else:
                recon_loss = F.mse_loss(reconstruction, original_features)

            losses["anomaly_loss"] = recon_loss
            losses["total_loss"] = cls_loss + self.anomaly_weight * recon_loss

        return losses
