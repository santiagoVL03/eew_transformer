"""
Advanced loss functions for earthquake detection.

Includes:
- Focal Loss: Handles class imbalance better than BCE
- Label Smoothing: Prevents overconfidence
- Combined loss strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Focuses learning on hard examples by down-weighting easy examples.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for positive class (0-1)
            gamma: Focusing parameter (gamma >= 0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (batch_size, 1)
            targets: Ground truth labels (batch_size, 1)
        
        Returns:
            Focal loss
        """
        # Compute probability
        probs = torch.sigmoid(inputs)
        
        # Compute BCE loss (without reduction)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute pt (probability of true class)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Compute focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute alpha weight
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Combine: FL = alpha * (1 - pt)^gamma * BCE
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingBCELoss(nn.Module):
    """
    Binary Cross Entropy with Label Smoothing.
    
    Instead of hard targets (0, 1), use soft targets (epsilon, 1-epsilon).
    Prevents overconfident predictions and improves generalization.
    """
    
    def __init__(self, smoothing=0.1, pos_weight=None):
        """
        Args:
            smoothing: Label smoothing factor (0.0 = no smoothing, typical: 0.1)
            pos_weight: Optional positive class weight
        """
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (batch_size, 1)
            targets: Ground truth labels (batch_size, 1)
        
        Returns:
            Label-smoothed BCE loss
        """
        # Apply label smoothing
        # 0 -> epsilon, 1 -> 1 - epsilon
        targets_smooth = targets * (1 - self.smoothing) + self.smoothing / 2
        
        # Compute BCE with smoothed targets
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                inputs, targets_smooth, pos_weight=self.pos_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(inputs, targets_smooth)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: Focal Loss + Label Smoothing.
    
    Combines the benefits of both:
    - Focal loss handles class imbalance
    - Label smoothing prevents overconfidence
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.1, focal_weight=0.7):
        """
        Args:
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            smoothing: Label smoothing factor
            focal_weight: Weight for focal loss (1 - focal_weight for BCE)
        """
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.bce_loss = LabelSmoothingBCELoss(smoothing=smoothing)
        self.focal_weight = focal_weight
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (batch_size, 1)
            targets: Ground truth labels (batch_size, 1)
        
        Returns:
            Combined loss
        """
        focal = self.focal_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        
        return self.focal_weight * focal + (1 - self.focal_weight) * bce


def get_loss_function(loss_type='focal', **kwargs):
    """
    Factory function to create loss function.
    
    Args:
        loss_type: 'focal', 'label_smoothing', 'combined', or 'bce'
        **kwargs: Loss-specific arguments
    
    Returns:
        Loss function
    """
    if loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('alpha', 0.25),
            gamma=kwargs.get('gamma', 2.0)
        )
    elif loss_type == 'label_smoothing':
        return LabelSmoothingBCELoss(
            smoothing=kwargs.get('smoothing', 0.1),
            pos_weight=kwargs.get('pos_weight', None)
        )
    elif loss_type == 'combined':
        return CombinedLoss(
            alpha=kwargs.get('alpha', 0.25),
            gamma=kwargs.get('gamma', 2.0),
            smoothing=kwargs.get('smoothing', 0.1),
            focal_weight=kwargs.get('focal_weight', 0.7)
        )
    elif loss_type == 'bce':
        pos_weight = kwargs.get('pos_weight', None)
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight])
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
