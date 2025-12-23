import os
import random

import numpy as np
import torch


def seed_everything(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_iou(pred_logits, masks, threshold=0.0):
    """
    Calculates Intersection over Union (IoU) for binary segmentation.

    Args:
        pred_logits (torch.Tensor): Raw model output [B, 1, H, W]
        masks (torch.Tensor): Ground truth [B, 1, H, W]
        threshold (float): Threshold for logits (0.0 implies sigmoid=0.5)
    """
    pred_mask = (pred_logits > threshold).float()
    masks = (masks > 0.5).float()

    intersection = (pred_mask * masks).sum(dim=(1, 2, 3))
    union = (pred_mask + masks).sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + 1e-7) / (union + 1e-7)

    return iou.mean().item()
