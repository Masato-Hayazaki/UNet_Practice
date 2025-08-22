# -*- coding: utf-8 -*-
# メトリクス（Dice）

import torch
import numpy as np

@torch.no_grad()
def dice_coeff(pred_logits: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> float:
    # 2値化してDiceを計算
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    target = (target_mask > 0.5).float()
    inter = (pred * target).sum(dim=[1, 2, 3])
    union = pred.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3])
    dice = ((2 * inter + eps) / (union + eps)).mean().item()
    return float(dice)
