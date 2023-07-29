# ------------------------------------------------------------------------
# 3DMOTFormer
# Copyright (c) 2023 Shuxiao Ding. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Pytorch Template Project (https://github.com/victoresque/pytorch-template)
# Copyright (c) 2018 Victor Huang. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import sigmoid_focal_loss

class Loss(nn.Module):

    def __init__(self, gamma, velo_loss_weight, normalize_by_positive=False):
        super().__init__()
        self.bce = True
        if gamma is not None:
            self.gamma = gamma
            self.bce = False
        self.velo_loss_weight = velo_loss_weight
        self.normalize_by_positive = normalize_by_positive
    
    def forward(self, prediction, target, pred_velo, target_velo, velo_mask):

        losses = 0.0
        normalizer = target.count_nonzero() if self.normalize_by_positive \
                else target.size(0)
        for pred in prediction:
            if self.bce:
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            else:
                loss = sigmoid_focal_loss(pred, target, alpha=-1, gamma=self.gamma, reduction='none')
            losses += torch.sum(loss) / normalizer

        losses = losses / len(prediction)

        velo_loss = F.smooth_l1_loss(pred_velo, target_velo, reduction='none')
        velo_loss *= velo_mask.float().unsqueeze(1)
        velo_loss = torch.sum(velo_loss) / (velo_loss.size(0) * velo_loss.size(1))
        losses += self.velo_loss_weight * velo_loss

        return losses
    
    def generate_target(self, track_gt, det_gt, edge_index_inter):
        track_idx = edge_index_inter[0, :]
        det_idx = edge_index_inter[1, :]

        track_gt_scattered = track_gt[track_idx]
        det_gt_scattered = det_gt[det_idx]

        track_cond = track_gt_scattered >= 0
        valid_mask = det_gt_scattered >= 0
        valid_mask = torch.logical_and(track_cond, valid_mask)

        target = torch.eq(track_gt_scattered, det_gt_scattered)
        target = torch.logical_and(target, valid_mask)
        return target.float().unsqueeze(1)
