# losses/segmentation_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        weights,
        points,
        keypoints
    ):
        """
        weights   [B,N,P]
        points    [B,N,3]
        keypoints [B,P,3]
        """

        dist = torch.cdist(
            points,
            keypoints
        )

        nearest = dist.argmin(
            dim=-1
        )

        target = F.one_hot(
            nearest,
            num_classes=keypoints.shape[1]
        ).float()

        loss = (
            (weights - target) ** 2
        ).mean()

        return loss