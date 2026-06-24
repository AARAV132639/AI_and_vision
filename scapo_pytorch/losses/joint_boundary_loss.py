import torch
import torch.nn as nn


class JointBoundaryLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pivots,
        keypoints
    ):
        """
        pivots     [B,P,3]
        keypoints  [B,P,3]
        """

        dist = (
            pivots
            - keypoints
        ).norm(
            dim=-1
        )

        return dist.mean()