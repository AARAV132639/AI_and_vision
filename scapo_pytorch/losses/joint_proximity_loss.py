# losses/joint_proximity_loss.py

import torch
import torch.nn as nn


class JointProximityLoss(nn.Module):

    def __init__(
        self,
        lam=30
    ):
        super().__init__()

        self.lam = lam

    def forward(
        self,
        pivots,
        points
    ):
        """
        pivots [B,P,3]
        points [B,N,3]
        """

        dist = torch.cdist(
            pivots,
            points
        )

        nearest = dist.min(
            dim=-1
        )[0]

        loss = (
            1
            - torch.exp(
                -self.lam
                * nearest.mean()
            )
        )

        return loss