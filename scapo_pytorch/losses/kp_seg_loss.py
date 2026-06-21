# losses/kp_seg_loss.py

import torch
import torch.nn as nn


class KeypointSegLoss(nn.Module):

    def __init__(
        self,
        threshold=5
    ):
        super().__init__()

        self.threshold = threshold

    def forward(
        self,
        points,
        weights,
        keypoints
    ):
        """
        points    [B,N,3]
        weights   [B,N,P]
        keypoints [B,P,3]
        """

        B,N,P = weights.shape

        loss = 0

        count = 0

        for p in range(P):

            wp = weights[:,:,p]

            support = wp.sum(dim=1)

            centroid = (
                wp.unsqueeze(-1)
                * points
            ).sum(dim=1)

            centroid = centroid / (
                support.unsqueeze(-1)
                + 1e-8
            )

            valid = support > self.threshold

            if valid.any():

                diff = (
                    keypoints[:,p]
                    - centroid
                )

                loss += (
                    diff[valid]
                    .pow(2)
                    .mean()
                )

                count += 1

        return loss / max(count,1)