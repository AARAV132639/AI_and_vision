import torch
import torch.nn as nn


class ShapeVarianceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        keypoints
    ):
        """
        keypoints [B,P,3]
        """

        dist = torch.cdist(
            keypoints,
            keypoints
        )

        P = keypoints.shape[1]

        mask = (
            1
            - torch.eye(
                P,
                device=keypoints.device
            )
        )

        mean_dist = (
            dist * mask
        ).sum() / (
            mask.sum()
            * keypoints.shape[0]
        )

        loss = 1.0 / (
            mean_dist + 1e-6
        )

        return loss