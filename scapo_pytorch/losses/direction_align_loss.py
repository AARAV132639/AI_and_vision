import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionAlignmentLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        axis,
        motion_dir
    ):
        """
        axis       [B,P,3]
        motion_dir [B,P,3]
        """

        axis = F.normalize(
            axis,
            dim=-1
        )

        motion_dir = F.normalize(
            motion_dir,
            dim=-1
        )

        cosine = (
            axis * motion_dir
        ).sum(
            dim=-1
        )

        loss = (
            1 - cosine.abs()
        ).mean()

        return loss