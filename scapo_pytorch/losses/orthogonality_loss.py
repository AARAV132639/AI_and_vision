import torch
import torch.nn as nn


class OrthogonalityLoss(nn.Module):
    """
    Enforces:

        R^T R = I

    for predicted rotation matrices.

    Input
    -----
    R : [B,3,3]

    Output
    ------
    scalar loss
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        R
    ):

        B = R.shape[0]

        I = torch.eye(
            3,
            device=R.device
        ).unsqueeze(0).repeat(
            B,
            1,
            1
        )

        RtR = torch.matmul(
            R.transpose(1, 2),
            R
        )

        loss = (
            (RtR - I) ** 2
        ).mean()

        return loss