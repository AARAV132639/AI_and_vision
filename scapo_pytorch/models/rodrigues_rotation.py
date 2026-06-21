# models/rodrigues_rotation.py

import torch
import torch.nn as nn


class RodriguesRotation(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, axis, angle):
        """
        axis  : [B,P,3]
        angle : [B,P,1]

        returns

        R : [B,P,3,3]
        """

        axis = axis / (
            torch.norm(axis, dim=-1, keepdim=True)
            + 1e-8
        )

        x = axis[..., 0]
        y = axis[..., 1]
        z = axis[..., 2]

        zeros = torch.zeros_like(x)

        K = torch.stack(
            [
                zeros, -z, y,
                z, zeros, -x,
                -y, x, zeros
            ],
            dim=-1
        )

        K = K.view(
            *axis.shape[:-1],
            3,
            3
        )

        I = torch.eye(
            3,
            device=axis.device
        ).view(
            1, 1, 3, 3
        )

        sin_theta = torch.sin(angle)[..., None]
        cos_theta = torch.cos(angle)[..., None]

        R = (
            I
            + sin_theta * K
            + (1 - cos_theta) * (K @ K)
        )

        return R