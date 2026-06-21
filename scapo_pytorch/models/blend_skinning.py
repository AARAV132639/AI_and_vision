# models/blend_skinning.py

import torch
import torch.nn as nn


class BlendSkinning(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        points,      # [B,N,3]
        weights,     # [B,N,P]
        transforms   # [B,P,4,4]
    ):
        """
        Returns

        deformed_points : [B,N,3]
        """

        B, N, _ = points.shape
        P = transforms.shape[1]

        ones = torch.ones(
            B,
            N,
            1,
            device=points.device
        )

        points_h = torch.cat(
            [points, ones],
            dim=-1
        )

        # [B,N,P,4]
        points_h = points_h.unsqueeze(2).repeat(
            1, 1, P, 1
        )

        # [B,N,P,4]
        transformed = torch.matmul(
            transforms.unsqueeze(1),
            points_h.unsqueeze(-1)
        ).squeeze(-1)

        # weighted blending
        weights_exp = weights.unsqueeze(-1)

        blended = (
            transformed * weights_exp
        ).sum(dim=2)

        return blended[..., :3]