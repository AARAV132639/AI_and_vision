# models/bone_transform.py

import torch
import torch.nn as nn

from models.rodrigues_rotation import RodriguesRotation


class BoneTransform(nn.Module):

    def __init__(self):
        super().__init__()

        self.rotation_layer = RodriguesRotation()

    def forward(
        self,
        pivot,     # [B,P,3]
        axis,      # [B,P,3]
        angle      # [B,P,1]
    ):
        """
        Returns

        T : [B,P,4,4]
        """

        B, P, _ = pivot.shape

        R = self.rotation_layer(
            axis,
            angle
        )

        T = torch.eye(
            4,
            device=pivot.device
        ).view(
            1,1,4,4
        ).repeat(
            B,P,1,1
        )

        T[..., :3, :3] = R

        #
        # translation component
        #
        # t = c - R*c
        #

        Rc = torch.matmul(
            R,
            pivot.unsqueeze(-1)
        ).squeeze(-1)

        t = pivot - Rc

        T[..., :3, 3] = t

        return T