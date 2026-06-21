# models/pose_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseNet(nn.Module):

    def __init__(
        self,
        latent_dim=1024,
        num_parts=4
    ):
        super().__init__()

        self.num_parts = num_parts

        self.shared = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.pivot_head = nn.Linear(
            256,
            num_parts * 3
        )

        self.axis_head = nn.Linear(
            256,
            num_parts * 3
        )

        self.angle_head = nn.Linear(
            256,
            num_parts
        )

    def forward(self, z):

        B = z.shape[0]

        feat = self.shared(z)

        pivot = self.pivot_head(feat)

        axis = self.axis_head(feat)

        angle = self.angle_head(feat)

        pivot = pivot.view(
            B,
            self.num_parts,
            3
        )

        axis = axis.view(
            B,
            self.num_parts,
            3
        )

        axis = F.normalize(
            axis,
            dim=-1
        )

        angle = angle.view(
            B,
            self.num_parts,
            1
        )

        return pivot, axis, angle