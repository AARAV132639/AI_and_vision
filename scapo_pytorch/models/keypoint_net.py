# models/keypoint_net.py

import torch
import torch.nn as nn


class KeypointNet(nn.Module):

    def __init__(
        self,
        latent_dim=1024,
        num_parts=4
    ):
        super().__init__()

        self.num_parts = num_parts

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(
                256,
                num_parts * 3
            )
        )

    def forward(self, z):

        B = z.shape[0]

        keypoints = self.mlp(z)

        keypoints = keypoints.view(
            B,
            self.num_parts,
            3
        )

        return keypoints