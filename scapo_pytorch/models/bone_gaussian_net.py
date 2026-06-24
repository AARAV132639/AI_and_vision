import torch
import torch.nn as nn
import torch.nn.functional as F


class BoneGaussianNet(nn.Module):

    def __init__(
        self,
        latent_dim=1024,
        num_parts=4
    ):
        super().__init__()

        self.num_parts = num_parts

        self.shared = nn.Sequential(
            nn.Linear(latent_dim,512),
            nn.ReLU(),

            nn.Linear(512,256),
            nn.ReLU()
        )

        #
        # orientation matrix
        #

        self.rotation_head = nn.Linear(
            256,
            num_parts * 9
        )

        #
        # scale values
        #

        self.scale_head = nn.Linear(
            256,
            num_parts * 3
        )

    def forward(self,z):

        B = z.shape[0]

        feat = self.shared(z)

        V = self.rotation_head(feat)

        V = V.view(
            B,
            self.num_parts,
            3,
            3
        )

        #
        # orthogonalize
        #

        Qr,_ = torch.linalg.qr(V)

        scales = self.scale_head(feat)

        scales = F.softplus(
            scales
        ) + 1e-4

        scales = scales.view(
            B,
            self.num_parts,
            3
        )

        Lambda = torch.diag_embed(
            scales
        )

        Q = (
            Qr.transpose(-1,-2)
            @ Lambda
            @ Qr
        )

        return Q