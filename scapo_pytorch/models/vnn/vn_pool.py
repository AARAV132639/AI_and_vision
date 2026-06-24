import torch
import torch.nn as nn
import torch.nn.functional as F


class VNMaxPool(nn.Module):
    """
    Vector Neuron Max Pool

    Input
    -----
    x : [B,N,C,3]

    Output
    ------
    y : [B,C,3]
    """

    def __init__(self, channels):
        super().__init__()

        self.direction = nn.Linear(
            channels,
            channels,
            bias=False
        )

    def forward(self, x):

        #
        # x
        # [B,N,C,3]
        #

        d = x.transpose(-1, -2)

        #
        # [B,N,3,C]
        #

        d = self.direction(d)

        #
        # [B,N,3,C]
        #

        d = d.transpose(-1, -2)

        #
        # [B,N,C,3]
        #

        d = F.normalize(
            d,
            dim=-1,
            eps=1e-6
        )

        #
        # projection score
        #

        score = (
            x * d
        ).sum(
            dim=-1
        )

        #
        # [B,N,C]
        #

        idx = score.argmax(
            dim=1
        )

        #
        # gather max vectors
        #

        B, N, C, _ = x.shape

        idx = idx.unsqueeze(-1).unsqueeze(-1)

        idx = idx.expand(
            B,
            C,
            1,
            3
        )

        x_perm = x.permute(
            0,
            2,
            1,
            3
        )

        pooled = torch.gather(
            x_perm,
            2,
            idx
        )

        pooled = pooled.squeeze(2)

        #
        # [B,C,3]
        #

        return pooled