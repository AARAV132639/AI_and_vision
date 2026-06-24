import torch
import torch.nn as nn
import torch.nn.functional as F


class VNReLU(nn.Module):
    """
    Vector Neuron ReLU

    Input
    -----
    x : [B,N,C,3]

    Output
    ------
    y : [B,N,C,3]
    """

    def __init__(self, channels):
        super().__init__()

        #
        # learn direction
        #

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

        #
        # compute direction
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
        # inner product
        #

        dot = (
            x * d
        ).sum(
            dim=-1,
            keepdim=True
        )

        #
        # reflection
        #

        reflected = (
            x
            - dot * d
        )

        y = torch.where(
            dot >= 0,
            x,
            reflected
        )

        return y