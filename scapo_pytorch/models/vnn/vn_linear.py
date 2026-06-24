import torch
import torch.nn as nn


class VNLinear(nn.Module):
    """
    Vector Neuron Linear Layer

    Input
    -----
    x : [B,N,Cin,3]

    Output
    ------
    y : [B,N,Cout,3]

    SO(3)-equivariant because
    weights act only on channels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels
            ) * 0.01
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    out_channels,
                    3
                )
            )
        else:
            self.bias = None

    def forward(self, x):

        #
        # x
        # [B,N,Cin,3]
        #

        y = torch.einsum(
            "oi,bnig->bnog",
            self.weight,
            x
        )

        if self.bias is not None:

            y = (
                y
                + self.bias.view(
                    1,
                    1,
                    self.out_channels,
                    3
                )
            )

        return y