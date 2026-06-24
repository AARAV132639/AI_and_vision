import torch
import torch.nn as nn

from models.vnn.vn_linear import VNLinear
from models.vnn.vn_relu import VNReLU
from models.vnn.vn_pool import VNMaxPool


class VNPointNetEncoder(nn.Module):
    """
    Input
    -----
    x : [B,N,3]

    Output
    ------
    z_inv : [B,1024]

    z_vec : [B,1024,3]
    """

    def __init__(self):

        super().__init__()

        self.vn1 = VNLinear(
            in_channels=1,
            out_channels=64
        )

        self.act1 = VNReLU(64)

        self.vn2 = VNLinear(
            64,
            128
        )

        self.act2 = VNReLU(128)

        self.vn3 = VNLinear(
            128,
            1024
        )

        self.act3 = VNReLU(1024)

        self.pool = VNMaxPool(
            1024
        )

    def forward(self, x):

        #
        # [B,N,3]
        #
        x = x.unsqueeze(2)

        #
        # [B,N,1,3]
        #

        x = self.vn1(x)
        x = self.act1(x)

        x = self.vn2(x)
        x = self.act2(x)

        x = self.vn3(x)
        x = self.act3(x)

        #
        # Global vector feature
        #
        z_vec = self.pool(x)

        #
        # [B,1024,3]
        #

        #
        # Invariant magnitude
        #
        z_inv = torch.norm(
            z_vec,
            dim=-1
        )

        #
        # [B,1024]
        #

        return z_inv, z_vec