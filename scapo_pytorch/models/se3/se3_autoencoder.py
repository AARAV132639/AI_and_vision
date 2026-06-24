import torch
import torch.nn as nn

from models.se3.vnt import VNT
from models.se3.rotation_head import RotationHead
from models.se3.canonicalizer import Canonicalizer

from models.vnn.vn_pointnet import VNPointNetEncoder


class SE3AutoEncoder(nn.Module):
    """
    Stage-1 Canonicalization Module

    Input
    -----
    points : [B,N,3]

    Output
    ------
    dict containing:

    canonical : [B,N,3]

    tg : [B,3]

    Rg : [B,3,3]

    z_inv : [B,1024]

    z_vec : [B,1024,3]
    """

    def __init__(self):

        super().__init__()

        self.vnt = VNT()

        self.encoder = VNPointNetEncoder()

        self.rotation_head = RotationHead()

        self.canonicalizer = Canonicalizer()

    def forward(
        self,
        points
    ):

        #
        # ------------------------------------------------
        # Translation Estimation
        # ------------------------------------------------
        #

        tg, centered = self.vnt(
            points
        )

        #
        # ------------------------------------------------
        # Equivariant Encoding
        # ------------------------------------------------
        #

        z_inv, z_vec = self.encoder(
            centered
        )

        #
        # ------------------------------------------------
        # Rotation Estimation
        # ------------------------------------------------
        #

        Rg = self.rotation_head(
            z_vec
        )

        #
        # ------------------------------------------------
        # Canonicalization
        # ------------------------------------------------
        #

        canonical = self.canonicalizer(
            points,
            Rg,
            tg
        )

        return {

            "canonical": canonical,

            "translation": tg,

            "rotation": Rg,

            "z_inv": z_inv,

            "z_vec": z_vec
        }