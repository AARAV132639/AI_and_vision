import torch
import torch.nn as nn


class Canonicalizer(nn.Module):
    """
    SCAPO Canonicalization Module

    Equation:

    S_obj = Rg^T (X - tg)

    Inputs
    ------
    points : [B,N,3]

    R : [B,3,3]

    t : [B,3]

    Outputs
    -------
    canonical : [B,N,3]
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        points,
        R,
        t
    ):

        #
        # remove translation
        #

        centered = (
            points
            - t.unsqueeze(1)
        )

        #
        # apply inverse rotation
        #
        # R^-1 = R^T
        #

        canonical = torch.bmm(
            centered,
            R
        )

        return canonical