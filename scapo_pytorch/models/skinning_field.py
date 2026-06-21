# models/skinning_field.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkinningField(nn.Module):
    """
    Converts Mahalanobis distances into
    soft skinning weights.

    Input
    -----
    distance : [B, N, P]

    Output
    ------
    weights  : [B, N, P]

    Sum along P = 1
    """

    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, distance):

        weights = F.softmax(
            -distance / self.gamma,
            dim=-1
        )

        return weights