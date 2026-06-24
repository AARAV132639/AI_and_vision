import torch
import torch.nn as nn


class VNT(nn.Module):
    """
    Vector Neuron Translation Module

    Predicts global translation tg.

    Input
    -----
    x : [B,N,3]

    Output
    ------
    tg : [B,3]

    x_centered : [B,N,3]
    """

    def __init__(self):

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.translation_head = nn.Linear(
            256,
            3
        )

    def forward(self, x):

        #
        # x
        # [B,N,3]
        #

        feat = self.mlp(x)

        #
        # [B,N,256]
        #

        global_feat = feat.mean(
            dim=1
        )

        #
        # [B,256]
        #

        tg = self.translation_head(
            global_feat
        )

        #
        # [B,3]
        #

        x_centered = (
            x
            - tg.unsqueeze(1)
        )

        return tg, x_centered