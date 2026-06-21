# losses/cycle_loss.py

import torch
import torch.nn as nn


class CycleLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(
        self,
        reconstructed,
        target
    ):
        return self.loss(
            reconstructed,
            target
        )