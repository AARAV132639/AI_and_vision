import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """
    Stage-1 Reconstruction Loss

    Encourages latent representation
    to preserve canonical shape
    information.

    Inputs
    ------
    pred : [B,N,3]

    target : [B,N,3]

    Output
    ------
    scalar loss
    """

    def __init__(
        self,
        loss_type="l1"
    ):
        super().__init__()

        if loss_type == "l1":

            self.loss_fn = nn.L1Loss()

        elif loss_type == "mse":

            self.loss_fn = nn.MSELoss()

        else:

            raise ValueError(
                f"Unknown loss type: {loss_type}"
            )

    def forward(
        self,
        pred,
        target
    ):

        return self.loss_fn(
            pred,
            target
        )