import torch
import torch.nn as nn


class AugmentationLoss(nn.Module):
    """
    Augmentation Consistency Loss

    Encourages canonical representations
    to remain stable under random
    augmentations.

    Inputs
    ------
    canon_a : [B,N,3]

    canon_b : [B,N,3]

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
        canon_a,
        canon_b
    ):

        return self.loss_fn(
            canon_a,
            canon_b
        )