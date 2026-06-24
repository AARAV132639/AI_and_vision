import torch
import torch.nn as nn


class CanonicalConsistencyLoss(nn.Module):
    """
    Encourages different transformed views
    of the same object to map to the same
    canonical representation.

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