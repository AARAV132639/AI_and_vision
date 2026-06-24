import torch
import torch.nn as nn
import torch.nn.functional as F


class RotationHead(nn.Module):
    """
    Predict global rotation matrix.

    Input
    -----
    z_vec : [B,C,3]

    Output
    ------
    R : [B,3,3]
    """

    def __init__(
        self,
        channels=1024
    ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(
                channels * 3,
                512
            ),
            nn.ReLU(),

            nn.Linear(
                512,
                256
            ),
            nn.ReLU(),

            nn.Linear(
                256,
                6
            )
        )

    def forward(
        self,
        z_vec
    ):

        B = z_vec.shape[0]

        #
        # [B,C,3]
        #
        x = z_vec.reshape(
            B,
            -1
        )

        #
        # [B,6]
        #
        x = self.fc(x)

        a1 = x[:, 0:3]
        a2 = x[:, 3:6]

        #
        # Gram-Schmidt
        #

        b1 = F.normalize(
            a1,
            dim=-1
        )

        proj = (
            b1 * a2
        ).sum(
            dim=-1,
            keepdim=True
        )

        b2 = a2 - proj * b1

        b2 = F.normalize(
            b2,
            dim=-1
        )

        b3 = torch.cross(
            b1,
            b2,
            dim=-1
        )

        R = torch.stack(
            [
                b1,
                b2,
                b3
            ],
            dim=-1
        )

        return R