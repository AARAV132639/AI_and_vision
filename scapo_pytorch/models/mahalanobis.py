# models/mahalanobis.py

import torch
import torch.nn as nn


class MahalanobisDistance(nn.Module):
    """
    Computes Mahalanobis distance between points and bones.

    Paper Eq. (6):
        W_i^p = (s_i - O_p)^T Q_p (s_i - O_p)

    Inputs
    ------
    points  : [B, N, 3]
        N points in the point cloud

    centers : [B, P, 3]
        Bone centers / keypoints

    Q       : [B, P, 3, 3]
        Precision matrices

    Outputs
    -------
    distance : [B, N, P]
        Mahalanobis distance of every point to every bone
    """

    def __init__(self):
        super().__init__()

    def forward(self, points, centers, Q):

        B, N, _ = points.shape
        _, P, _ = centers.shape

        # [B, N, 1, 3]
        points_exp = points.unsqueeze(2)

        # [B, 1, P, 3]
        centers_exp = centers.unsqueeze(1)

        # [B, N, P, 3]
        diff = points_exp - centers_exp

        # Compute:
        # diff^T * Q * diff

        # [B, N, P, 3]
        temp = torch.einsum(
            "bnpj,bpji->bnpi",
            diff,
            Q
        )

        # [B, N, P]
        distance = torch.einsum(
            "bnpi,bnpi->bnp",
            temp,
            diff
        )

        return distance