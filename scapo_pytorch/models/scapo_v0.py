import torch
import torch.nn as nn

from models.pointnet_encoder import PointNetEncoder
from models.keypoint_net import KeypointNet
from models.pose_net import PoseNet

from models.mahalanobis import MahalanobisDistance
from models.skinning_field import SkinningField

from models.bone_transform import BoneTransform
from models.blend_skinning import BlendSkinning

## upgrade 1: Bone transform
from models.bone_gaussian_net import BoneGaussianNet


class SCAPOv0(nn.Module):

    def __init__(
        self,
        num_parts=4,
        gamma=0.1
    ):
        super().__init__()

        self.num_parts = num_parts

        #
        # Stage A
        #

        self.encoder = PointNetEncoder()

        self.keypoint_net = KeypointNet(
            latent_dim=1024,
            num_parts=num_parts
        )

        self.pose_net = PoseNet(
            latent_dim=1024,
            num_parts=num_parts
        )

        #
        # Stage B
        #

        self.mahal = MahalanobisDistance()

        #Adding bone gaussian
        self.bone_gaussian = BoneGaussianNet(
                latent_dim=1024,
                num_parts=num_parts
                )

        self.skinning = SkinningField(
            gamma=gamma
        )

        self.bone_transform = BoneTransform()

        self.blend_skinning = BlendSkinning()

    
    def forward(
        self,
        points
    ):
        """
        points

        [B,N,3]
        """

        B = points.shape[0]

        #
        # Encoder
        #

        z = self.encoder(points)

        #
        # Predict keypoints
        #

        keypoints = self.keypoint_net(z)

        #
        # Predict joints
        #

        pivot, axis, angle = self.pose_net(z)

        #
        # Mahalanobis field
        #

        Q = self.bone_gaussian(z) #using bone gaussian

        distance = self.mahal(
            points,
            keypoints,
            Q
        )

        #
        # Skinning weights
        #

        weights = self.skinning(
            distance
        )

        #
        # Bone transforms
        #

        transforms = self.bone_transform(
            pivot,
            axis,
            angle
        )

        #
        # Articulation
        #

        deformed = self.blend_skinning(
            points,
            weights,
            transforms
        )

        return {
            "latent": z,
            "keypoints": keypoints,
            "Q": Q,
            "pivot": pivot,
            "axis": axis,
            "angle": angle,
            "distance": distance,
            "weights": weights,
            "transforms": transforms,
            "deformed": deformed
        }