import torch

from models.bone_transform import BoneTransform
from models.blend_skinning import BlendSkinning

def create_laptop():

    # Base
    base_x = torch.linspace(-1, 1, 40)
    base_y = torch.linspace(-0.6, 0.6, 20)

    bx, by = torch.meshgrid(
        base_x,
        base_y,
        indexing="ij"
    )

    bz = torch.zeros_like(bx)

    base = torch.stack(
        [bx, by, bz],
        dim=-1
    ).reshape(-1, 3)

    # Lid

    lid_x = torch.linspace(-1, 1, 40)
    lid_z = torch.linspace(0, 1.5, 30)

    lx, lz = torch.meshgrid(
        lid_x,
        lid_z,
        indexing="ij"
    )

    ly = torch.zeros_like(lx)

    lid = torch.stack(
        [lx, ly, lz],
        dim=-1
    ).reshape(-1, 3)

    return base, lid

# Creating point cloud

base, lid = create_laptop()

points = torch.cat(
    [base, lid],
    dim=0
)

#manual bone weights
N_base = len(base)
N_lid = len(lid)

weights = torch.zeros(
    1,
    N_base + N_lid,
    2
)

weights[:, :N_base, 0] = 1.0
weights[:, N_base:, 1] = 1.0

# Bone transform

pivot = torch.tensor(
[
[
[0.,0.,0.],   # base

[0.,0.,0.]    # lid hinge
]
]
)

axis = torch.tensor(
[
[
[1.,0.,0.],
[1.,0.,0.]
]
]
)

import math

angle = torch.tensor(
[
[
[0.0],
[math.pi/3]
]
]
)

# Transform matrix
bone_transform = BoneTransform()

T = bone_transform(
    pivot,
    axis,
    angle
)

# Blend skinning

blend = BlendSkinning()

deformed = blend(
    points.unsqueeze(0),
    weights,
    T
)

# visualize

import numpy as np


#display
import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(
    projection="3d"
)

orig = points.numpy()

deformed_np = (
    deformed[0]
    .detach()
    .numpy()
)

ax.scatter(
    orig[:,0],
    orig[:,1],
    orig[:,2],
    s=1
)

ax.scatter(
    deformed_np[:,0],
    deformed_np[:,1],
    deformed_np[:,2],
    s=1
)

plt.show()