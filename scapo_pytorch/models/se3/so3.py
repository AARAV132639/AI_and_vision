"""
"The encoder produces affine-invariant features Zx and estimates global rotation Rg and translation tg using VNN and VNT layers.
"""

import torch
import math


# =====================================================
# Skew Symmetric Matrix
# =====================================================

def skew(v):
    """
    v : [...,3]

    Returns
    -------
    [...,3,3]
    """

    x = v[..., 0]
    y = v[..., 1]
    z = v[..., 2]

    O = torch.zeros_like(x)

    K = torch.stack(
        [
            O, -z,  y,
            z,  O, -x,
           -y,  x,  O
        ],
        dim=-1
    )

    return K.reshape(*v.shape[:-1], 3, 3)


# =====================================================
# Rodrigues Formula
# =====================================================

def axis_angle_to_matrix(axis, angle):
    """
    axis  : [...,3]
    angle : [...,1] or [...]

    Returns
    -------
    R : [...,3,3]
    """

    axis = axis / (
        torch.norm(axis, dim=-1, keepdim=True)
        + 1e-8
    )

    angle = angle.squeeze(-1)

    K = skew(axis)

    I = torch.eye(
        3,
        device=axis.device,
        dtype=axis.dtype
    )

    I = I.expand(
        *axis.shape[:-1],
        3,
        3
    )

    sin = torch.sin(angle)[..., None, None]
    cos = torch.cos(angle)[..., None, None]

    R = (
        I
        + sin * K
        + (1 - cos) * (K @ K)
    )

    return R


# =====================================================
# Random SO(3)
# =====================================================

def random_rotation_matrix(
    batch_size=1,
    device="cpu"
):
    """
    Uniform random rotations.
    """

    q = torch.randn(
        batch_size,
        4,
        device=device
    )

    q = q / (
        torch.norm(q, dim=-1, keepdim=True)
        + 1e-8
    )

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R = torch.zeros(
        batch_size,
        3,
        3,
        device=device
    )

    R[:,0,0] = 1 - 2*(y*y + z*z)
    R[:,0,1] = 2*(x*y - z*w)
    R[:,0,2] = 2*(x*z + y*w)

    R[:,1,0] = 2*(x*y + z*w)
    R[:,1,1] = 1 - 2*(x*x + z*z)
    R[:,1,2] = 2*(y*z - x*w)

    R[:,2,0] = 2*(x*z - y*w)
    R[:,2,1] = 2*(y*z + x*w)
    R[:,2,2] = 1 - 2*(x*x + y*y)

    return R


# =====================================================
# Apply Rotation
# =====================================================

def apply_rotation(
    points,
    R
):
    """
    points : [B,N,3]
    R      : [B,3,3]

    Returns
    -------
    rotated : [B,N,3]
    """

    return torch.bmm(
        points,
        R.transpose(1, 2)
    )


# =====================================================
# Apply SE(3)
# =====================================================

def apply_transform(
    points,
    R,
    t
):
    """
    points : [B,N,3]
    R      : [B,3,3]
    t      : [B,3]

    Returns
    -------
    transformed : [B,N,3]
    """

    rotated = apply_rotation(
        points,
        R
    )

    transformed = (
        rotated
        + t.unsqueeze(1)
    )

    return transformed


# =====================================================
# SO(3) Orthonormality Loss
# =====================================================

def orthogonality_loss(R):
    """
    R : [B,3,3]

    Returns
    -------
    scalar loss
    """

    I = torch.eye(
        3,
        device=R.device
    )

    I = I.unsqueeze(0)

    RtR = torch.bmm(
        R.transpose(1,2),
        R
    )

    return (
        (RtR - I)**2
    ).mean()


# =====================================================
# Determinant Check
# =====================================================

def determinant_loss(R):
    """
    Forces det(R)=1
    """

    det = torch.det(R)

    return (
        (det - 1.0)**2
    ).mean()