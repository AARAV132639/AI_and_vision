# SCAPOv0 Training README

## Overview

SCAPOv0 is a simplified reproduction of the **Joint-Aware Deformation** stage of the SCAPO paper.

The goal of this implementation is **not** to fully reproduce the CVPR 2026 paper, but to:

1. Understand articulated object modeling.
2. Implement blend skinning from scratch.
3. Learn keypoints, segmentation, and joint parameters.
4. Train on synthetic articulated objects.
5. Create a foundation for future extensions such as:

   * Learnable Gaussian Bones
   * SO(3)/SE(3)-Equivariant Encoders
   * Edge-SCAPO
   * Defense-oriented articulated object understanding

---

# Current Pipeline

```text
Input Point Cloud
      │
      ▼
PointNet Encoder
      │
      ▼
1024-D Latent Feature
      │
      ├─────────────► KeypointNet
      │                    │
      │                    ▼
      │               Bone Centers
      │
      └─────────────► PoseNet
                           │
                           ▼
                Pivot / Axis / Angle

Bone Centers
      │
      ▼
Mahalanobis Distance
      │
      ▼
Skinning Field
      │
      ▼
Skinning Weights

Pivot + Axis + Angle
      │
      ▼
Bone Transform
      │
      ▼
Transform Matrices

Transforms + Weights
      │
      ▼
Blend Skinning
      │
      ▼
Deformed Point Cloud
```

---

# Files Used During Training

```text
models/
│
├── pointnet_encoder.py
├── keypoint_net.py
├── pose_net.py
├── mahalanobis.py
├── skinning_field.py
├── rodrigues_rotation.py
├── bone_transform.py
├── blend_skinning.py
└── scapo_v0.py

datasets/
│
└── synthetic_laptop.py

losses/
│
├── cycle_loss.py
├── segmentation_loss.py
└── kp_seg_loss.py

train_v0.py
```

---

# Dataset

## Synthetic Laptop Dataset

Each sample contains:

```python
{
    "points": articulated_laptop,
    "canonical": canonical_laptop,
    "angle": articulation_angle
}
```

### Input

Randomly articulated laptop.

```text
0° → 90°
```

### Target

Canonical laptop.

```text
Fixed reference pose.
```

---

# Model Components

## 1. PointNet Encoder

Purpose:

```text
Point Cloud
    ↓
Compact Representation
```

Input:

```python
[B,N,3]
```

Output:

```python
[B,1024]
```

Example:

```python
torch.Size([8,1024])
```

Meaning:

```text
8 objects

Each object compressed into
1024 learned geometric features.
```

---

## 2. KeypointNet

Purpose:

```text
Locate rigid part centers.
```

Output:

```python
[B,P,3]
```

Example:

```python
[B,4,3]
```

Represents:

```text
4 keypoints

Each keypoint:
x,y,z
```

Used as:

```text
Bone anchors.
```

---

## 3. PoseNet

Purpose:

```text
Predict articulation parameters.
```

Outputs:

### Pivot

```python
[B,P,3]
```

Joint location.

### Axis

```python
[B,P,3]
```

Joint rotation axis.

### Angle

```python
[B,P,1]
```

Joint articulation state.

---

## 4. Mahalanobis Distance

Purpose:

```text
Measure point-to-bone affinity.
```

Formula:

W = (x - O)^T Q (x - O)

Current implementation:

```text
Q = Identity Matrix
```

Therefore:

```text
Mahalanobis Distance
=
Squared Euclidean Distance
```

---

## 5. Skinning Field

Purpose:

```text
Convert distances into
soft part assignments.
```

Formula:

```text
Softmax(-distance / gamma)
```

Output:

```python
[B,N,P]
```

Each point receives:

```text
Probability of belonging
to each bone.
```

---

## 6. Bone Transform

Purpose:

```text
Generate rigid motion
for each bone.
```

Uses:

```text
Pivot
Axis
Angle
```

Produces:

```python
[B,P,4,4]
```

Homogeneous transformation matrices.

---

## 7. Blend Skinning

Purpose:

```text
Move every point
using weighted bone motion.
```

Formula:

```text
Point Motion
=
Σ(weight × transform)
```

Produces:

```python
[B,N,3]
```

Deformed point cloud.

---

# Loss Functions

## Cycle Loss

Purpose:

```text
Reconstruct canonical shape.
```

Formula:

```python
L1(deformed, canonical)
```

Weight:

```text
10.0
```

Most important loss.

---

## Segmentation Loss

Purpose:

```text
Encourage meaningful
part assignments.
```

Uses:

```text
Nearest keypoint
→ pseudo label
```

to supervise:

```text
Skinning weights.
```

Weight:

```text
1.0
```

---

## Keypoint Segmentation Loss

Purpose:

```text
Force keypoints
towards part centroids.
```

Weight:

```text
1.0
```

---

# Total Loss

```python
loss =
    10 * cycle_loss
  + 1 * seg_loss
  + 1 * kp_seg_loss
```

---

# Training Configuration

```python
Optimizer:
Adam
```

```python
Learning Rate:
1e-4
```

```python
Batch Size:
8
```

```python
Epochs:
100
```

---

# Expected Training Behaviour

Beginning:

```text
Loss ≈ 5-10
```

Middle:

```text
Loss ≈ 1-2
```

End:

```text
Loss ≈ 0.1
```

Typical convergence:

```text
Epoch 096 | Loss 0.0981
Epoch 097 | Loss 0.0951
Epoch 098 | Loss 0.1030
Epoch 099 | Loss 0.0963
Epoch 100 | Loss 0.0987
```

Interpretation:

```text
Model has converged.

Small oscillations are normal.

No sign of instability.
```

---

# What Has Been Reproduced?

Successfully implemented:

```text
✓ Mahalanobis Distance

✓ Skinning Field

✓ Rodrigues Rotation

✓ Bone Transform

✓ Blend Skinning

✓ Keypoint Prediction

✓ Joint Prediction

✓ Synthetic Training Pipeline
```

This covers most of the Joint-Aware Deformation stage described in SCAPO.

---

# Known Simplifications

Current implementation does NOT include:

```text
✗ VNN Encoder

✗ VNT Layers

✗ SE(3)-Equivariant Autoencoder

✗ Canonical Template

✗ Cross-space Consistency

✗ Shape Variance Decoder

✗ Learnable Gaussian Bones
```

Therefore this implementation is:

```text
SCAPOv0

=
Minimal Articulation Learning System

NOT

Full SCAPO
```

---

# Future Roadmap

## Phase 1

Learnable Gaussian Bones

```text
Q = VᵀΛV
```

instead of:

```text
Q = I
```

---

## Phase 2

Rotation Robustness Experiments

Evaluate sensitivity to:

```text
SO(3) rotations.
```

---

## Phase 3

Real Point Clouds

```text
Laptop
Drawer
Safe
```

from CAD models.

---

## Phase 4

SE(3)-Equivariant Canonicalization

Replace:

```text
PointNet
```

with:

```text
VNN + VNT
```

to move closer to the original SCAPO architecture.

---

# Current Status

```text
SCAPOv0 Successfully Trained

Loss Converged ≈ 0.1

Ready for:
    Learnable Gaussian Bones
    Rotation Robustness Studies
    Real Point Cloud Experiments
```
