# SCAPO PyTorch Reproduction Notes

# Overview

This repository focuses on reproducing the **Joint-Aware Deformation** stage of SCAPO before implementing the full neural architecture.

The objective is:

```text
Point Cloud
    ↓
Bone Assignment
    ↓
Joint Motion
    ↓
Blend Skinning
    ↓
Articulated Object Reconstruction
```

By implementing these geometric modules first, we verify the mathematical correctness of the articulation engine before introducing learning components.

---

# Repository Structure

```text
scapo/
│
├── models/
│   ├── mahalanobis.py
│   ├── skinning_field.py
│   ├── rodrigues_rotation.py
│   ├── bone_transform.py
│   ├── blend_skinning.py
│   ├── keypoint_net.py
│   ├── pose_net.py
│   └── losses.py
│
├── datasets/
├── train.py
└── test.py
```

---

# SCAPO Pipeline Flow

```text
                    INPUT POINT CLOUD
                           X
                           │
                           ▼
                 ┌──────────────────┐
                 │  KeypointNet     │
                 │ Predict Bones    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Mahalanobis      │
                 │ Distance         │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Skinning Field   │
                 │ Soft Bone Weights│
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ PoseNet          │
                 │ Joint Params     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Rodrigues        │
                 │ Rotation         │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Bone Transform   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Blend Skinning   │
                 └────────┬─────────┘
                          │
                          ▼
                 DEFORMED POINT CLOUD
```

---

# 1. mahalanobis.py

## Purpose

Computes the Mahalanobis distance between every point and every bone.

SCAPO Equation:

```math
W_i^{(p)}
=
(s_i - O_p)^T
Q_p
(s_i - O_p)
```

where:

* s_i = point
* O_p = bone center
* Q_p = bone precision matrix

---

## Input

```python
points  : [B,N,3]
centers : [B,P,3]
Q       : [B,P,3,3]
```

---

## Output

```python
distance : [B,N,P]
```

---

## Role in SCAPO

Determines how strongly each bone influences each point.

Without this module:

```text
No bone influence estimation
No segmentation
No articulation
```

---

## Analogy

```text
Point ---> Which bone am I closest to?
```

---

# 2. skinning_field.py

## Purpose

Converts distances into soft assignments.

SCAPO Equation:

```math
w_i
=
softmax
(
-W_i / γ
)
```

---

## Input

```python
distance : [B,N,P]
```

---

## Output

```python
weights : [B,N,P]
```

---

## Role in SCAPO

Creates soft segmentation.

Example:

```text
Point A

Bone 1 : 0.95
Bone 2 : 0.03
Bone 3 : 0.02
```

Point A mostly belongs to Bone 1.

---

## Analogy

```text
Bone Voting System
```

---

# 3. rodrigues_rotation.py

## Purpose

Converts axis-angle representation into a rotation matrix.

SCAPO Equation:

```math
R
=
ExpSO3
(
[d]_x a
)
```

---

## Input

```python
axis  : [B,P,3]
angle : [B,P,1]
```

---

## Output

```python
R : [B,P,3,3]
```

---

## Role in SCAPO

Represents revolute joints.

Examples:

```text
Laptop Hinge
Door Hinge
Scissor Joint
Robot Arm Joint
```

---

## Analogy

```text
Joint Axis + Angle
            ↓
      Rotation Matrix
```

---

# 4. bone_transform.py

## Purpose

Constructs rigid transformations around a joint pivot.

Transformation:

```text
Move To Pivot
      ↓
Rotate
      ↓
Move Back
```

---

## Input

```python
pivot
axis
angle
```

---

## Output

```python
T : [B,P,4,4]
```

---

## Role in SCAPO

Creates actual bone motion.

Without this module:

```text
Joint exists
But cannot move
```

---

## Analogy

```text
Laptop Hinge Mechanism
```

---

# 5. blend_skinning.py

## Purpose

Applies weighted bone transformations.

SCAPO Equation:

```math
T_i
=
Σ w_i^(p) T^(p)
```

---

## Input

```python
points
weights
transforms
```

---

## Output

```python
deformed_points
```

---

## Role in SCAPO

Core articulation engine.

This is where:

```text
Door Opens
Laptop Opens
Drawer Slides
Scissors Rotate
```

actually happens.

---

## Analogy

```text
Each point follows
multiple bones simultaneously
```

---

# 6. keypoint_net.py

## Purpose

Predicts bone centers.

SCAPO predicts:

```text
Part Centroids
```

which act as:

```text
Bone Anchors
```

---

## Input

```python
latent feature
```

---

## Output

```python
[B,P,3]
```

---

## Role in SCAPO

Determines:

```text
Where bones exist
```

---

## Analogy

```text
Automatic Skeleton Extraction
```

---

# 7. pose_net.py

## Purpose

Predicts joint parameters.

Outputs:

```python
pivot
axis
angle
```

---

## Role in SCAPO

Learns:

```text
How parts move
```

rather than manually specifying motion.

---

## Analogy

```text
Joint Discovery Network
```

---

# 8. losses.py

Contains all training objectives.

---

## Cycle Consistency Loss

```math
X
→ Canonical
→ Observation
≈ X
```

Purpose:

```text
Ensure reversible articulation
```

---

## Segmentation Loss

Purpose:

```text
Correct bone assignment
```

---

## Keypoint Consistency Loss

Purpose:

```text
Keep keypoints near part centers
```

---

## Joint Direction Loss

Purpose:

```text
Physically meaningful joint axes
```

---

## Joint Proximity Loss

Purpose:

```text
Keep pivots close to object surface
```

---

## Boundary Attraction Loss

Purpose:

```text
Place pivots near articulation boundaries
```

---

# Development Roadmap

## Phase 1 — Geometry Engine

Implemented first:

```text
MahalanobisDistance
SkinningField
RodriguesRotation
BoneTransform
BlendSkinning
```

Goal:

```text
Open laptop
Rotate door
Slide drawer
```

without learning.

---

## Phase 2 — Learning Engine

Implement:

```text
KeypointNet
PoseNet
```

Goal:

```text
Predict joints automatically
```

---

## Phase 3 — Self-Supervised Learning

Implement:

```text
Cycle Loss
Segmentation Loss
Joint Losses
```

Goal:

```text
Train SCAPO end-to-end
```

---

# Complete SCAPO Flow

```text
Input Point Cloud
        │
        ▼
 SE(3) Canonicalization
        │
        ▼
   KeypointNet
        │
        ▼
 Mahalanobis Distance
        │
        ▼
  Skinning Weights
        │
        ▼
     PoseNet
        │
        ▼
 Joint Parameters
        │
        ▼
 Rodrigues Rotation
        │
        ▼
 Bone Transform
        │
        ▼
 Blend Skinning
        │
        ▼
 Deformed Point Cloud
        │
        ▼
 Cycle Reconstruction
        │
        ▼
 Self-Supervised Losses
```

---

# Current Status

```text
Geometry Layer

✓ MahalanobisDistance
✓ SkinningField
✓ RodriguesRotation
✓ BoneTransform
✓ BlendSkinning

Next:
→ KeypointNet
→ PoseNet
→ Losses
→ Full SCAPO
```
