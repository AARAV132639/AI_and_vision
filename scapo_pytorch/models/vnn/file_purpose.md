# SO(3) & VNN Module Notes

---

# Why are we implementing these modules?

The original SCAPO architecture contains a Stage-1:

```text
Input Point Cloud
        ↓
SE(3)-Equivariant Autoencoder
        ↓
Canonical Shape
        ↓
Joint-Aware Deformation
```

Our SCAPOv0 already reproduces the Joint-Aware Deformation stage.

The purpose of the SO(3) and VNN modules is to reproduce the missing Stage-1.

---

# Important Terminology

---

## SO(3)

SO(3) stands for:

```text
Special Orthogonal Group in 3D
```

Represents:

```text
All possible 3D rotations
```

Properties:

```text
RᵀR = I

det(R) = 1
```

Examples:

```text
Rotate object 45°

Rotate object 90°

Rotate object 180°
```

These are all elements of SO(3).

---

## SE(3)

SE(3) stands for:

```text
Special Euclidean Group
```

Represents:

```text
Rotation
+
Translation
```

Examples:

```text
Rotate laptop 45°
and move it 2m right
```

SCAPO aims to become invariant/equivariant to SE(3).

---

## Invariance

A feature is invariant if:

```text
Input changes

Output remains same
```

Example:

```text
Laptop
↓
Rotate laptop
↓
Feature remains unchanged
```

Mathematically:

f(Rx) = f(x)

---

## Equivariance

A feature is equivariant if:

```text
Input rotates

Output rotates similarly
```

Mathematically:

f(Rx) = Rf(x)

Example:

```text
Rotate object 90°
↓
Feature rotates 90°
```

instead of changing arbitrarily.

---

## Vector Neuron

Traditional neural networks process:

```text
Scalar Features

2.1
-3.4
5.7
```

Vector Neurons process:

```text
3D Vectors

[x,y,z]
```

Example:

```text
[1,2,3]
```

instead of:

```text
5.6
```

Purpose:

```text
Preserve geometric structure
under rotation.
```

---

## Canonicalization

Goal:

```text
Different views

→

Same representation
```

Example:

```text
Laptop at 0°

Laptop at 90°

Laptop at 180°
```

All become:

```text
Canonical Laptop
```

before articulation learning.

---

# File Descriptions

---

# models/se3/so3.py

## Objective

Provide all mathematical operations related to:

```text
SO(3)
SE(3)
Rotations
Transformations
```

This file contains no learning.

Only geometry.

---

## Functions

### skew()

Purpose:

Convert vector into skew-symmetric matrix.

Input:

```python
[wx, wy, wz]
```

Output:

```text
      [ 0  -z   y ]
K  =  [ z   0  -x ]
      [-y   x   0 ]
```

Used by:

```text
Rodrigues Rotation Formula
```

---

### axis_angle_to_matrix()

Purpose:

Convert:

```text
Axis
+
Angle
```

into:

```text
Rotation Matrix
```

Used in:

```text
BoneTransform
Joint Rotation
```

---

### random_rotation_matrix()

Purpose:

Generate random SO(3) rotations.

Used for:

```text
Equivariance Testing
Data Augmentation
```

---

### apply_rotation()

Purpose:

Apply:

```text
R
```

to point cloud.

Input:

```python
[B,N,3]
```

Output:

```python
[B,N,3]
```

---

### apply_transform()

Purpose:

Apply:

```text
Rotation
+
Translation
```

to point cloud.

Represents:

```text
SE(3)
```

transform.

---

### orthogonality_loss()

Purpose:

Enforce:

```text
RᵀR = I
```

during training.

---

### determinant_loss()

Purpose:

Enforce:

```text
det(R)=1
```

to ensure valid rotations.

---

## Role in SCAPO

Provides:

```text
SO(3) Geometry
```

needed by:

```text
VNN
VNT
Canonicalization
```

---

# models/vnn/vn_linear.py

## Objective

SO(3)-Equivariant version of:

```python
nn.Linear
```

---

## Traditional Linear Layer

Input:

```text
[B,N,C]
```

Output:

```text
[B,N,Cout]
```

Processes:

```text
Scalars
```

---

## VNLinear

Input:

```text
[B,N,C,3]
```

Output:

```text
[B,N,Cout,3]
```

Processes:

```text
Vectors
```

instead of scalars.

---

## Key Idea

Weights operate on:

```text
Channels
```

NOT

```text
xyz coordinates
```

Therefore:

```text
Rotation before layer

=

Rotation after layer
```

---

## Role in SCAPO

Acts as:

```text
SO(3)-Equivariant MLP
```

inside encoder.

---

# models/vnn/vn_relu.py

## Objective

SO(3)-Equivariant activation function.

Replacement for:

```python
ReLU()
```

---

## Problem

Normal ReLU:

```text
Acts independently on:

x
y
z
```

Breaks equivariance.

---

## Solution

VNReLU activates vectors based on:

```text
Direction
```

instead of coordinates.

Uses:

```text
Learned direction vector
```

to decide:

```text
Keep vector

or

Reflect vector
```

---

## Role in SCAPO

Provides:

```text
Non-linearity
```

without destroying:

```text
SO(3)-equivariance
```

---

# models/vnn/vn_pool.py

## Objective

SO(3)-Equivariant version of:

```python
MaxPool
```

---

## Problem

Traditional PointNet:

```python
torch.max()
```

breaks equivariance.

---

## Solution

VNMaxPool:

```text
Project vectors
onto learned direction

Choose strongest vector
```

instead of:

```text
Largest scalar value
```

---

## Input

```text
[B,N,C,3]
```

---

## Output

```text
[B,C,3]
```

Global vector feature.

---

## Role in SCAPO

Equivalent of:

```text
PointNet Global Pooling
```

for vector neurons.

---

# models/vnn/vn_pointnet.py

## Objective

SO(3)-Equivariant replacement for:

```python
PointNetEncoder
```

---

## Architecture

```text
Point Cloud
      ↓
VNLinear
      ↓
VNReLU
      ↓
VNLinear
      ↓
VNReLU
      ↓
VNLinear
      ↓
VNReLU
      ↓
VNMaxPool
```

---

## Input

```text
[B,N,3]
```

Point cloud.

---

## Outputs

### z_vec

```text
[B,1024,3]
```

Equivariant vector feature.

Property:

```text
Rotate input
↓
Feature rotates similarly
```

---

### z_inv

```text
[B,1024]
```

Invariant feature.

Computed using:

```python
torch.norm(z_vec)
```

Property:

```text
Rotate input
↓
Feature remains same
```

---

## Why Two Outputs?

SCAPO Stage-1 requires:

```text
Invariant Features
```

for:

```text
Canonical Shape Learning
```

and

```text
Equivariant Features
```

for:

```text
Frame Prediction
Rotation Estimation
```

---

## Role in SCAPO

This is the first true component of:

```text
SE(3)-Equivariant Encoder
```

and replaces:

```python
PointNetEncoder()
```

used in SCAPOv0.

---

# Overall Flow

```text
Point Cloud
      ↓
VNPointNetEncoder
      ↓
────────────────────────────
│                          │
│ z_inv                    │ z_vec
│ Invariant                │ Equivariant
│ Features                 │ Features
│                          │
────────────────────────────
      ↓
Future VNT Layer
      ↓
Canonical Frame Prediction
      ↓
Canonical Point Cloud
      ↓
SCAPO Stage-2
(Keypoints + Gaussian Bones + Skinning)
```

---

# Current Reproduction Status

```text
✓ Joint-Aware Deformation

✓ Mahalanobis Skinning

✓ Learnable Gaussian Bones

✓ SO(3) Utilities

✓ VNLinear

✓ VNReLU

✓ VNMaxPool

✓ VNPointNetEncoder

Next:
VNT Layer
↓
Canonical Frame Prediction
↓
SE(3)-Equivariant Autoencoder
```