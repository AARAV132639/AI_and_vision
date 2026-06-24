# SE(3) Module Notes

## Objective

The original SCAPO paper contains a Stage-1 module whose purpose is:

```text
Input Point Cloud
        ↓
Remove Translation
        ↓
Remove Rotation
        ↓
Canonical Shape
```

The intuition is:

```text
Laptop Opened 30°
Laptop Opened 60°
Laptop Rotated 90°
Laptop Rotated 180°

↓

All should map to the same canonical representation.
```

This allows Stage-2 to focus only on:

```text
Articulation
```

instead of wasting capacity handling arbitrary object orientations.

---

# Why SE(3)?

SE(3) = Special Euclidean Group

Represents:

```text
Translation
+
Rotation
```

in 3D space.

Mathematically:

SE(3)
=
SO(3) + Translation

where

SO(3)
=
3D Rotations
```

Stage-1 tries to remove SE(3) transformations.

---

# Overall Pipeline

```text
Input Point Cloud X
          │
          ▼
        VNT
          │
          ▼
   Translation tg
          │
          ▼
  Translation-Free Shape
          │
          ▼
   VNPointNetEncoder
          │
          ▼
   Equivariant Features
          │
          ▼
    RotationHead
          │
          ▼
      Rotation Rg
          │
          ▼
    Canonicalizer
          │
          ▼
 Canonical Shape Sobj
```

---

# File Structure

```text
models/se3/

├── so3.py
├── vnt.py
├── rotation_head.py
├── canonicalizer.py
└── se3_autoencoder.py
```

---

# 1. so3.py

## Purpose

Provides mathematical operations for SO(3).

Used everywhere rotation is involved.

---

## What is SO(3)?

SO(3) = Set of all valid 3D rotation matrices.

Properties:

```text
RᵀR = I

det(R) = 1
```

---

## Functions

### skew()

Creates skew-symmetric matrix.

Input:

```python
v = [x,y,z]
```

Output:

```text
[ 0 -z  y]
[ z  0 -x]
[-y  x  0]
```

Used by Rodrigues formula.

---

### axis_angle_to_matrix()

Converts:

```text
Axis + Angle
```

into:

```text
Rotation Matrix
```

using Rodrigues rotation formula.

---

### random_rotation_matrix()

Generates random SO(3) rotations.

Used for:

```text
Data Augmentation
Rotation Robustness Tests
```

---

### apply_rotation()

Applies:

```text
R x
```

to point clouds.

---

### apply_transform()

Applies:

```text
R x + t
```

(SE(3) transform)

---

### orthogonality_loss()

Ensures:

```text
RᵀR ≈ I
```

during training.

---

### determinant_loss()

Ensures:

```text
det(R) ≈ 1
```

during training.

---

# 2. vnt.py

## Purpose

Estimate object translation.

---

## Problem

Suppose:

```text
Laptop A
center = (0,0,0)

Laptop B
center = (5,2,1)
```

They are identical objects.

Only location changed.

---

Without VNT

Network sees:

```text
Different Objects
```

---

With VNT

Network predicts:

```text
tg
```

and computes:

```text
X - tg
```

making object location-independent.

---

## Input

```python
[B,N,3]
```

---

## Output

### tg

```python
[B,3]
```

Predicted translation.

---

### centered

```python
[B,N,3]
```

Point cloud with translation removed.

---

# 3. rotation_head.py

## Purpose

Estimate global rotation.

---

## Problem

Suppose:

```text
Laptop facing east

Laptop facing north
```

Same laptop.

Different orientation.

---

We want:

```text
Rg
```

such that:

```text
Rg⁻¹ X
```

aligns object to canonical orientation.

---

## Input

```python
z_vec
```

from VNPointNet.

Shape:

```python
[B,1024,3]
```

---

## Output

```python
R
```

Shape:

```python
[B,3,3]
```

Valid SO(3) rotation matrix.

---

## Technique

Uses:

```text
6D Rotation Representation
+
Gram-Schmidt
```

to guarantee:

```text
RᵀR = I

det(R)=1
```

---

# 4. canonicalizer.py

## Purpose

Actually remove translation and rotation.

---

## SCAPO Equation

Canonical shape:

Sobj = Rgᵀ(X − tg)

---

## Input

### points

```python
[B,N,3]
```

### R

```python
[B,3,3]
```

### t

```python
[B,3]
```

---

## Output

### canonical

```python
[B,N,3]
```

---

## Meaning

Transforms:

```text
Observed Shape
```

into:

```text
Canonical Shape
```

---

Example

```text
Laptop Rotated 90°
```

↓

```text
Canonical Laptop
```

---

# 5. se3_autoencoder.py

## Purpose

Combines all Stage-1 modules.

---

## Pipeline

```text
Input Shape
      │
      ▼
    VNT
      │
      ▼
 Translation-Free Shape
      │
      ▼
 VNPointNetEncoder
      │
      ▼
 Equivariant Features
      │
      ▼
 RotationHead
      │
      ▼
      Rg
      │
      ▼
 Canonicalizer
      │
      ▼
 Canonical Shape
```

---

## Inputs

```python
points
```

Shape:

```python
[B,N,3]
```

---

## Outputs

### canonical

```python
[B,N,3]
```

Canonical shape.

---

### translation

```python
[B,3]
```

Estimated translation.

---

### rotation

```python
[B,3,3]
```

Estimated rotation.

---

### z_inv

```python
[B,1024]
```

Rotation-invariant latent feature.

---

### z_vec

```python
[B,1024,3]
```

Rotation-equivariant latent feature.

---

# What Have We Reproduced?

Current Stage-1:

```text
✓ Translation Estimation

✓ Rotation Estimation

✓ Canonicalization

✓ SO(3) Utilities

✓ Equivariant Feature Extraction
```

---

# What Is Still Missing?

Full SCAPO Stage-1 also contains:

```text
□ Decoder

□ Reconstruction Loss

□ Shape Variance Decoder

□ Augmentation Loss

□ Canonical Consistency Loss
```

These modules teach the network to actually learn a stable canonical representation.

---

# Why Are We Building This?

Current SCAPOv0:

```text
Point Cloud
      ↓
PointNet
      ↓
Articulation Learning
```

Problem:

```text
Sensitive to Rotation
Sensitive to Translation
```

---

SCAPOv1:

```text
Point Cloud
      ↓
SE(3) Canonicalization
      ↓
Canonical Shape
      ↓
Articulation Learning
```

Advantage:

```text
Same object

regardless of pose

→ same latent representation
```

This is the core idea behind the original SCAPO paper and the reason Stage-1 exists before the articulation module.