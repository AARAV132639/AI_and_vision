# SCAPO Loss Functions Notes

## Overview

SCAPO learns articulated object structure **without ground-truth segmentation, joints, or articulation labels**.

The losses provide the supervision signal.

The overall optimization objective can be viewed as:

[
L_{total}
=========

\lambda_{cycle}L_{cycle}
+
\lambda_{recon}L_{recon}
+
\lambda_{kp}L_{kp-seg}
+
\lambda_{seg}L_{seg}
+
\lambda_{shape}L_{shape-var}
+
\lambda_{dir}L_{dir-align}
+
\lambda_{prox}L_{joint-prox}
+
\lambda_{boundary}L_{joint-boundary}
]

---

# 1. Cycle Consistency Loss

## Purpose

Ensures that:

```text
Observed Shape
      ↓
Canonical Shape
      ↓
Observed Shape

≈ Original Observation
```

The deformation process should be reversible.

Without this loss:

* articulation becomes unstable
* canonical space drifts
* deformations become non-invertible

---

## Formula

[
L_{cycle}
=========

|\hat X - X|_1
]

where:

* (X) = original point cloud
* (\hat X) = reconstructed point cloud after full cycle

---

## Flow

```text
Observed X
     ↓
Canonicalization
     ↓
S*
     ↓
Forward Deformation
     ↓
X^

Lcycle = |X^ - X|
```

---

## PyTorch

```python
loss = F.l1_loss(
    reconstructed,
    original
)
```

---

# 2. Reconstruction Loss

## Purpose

Aligns canonicalized observations with the learnable category template.

Without it:

* template and instance geometry diverge
* category-level alignment fails

---

## Formula

[
L_{recon}
=========

CD(\hat S_{x\rightarrow y}, Y_X)
]

where:

* (CD) = Chamfer Distance
* (Y_X) = canonical template

---

## Flow

```text
Input Shape
      ↓
Canonicalization
      ↓
Cross-space Deformation
      ↓
Predicted Template

Compare

Predicted Template
vs
Canonical Template
```

---

# 3. Keypoint-Segmentation Consistency Loss

## Purpose

Forces predicted keypoints to lie at the center of their assigned part.

Without this loss:

* keypoints drift
* bones no longer represent parts

---

## Formula

Part centroid:

[
m^{(p)}
=======

\frac{
\sum_i
w_i^{(p)} s_i
}{
\sum_i
w_i^{(p)}
}
]

Loss:

[
L_{kp-seg}
==========

\frac1P
\sum_{p=1}^{P}
|k^{(p)} - m^{(p)}|_2^2
]

where:

* (k^{(p)}) = predicted keypoint
* (m^{(p)}) = weighted centroid

---

## Flow

```text
Skinning Weights
      ↓
Part Centroid
      ↓
Compare
      ↓
Predicted Keypoint
```

---

## Interpretation

This turns every keypoint into a geometric center of a rigid part.

---

# 4. Segmentation Loss

## Purpose

Provides pseudo-labels for segmentation.

No segmentation annotations are available.

SCAPO generates labels from nearest keypoints.

---

## Formula

Nearest keypoint:

[
p_i^*
=====

\arg\min_p
|s_i-k^{(p)}|^2
]

One-hot label:

[
y_i
]

Loss:

[
L_{seg}
=======

\frac1N
\sum_i
|w_i-y_i|^2
]

where:

* (w_i) = predicted skinning weights
* (y_i) = pseudo-label

---

## Flow

```text
Point
 ↓

Nearest Keypoint
 ↓

Pseudo Label
 ↓

Compare
 ↓

Skinning Weights
```

---

## Interpretation

Nearest keypoint acts as a self-generated segmentation label.

---

# 5. Shape Variance Loss

## Purpose

Prevents the residual shape decoder from creating unrealistic deformations.

Without it:

* category template collapses
* network memorizes every object

---

## Formula

[
L_{shape-var}
=============

1-
\exp
\left(
-\lambda_{var}
|V|^2
\right)
]

where:

* (V) = residual shape offsets

---

## Flow

```text
Category Template
      +
Residual Shape
      ↓

Final Shape
```

Penalize large residuals.

---

## Interpretation

Keeps instances close to the shared category geometry.

---

# 6. Joint Direction Alignment Loss

## Purpose

Encourages predicted joint axes to align with actual articulation boundaries.

Without it:

* hinge direction becomes arbitrary

---

## Formula

[
L_{dir-align}
=============

## 1

|
\langle
d,
\tilde d
\rangle
|
]

where:

* (d) = predicted axis
* (\tilde d) = PCA direction from boundary points

---

## Flow

```text
Boundary Points
      ↓
PCA
      ↓
Reference Axis
      ↓
Compare
      ↓
Predicted Joint Axis
```

---

## Interpretation

For a laptop:

```text
Expected Axis
============
Hinge Direction
```

not random 3D directions.

---

# 7. Joint-to-Shape Proximity Loss

## Purpose

Keeps joint pivots close to the object surface.

Without it:

* pivots float in empty space

---

## Formula

[
L_{joint-prox}
==============

## 1

\exp
\left(
-\lambda_{prox}
,
\mathbb E
\left[
\min_i
|c-s_i|
\right]
\right)
]

where:

* (c) = pivot
* (s_i) = point cloud point

---

## Flow

```text
Joint Pivot
      ↓

Nearest Point
      ↓

Distance Penalty
```

---

## Interpretation

Laptop hinge should lie on the laptop.

Not 2 meters away in space.

---

# 8. Joint Boundary Attraction Loss

## Purpose

Pulls joint pivots toward segmentation boundaries.

Articulation usually occurs where parts meet.

Examples:

```text
Laptop Hinge
Drawer Rail
Scissor Joint
Door Hinge
```

---

## Boundary Entropy

[
H_i
===

*

\sum_p
w_i^{(p)}
\log
w_i^{(p)}
]

High entropy:

```text
Boundary Region
```

Low entropy:

```text
Part Interior
```

---

## Attention

[
\alpha_{j,i}
============

\frac{
e^{-|c_j-s_i|}
}{
\sum_k
e^{-|c_j-s_k|}
}
]

---

## Loss

[
L_{joint-boundary}
==================

*

\frac1J
\sum_j
\sum_i
\alpha_{j,i}
H_i
]

---

## Flow

```text
Segmentation
      ↓
Entropy
      ↓
Boundary Regions
      ↓
Attract Joint Pivot
```

---

# Complete SCAPO Training Pipeline

```text
Point Cloud X
      ↓
SE(3) Encoder
      ↓
Latent Zx
      ↓
─────────────────────────────
      ↓
KeypointNet
      ↓
Keypoints
      ↓
Mahalanobis Distance
      ↓
Skinning Field
      ↓
Segmentation Weights
─────────────────────────────
      ↓
PoseNet
      ↓
Pivot Axis Angle
      ↓
Bone Transform
─────────────────────────────
      ↓
Blend Skinning
      ↓
Canonical Shape
      ↓
Forward Deformation
      ↓
Reconstruction
─────────────────────────────
      ↓
Losses

Lcycle
Lrecon
Lkp-seg
Lseg
Lshape-var
Ldir-align
Ljoint-prox
Ljoint-boundary
```

# Intuition

The losses can be grouped into three categories:

### Geometry

[
L_{cycle},
L_{recon},
L_{shape-var}
]

Ensure shape consistency.

---

### Segmentation

[
L_{kp-seg},
L_{seg}
]

Ensure meaningful rigid parts.

---

### Articulation

[
L_{dir-align},
L_{joint-prox},
L_{joint-boundary}
]

Ensure physically realistic joints.

Together these losses allow SCAPO to discover articulated structure without any ground-truth joint annotations or segmentation labels.
