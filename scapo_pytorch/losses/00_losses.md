# SCAPO Loss Functions – Complete Reference

## Overview

The SCAPO framework learns articulated object representations through two major stages:

### Stage 1 – SE(3) Canonicalization

Goal:

```text
Observed Shape
↓
Remove Translation
↓
Remove Rotation
↓
Canonical Shape
```

This stage learns pose invariance.

---

### Stage 2 – Articulation Discovery

Goal:

```text
Canonical Shape
↓
Keypoint Discovery
↓
Bone Assignment
↓
Joint Prediction
↓
Articulated Object Understanding
```

This stage learns articulation structure.

---

# Stage 1 Losses

---

# 1. Reconstruction Loss

File:

```text
reconstruction_loss.py
```

## Purpose

Forces latent features to preserve shape information.

Pipeline:

```text
z_inv
↓
Decoder
↓
Reconstructed Shape
```

Loss:

[
L_{rec}
=======

||S_{pred}-S_{gt}||
]

Typically:

```text
L1 Loss
```

or

```text
MSE Loss
```

---

## What It Does In Paper

Ensures the latent representation contains enough information to reconstruct the canonical object.

Without it:

```text
Latent can collapse
```

into meaningless features.

---

## Current Implementation

```python
L_rec =
L1(
    reconstruction,
    canonical_gt
)
```

---

## Limitation

Current decoder tends to learn:

```text
Average Laptop
```

because dataset diversity is low.

Therefore reconstruction loss currently provides weak supervision.

---

# 2. Orthogonality Loss

File:

```text
orthogonality_loss.py
```

## Purpose

Ensures predicted rotation matrix remains a valid SO(3) rotation.

Valid rotation:

[
R^TR = I
]

Loss:

[
L_{ortho}
=========

||R^TR-I||^2
]

---

## What It Does In Paper

Prevents invalid rotations.

Keeps canonicalization mathematically correct.

---

## Current Implementation

```python
RtR = R.T @ R
loss = (RtR-I)^2
```

---

## Limitation

Does NOT ensure:

[
det(R)=1
]

although Gram-Schmidt largely handles this.

---

# 3. Canonical Consistency Loss

File:

```text
canonical_consistency_loss.py
```

## Purpose

Different views of the same object should map to the same canonical representation.

Example:

```text
Laptop Rotated 90°
Laptop Rotated 180°

↓

Same Canonical Shape
```

Loss:

[
L_{can}
=======

||C_1-C_2||
]

---

## What It Does In Paper

Directly teaches pose invariance.

One of the most important Stage-1 losses.

---

## Current Implementation

```python
L1(
    canonical_a,
    canonical_b
)
```

---

## Limitation

Currently requires manually generating two views.

Not yet integrated into training.

---

# 4. Augmentation Loss

File:

```text
augmentation_loss.py
```

## Purpose

Canonical shape should remain stable under perturbations.

Examples:

```text
Noise
Dropout
Jitter
```

Loss:

[
L_{aug}
=======

||C_{aug1}-C_{aug2}||
]

---

## What It Does In Paper

Improves robustness.

Prevents overfitting to exact point locations.

---

## Current Implementation

Mathematically identical to canonical consistency.

---

## Limitation

Currently does not use actual augmentation operators.

Only compares two tensors.

---

# Stage 2 Losses

---

# 5. Segmentation Loss

File:

```text
segmentation_loss.py
```

## Purpose

Assign every point to a part.

Pipeline:

```text
Point
↓
Nearest Keypoint
↓
Pseudo Label
↓
Compare with Predicted Weight
```

Loss:

[
L_{seg}
=======

||W-W_{target}||
]

---

## What It Does In Paper

Encourages meaningful bone assignments.

---

## Current Implementation

Nearest keypoint becomes segmentation label.

Uses:

```python
torch.cdist()
```

and MSE.

---

## Limitation

Nearest keypoint is only a proxy.

Real articulated datasets contain more complex boundaries.

---

# 6. Keypoint Segmentation Loss

File:

```text
kp_seg_loss.py
```

## Purpose

Encourages keypoints to stay inside their assigned segments.

Pipeline:

```text
Segment
↓
Weighted Centroid
↓
Keypoint
```

Loss:

[
L_{kp}
======

||K-C||
]

where:

```text
K = keypoint
C = segment centroid
```

---

## What It Does In Paper

Aligns discovered keypoints with discovered parts.

---

## Current Implementation

Weighted centroid of segmentation weights.

---

## Limitation

Paper uses segmentation-guided learning.

Centroid matching is an approximation.

---

# 7. Joint Proximity Loss

File:

```text
joint_proximity_loss.py
```

## Purpose

Joint pivots should lie close to object geometry.

Pipeline:

```text
Pivot
↓
Nearest Point
↓
Distance
```

Loss:

[
L_{prox}
========

f(d)
]

---

## What It Does In Paper

Prevents floating joints.

---

## Current Implementation

```python
1-exp(-λd)
```

---

## Limitation

Current formulation saturates quickly.

Large λ values may produce unstable gradients.

Future version should use:

```python
mean(d)
```

or

```python
d²
```

instead.

---

# 8. Joint Boundary Loss

File:

```text
joint_boundary_loss.py
```

## Purpose

Encourages joints to appear near articulation boundaries.

Pipeline:

```text
Pivot
↓
Boundary Region
```

Loss:

[
L_{boundary}
============

||Pivot-Keypoint||
]

---

## What It Does In Paper

Helps discover realistic hinge locations.

---

## Current Implementation

Distance between pivot and keypoint.

---

## Limitation

Extremely simplified.

Does not explicitly model boundaries.

---

# 9. Direction Alignment Loss

File:

```text
direction_align_loss.py
```

## Purpose

Predicted motion axis should align with actual articulation direction.

Loss:

[
L_{dir}
=======

1-|a^Tb|
]

where:

```text
a = predicted axis
b = motion direction
```

---

## What It Does In Paper

Encourages physically meaningful articulation axes.

---

## Current Implementation

Cosine similarity loss.

---

## Limitation

Requires motion direction labels.

Synthetic dataset currently lacks these labels.

---

# 10. Shape Variance Loss

File:

```text
shape_variance_loss.py
```

## Purpose

Prevents all keypoints from collapsing into one location.

Loss:

[
L_{var}
=======

\frac{1}{\text{mean pairwise distance}}
]

---

## What It Does In Paper

Encourages coverage of the object.

Improves articulation discovery.

---

## Current Implementation

Pairwise keypoint distance regularization.

---

## Limitation

Much simpler than paper's Shape Variance Decoder.

Acts only as a geometric spread constraint.

---

# Current Reproduction Status

## Stage 1

```text
✓ Reconstruction Loss

✓ Orthogonality Loss

✓ Canonical Consistency Loss

✓ Augmentation Loss
```

---

## Stage 2

```text
✓ Segmentation Loss

✓ Keypoint Segmentation Loss

✓ Joint Proximity Loss

✓ Joint Boundary Loss

✓ Direction Alignment Loss

✓ Shape Variance Loss
```

---

# Important Research Note

The current implementations are intentionally lightweight reproductions.

They capture the core intuition of the paper but are not exact ECCV-level implementations.

The primary goal was:

```text
Understand
↓
Reproduce
↓
Experiment
↓
Improve
```

Future work should focus on:

1. Better dataset diversity
2. Stronger canonicalization supervision
3. Exact Shape Variance Decoder
4. Improved joint boundary modeling
5. Real Shape2Motion / HOI4D training
6. Edge-AI and Defence deployment adaptations

```
```
