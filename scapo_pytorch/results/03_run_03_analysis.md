# Stage-1 Experiment #3 Analysis

## Goal

Train SE3AutoEncoder to recover a canonical laptop shape.

Pipeline:

Observed Shape
↓
VNT
↓
VNPointNet
↓
RotationHead
↓
Canonicalizer
↓
Canonical Shape

Loss:

L1(
    canonical_pred,
    canonical_gt
)

---

# Dataset

Each sample:

Canonical Laptop
↓
Articulation
↓
Random Rotation
↓
Random Translation
↓
Observed Shape

Target:

Canonical Laptop

---

# Training Result

Final Loss:

~0.47

Det(R):

1.0000

Training converged but plateaued.

---

# What Worked?

## 1. Rotation Prediction is Valid

det(R) ≈ 1

This means:

- Gram-Schmidt works
- Rotation matrices are valid
- SO(3) constraints are satisfied

No numerical instability.

---

## 2. VNN Stack Runs Correctly

Previously verified:

VNLinear Equivariance Error

≈ 1e-9

Therefore:

VN layers are mathematically correct.

---

## 3. Canonicalizer Works

Unit tests passed.

Equation:

Sobj = Rgᵀ(X − tg)

is implemented correctly.

---

# Why Did Loss Plateau?

The current Stage-1 is:

Observed Shape
↓
Predict tg
↓
Predict Rg
↓
Canonicalizer
↓
Canonical Shape

No decoder exists.

No reconstruction branch exists.

No latent supervision exists.

---

# Core Problem

The model only receives supervision at the very end.

Loss:

L1(
    canonical_pred,
    canonical_gt
)

This means:

RotationHead
VNPointNet
VNT

must learn everything indirectly.

Gradients are weak.

Optimization becomes difficult.

---

# Missing Components

The original SCAPO paper contains:

1. Decoder
2. Reconstruction Loss
3. Canonical Consistency Loss
4. Augmentation Loss
5. Shape Variance Decoder

These components create stronger learning signals.

---

# Why Decoder Is Needed

Current:

z_inv
↓
unused

The latent representation is never forced to encode shape.

Therefore:

VNPointNet can produce poor features.

---

With Decoder:

z_inv
↓
Decoder
↓
Reconstructed Canonical Shape

Loss:

L_recon

directly supervises the latent space.

---

# Interpretation

Stage-2 succeeded because:

- KeypointNet
- PoseNet
- Gaussian Bones

have direct supervision.

Stage-1 failed because:

- Translation estimation
- Rotation estimation
- Canonicalization

are only supervised indirectly.

---

# Conclusion

The Stage-1 plateau around 0.47 is expected.

The current implementation reproduces:

✓ SO(3) utilities

✓ VNN primitives

✓ Translation estimation

✓ Rotation estimation

✓ Canonicalization

but does not yet reproduce the full Stage-1 learning framework.

Next step:

Implement Decoder and Reconstruction Loss.