scapo/

├── datasets/
│   ├── shape2motion.py
│   ├── hoi4d.py
│   └── synthetic_dataset.py
│
├── models/
│
│   ├── se3/
│   │
│   │   ├── so3.py
│   │   ├── vnt.py
│   │   ├── canonicalizer.py
│   │   └── se3_autoencoder.py
│   │
│   ├── vnn/
│   │
│   │   ├── vn_linear.py
│   │   ├── vn_relu.py
│   │   ├── vn_pool.py
│   │   └── vn_pointnet.py
│   │
│   ├── keypoint_net.py
│   ├── pose_net.py
│   ├── bone_gaussian.py
│   ├── mahalanobis.py
│   ├── bone_transform.py
│   ├── blend_skinning.py
│   │
│   ├── scapo_v0.py
│   └── scapo_v1.py
│
├── losses/
│
│   ├── reconstruction_loss.py
│   ├── cycle_loss.py
│   ├── orthogonality_loss.py
│   ├── augmentation_loss.py
│   ├── canonical_loss.py
│   │
│   ├── kp_seg_loss.py
│   ├── seg_loss.py
│   ├── shape_var_loss.py
│   ├── joint_prox_loss.py
│   ├── joint_boundary_loss.py
│   └── dir_align_loss.py
│
├── training/
│
│   ├── train_stage1.py
│   ├── train_stage2.py
│   └── train_full.py
│
├── tests/
│
│   ├── test_so3.py
│   ├── test_vn_linear.py
│   ├── test_vn_relu.py
│   ├── test_vn_pool.py
│   ├── test_vn_pointnet.py
│   ├── test_stage1.py
│   └── test_stage2.py
│
├── configs/
│   ├── stage1.yaml
│   ├── stage2.yaml
│   └── full.yaml
│
└── results/