import numpy as np

from utils.mock_data import generate_mock_articulated_object

from core.canonicalization import(transform_to_canonical, reconstruct_observation)

from core.kinematics import(compute_bone_transformation)

from core.skinning import(get_skinning_weights, linear_blend_skinning)

def main():

    print("Scapo Numpy Pipeline simulation:")

    X_observed, true_seg = generate_mock_articulated_object()

    print(f"[Data]Loaded point cloud with"f"{X_observed.shape[0]}points")

    # Stage 1: Global cannonicalization

    mock_tg = np.array([0.1,-0.2,0.0])

    theta_g = np.pi/4 #45deg

    mock_Rg = np.array(
        [
            [np.cos(theta_g),0,np.sin(theta_g)],
            [0,1,0],
            [-np.sin(theta_g),0,np.cos(theta_g)]
        ]
    )

    S_obj = transform_to_canonical(
        X_observed,
        mock_Rg,
        mock_tg
    )

    print(
        "[Stage 1] Global pose removed"
        "Points shifted to canonical frame"
    )

    # Stage 2 : Bone Anchors & Skinning
    # --------------------------------------------------

    keypoints = [
        np.array([0.0, 0.0, -0.25]),  # Base
        np.array([0.0, 0.0, 0.25])    # Lid
    ]

    orientations = [
        np.eye(3),
        np.eye(3)
    ]

    scales = [
        np.eye(3) * 10.0,
        np.eye(3) * 10.0
    ]

    W = get_skinning_weights(
        S_obj,
        keypoints,
        orientations,
        scales,
        gamma=0.05
    )

    print(
        "[Stage 2] Skinning matrix W "
        "computed from Mahalanobis distances."
    )

    # --------------------------------------------------
    # 4. Stage 2 : Joint Articulation
    # --------------------------------------------------

    pivot_1 = np.array([0.0, 0.0, 0.0])

    axis_1 = np.array([1.0, 0.0, 0.0])

    angle_1 = np.pi / 6  # 30°

    T0 = np.eye(4)

    T1 = compute_bone_transformation(
        pivot_1,
        axis_1,
        angle_1,
        joint_type="revolute"
    )

    transformations = [T0, T1]

    # --------------------------------------------------
    # 5. Linear Blend Skinning
    # --------------------------------------------------

    S_articulated = linear_blend_skinning(
        S_obj,
        W,
        transformations
    )

    print(
        "[Stage 2] Applied Linear Blend "
        "Skinning deformation."
    )

    # --------------------------------------------------
    # 6. Reconstruction
    # --------------------------------------------------

    X_reconstructed = reconstruct_observation(
        S_articulated,
        mock_Rg,
        mock_tg
    )

    print(
        "[Pipeline] Forward transformation "
        "complete."
    )

    # --------------------------------------------------
    # 7. Verification Metrics
    # --------------------------------------------------

    mean_displacement = np.mean(np.linalg.norm(X_reconstructed - X_observed, axis=1 ))

    print(
        f"\n[Metrics] Mean point displacement: "
        f"{mean_displacement:.4f}"
    )

    sample_idx = np.where(true_seg == 1 )[0][0]

    print(
        f"[Verification] Point {sample_idx}"
        f"\nOriginal     : "
        f"{np.round(X_observed[sample_idx], 3)}"
    )

    print(
        f"Articulated  : "
        f"{np.round(X_reconstructed[sample_idx], 3)}"
    )


if __name__ == "__main__":
    main()