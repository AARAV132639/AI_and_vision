import numpy as np

def get_skew_symmetric_matrix(vector):

    """
    - Computes 3x3 skey-symmetric matrix of a 3D vector
    - Used to implement cross- product operations as matrix multiplication
    - Turns vector physics into simple, clean matrix multiplication
    """

    x,y,z= vector

    return np.array([
        [0,-z,y],
        [z,0,-x],
        [-y,x,0]
    ])

def rodrigures_rotation(axis, angle):

    """
    
    1. Rodrigues' Rotation Formula (SO(3) exponentioal map)

    - R= I + sin(theta)[K]x +(1-cos(theta))[K]x^2

    2. Parameters:
    - axis: rotation axis(3D vector)
    - ange: rotation angle (radians)

    3. Returns: 3x3 rotation matrix
    
    """

    # Normalize axis
    axis= axis/np.linalg.norm(axis)

    I= np.eye(3)
    K= get_skew_symmetric_matrix(axis)

    R=(
        I
        + np.sin(angle)*K
        +(1-np.cos(angle))*np.dot(K,K)
    )

    return R

def compute_bone_transformation(
        pivot,
        axis,
        angle,
        joint_type="revolute"
):
    
    """
    1. Computes a 4x4 SE(3) transformation matrix

    2. Revolute Joint:
        - Translate pivot to origin
        - Rotate around axis
        - Translate back

    3. Prismatic Joint: Slides along the given axis

    4. Parameters:
        - pivot: joint center
        - axis: rotation/translation axis
        - angle: rotation angle(or translation distance)
        - joint_type: 'revolute' or 'prismatic' 

    5. Returns: 4x4 transformation matrix

    6. By unifying rotation (R) and translation(t) into a single 4x4 array we can apply complex rigit movements to millions of 3D coordignates simulateneously
    """

    T= np.eye(4)

    if joint_type=="revolute":

        # Rotation matrix
        R= rodrigures_rotation(axis, angle)

        # Translation needed to keep pivot fixed
        t = pivot- np.dot(R,pivot)

    elif joint_type=="prismatic":

        #No rotation
        R= np.eye(3)

        # Translation along axis
        t= axis*angle

    else:
        raise ValueError(
            "Unknown joint type. Choose revolute or 'prismatic'"
        )
    
    # Assemble SE(3) matrix
    T[:3,:3] =R
    T[:3,3] = t

    return T

if __name__ =="__main__":

    # example: rotate 90 around X-axis

    pivot = np.array([0.0,0.0,0.0])
    axis= np.array([1.0,0.0,0.0,])

    angle= np.pi/2

    T_matrix= compute_bone_transformation(
        pivot,
        axis,
        angle 
    )

    print("Generate 4x4 SE(3) Matrix:")
    print(np.round(T_matrix,4))


