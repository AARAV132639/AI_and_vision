"""
1. This module is where we combine the point segments together.

2. It assigns each point a soft wegith representing which part it belongs to (the skinning field) and then blends the rigid transformations together u
    using Linear Blend Skinning(LBS)
"""

import numpy as np

def compute_mahalanobis_distance(points, keypoint, Q):

    """
    1. Computes the squared Mahalanobis distance between each point and a bone center.

    - D^2 = (s_i- O_p)^Q_p(s_i-O_p)

    2. Parameters:
        - points: (N,3) point coud
        - keypoint: (3,) bone center
        - Q : (3,3) precision matrix
    
    3. Returns: distances: (N,) squared Mahalanobis distances
    """

    #shift points relative to keypoint
    diff= points- keypoint

    # Efficient computation of xTQx for all points
    distance = np.sum((diff@Q)*diff,aix=1)

    return distance 

def get_skinning_weights(
        points,
        keypoints,
        orientations,
        scales,
        gamma=0.1
):
    """
    1. Computes soft skinning weights using a temperature-controlled softmax

    Parameters:
                - points: (N,3)
                - keypoints: list of P bone centers
                - orientations: list of P rotation matrics (3x3)
                - scales: list of P diagonal scale matrices (3x3)
                - gamma: softmax temperature
    
    Returns: 
            - weights: (N,P)

    Each Row sums to 1
    """

    N= points.shape[0]
    P= len(keypoints)

    raw_distance= np.zeros((N,P))

    for p in range(P):

        V= orientations[p]
        Lambda = scales[p]

        #Precision matrix
        Q= V.T@Lambda@V

        raw_distance[:,p] = compute_mahalanobis_distance(
            points,
            keypoints[p],
            Q
        )

        # Temperature- controlled softmax
        scaled_dist = -raw_distance/gamma

        #Numerical stability trick
        max_val = np.max(
            scaled_dist,
            axis=1,
            keepdims= True 
        )

        exp_weights = np.exp(
            scaled_dist = - max_val 
        )

        weights= exp_weights/ np.sum(
            exp_weights,
            axis=1,
            keepdims= True 
        )

    return weights 

def linear_blend_skinning(
        points,
        weights,
        transformations
):
    """
    1. Applies Linear Blend Skinning (LBS)

    - For each points:

                    - T_blended = Summation(w_i,p*t_p)
                    - transformed_points= T_blended@[x,y,z,1]^T
    
    - Parameters:
                - points : (N,3)
                - weights : (N,P)
                - transformations: List of P SE(3) matrices
    
    - Returns : transformed_points: (N,3)
    """

    N= points.shape[0]

    transformed_points = np.zeros_like(points)

    # Convert to homogenous co ordinates
    homogenous_points = np.hstack([
        points,
        np.ones((N,1))
    ])

    for i in range(N):

        T_blended = np.zeros((4,4))

        for p in range(len(transformations)):

            T_blended+= (
                weights[i,p]* transformations[p]
            )

            transformed_points[i]= (T_blended@homogenous_points[i])[:3]

    return transformed_points

if __name__=="__main__":

    #two sample points

    mock_points = np.array([
        [0.0,0.0,0.2],
        [0.0,0.0,-0.2]
    ]) 

    # skinning weights
    mock_weights = np.array([
        [0.9,0.1],
        [0.1,0.9] 
    ])   

    #Identity transforms
    mock_transformations = [
        np.eye(4), np.eye(4)
    ]   

    output= linear_blend_skinning(
        mock_points,
        mock_weights,
        mock_transformations 
    )

    print(
        "Blended output points"
        "(should match input):"
    )
    print (output)
    
