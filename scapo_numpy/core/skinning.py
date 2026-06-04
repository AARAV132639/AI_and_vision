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