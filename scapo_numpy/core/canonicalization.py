import numpy as np

def transform_to_canonical(X,R_g, t_g):

    """
    
    1. Transform an observed point cloud into the canonical forms

    2. Fromula: S_obj = (X - t_g)@R_g

    3. Parameters:
                - X : (N,3) observed point cloud
                - R_g: (3,3) global rotation matrix
                - t_g : (3,) global translation vector
    
    4. Returns: S_obj: (N,3) canonical point cloud
    
    """

    #Remove global translation
    
    centered_X = X-t_g

    # Apply inverse rotation
    #Since R_g is orthogonal
    # R_g-1 = R_gT

    S_obj= centered_X@R_g

    return S_obj

def reconstruct_observation(S_obj, R_g, t_g):

    """
    1. Transform a canonical point cloud back to observation space

    2. Formula: X_hat = S_obj @ R_gT +t_g

    3. Parameters:
                - S_obj : (N,3) canonical point cloud
                - R_g : (3,3) global rotation matrix
                - t_g: (3,) global translation vector
    
    4. Returns: X_hat: (N,3) reconstruted observation
    """
    X_hat = (S_obj@R_g.T) + t_g

    return X_hat

if __name__=="__main__":

    np.random.seed(0)

    #random point cloud
    input_points = np.random.rand(5,3)

    #90 degree rotation around z-axis
    R = np.array([
        [0,-1,0],
        [1,0,0],
        [0,0,1]
    ])

    #Translation
    t = np.array([1.0,2.0,-3.0])

    #Observation--->canonical
    canoncial_points = transform_to_canonical(input_points, R, t)

    # Canonical----> observation
    reconstructed_points = reconstruct_observation(canoncial_points, R, t)

    print("Perfect reconstruction?")

    print(np.allclose(input_points, reconstructed_points))