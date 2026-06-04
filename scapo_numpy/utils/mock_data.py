import numpy as np

def generate_mock_articulated_object(num_points=1024):

    """
     - Generates a mock two-part articulated object(laptop/box hinge) centered at origin
     - Part 0 (Base): Lower half(z<0)
     - Part 1 (Lid): Upper half(z>=0)

     returns:
     - points(np.ndarray): Shape(num_points,3)---> the composite 3D point cloud.
     - ture_segmentation(np.ndarray): shape(num_points,)- Integer array indicating Part 0 or 1,
    
    """

    np.random.seed(42) #For reproducible geometry
    
    #Generate random points inside a unit bounding box [-0.5,0.5]

    raw_points= np.random.uniform(-0.5,0.5,size=(num_points,3))

    # Create a clear geometric boundary at the Z-axis plane (z=0)
    # Points with z>=0 are designated as the moving part(part 1)
    # points with z<0 are designated as the static base (Part 0)

    true_segmentation= (raw_points[:,2]>=0).astype(int)

    return raw_points, true_segmentation

if __name__=="__main__":

    pts,seg= generate_mock_articulated_object()

    print(f"Generated point cloud shape:{pts.shape}")

    print(f"Points in Base(Part 0):{np.sum(seg==0)}")

    print(f"Points in Lid (Part 1): {np.sum(seg==1)}")