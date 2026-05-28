import numpy as np

def relu(x):

    """
    Applying ReLu function. ReLU(x)= max(0,x)    
    """
    return np.maximum(0,x)

if __name__=="__main__":
    sample = np.array([
        [-3,3,-1],
        [5,-7,8]
    ])
    result= relu(sample)
    print (result)