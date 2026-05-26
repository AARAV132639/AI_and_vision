import numpy as np

def zero_pad(image, pad):
    """
    1. Adds zero padding around a 2d image
    2. Parameters:
        image: numpy array of shape(H,W)
        pad: number of zeros to add around border

        returns: padded image
    """

    height, width= image.shape

    padded_height= height+2*pad
    padded_width= width+2*pad

    padded_image= np.zeros((padded_height, padded_height))

    padded_image[pad:pad+ height, pad: pad+width] = image

    return padded_image

if __name__=="__main__":
    sample= np.array([
        [1,2],
        [3,4]
    ])
    result = zero_pad(sample,1)
    print(result)
