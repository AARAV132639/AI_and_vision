import numpy as np

def multi_channel_convolve(image, kernel, stride=1, padding=0):
    """
    Performing convolution on multi-channel image.

    Parameters:
                i. image: Shape(C,H,W)---> Channel height width
                ii. kernal: Shape(C,KH,KW)---> Kernel should match the dimension of input
                iii. stride: movement setup
                iv. zero: padding
    
    Returns: output feature map
    """

    channels, height, width= image.shape
    kernel_channels, kernel_h,kernel_w= kernel.shape

    #Sanity check:
    if channels!= kernel_channels:
        raise ValueError("Image and kernel channels must match")
    
    # Applying padding:

    if padding>0:
        image= np.pad(
            image,
            ((0,0),(padding,padding),(padding,padding)),
            mode= 'constant'
        )
    """
    1. Explain the padding code block in your own words
    2. Why are we using 0 padding?
    """

    _, padded_h, padded_w= image.shape
    output_h=((padded_h-kernel_h)//stride)+1
    output_w= ((padded_w-kernel_w)//stride)+1

    output= np.zeros((output_h, output_w))

    # Sliding kernel
    for i in range(output_h):
        for j in range(output_w):

            vert_start= i*stride
            vert_end= vert_start+kernel_h

            horiz_start= j*stride
            horiz_end= horiz_start+kernel_w

            total=0

            for c in range (channels):
                image_slice= image[c,vert_start:vert_end,horiz_start:horiz_end]
                total+= np.sum(image_slice*kernel[c])

            output[i,j]= total

    return output

"""
What changed from Phase_01/src/datset.py?

- Earlier: window*kernel

- NOw--->
            for each channel:
                window[channel]*kernel[channel]
            sum everything
        
    So basically, Ouput= Conv(R,Kr)+ Conv(G,Kg)+Conv(B,Kb)

- Why are we doing this?

Because an RGB pixel has 3 channels. CNN learns cross-channel features like: Color contrast, texture boundaries, object edges etc

"""