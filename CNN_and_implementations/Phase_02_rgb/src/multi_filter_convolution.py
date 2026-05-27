import numpy as np

def multi_filter_convolve(image, kernels, stride=1, padding=0):

    """
    Multi- filter convolution

    Parameters:
                1. Image:(C,H,W)
                2. kernels: (num_filters, C,KH, KW)
    
    Returns:
            feature maps:(num_filters, H_out, W_out)
    """

    channels, heights, width= image.shape
    num_filters, kernel_channels, kernel_h, kernel_w= kernels.shape

    if channels!= kernel_channels:
        raise ValueError("Channels mismatch")
    
    # padding
    if padding>0:
        image= np.pad(
            image, ((0,0), (padding,padding), (padding,padding)),
            mode= 'constant'
        )
    
    _,padded_h, padded_w= image.shape

    output_h =((padded_h- kernel_h)//stride)+1
    output_w= ((padded_w- kernel_w)//stride)+1

    output= np.zeros((num_filters, output_h,output_w))

    for f in range(num_filters):
        kernel= kernels[f]

        for i in range (output_h):
            for j in range(output_w):
                vert_start= i*stride
                vert_end= vert_start+kernel_h

                horiz_start = j*stride
                horiz_end= horiz_start+kernel_w

                total =0

                for c in range(channels):
                    image_slice = image[c,vert_start:vert_end, horiz_start:horiz_end]
                    total+= np.sum(image_slice*kernel[c])
                
                output[f,i,j]= total

    return output

"""
What is the difference now?

1. Earlier---> (3,3,3)
2. now---> kernels.shape(8,3,3,3). Meaning 8 seperate RGB kernels and each producing a feature map

Why this matters though?
Well my friend then let me inform you gladly...
We are entering the field where we'll resemble LeNET, AlexNet, VGG, ResNet shortly!

because Cnn layers are stacks of learned filters
"""


