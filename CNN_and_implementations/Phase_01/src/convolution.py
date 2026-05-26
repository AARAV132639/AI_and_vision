import numpy as np

def convolve2d(image, kernel, stride=1, padding=0):
    """
    Performing 2D convolution manually

    Parameters:
        - image input image(H,W)
        - Kernel: filter/kernel(KH,KW)
        - stride: movement set up
        - padding: zero padding size
    
        returns: convolved output
    """

    # Step1: Apply padding if needed
    if padding>0:
        image= np.pad(
            image, ((padding,padding),(padding,padding)),
            mode= 'constant'
        )
    
    image_h, image_w = image.shape
    kernel_h, kernel_w= kernel.shape

    # Step2: Compute output dimensions
    output_h= ((image_h-kernel_h)//stride)+1
    output_w= ((image_w-kernel_w)//stride)+1

    output = np.zeros((output_h, output_w))

    # step3: Slide kernel over image
    for i in range(output_h):
        for j in range(output_w):

            vert_start= i*stride
            vert_end= vert_start+kernel_h

            horiz_start= j*stride
            horiz_end= horiz_start+kernel_w
            
            image_slice = image[vert_start:vert_end, horiz_start: horiz_end]
            output[i,j] = np.sum(image_slice*kernel)
    
    return output

if __name__ == "__main__":
    sample_image = np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ])

    sample_kernel = np.array([
        [1,0],
        [0,1]
    ])

    result = convolve2d(sample_image, sample_kernel)
    print(result)
        