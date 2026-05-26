import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import FashionMNIST
from convolution import convolve2d

# Load dataset

dataset= FashionMNIST(
    root="datasets",
    train= True,
    download= False
)

# Get first image
image, label= dataset[0]

#Convert PIL image to numpy
image= np.array(image)

#Simple edge detection kernel
kernel = np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])

# Running convolution
output= convolve2d(
    image= image,
    kernel= kernel,
    stride=1,
    padding=1
)

#Show original +result
plt.figure(figsize= (8,4))

plt.subplot(1,2,1)
plt.imshow(image, cmap= 'gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(np.abs(output),cmap='gray') 

"""
- By doing np.abs(output) negative edges were visible too
- with using only "output" only positive edges were visible

- what do you mean by negative edges?
"""
plt.title("Convolved")
plt.axis("off")

plt.show()