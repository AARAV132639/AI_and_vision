import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

from src.multi_filter_convolution import multi_filter_convolve
#from src.multi_channel_convolution import multi_channel_convolve
from src.pooling import max_pool_multi_channel


# Loading dataset
dataset= CIFAR10(
    root="./src/datasets",
    train= True,
    download= False
)

# Getting first sample
image, label= dataset[0]

#Convert PIL image---> numpy
image= np.array(image)

print("Original image shape:", image.shape)

#converting (H,W,C) ---> (C,H,W)
image= np.transpose(image,(2,0,1))
print ("Transposed image shape:", image.shape)

## Using single filter:
"""
# Random RGB kernel
kernel= np.random.randn(3,3,3)

# Running convolution
output= multi_channel_convolve(
    image= image,
    kernel= kernel,
    stride=1,
    padding=1
)
"""

#Create 8 random RGB filters
kernels= np.random.randn(8,3,3,3)

# Run covoution:
feature_maps= multi_filter_convolve(
    image= image,
    kernels= kernels,
    stride=1,
    padding=1
)

print("Feature maps shape:", feature_maps.shape)

## Update 4: Adding activation function
from src.activations import relu
feature_maps= relu(feature_maps)

pooled_maps= max_pool_multi_channel(feature_maps, pool_size=2, stride=2)
print("After pooling:", pooled_maps.shape)

## Update 3: Adding 2nd convolution layer
second_layer_kernels= np.random.randn(4,8,3,3)
second_feature_maps= multi_filter_convolve(
    image= feature_maps,
    kernels= second_layer_kernels,
    stride=1,
    padding=1
)

#Appling ReLu on 2nd layer
second_feature_maps= relu(second_feature_maps)


# Normalizing display
"""
- Why are we normalizing dispaly though?

"""
normalized_maps= []

for fmap in feature_maps:
    norm= (fmap-fmap.min())/(fmap.max()-fmap.min())
    normalized_maps.append(norm)


# plot original +feature maps
plt.figure(figsize=(14,8))
plt.subplot(3,3,1)
plt.imshow(np.transpose(image,(1,2,0)))
plt.title("Original RGB")
plt.axis("off")

for i in range(8):
    plt.subplot(3,3,i+2)
    plt.imshow(normalized_maps[i], cmap='gray')
    plt.title(f"Filter{i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# plotting pooled image
plt.figure(figsize=(12,8))

for i in range(8):
    fmap= pooled_maps[i]

    norm= (fmap-fmap.min())/(fmap.max()-fmap.min())

    plt.subplot(2,4,i+1)
    plt.imshow(norm,cmap='gray')
    plt.title(f"pooled{i+1}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# plotting 2nd convolution layer
plt.figure(figsize=(10,8))

for i in range(4):
    fmap= second_feature_maps[i]
    norm= (fmap-fmap.min())/(fmap.max()-fmap.min())

    plt.subplot(2,2,i+1)
    plt.imshow(norm,cmap='gray')
    plt.title(f"Layer 2 Map{i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()


"""

## Run 1: Applied 1 kernel with random weights to test the pipeline mechanics by importing multi_channel_convolution.py
    - Implemented the output in gray scale
    - Observation: Ironically, both the orignal and output were blurry on dispaly
    - Reason: CIFAR10 images are small in size having dimension 32x32 however when matplotlib dispalays it upscales it. Hence we got minecraft resolution dispaly

## Run 2: Applied 8 kernels with random weights and got 8 different features map of same
    - 1 RGB image
    - 8 filters
    - 8 feature maps

### Question time :)
    - Why are we getting the output of feature map in gray scale?
    Ans is simple dear, because the feature map is NOT AN RGB IMAGE!

    Yeah, that's right.
    - Input: (3,32,32)----> Meaning: 1. Red channel 2. Green channel 3. Blue channel
    - But after applying a filter we are technically doing: summation of all convolution applied on each channel
    - This sum produces: One scalar per pixel location and this becomes a single activation map. Hence grayscale
    - Which is why we are using 8 maps where each filter asks different questions like 
                                                                                        1. Do I see vertical edge?
                                                                                        2. Do I see red-green contrast?
                                                                                        3. Do I see texture etc etc
    So what did we learn?
    - Grayscale output is not a limitation but mathematically correct CNN behaviour

## Run 3: First CNN stack:
    - Input---> conv layer 1---> conv layer 2

## Run 4: Applied activation function: ReLU and Pooling
    - Many of the feature maps turned completely blank
    - Reason: negative signals and noise got suppresed. Only positive and strongest evidence survived
    - Earlier the activation functions like sigmoid, tanh caused 1. Vanishing gradient 2. Slow training. ReLu resolved much of this
    - However there are still limitations to it because of its binary nature time behaviour which actually leads to signal loss at higher depth.

"""