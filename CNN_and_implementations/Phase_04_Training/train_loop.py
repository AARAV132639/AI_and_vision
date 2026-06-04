#Implementing full training loop on 20 Inputs. The loss eventually reaches a minimum with number of epochs
#update rule is: W_new = W - a(dL_dW)

import numpy as np

import sys
import os
# This tells python to look one folder up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from Phase_03_classifier.src.dense import DenseLayer
from Phase_03_classifier.src.softmax import softmax

from src.entropy_loss import cross_entropy_loss
from src.gradients import softmax_crossentropy_gradient

#HyperParameters
INPUT_SIZE= 20
NUM_CLASSES =3
LEARNING_RATE =0.1
EPOCHS= 10000

# Dense layer
dense= DenseLayer(
    input_size= INPUT_SIZE,
    output_size= NUM_CLASSES
)

# Fixed sample
x= np.random.randn(INPUT_SIZE)

# Correct class
true_class =2

for epoch in range (EPOCHS):

    #forward
    logits= dense.forward(x)

    probs= softmax(logits)

    loss= cross_entropy_loss(
        probs,
        true_class
    )

    #backward
    gradient = softmax_crossentropy_gradient(
        probs,
        true_class 
    )

    dense.backward(
        gradient,
        LEARNING_RATE 
    )

    if epoch%1000==0:
        print(
            f"Epoch{epoch:3d}|Loss={loss:.6f}"
        )