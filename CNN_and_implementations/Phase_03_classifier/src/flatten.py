import numpy as np

def flatten(feature_maps):

    """
    Flatten multi-channel feature maps into 1 D vetor

    - Input: (C,H,W)
    - Output: (C*H*W)

    - Why flattening matters?
    Ans. Because CNN layers understand spatial structure. However dense layers understnand feature vectors.
        - Flatten is the bridge between them
    
    """
    return feature_maps.flatten()


if __name__=="__main__":
        sample= np.random.randn(8,16,16)

        output= flatten(sample)

        print("Original shape:", sample.shape)
        print("Flattened shape:", output.shape)