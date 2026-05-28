 
 ### Ref: multi_channel_convolution.py
 if padding>0:
        image= np.pad(
            image,
            ((0,0),(padding,padding),(padding,padding)),
            mode= 'constant'
        )
   
    1. Explain the padding code block in your own words
    2. Why are we using 0 padding?
    
### Ref: main.py
- Why are we changing the dimension of image

 To-do:
    
    2. Pooling on multichannel feature maps
    3. Build full CNN blocks