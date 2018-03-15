We upload our project files here for the course GA-2569 Inference and Representation. 
The details about the course can be found [here](https://inf16nyu.github.io/home/). 

In this project, we adapted the convolutional neural network architecture of [U-Net](https://arxiv.org/pdf/1505.04597.pdf) to detect occluded area between two successive video frames. 
For this supervised learning problem, we have constructed fully convolutional networks. 
During the convolution phase, we built 5 levels and 3 sublayers per level. 
At each level, we used 2x2 pooling and 3x3 kernels. 
During the deconvolution phase, we concatenated with a corresponding convolution layer. 

The model was able to detect the shape of occlusion. 
As a quantitative evaluation, we compared each pixel to the ground truth. 
We counted how many pixels were predicted correctly. The matching rate was approximately 90\% on average. 
Since occluded area is relatively small in a whole image, this value would be higher than the truth. 
Thus, we adopted qualitative analysis for better evaluation.

![eval](https://user-images.githubusercontent.com/24445335/37490949-98eae1e8-2872-11e8-888f-2cd60a08abbb.png)

The left are the predicted images and the right are the ground truth. It shows that the model captured the shape of occluded area, but it did not capture the subtle change between frames. For more details, please refer to the project poster.
