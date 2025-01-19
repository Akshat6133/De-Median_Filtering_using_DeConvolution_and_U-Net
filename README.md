# Median Filtering Detection using U-Net and Deconvolution
- Presented By Akshat Arora (B.Tech 2021-25) @IIT_Bhilai


the dataset generated at different steps are mentioned in the presentation pdf. the path to the dataset are given at the start of the code files. Modify the path to dataset directories to that of the actual paths in your system.


# Variational Deconvolution Network
 A variational deconvolutional framework is a method used in image processing to recover clear images from blurry ones. It combines deep learning techniques with variational principles to improve the quality of deblurred images1.

Here's a simplified explanation:

Deep Image Prior (DIP): This approach uses a deep neural network to act as a regularizer for a single image, helping to recover sharp images from blurry ones.

Variational Deep Image Prior (VDIP): This method enhances DIP by adding constraints on both the blur kernel and the recovered image, improving the optimization process.

Total Generalized Variation (TGV): This regularization technique helps to reduce artifacts and maintain sharp edges in the recovered images by considering the characteristics of partial derivatives at different scales.

By combining these techniques, the variational deconvolutional framework can effectively recover image details and edges, resulting in higher-quality deblurred images.



Variational Deconvolution Network  = Variational + Deconvolution

# Deconvolution

A deconvolutional framework is used in image processing and deep learning to reverse the effects of convolution. Convolution is a mathematical operation that blends two functions, often used in image processing to apply filters like blurring, sharpening, and edge detection. Deconvolution, on the other hand, aims to reconstruct the original image from the convolved (blurred) image.

Here's a more technical breakdown:

Deconvolutional Layers: These layers are used to perform the reverse operation of convolution. They use the same receptive fields from the convolution layer but flip them 180° horizontally and vertically1.

Applications: Deconvolutional frameworks are commonly used in tasks like image denoising, super-resolution, and feature visualization in deep learning models.

Deep Learning: In deep learning, deconvolutional networks (also known as transposed convolutional networks) are used to upsample feature maps, helping in tasks like semantic segmentation and generating detailed images from lower-resolution inputs.


# what is the variational part in variational deconvolutional framework

The variational part in a variational deconvolutional framework refers to the use of variational principles to impose constraints on the optimization process. This involves formulating the problem as a convex optimization task, where the goal is to minimize a cost function that includes terms for both the data fidelity and regularization1.

In the context of image deconvolution, the variational approach typically involves:

Defining a cost function: This function includes terms for the fidelity of the reconstructed image to the observed blurry image and regularization terms that impose smoothness or other priors on the solution.

Imposing constraints: Variational methods add constraints on the blur kernel and the recovered image, ensuring that the solution adheres to certain properties, such as sparsity or edge preservation.

Optimization: The optimization process is often solved using techniques like the Alternating Direction Method of Multipliers (ADMM), which combines traditional optimization algorithms with deep learning methods.

By incorporating these variational principles, the framework can effectively recover image details and edges while reducing artifacts.




