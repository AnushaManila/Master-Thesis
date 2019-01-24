# Generation of Faces with GANs-The DCGAN
One of the strong candidates for unsupervised learning was introduced in by Radford et al., in 2016. They presented Deep Convolutional Generative Adversarial Network (DCGAN)s which employ fractionally-strided convolutions to upsample images. Though GANs were both deep and convolutional prior to DCGANs, the name DCGAN is useful to refer to this specific style of architecture. Since then, DCGAN has been widely used by computer vision researchers. However, there is no referable document presenting and summarizing the experiments on altering the intermediate layers of DCGAN network and extending the model architecture. Starting with an overview, this section aims to provide a discussion on several experiments directed in this regard.
## Getting Started

The instructions in acknowledgement, will provide details on how to deploy the project on a live system. The folder named 'Experiments' includes the implementation of all the DCGAN experiments and the folder named 'Results' has GIF images of generated output of each experiment.

## DCGAN experiments
Original DCGAN Architecture:
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide23.jpg)</br>
A 100 dimensional uniform distribution Z is projected to a small spatial extent convolutional representation with many feature maps. A series of four fractionally-strided convolutions then convert this high level representation into a 64×64 pixel image. Notably, no fully connected or pooling layers are used. [Further details](http://bamos.github.io/2016/08/09/deep-completion/) </br>

Several experiments are carried out to understand what happens when the parameters of DCGAN architecture are altered and are classified into four categories mentioned below:

```
1. Changing the depth of intermediate layers
2. Altering the latent space dimension
3. Up-sampling based on interpolations
4. Extending the DCGAN architecture
```
#### 1. Changing the Depth of Intermediate layers of DCGAN:
The DCGAN generator architecture shown above has the depths (1024, 512, 256, 128) of each intermediate layers which are formed by transposed convolution operations. These depth values are independent of the final output resolution. However, changing these depth values might affect the quality of generated results. In order to examine the effect of modifying the depth values, experiment 1 to 5 are executed. The red box shows the changes made to the generator architecture. And the corresponding discriminator network is also altered accordingly (just the reverse process of the generator).
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide24.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide25.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide26.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide27.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide28.jpg)</br>
Changing the depth values of the generator architecture (Experiment 1 to 5) and keeping the latent space dimension constant, impacts the final image quality as shown below:
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide29.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide30.jpg)</br></br>
#### 2. Altering the latent space dimension:
The generator network of DCGAN takes a latent vector (also called as z vector) as input and generates fake images. Commensing with the smallest possible latent space dimension i.e, 1, the dimension value increased up to 512 (z in R1 to z in R512) in experiment 6 to 10. In these experiments, the random points for the latent vectors are uniformly sampled over [0 to 1] from different dimensional hyperspheres. 
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide31.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide32.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide33.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide34.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide35.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide36.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide37.jpg)
</br></br>
#### 3. Up-sampling based on interpolations:
In addition to Transposed Convolution for Upsampling, Nearest Neighbor and Bilinear interpolations are employed to upsample the feature maps of the generator and corresponding results are as shown in the figure below.
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide39.jpg)</br></br>
Certainly, DCGAN architecture with nearest neighbor up-sampling generates comparable results to the experiment of transposed convolution operation and the generated images are better than the bilinear interpolation based model results. The model based on nearest neighbor up-sampling (the curve in orange) converges better as seen from the Generator and Discriminator loss curves.
#### 4. Extending the DCGAN architecture:
The original DCGAN architecture proposed by Radford et.al., generates low resolution images (64X64 pixel only). To generate images with higher resolution, the Generator and Discriminator architectures of DCGAN are extended. 
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide40.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide42.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide44.jpg)</br>
When the resolution doubled to 128X128 pixel, total trainable parameters become 47.8M and 44.6M for generator and discriminator individually. The sample results are appear to be consistent with the lower resolution results. However, at 256X256 pixel resolution, total trainable parameters escalate to 185.3M and 178.8M for generator and discriminator respectively. Moreover, the image features gradually become undesirable. Similarly, up-sampling the resolution to 512X512 by extending the original architecture and using the CelebA dataset generated no good results. This is evident because generating higher quality images by training low resolution dataset is not practicable. However, additional experiment with the smaller but high quality dataset (CelebA HQ), shows better results.
![](https://github.com/AnushaManila/Master-Thesis/blob/master/03_DCGAN/Results/DCGAN_128.png)
![   ](https://github.com/AnushaManila/Master-Thesis/blob/master/03_DCGAN/Results/DCGAN_512_CelebAHQ.png)

Note: The above pictures are directly taken from my master's thesis report.
## Acknowledgment

* https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN


