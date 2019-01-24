# Generation of Faces with GANs-The DCGAN
One of the strong candidates for unsupervised learning was introduced in by Radford et al., in 2016. They presented Deep Convolutional Generative Adversarial Network (DCGAN)s which employ fractionally-strided convolutions to upsample images. Though GANs were both deep and convolutional prior to DCGANs, the name DCGAN is useful to refer to this specific style of architecture. Since then, DCGAN has been widely used by computer vision researchers. However, there is no referable document presenting and summarizing the experiments on altering the intermediate layers of DCGAN network and extending the model architecture. Starting with an overview, this section aims to provide a discussion on several experiments directed in this regard.
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See acknowledgement for details on how to deploy the project on a live system.

### Details of the DCGAN experiments
Original DCGAN Architecture:
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide23.jpg)</br>
A 100 dimensional uniform distribution Z is projected to a small spatial extent convolutional representation with many feature maps. A series of four fractionally-strided convolutions then convert this high level representation into a 64×64 pixel image. Notably, no fully connected or pooling layers are used. [Further details](http://bamos.github.io/2016/08/09/deep-completion/) </br>
The experiments are carried out to understand what happens when the parameters of DCGAN architecture are altered and are classified into four categories mentioned below:

```
1. Changing the depth of intermediate layers
2. Altering the latent space dimension
3. Up-sampling based on interpolations
4. Extending the DCGAN architecture
```
The details of the above experiments are in the section below.
### DCGAN Experiment Results

#### 1. Changing the Depth of Intermediate layers of DCGAN:
The red box shows the changes made to the original architecture.
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide24.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide25.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide26.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide27.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide28.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide29.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide30.jpg)</br></br>
#### 2. Altering the latent space dimension:
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide31.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide32.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide33.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide34.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide35.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide36.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide37.jpg)
</br></br>
#### 1. Up-sampling based on interpolations:
In addition to Transposed Convolution for Upsampling, Nearest Neighbor and Bilinear interpolations are employed to upsample the feature maps of the generator and corresponding results are as shown in the figure below.
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide39.jpg)</br></br>
#### 4. Extending the DCGAN architecture:
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide40.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide42.jpg)
![ ](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide44.jpg)
## Acknowledgment

* https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN


