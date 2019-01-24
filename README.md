# Master-Thesis
Recent works have shown the power of Convolutional Neural Network (CNN)s to reconstruct an image with a target style by separating and recombining the content and style of two images. This process of extracting the semantic content of one image and applying the style or texture of another image is referred as Neural Style Transfer. Since the recent successful demonstration of [Neural Style Transfer](https://arxiv.org/abs/1508.06576), it has become an active research topic in both the academia and industry. We experiment with Gram matrices to control the amount of style texture by reducing the dimensionality and also try to extend the existing models to combine multiple instances of a style. Furthermore, artificial image generation experiments are carried out based on popular Generative Adversarial Network (GAN) architectures namely, Deep Convolutional GAN [DCGAN](https://arxiv.org/abs/1511.06434) introduced in 2016 and recently presented Progressive growing of GANs [Progressive GAN](https://arxiv.org/abs/1710.10196).

This repository includes no datasets or any trained models but only the implementation and generated sample results of my Master's Thesis executed under the guidance of Prof.Björn Ommer at ['Computer Vision Lab, HCI'](https://hci.iwr.uni-heidelberg.de/home), University of Heidleberg, Germany.
<br />

### Folder information
**01_Neural_Style-Transfer:** includes the implementation of Neural Style transfer experiments.<br />
**02_real-time-style-transfer:** includes the implementation of [real time style transfer](https://arxiv.org/abs/1603.08155).<br />
**03_DCGAN:** contains all the experiments on Deep Convolutional Generative Adversarial Networks.<br />
**04_ProgressiveGAN:** includes the pytorch implementation of Progressive growing of GANs.<br />

### Prerequisites
Development Environment:
```
	Ubuntu 16.04 LTS
	NVIDIA Titan X (Pascal/Maxwell)/Quadro P5000
	cuda 8.0
	Python 2.7.6
	pytorch 0.1.12
	torchvision 0.1.8
	matplotlib 1.3.1
	imageio 2.2.0
	Tensorboard 1.6.0
```
### Conclusion
##### Neural style transfer:
	Only top 5 to 10 % of eigenvalues of gram matrices are significant (Dimensionality Reduction)
	Eigen components can control the amount of style texture
##### DCGAN
	Suitable depth of intermediate layers: {1024, 512, 256, 128}
	The best choice of latent space dimension:  >= 100  
	Up-sampling: Transposed convolution or Nearest neighbor interpolation
	DCGAN architecture extension up to 512X512 resolution
##### Progressive GAN
	Pros: 
		High quality image generation up to 1024X1024 resolution
		Improved training stability
	Cons:
		Longer training time
