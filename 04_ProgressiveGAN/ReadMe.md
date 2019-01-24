## Progressive Growing of GANs
The generation of high-resolution images is difficult because higher resolution makes it easier to differentiate the generated images apart from training images, thus drastically amplifying the gradient problem. Large resolutions also necessitate using smaller mini-batches due to memory constraints, further compromising training stability. The key idea of [Karras et al.](https://arxiv.org/abs/1710.10196), is to grow both the generator and discriminator progressively: starting from easier low resolution images, add new layers that introduce higher-resolution details as the training progresses as visualized in Figure below.
![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide46.jpg)

Inspired by the work of [Karras et al.](https://arxiv.org/abs/1710.10196), we reproduced the idea and generated 512X512 resolution images. By training the model using a high quality version of CelebA dataset which is an additional contribution by Karras et al., we obtained results as shown in figure below and generated images are at 512X512 resolution. The base code is taken from [pggan](https://github.com/nashory/pggan-pytorch.git) where the model is trained with CelebA low resolution dataset.
![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide50.jpg)
![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide51.jpg)
![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide52.jpg)

#### Progressive GAN Latent Space Interpolation
[My Youtube Link](https://www.youtube.com/playlist?list=PLXWEtY4zQuFJ9v5SeitUW1VoyWw_338Hg)

### Conclusion
Progressive GAN experiments have latent vectors sampled from the normal distribution with zero mean and standard deviation 1 both during training and testing. Training the GAN progressively has several benefits over traditional methods. Beginning with the smaller resolution, it is substantially more stable because there is less class information. Doubling the resolution in steps, we are continuously asking a much simpler question compared to the ultimate goal of discovering a mapping from latent vectors to the final resolution. The consequences are better understood by comparing the results obtained from training the GAN non progressively:
![](https://github.com/AnushaManila/Master-Thesis/blob/master/04_ProgressiveGAN/non_progressive_GAN_256.png)
