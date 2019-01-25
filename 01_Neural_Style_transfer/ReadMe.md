# Neural Style Transfer using Pytorch
Artistic Style Transfer using CNN has shown appealing results with new forms of image manipulation. The prior optimization-based approaches are approximated by learning a feed-forward generative network and such models have shown real-time performance. Most of the existing models are capable of applying single style at a time. However, separate networks are required to train different style targets. In order to extend the scalability of this innovative algorithm, we train multiple instances of the same style and we experiment with Gram matrices to control the amount of style texture. </br>
![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide04.jpg)

Beginning with the random noise, the idea is to update pixels in the stylized image (yet unknown) iteratively through backpropagation. The objective of image iteration is to minimize the total loss such that the stylized image simultaneously emulates the semantic essence of the content image and the style representation of the style image.

## Getting Started

The basic idea of Style Transfer is from ['Image Style Transfer using CNN'](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)


### Investigations on Gram matrices

The investigations are classified into four categories:

```
1. Eigen decomposition 
2. Descriptive statistics 
3. Style loss modification to reduce dimension
4. Style image patches
```


### Acknowledgment

* https://github.com/leongatys/PytorchNeuralStyleTransfer
