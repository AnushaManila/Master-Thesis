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
![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide10.jpg)
A Gram (or Grammian) matrix G in the context of style transfer contains the un-shifted cross-correlation between all pairs of feature maps within the same layers of the CNN. The feature map contains a high-level presentation of features such as colors, texture, edges, and shapes. A cross-correlation of feature maps on an image with respect to itself gives a measure of which features in an image occur together, and these correlated features constitute the style of an image intuitively. The essence of neural style transfer is to match the feature distributions between the style images and the generated stylized images. G is a square (N⇥N) matrix with linearly independent eigenvectors p_i(i=1,....,N). The columns of P are the eigenvectors and lambda is the diagonal matrix whose diagonal elements are the corresponding eigenvalues. With Gatys et al as the baseline and using all the eigenvalues and eigenvectors of Gram matrices for the reconstruction, the recreation in above sub-figure is exactly same as the stylized image obtained by using original Gram matrices directly without eigen decomposition.

![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide11.jpg)
![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide12.jpg)
![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide13.jpg)
![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide18.jpg)
![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide14.jpg)
![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide15.jpg)
![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide16.jpg)
![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide17.jpg)
![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide20.jpg)
![](https://github.com/AnushaManila/Master-Thesis/blob/master/05_Thesis_Slides/Slide21.jpg)

### Acknowledgment

* https://github.com/leongatys/PytorchNeuralStyleTransfer
