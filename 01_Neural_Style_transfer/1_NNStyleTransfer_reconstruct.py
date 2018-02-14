#**********************************************
# Neural Style Transfer (Gatys et al) in Pytorch
# Base: https://github.com/leongatys/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
# Added:  1. Functionality to save the gram matrices for further experiments
        # 2. Created random image for texture synthesis
        # 3. Eigen value decomposition of gram matrices
        # 4. Sorting the eigen components

#**********************************************
import time
import os 
image_dir = os.getcwd() + '/Images/rainprincess/'
model_dir = os.getcwd() + '/Models/'

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

from PIL import Image
from collections import OrderedDict
from scipy.misc import imsave
import numpy as np
from numpy import linalg as LA

# vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]

# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2))
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        #out = nn.MSELoss()(GramMatrix()(input), target)
        out = nn.MSELoss()(input, target)
        return(out)

# sort the eigen vectors and values in descending order
def eig_vec_sort(gram_mat, n):
    eigenValues, eigenVectors  = LA.eig(gram_mat)
    #sort eigenvalues and associated eigen vectors
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenValues[0:n], eigenVectors[:, 0:n]

# pre and post processing for images
img_size = 512
prep = transforms.Compose([transforms.Scale(img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
postpb = transforms.Compose([transforms.ToPILImage()])
def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(t)
    return img

#get network
vgg = VGG()
vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()

#load images, ordered as [style_image, content_image]
img_dirs = [image_dir, image_dir]
img_names = ['1.jpg', 'random.jpg']
imgs = [Image.open(img_dirs[i] + name) for i,name in enumerate(img_names)]
imgs_torch = [prep(img) for img in imgs]
if torch.cuda.is_available():
    imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
else:
    imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
style_image, content_image = imgs_torch

# opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init
opt_img = Variable(content_image.data.clone(), requires_grad=True)

"""
a = np.random.rand(512,512,3) * 255
im_out = Image.fromarray(a.astype('uint8')).convert('RGBA')
im_out.save('random.jpg') 
"""


# define layers, loss functions, weights and compute optimization targets
style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
content_layers = ['r42']
loss_layers = style_layers #+ content_layers
loss_fns = [GramMSELoss()] * len(style_layers) #+ [nn.MSELoss()] * len(content_layers)
if torch.cuda.is_available():
    loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

# these are good weights settings:
style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
#content_weights = [1e0]
weights = style_weights #+ content_weights

#compute optimization targets
style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
'''
print('\n storing gram matrices')
for i in range(5):
    numpy_style_targets =  np.squeeze(style_targets[i].data.cpu().numpy(), axis=(0,))
    numpy_style_targets.dump('/export/home/amanila/00_thesis/00_pytorch_neuralstyletransfer/PytorchNeuralStyleTransfer/gram/Bruegel_2resized_jpg_relu'+str(i+1)+'_1')
    print('style_targets shape:', numpy_style_targets.shape)
    print('\n')
print('\n')

style_targets_ = []
for counter in range(5):
    style_gram_ = np.load('/export/home/amanila/00_thesis/00_pytorch_neuralstyletransfer/PytorchNeuralStyleTransfer/gram/Bruegel_2jpg_relu'+str(counter+1)+'_1')
    style_gram = np.expand_dims(style_gram_, axis=0)
    style_gram = style_gram.astype(int)
    style_gram = Variable(torch.cuda.FloatTensor(style_gram))
    style_targets_.append(style_gram)
'''
#content_targets = [A.detach() for A in vgg(content_image, content_layers)]
targets = style_targets #+ content_targets
print('targets shape:', targets[0].size())



# run style transfer
max_iter = 500
show_iter = 50
optimizer = optim.LBFGS([opt_img]);
n_iter = [0]

while n_iter[0] <= max_iter:

    def closure():
        optimizer.zero_grad()
        out = vgg(opt_img, loss_layers)
        #print('out:', out)
        # ____________________
        layer_losses = []
        for a, A in enumerate(out):
            # targets => style image => G1 = P1*lamda1*Pinv1
            if a == 5:
                layer_losses.append(weights[a]*loss_fns[a](A,targets[a]))
                break
            s = targets[a].size()[1]
            numpy_tar = np.squeeze(targets[a].data.cpu().numpy(), axis=(0,))
            lamda, P = eig_vec_sort(numpy_tar,s)
            #lamda[-1] = 1e-09
            P_lamda = np.dot(P, np.diag(lamda))
            P_inverse = LA.inv(P)
            P_lamda_Pinv = np.dot(P_lamda, P_inverse)

            G1 = Variable(torch.from_numpy(P_lamda_Pinv),requires_grad=False)
            G1 = G1.type(torch.cuda.FloatTensor)
          
            mat = GramMatrix()(A)
            mat_ = mat.type(torch.cuda.FloatTensor)
            

            layer_losses.append(weights[a]*loss_fns[a](mat_, G1)) # 
            # ____________________
        #layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        n_iter[0] += 1
        # print loss
        if n_iter[0] % show_iter == (show_iter - 1):
            print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.data[0]))
        # print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
        return loss


    optimizer.step(closure)

# display result
out_img = postp(opt_img.data[0].cpu().squeeze())
imsave('/export/home/amanila/00_thesis/00_pytorch_neuralstyletransfer/PytorchNeuralStyleTransfer/outputs/rainprincess_1_texture_G2_eig_all.png', out_img)

#make the image high-resolution as described in
#"Controlling Perceptual Factors in Neural Style Transfer", Gatys et al.
#(https://arxiv.org/abs/1611.07865)

#hr preprocessing
img_size_hr = 800 #works for 8GB GPU, make larger if you have 12GB or more
prep_hr = transforms.Compose([transforms.Scale(img_size_hr),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
#prep hr images
imgs_torch = [prep_hr(img) for img in imgs]
if torch.cuda.is_available():
    imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
else:
    imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
style_image, content_image = imgs_torch

#now initialise with upsampled lowres result
opt_img = prep_hr(out_img).unsqueeze(0)
opt_img = Variable(opt_img.type_as(content_image.data), requires_grad=True)

#compute hr targets
style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
#content_targets = [A.detach() for A in vgg(content_image, content_layers)]
'''
style_targets_ = []
for counter in range(5):
    style_gram_ = np.load('/export/home/amanila/00_thesis/00_pytorch_neuralstyletransfer/PytorchNeuralStyleTransfer/gram/Bruegel_2jpg_relu'+str(counter+1)+'_1')
    style_gram = np.expand_dims(style_gram_, axis=0)
    style_gram = style_gram.astype(int)
    style_gram = Variable(torch.cuda.FloatTensor(style_gram))
    style_targets_.append(style_gram)
'''
targets = style_targets #+ content_targets

# run style transfer for high res
max_iter_hr = 200
optimizer = optim.LBFGS([opt_img]);
n_iter = [0]
while n_iter[0] <= max_iter_hr:

    def closure():
        optimizer.zero_grad()
        out = vgg(opt_img, loss_layers)
        # ____________________
        layer_losses = []
        for a, A in enumerate(out):
            # targets => style image => P*lamda = lamdaP
            if a == 5:
                layer_losses.append(weights[a]*loss_fns[a](A,targets[a]))
                break
            s = targets[a].size()[1]
            numpy_tar = np.squeeze(targets[a].data.cpu().numpy(), axis=(0,))
            lamda, P = eig_vec_sort(numpy_tar,s)
            #lamda[-1] = 1e-09
            P_lamda = np.dot(P, np.diag(lamda))
            P_inverse = LA.inv(P)
            P_lamda_Pinv = np.dot(P_lamda, P_inverse)

            G1 = Variable(torch.from_numpy(P_lamda_Pinv),requires_grad=False)
            G1 = G1.type(torch.cuda.FloatTensor)

            mat = GramMatrix()(A)
            mat_ = mat.type(torch.cuda.FloatTensor)
            
          
            layer_losses.append(weights[a]*loss_fns[a](mat_, G1))  # 
            # ____________________
        #layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        n_iter[0] += 1
        # print loss
        if n_iter[0] % show_iter == (show_iter - 1):
            print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.data[0]))
        # print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
        return loss


    optimizer.step(closure)

# display result
out_img_hr = postp(opt_img.data[0].cpu().squeeze())
imsave('/export/home/amanila/00_thesis/00_pytorch_neuralstyletransfer/PytorchNeuralStyleTransfer/outputs/rainprincess_1_texture_G2_eig_all_hd.png', out_img_hr)

