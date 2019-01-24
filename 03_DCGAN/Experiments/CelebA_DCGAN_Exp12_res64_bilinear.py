#**********************************************
# DCGAN in Pytorch Experiment 11 $z \in R^{100}$ (depth --> 1024, 512, 256, 128, 3) 
# Base: https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN
# Added: #1. Prepared and trained the celebA dataset
         #2. Upsampling based on bilinear interpolation
#**********************************************

import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from scipy.misc import imsave
import numpy as np
import datetime
from tensorboard import summary
from tensorboard import FileWriter

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'), 
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block


# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, ngf = 128):
        super(generator, self).__init__()

        self.gf_dim = ngf
        self.upscale = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear')
            )

        self.upsample0 = upBlock(100, ngf*8)
          # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(ngf*8, ngf*4)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf*4, ngf*2)
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock(ngf*2, ngf)
        # -> ngf/16 x 64 x 64
        self.upsample4 = upBlock(ngf, ngf // 2)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            conv3x3(ngf // 2, 3),
            nn.Tanh())

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):

        h_code = self.upscale(input)
        #print('h_code0:',h_code)
        h_code = self.upsample0(h_code)
        #print('h_code2:',h_code) #128x1024x4x4
        h_code = self.upsample1(h_code)
        #print('h_code3:',h_code) #128x512x8x8
        h_code = self.upsample2(h_code)
        #print('h_code4:',h_code) #128x256x16x16
        h_code = self.upsample3(h_code)
        #print('h_code5:',h_code) #128x128x32x32
        h_code = self.upsample4(h_code)
        #print('h_code6:',h_code) #128x64x64x64
        # state size 3 x 64 x 64
        fake_img = self.img(h_code)
        #print('fake_img:',fake_img) #128x3x64x64

        return fake_img

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20

# data_loader
img_size = 64
isCrop = False
if isCrop:
    transform = transforms.Compose([
        transforms.Scale(108),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
data_dir = '/export/home/amanila/00_thesis/08_DCGAN/01_MNIST-CelebA-DCGAN/data/resized_celebA'          # this path depends on your computer
dset = datasets.ImageFolder(data_dir, transform)
train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True)
temp = plt.imread(train_loader.dataset.imgs[0][0])
if (temp.shape[0] != img_size) or (temp.shape[0] != img_size):
    sys.stderr.write('Error! image size is not 64 x 64! run \"celebA_data_preprocess.py\" !!!')
    sys.exit(1)

home_path = "/net/hci-storage02/userfolders/amanila/00_thesis/08_DCGAN/03_CelebA_DCGAN_Upsample/"      # this path depends on your computer
log_dir = home_path + 'logs/12_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
summary_writer = FileWriter(log_dir) #tensorboard


# network
G = generator(128)
D = discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('CelebA_DCGAN_results'):
    os.mkdir('CelebA_DCGAN_results')
if not os.path.isdir('CelebA_DCGAN_results/Exp12'):
    os.mkdir('CelebA_DCGAN_results/Exp12')
if not os.path.isdir('CelebA_DCGAN_results_10'):
    os.mkdir('CelebA_DCGAN_results_10')
    if not os.path.isdir('CelebA_DCGAN_results_10/Exp12'):
    os.mkdir('CelebA_DCGAN_results_10/Exp12')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('Training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []

    # learning rate decay
    if (epoch+1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch+1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    num_iter = 0

    epoch_start_time = time.time()
    for iter, (x_, _) in enumerate(train_loader):
        # train discriminator D
        D.zero_grad()
        
        if isCrop:
            x_ = x_[:, :, 22:86, 22:86]

        mini_batch = x_.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        D_result = D(x_).squeeze()
        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda()) #.data.normal_(0, 1)
        G_result = G(z_)
        #print('G_result:',G_result) # 128X3X16X16

        D_result = D(G_result).squeeze()
        D_fake_loss = BCE_loss(D_result, y_fake_)
        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.data[0])

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())

        G_result = G(z_)
        D_result = D(G_result).squeeze()
        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data[0])

        if (iter+1)%100 == 0:
            p_instance_ = 'CelebA_DCGAN_results_10/Exp7/CelebA_DCGAN__' + str(iter + 1) + '.png'
            vutils.save_image(G_result[0:10,:,:,:].data,'%s' % (p_instance_), nrow=10,normalize=True)

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time


    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    p = 'CelebA_DCGAN_results/Exp12/CelebA_DCGAN_' + str(epoch + 1) + '.png'
    vutils.save_image(G_result.data,'%s' % (p), normalize=True)
    p_instance = 'CelebA_DCGAN_results_10/Exp12/CelebA_DCGAN_' + str(epoch + 1) + '.png'
    vutils.save_image(G_result[0:10,:,:,:].data,'%s' % (p_instance), nrow=10,normalize=True)

    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    # tensorboard
    summary_D = summary.scalar('D_loss', torch.mean(torch.FloatTensor(D_losses)))
    summary_G = summary.scalar('G_loss', torch.mean(torch.FloatTensor(G_losses)))
    summary_writer.add_summary(summary_D, (epoch + 1))
    summary_writer.add_summary(summary_G, (epoch + 1))

summary_writer.close()
end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "CelebA_DCGAN_results/Exp12/generator_param.pkl")
torch.save(D.state_dict(), "CelebA_DCGAN_results/Exp12/discriminator_param.pkl")
with open('CelebA_DCGAN_results/Exp12/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

#show_train_hist(train_hist, save=True, path='CelebA_DCGAN_results/CelebA_DCGAN_train_hist.png')

images = []
for e in range(train_epoch):
    img_name = 'CelebA_DCGAN_results/Exp12/CelebA_DCGAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('CelebA_DCGAN_results/Exp12_generation_animation.gif', images, fps=5)
