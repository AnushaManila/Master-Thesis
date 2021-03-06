#**********************************************
# DCGAN in Pytorch Experiment 14
# Base: https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN
# Added: #1. Prepared and trained the celebA High resolution dataset from Karras et al
         #2. Extend the architectures of G and D to generate 512X512 resolution images
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

sys.path.insert(0, " /export/home/amanila/00_thesis/09_pggan-pytorch/celebhq_512/")
# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=32):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*32, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*32)
        self.deconv2 = nn.ConvTranspose2d(d*32, d*16, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*16)
        self.deconv3 = nn.ConvTranspose2d(d*16, d*8, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*8)
        self.deconv4 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d*4)
        self.deconv5 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d*2)
        self.deconv6 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(d)
        self.deconv7 = nn.ConvTranspose2d(d, d/2, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(d/2)
        self.deconv8 = nn.ConvTranspose2d(d/2, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        x = F.relu(self.deconv6_bn(self.deconv6(x)))
        x = F.relu(self.deconv7_bn(self.deconv7(x)))
        x = F.tanh(self.deconv8(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=32):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, d/2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d/2, d, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d)
        self.conv3 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*2)
        self.conv4 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*4)
        self.conv5 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d*8)
        self.conv6 = nn.Conv2d(d*8, d*16, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d*16)
        self.conv7 = nn.Conv2d(d*16, d*32, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d*32)
        self.conv8 = nn.Conv2d(d*32, 1, 4, 1, 0)

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
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.conv6_bn(self.conv6(x)), 0.2)
        x = F.leaky_relu(self.conv7_bn(self.conv7(x)), 0.2)
        x = F.sigmoid(self.conv8(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# training parameters
batch_size = 2
lr = 0.00038 
train_epoch = 20

# data_loader
img_size = 512
isCrop = False
if isCrop:
    transform = transforms.Compose([
        transforms.Scale(108),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
else:
    transform = transforms.Compose([
        transforms.Scale(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
transform = transforms.Compose([
        transforms.Scale(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
        
data_dir = '/export/home/amanila/00_thesis/09_pggan-pytorch/celebhq_512'                # this path depends on your computer
dset = datasets.ImageFolder(data_dir, transform)
train_loader = torch.utils.data.DataLoader(dset, batch_size=2, shuffle=True)
temp = plt.imread(train_loader.dataset.imgs[0][0])
if (temp.shape[0] != img_size) or (temp.shape[0] != img_size):
    sys.stderr.write('Error! image size is not 512 x 512! run \"celebA_data_preprocess.py\" !!!')
    sys.exit(1)

home_path = "/net/hci-storage02/userfolders/amanila/00_thesis/08_DCGAN/02_DCGAN_extend/" # this path depends on your computer
log_dir = home_path + 'logs/15_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
summary_writer = FileWriter(log_dir) #tensorboard


# network
G = generator(32)
D = discriminator(32)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

#compute total no. of trainable parametrs
G_parameters = filter(lambda p: p.requires_grad, G.parameters())
G_params = sum([np.prod(p.size()) for p in G_parameters])
print('G_params:',G_params)
D_parameters = filter(lambda p: p.requires_grad, D.parameters())
D_params = sum([np.prod(p.size()) for p in D_parameters])
print('D_params:',D_params)

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('CelebA_DCGAN_results'):
    os.mkdir('CelebA_DCGAN_results')
if not os.path.isdir('CelebA_DCGAN_results/Exp15'):
    os.mkdir('CelebA_DCGAN_results/Exp15')
if not os.path.isdir('CelebA_DCGAN_results_10'):
    os.mkdir('CelebA_DCGAN_results_10')
if not os.path.isdir('CelebA_DCGAN_results_10/Exp15'):
    os.mkdir('CelebA_DCGAN_results_10/Exp15')

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
        G_optimizer.param_groups[0]['lr'] *= 0.87
        D_optimizer.param_groups[0]['lr'] *= 0.87
        print("learning rate change!")

    if (epoch+1) == 16:
        G_optimizer.param_groups[0]['lr'] *= 0.87
        D_optimizer.param_groups[0]['lr'] *= 0.87
        print("learning rate change!")

    num_iter = 0

    epoch_start_time = time.time()
    for iter, (x_, _) in enumerate (train_loader):
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
        z_ = Variable(z_.cuda())
        G_result = G(z_)

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

        if (iter+1) % 500 == 0:
            p_instance_ = 'CelebA_DCGAN_results_10/Exp12/CelebA_DCGAN_' + str(iter + 1) + '.png'
            vutils.save_image(G_result[0:6,:,:,:].data,'%s' % (p_instance_), nrow=2,normalize=True)
    
        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    torch.save(G.state_dict(), "CelebA_DCGAN_results/Exp15/generator_param.pkl")
    torch.save(D.state_dict(), "CelebA_DCGAN_results/Exp15/discriminator_param.pkl")
    with open('CelebA_DCGAN_results/Exp15/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    p = 'CelebA_DCGAN_results/Exp15/CelebA_DCGAN_' + str(epoch + 1) + '.png'
    vutils.save_image(G_result.data,'%s' % (p), normalize=True)
    p_instance = 'CelebA_DCGAN_results_10/Exp15/CelebA_DCGAN_' + str(epoch + 1) + '.png'
    vutils.save_image(G_result[0:9,:,:,:].data,'%s' % (p_instance), nrow=3,normalize=True)
    
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


#show_train_hist(train_hist, save=True, path='CelebA_DCGAN_results/CelebA_DCGAN_train_hist.png')

images = []
for e in range(train_epoch):
    img_name = 'CelebA_DCGAN_results/Exp15/CelebA_DCGAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('CelebA_DCGAN_results/Exp15_generation_animation.gif', images, fps=5)
