# generate interpolated images.


import os,sys
import torch
from config import config
from torch.autograd import Variable
import utils as utils
import imageio
imageio.plugins.ffmpeg.download()

import numpy as np 
import scipy.ndimage

use_cuda = True
checkpoint_path = 'repo/model/gen_R9_T5000.pth.tar'
n_intp = 50


# load trained model.
import network as net
test_model = net.Generator(config)
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    test_model = torch.nn.DataParallel(test_model).cuda(0)
else:
    torch.set_default_tensor_type('torch.FloatTensor')

for resl in range(3, config.max_resl+1):
    test_model.module.grow_network(resl)
    test_model.module.flush_network()
print(test_model)


print('load checkpoint form ... {}'.format(checkpoint_path))
checkpoint = torch.load(checkpoint_path)
test_model.module.load_state_dict(checkpoint['state_dict'])


# create folder.
for i in range(8192):
    name = 'repo/interpolation/try_{}'.format(i)
    #name2 = 'repo/extra/try_{}'.format(i)
    if not os.path.exists(name):
        os.system('mkdir -p {}'.format(name))
        #os.system('mkdir -p {}'.format(name2))
        break;


# Generate latent vectors 
latents = np.random.randn(900, 1, 512).astype(np.float32)
latents = scipy.ndimage.gaussian_filter(latents, [30] + [0] * (2), mode='wrap')
latents /= np.sqrt(np.mean(latents ** 2))
print('latents shape:', latents.shape)



# interpolate between twe noise(z1, z2).
z_intp = torch.FloatTensor(1, config.nz)
z1 = torch.from_numpy(latents)

if use_cuda:
    z_intp = z_intp.cuda()
    z1 = z1.cuda()
    test_model = test_model.cuda()

z_intp = Variable(z_intp)


for i in range(1, 900):
    z_intp.data = z1[i,:,:]
    fake_im = test_model.module(z_intp)
    fname = os.path.join(name, '_intp{}.jpg'.format(i))
    utils.save_image_single(fake_im.data, fname, imsize=pow(2,config.max_resl))
    print('saved {}-th interpolated image ...'.format(i))
