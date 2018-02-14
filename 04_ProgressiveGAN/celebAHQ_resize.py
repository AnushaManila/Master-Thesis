import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize

# root path depends on your computer
home = "/net/hci-storage02/userfolders/amanila/00_thesis/"
root = home+'09_pggan-pytorch/celebhq_512/celebA/'
save_root =  home+'09_pggan-pytorch/celebhq_224/'
resize_size = 224

if not os.path.isdir(save_root):
    os.mkdir(save_root)

img_list = os.listdir(root)

# ten_percent = len(img_list) // 10

img_arr = []
for i in range(0,30000):
    img = plt.imread(root + 'img' + str(i) +'.png')
    img = imresize(img, (resize_size, resize_size))
    #plt.imsave(fname=save_root + 'img' + str(i) +'.png', arr=img)
    img_arr.append(img)
    if (i % 1000) == 0:
        print('%d images complete' % i)
    np.save(save_root+'celebAhq_224.npy',img_arr)