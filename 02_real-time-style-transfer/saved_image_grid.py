#**********************************************
# Script to save images into grids
#**********************************************


from __future__ import print_function
import sys, os, pdb
sys.path.insert(0, 'src')
import scipy.misc
import os
import numpy as np
from utils import get_img

# get a row with first image being test image and K NN are appended along a row
def get_row(img_path, res_path):
    # save style image
    style_img = get_img(img_path, (224,224,3))
    img_shape = [224, 224, 3]
    padding = np.zeros((img_shape[0], 20, 3))

    row = [style_img, padding]
    for i in range(k):
        match = get_img(res_path+"/"+str(i+1)+".jpg", (224,224,3))
        row.append(match)
    row = np.concatenate(row, axis=1)
    return row



k = 3
row_num = 4
rows = []
img_path = '/export/home/amanila/00_thesis/00_fast-style-transfer-master/examples/style/00_rain_princess/'
res_path = '/export/home/amanila/00_thesis/00_fast-style-transfer-master/examples/results/00_rain_princess/'

for i in range(row_num):
    row = get_row(img_path+str(i+1)+".jpg", res_path+"0"+str(i+1))
    rows.append(row)
superimage = np.concatenate(rows, axis = 0)
scipy.misc.imsave('superimage'+str(row_num)+'_'+str(k)+'.png', superimage)
