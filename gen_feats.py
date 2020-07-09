# -*- coding:utf-8 -*-
import numpy as np
import os
import rasterio

from config import config
from utils import *

def read_img_data_dict(imgs_dir):
    imgs_names = os.listdir(imgs_dir)
    imgs_data = []
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    for i in range(len(imgs_names)):
        img_name = imgs_names[i]
        if not img_name.endswith('.tif') and not img_name.endswith('.TIF'):
            continue
        img_path = os.path.join(imgs_dir, img_name)
        ds = rasterio.open(img_path)
        img = ds.read()
        imgs_data.append()
    imgs_data = np.concatenate(imgs_data, axis=0)
    
    return imgs_data

if __name__ == '__main__':    
    X_train = read_img_data_dict(config.train_imgs_path)
    pickle.dump(X_train, open(os.path.join(config.train_imgs_dump_path,'/X_train.p'), 'wb'), protocol=4)
    Y_train = read_img_data_dict(config.train_gt_path)
    pickle.dump(X_train, open(os.path.join(config.train_gt_dump_path,'/Y_train.p'), 'wb'), protocol=4)
    
    X_test = read_img_data_dict(config.test_imgs_path)
    pickle.dump(X_train, open(os.path.join(config.test_imgs_dump_path,'/X_train.p'), 'wb'), protocol=4)
    Y_test = read_img_data_dict(config.test_gt_path)
    pickle.dump(X_train, open(os.path.join(config.test_gt_dump_path,'/Y_train.p'), 'wb'), protocol=4)
    
    
    
    



