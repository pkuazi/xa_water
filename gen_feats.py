# -*- coding:utf-8 -*-
import numpy as np
import os
import rasterio
import pickle

def read_img_data_dict(imgs_dir,SIZE):
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
        resize = img[:,:SIZE, :SIZE]
        imgs_data.append(resize[None, ...])
    imgs_data = np.concatenate(imgs_data, axis=0)
    
    return imgs_data

if __name__ == '__main__':    
    BASE_DIR = '/mnt/rsimages/lulc/AISample'    
    
    SIZE = 240#<256 IS OK
    
    X_train = read_img_data_dict(os.path.join(BASE_DIR,"train_set/imgs"),SIZE)
    print(X_train.shape)
    X_train_file = os.path.join(BASE_DIR,'cache/train-set/X_train.p')
    print(X_train_file)
    pickle.dump(X_train, open(X_train_file,'wb'))
      
    Y_train = read_img_data_dict(os.path.join(BASE_DIR,'train_set/gt'),SIZE)
    Y_train_file = os.path.join(BASE_DIR,'cache/train-set/Y_train.p')
    pickle.dump(Y_train, open(Y_train_file,'wb'))
    
    X_test = read_img_data_dict(os.path.join(BASE_DIR,"test_set/imgs"),SIZE)
    X_test_file = os.path.join(BASE_DIR,'cache/test-set/X_test.p')
    print(X_test.shape)
    pickle.dump(X_test, open(X_test_file,'wb'))
    
    Y_test = read_img_data_dict(os.path.join(BASE_DIR,'test_set/gt'),SIZE)
    Y_test_file = os.path.join(BASE_DIR,'cache/test-set/Y_test.p')
    pickle.dump(Y_test, open(Y_test_file,'wb'))
    
    
    
    



