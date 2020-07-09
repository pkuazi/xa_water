# -*- coding:utf-8 -*-
import os
from easydict import EasyDict as edict

if not os.path.exists("./logs/"): os.makedirs("./logs/")
if not os.path.exists("./cache/model/"): os.makedirs("./cache/model/")  # 存放模型的地址

config = edict()

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '/mnt/rsimages/lulc/AISample'
# config.data_path = os.path.join(BASE_DIR,"Postdam/data/processed")
config.data_path = '/mnt/rsimages/lulc/AISample'

config.train_imgs_path = os.path.join(BASE_DIR,"train_set/imgs")
config.train_gt_path = os.path.join(BASE_DIR,'train_set/gt')
config.train_imgs_dump_path = os.path.join(BASE_DIR,'cache/train-set')
config.train_gt_dump_path = os.path.join(BASE_DIR,'cache/train-set')

config.test_imgs_path = os.path.join(BASE_DIR,"test_set/imgs")
config.test_gt_path = os.path.join(BASE_DIR,'test_set/gt')
config.test_gt_dump_path = os.path.join(BASE_DIR,"cachet/test-set")
config.test_imgs_dump_path = os.path.join(BASE_DIR,"cache/test-set")

config.test_pred_path = os.path.join(BASE_DIR,"test_set/pred")

config.model_path = os.path.join(BASE_DIR,"cache/model/")

config.class_num = 6
config.img_rows = 256
config.img_cols = 256
config.train_batch_num = 100 # 1个epoch有train_batch_num个样本
config.vali_batch_num = 50 # 1个epoch有train_batch_num个样本
config.batch_size = 2  # 深度模型 分批训练的批量大小
config.epochs = 100  # 总共训练的轮数（实际不会超过该轮次，因为有early_stop限制）
config.early_stop = 30  # 最优epoch的置信epochs
config.folds = 5 # 使用5折交叉验证

