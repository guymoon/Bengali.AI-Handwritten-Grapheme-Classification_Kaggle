import warnings
warnings.filterwarnings("ignore")

import gc
import os
from pathlib import Path
import random
import sys

from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold

import torch
import torch.nn as nn

import cv2

pd.options.display.max_rows = 10000
#pd.options.display.max_columns = None
pd.options.display.max_columns = 1000
pd.options.display.max_colwidth = 100

HEIGHT = 137
WIDTH = 236
IMG_SIZE = 224

TRAIN = ['/workspace/inyong/inyong_datasets/kaggle/bengaliai-cv19/train_image_data_0.parquet',
         '/workspace/inyong/inyong_datasets/kaggle/bengaliai-cv19/train_image_data_1.parquet',
         '/workspace/inyong/inyong_datasets/kaggle/bengaliai-cv19/train_image_data_2.parquet',
         '/workspace/inyong/inyong_datasets/kaggle/bengaliai-cv19/train_image_data_3.parquet']

df = pd.read_parquet(TRAIN[0])


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=128, pad=16):
    # crop a box around pixels large than the threshold
    # some images contain line at the sides
    ymin, ymax, xmin, xmax = bbox(img0[5:-5, 5:-5] > 40)
    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax, xmin:xmax]
    ## remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax - xmin, ymax - ymin
    l = max(lx, ly) + pad
    ## make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l - ly) // 2,), ((l - lx) // 2,)], mode='constant')
    return cv2.resize(img, (size, size))


def process(df):
    x_tot,x2_tot = [],[]
    resized = {}
    df = df.set_index('image_id')
    for i in tqdm(range(df.shape[0])):
        img0 = df.loc[df.index[i]].values.reshape(HEIGHT, WIDTH).astype(np.uint8)
        #img0 = crop_resize(img0, IMG_SIZE) #cv2.resize(img0, (IMG_SIZE,IMG_SIZE))
        x_tot.append((img0/255.0).mean())
        x2_tot.append(((img0/255.0)**2).mean())
        resized[df.index[i]] = img0.reshape(-1)
    resized = pd.DataFrame(resized).T.reset_index()
    resized.columns = resized.columns.astype(str)
    resized.rename(columns={'index':'image_id'},inplace=True)
    return resized, x_tot, x2_tot

x_tot,x2_tot = [],[]
for i, fname in enumerate(TRAIN):
    df = pd.read_parquet(fname)
    df, tmp_tot, tmp2_tot = process(df)
    x_tot.append(tmp_tot)
    x2_tot.append(tmp2_tot)
    #df.to_feather(f'/workspace/inyong/inyong_datasets/kaggle/bengaliai-cv19/train_image_data_{i}_{IMG_SIZE}_{IMG_SIZE}.feather')
    df.to_feather(
        f'/workspace/inyong/inyong_datasets/kaggle/bengaliai-cv19/train_image_data_{i}_{HEIGHT}_{WIDTH}.feather')
    del df

#image stats
img_avr =  np.array(x_tot).mean()
img_std =  np.sqrt(np.array(x2_tot).mean() - img_avr**2)
print('mean:',img_avr, ', std:', img_std)