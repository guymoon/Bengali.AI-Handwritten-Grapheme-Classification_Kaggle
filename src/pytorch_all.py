cimport warnings
warnings.filterwarnings("ignore")

import gc
import os
from pathlib import Path
import random
import sys
import math
import tifme
import multiprocessing

from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
#import seaborn as sns

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
import apex
from apex import amp
from apex.parallel import DistributedDataParallel

import albumentations as A
import cv2
from PIL import Image, ImageEnhance, ImageOps

pd.options.display.max_rows = 10000
#pd.options.display.max_columns = None
pd.options.display.max_columns = 1000
pd.options.display.max_colwidth = 100

from torchvision import transforms
import pretrainedmodels


os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

debug=False
submission=False
batch_per_cuda = 128
batch_size=batch_per_cuda*torch.cuda.device_count()
#device='cuda:0'
device='cuda'
out='.'
image_size=224 #128
image_height=137
image_width=236
arch='pretrained'
model_name='se_resnext50_32x4d'
#model_name='resnet34'
num_epochs = 150
experi_num = 'iy03'
val_fold = 0
num_workers = 2 #multiprocessing.cpu_count()//8

SEED = 42

print(f'batch_size:{batch_size}, num_workers:{num_workers}')
print(torch.cuda.device_count())

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

datadir = Path('/workspace/inyong/inyong_datasets/kaggle/bengaliai-cv19')
featherdir = Path('/workspace/inyong/inyong_datasets/kaggle/bengaliai-cv19')
logdir = Path('log')
modeldir = Path('model')

import feather
def prepare_image(datadir, featherdir, data_type='train',
                  submission=False, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
    if submission:
        image_df_list = [pd.read_parquet(datadir / f'{data_type}_image_data_{i}.parquet')
                         for i in indices]
    else:
        image_df_list = [feather.read_dataframe(featherdir / f'{data_type}_image_data_{i}_{image_height}_{image_width}.feather')
                     for i in indices]

    #image_df_list = [pd.read_feather(featherdir / f'{data_type}_image_data_{i}_{image_size}_{image_size}.feather')
        #                 for i in indices]

    print('image_df_list', len(image_df_list))
    images = [df.iloc[:, 1:].values.reshape(-1, image_height, image_width) for df in image_df_list]
    del image_df_list
    gc.collect()
    images = np.concatenate(images, axis=0)
    return images

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


train = pd.read_csv(datadir/'train_with_10_fold_42.csv')
le = preprocessing.LabelEncoder()
train['grapheme_enc'] = le.fit_transform(train['grapheme'])

grapheme_map = train[['grapheme_root','vowel_diacritic','consonant_diacritic','grapheme_enc']].drop_duplicates()
grapheme_map = grapheme_map.set_index('grapheme_enc')
print(grapheme_map.shape)

gr_map = grapheme_map.reset_index()[['grapheme_enc','grapheme_root']].to_dict(orient='records')
gr_map = {r['grapheme_enc']:r['grapheme_root'] for r in gr_map}

v_map = grapheme_map.reset_index()[['grapheme_enc','vowel_diacritic']].to_dict(orient='records')
v_map = {r['grapheme_enc']:r['vowel_diacritic'] for r in v_map}

c_map = grapheme_map.reset_index()[['grapheme_enc','consonant_diacritic']].to_dict(orient='records')
c_map = {r['grapheme_enc']:r['consonant_diacritic'] for r in c_map}

#%%time

train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic','grapheme_enc']].values
indices = [0] if debug else [0, 1, 2, 3]
train_images = prepare_image(datadir, featherdir, data_type='train', submission=False, indices=indices)
print(train_images.shape)


class BengaliAIDataset(Dataset):

    def __init__(self, images, labels=None, transform=None, indices=None):
        self.images = images
        self.labels = labels
        if indices is None:
            indices = np.arange(len(images))
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        i = self.indices[i]
        image = self.images[i]
        # image = np.stack((image, image, image), axis=-1)
        # image = cv2.cvtColor(image ,cv2.COLOR_GRAY2RGB)
        # image = image.reshape(image_size, image_size, -1)
        image = 1 - (image / 255.0).astype(np.float32)

        # if self.transform:
        #    image = self.transform(image=image)['image']
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        y = self.labels[i]
        if self.transform:
            aug_image = self.transform(image)
            image = image.reshape(image_height, image_width, -1)
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            aug_image = aug_image.reshape(image_height, image_width, -1)
            aug_image = np.transpose(aug_image, (2, 0, 1)).astype(np.float32)
            return image, aug_image, y[0], y[1], y[2], y[3]
        else:
            image = image.reshape(image_height, image_width, -1)
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            return image, y[0], y[1], y[2], y[3]

train_dataset = BengaliAIDataset(train_images, train_labels)
image, gr, v, c, g = train_dataset[0]
print(f'image.shape: {image.shape}, label:{gr},{v},{c},{g}')

# helper --
def make_grid_image(width,height, grid_size=16):

    image = np.zeros((height,width),np.float32)
    for y in range(0,height,2*grid_size):
        for x in range(0,width,2*grid_size):
             image[y: y+grid_size,x:x+grid_size] = 1

    # for y in range(height+grid_size,2*grid_size):
    #     for x in range(width+grid_size,2*grid_size):
    #          image[y: y+grid_size,x:x+grid_size] = 1

    return image

#---

def do_identity(image, magnitude=None):
    return image


# *** geometric ***

def do_random_projective(image, magnitude=0.5):
    mag = np.random.uniform(-1, 1) * 0.5*magnitude

    height, width = image.shape[:2]
    x0,y0=0,0
    x1,y1=1,0
    x2,y2=1,1
    x3,y3=0,1

    mode = np.random.choice(['top','bottom','left','right'])
    if mode =='top':
        x0 += mag;   x1 -= mag
    if mode =='bottom':
        x3 += mag;   x2 -= mag
    if mode =='left':
        y0 += mag;   y3 -= mag
    if mode =='right':
        y1 += mag;   y2 -= mag

    s = np.array([[ 0, 0],[ 1, 0],[ 1, 1],[ 0, 1],])*[[width, height]]
    d = np.array([[x0,y0],[x1,y1],[x2,y2],[x3,y3],])*[[width, height]]
    transform = cv2.getPerspectiveTransform(s.astype(np.float32),d.astype(np.float32))

    image = cv2.warpPerspective( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image


def do_random_perspective(image, magnitude=0.5):
    mag = np.random.uniform(-1, 1, (4,2)) * 0.25*magnitude

    height, width = image.shape[:2]
    s = np.array([[ 0, 0],[ 1, 0],[ 1, 1],[ 0, 1],])
    d = s+mag
    s *= [[width, height]]
    d *= [[width, height]]
    transform = cv2.getPerspectiveTransform(s.astype(np.float32),d.astype(np.float32))

    image = cv2.warpPerspective( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image


def do_random_scale( image, magnitude=0.5 ):
    s = 1+np.random.uniform(-1, 1)*magnitude*0.5

    height, width = image.shape[:2]
    transform = np.array([
        [s,0,0],
        [0,s,0],
    ],np.float32)
    image = cv2.warpAffine( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image



def do_random_shear_x( image, magnitude=0.5 ):
    sx = np.random.uniform(-1, 1)*magnitude

    height, width = image.shape[:2]
    transform = np.array([
        [1,sx,0],
        [0,1,0],
    ],np.float32)
    image = cv2.warpAffine( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_random_shear_y( image, magnitude=0.5 ):
    sy = np.random.uniform(-1, 1)*magnitude

    height, width = image.shape[:2]
    transform = np.array([
        [1, 0,0],
        [sy,1,0],
    ],np.float32)
    image = cv2.warpAffine( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_random_stretch_x(image, magnitude=0.5 ):
    sx = 1+np.random.uniform(-1, 1)*magnitude

    height, width = image.shape[:2]
    transform = np.array([
        [sx,0,0],
        [0, 1,0],
    ],np.float32)
    image = cv2.warpAffine( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_random_stretch_y(image, magnitude=0.5 ):
    sy = 1+np.random.uniform(-1, 1)*magnitude

    height, width = image.shape[:2]
    transform = np.array([
        [1, 0,0],
        [0,sy,0],
    ],np.float32)
    image = cv2.warpAffine( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_random_rotate(image, magnitude=0.5 ):
    angle = 1+np.random.uniform(-1, 1)*30*magnitude

    height, width = image.shape[:2]
    cx, cy = width // 2, height // 2

    transform = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    image = cv2.warpAffine( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


#----
def do_random_grid_distortion(image, magnitude=0.5 ):
    num_step = 5
    distort  = magnitude

    # http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    distort_x = [1 + random.uniform(-distort,distort) for i in range(num_step + 1)]
    distort_y = [1 + random.uniform(-distort,distort) for i in range(num_step + 1)]

    #---
    height, width = image.shape[:2]
    xx = np.zeros(width, np.float32)
    step_x = width // num_step

    prev = 0
    for i, x in enumerate(range(0, width, step_x)):
        start = x
        end   = x + step_x
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + step_x * distort_x[i]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    yy = np.zeros(height, np.float32)
    step_y = height // num_step
    prev = 0
    for idx, y in enumerate(range(0, height, step_y)):
        start = y
        end = y + step_y
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + step_y * distort_y[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image

#https://github.com/albumentations-team/albumentations/blob/8b58a3dbd2f35558b3790a1dbff6b42b98e89ea5/albumentations/augmentations/transforms.py

# https://ciechanow.ski/mesh-transforms/
# https://stackoverflow.com/questions/53907633/how-to-warp-an-image-using-deformed-mesh
# http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
def do_random_custom_distortion1(image, magnitude=0.5):
    distort=magnitude*0.3

    height,width = image.shape
    s_x = np.array([0.0, 0.5, 1.0,  0.0, 0.5, 1.0,  0.0, 0.5, 1.0])
    s_y = np.array([0.0, 0.0, 0.0,  0.5, 0.5, 0.5,  1.0, 1.0, 1.0])
    d_x = s_x.copy()
    d_y = s_y.copy()
    d_x[[1,4,7]] += np.random.uniform(-distort,distort, 3)
    d_y[[3,4,5]] += np.random.uniform(-distort,distort, 3)

    s_x = (s_x*width )
    s_y = (s_y*height)
    d_x = (d_x*width )
    d_y = (d_y*height)

    #---
    distort = np.zeros((height,width),np.float32)
    for index in ([4,1,3],[4,1,5],[4,7,3],[4,7,5]):
        point = np.stack([s_x[index],s_y[index]]).T
        qoint = np.stack([d_x[index],d_y[index]]).T

        src  = np.array(point, np.float32)
        dst  = np.array(qoint, np.float32)
        mat  = cv2.getAffineTransform(src, dst)

        point = np.round(point).astype(np.int32)
        x0 = np.min(point[:,0])
        x1 = np.max(point[:,0])
        y0 = np.min(point[:,1])
        y1 = np.max(point[:,1])
        mask = np.zeros((height,width),np.float32)
        mask[y0:y1,x0:x1] = 1

        mask = mask*image
        warp = cv2.warpAffine(mask, mat, (width, height),borderMode=cv2.BORDER_REPLICATE)
        distort = np.maximum(distort,warp)
        #distort = distort+warp

    return distort


# *** intensity ***
def do_random_contrast(image, magnitude=0.5):
    alpha = 1 + random.uniform(-1,1)*magnitude
    image = image.astype(np.float32) * alpha
    image = np.clip(image,0,1)
    return image


def do_random_block_fade(image, magnitude=0.5):
    size  = [0.1, magnitude]

    height,width = image.shape

    #get bounding box
    m = image.copy()
    cv2.rectangle(m,(0,0),(height,width),1,5)
    m = image<0.5
    if m.sum()==0: return image

    m = np.where(m)
    y0,y1,x0,x1 = np.min(m[0]), np.max(m[0]), np.min(m[1]), np.max(m[1])
    w = x1-x0
    h = y1-y0
    if w*h<10: return image

    ew, eh = np.random.uniform(*size,2)
    ew = int(ew*w)
    eh = int(eh*h)

    ex = np.random.randint(0,w-ew)+x0
    ey = np.random.randint(0,h-eh)+y0

    image[ey:ey+eh, ex:ex+ew] *= np.random.uniform(0.1,0.5) #1 #
    image = np.clip(image,0,1)
    return image


# *** noise ***
# https://www.kaggle.com/ren4yu/bengali-morphological-ops-as-image-augmentation
def do_random_erode(image, magnitude=0.5):
    s = int(round(1 + np.random.uniform(0,1)*magnitude*6))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((s,s)))
    image  = cv2.erode(image, kernel, iterations=1)
    return image

def do_random_dilate(image, magnitude=0.5):
    s = int(round(1 + np.random.uniform(0,1)*magnitude*6))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((s,s)))
    image  = cv2.dilate(image, kernel, iterations=1)
    return image

def do_random_sprinkle(image, magnitude=0.5):

    size = 16
    num_sprinkle = int(round( 1 + np.random.randint(10)*magnitude ))

    height,width = image.shape
    image = image.copy()
    image_small = cv2.resize(image, dsize=None, fx=0.25, fy=0.25)
    m   = np.where(image_small>0.25)
    num = len(m[0])
    if num==0: return image

    s = size//2
    i = np.random.choice(num, num_sprinkle)
    for y,x in zip(m[0][i],m[1][i]):
        y=y*4+2
        x=x*4+2
        image[y-s:y+s, x-s:x+s] = 0 #0.5 #1 #
    return image


#https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv
def do_random_noise(image, magnitude=0.5):
    height,width = image.shape
    noise = np.random.uniform(-1,1,(height,width))*magnitude*0.7
    image = image+noise
    image = np.clip(image,0,1)
    return image



def do_random_line(image, magnitude=0.5):
    num_lines = int(round(1 + np.random.randint(8)*magnitude))

    # ---
    height,width = image.shape
    image = image.copy()

    def line0():
        return (0,0),(width-1,0)

    def line1():
        return (0,height-1),(width-1,height-1)

    def line2():
        return (0,0),(0,height-1)

    def line3():
        return (width-1,0),(width-1,height-1)

    def line4():
        x0,x1 = np.random.choice(width,2)
        return (x0,0),(x1,height-1)

    def line5():
        y0,y1 = np.random.choice(height,2)
        return (0,y0),(width-1,y1)

    for i in range(num_lines):
        p = np.array([1/4,1/4,1/4,1/4,1,1])
        func = np.random.choice([line0,line1,line2,line3,line4,line5],p=p/p.sum())
        (x0,y0),(x1,y1) = func()

        color     = np.random.uniform(0,1)
        thickness = np.random.randint(1,5)
        line_type = np.random.choice([cv2.LINE_AA,cv2.LINE_4,cv2.LINE_8])

        cv2.line(image,(x0,y0),(x1,y1), color, thickness, line_type)

    return image



# batch augmentation that uses pairing, e.g mixup, cutmix, cutout #####################
def make_object_box(image):
    m = image.copy()
    cv2.rectangle(m,(0,0),(236, 137), 0, 10)
    m = m-np.min(m)
    m = m/np.max(m)
    h = m<0.5

    row = np.any(h, axis=1)
    col = np.any(h, axis=0)
    y0, y1 = np.where(row)[0][[0, -1]]
    x0, x1 = np.where(col)[0][[0, -1]]

    return [x0,y0],[x1,y1]




def do_random_batch_mixup(input, onehot):
    batch_size = len(input)

    alpha = 0.4 #0.2  #0.2,0.4
    gamma = np.random.beta(alpha, alpha, batch_size)
    gamma = np.maximum(1-gamma,gamma)

    # #mixup https://github.com/moskomule/mixup.pytorch/blob/master/main.py
    gamma = torch.from_numpy(gamma).float().to(input.device)
    perm  = torch.randperm(batch_size).to(input.device)
    perm_input  = input[perm]
    perm_onehot = [t[perm] for t in onehot]

    gamma = gamma.view(batch_size,1,1,1)
    mix_input  = gamma*input + (1-gamma)*perm_input
    gamma = gamma.view(batch_size,1)
    mix_onehot = [gamma*t + (1-gamma)*perm_t for t,perm_t in zip(onehot,perm_onehot)]

    return mix_input, mix_onehot, (perm_input, perm_onehot)


def do_random_batch_cutout(input, onehot):
    batch_size,C,H,W = input.shape

    mask = np.ones((batch_size,C,H,W ), np.float32)
    for b in range(batch_size):

        length = int(np.random.uniform(0.1,0.5)*min(H,W))
        y = np.random.randint(H)
        x = np.random.randint(W)

        y0 = np.clip(y - length // 2, 0, H)
        y1 = np.clip(y + length // 2, 0, H)
        x0 = np.clip(x - length // 2, 0, W)
        x1 = np.clip(x + length // 2, 0, W)
        mask[b, :, y0: y1, x0: x1] = 0
    mask  = torch.from_numpy(mask).to(input.device)

    input = input*mask
    return input, onehot, None

def valid_augment(image):
    return image


def train_augment(image):
    if 1:
        for op in np.random.choice([
            lambda image : do_identity(image),
            lambda image : do_random_projective(image, 0.4),
            lambda image : do_random_perspective(image, 0.4),
            lambda image : do_random_scale(image, 0.4),
            lambda image : do_random_rotate(image, 0.4),
            lambda image : do_random_shear_x(image, 0.5),
            lambda image : do_random_shear_y(image, 0.4),
            lambda image : do_random_stretch_x(image, 0.5),
            lambda image : do_random_stretch_y(image, 0.5),
            lambda image : do_random_grid_distortion(image, 0.4),
            lambda image : do_random_custom_distortion1(image, 0.5),
        ],1):
            image = op(image)

        for op in np.random.choice([
            lambda image : do_identity(image),
            lambda image : do_random_erode(image, 0.4),
            lambda image : do_random_dilate(image, 0.4),
            lambda image : do_random_sprinkle(image, 0.5),
            lambda image : do_random_line(image, 0.5),
        ],1):
            image = op(image)

        for op in np.random.choice([
            lambda image : do_identity(image),
            lambda image : do_random_contrast(image, 0.5),
            lambda image : do_random_block_fade(image, 0.5),
        ],1):
            image = op(image)

        #image = do_random_pad_crop(image, 3)
    return image

train_dataset = BengaliAIDataset(train_images, train_labels, transform=train_augment)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]
PRETRAIN_FILE = 'model/premodel/se_resnext50_32x4d-a260b3a4.pth'
#PRETRAIN_FILE = 'model/model_iy02_epoch_084_fold_0_recall_0.9934.pt'
NUM_CLASS = [168,11,7,1295]

CONVERSION=[
 'block0.0.weight',	(64, 3, 7, 7),	 'layer0.conv1.weight',	(64, 3, 7, 7),
 'block0.1.weight',	(64,),	 'layer0.bn1.weight',	(64,),
 'block0.1.bias',	(64,),	 'layer0.bn1.bias',	(64,),
 'block0.1.running_mean',	(64,),	 'layer0.bn1.running_mean',	(64,),
 'block0.1.running_var',	(64,),	 'layer0.bn1.running_var',	(64,),
 'block1.0.conv_bn1.conv.weight',	(128, 64, 1, 1),	 'layer1.0.conv1.weight',	(128, 64, 1, 1),
 'block1.0.conv_bn1.bn.weight',	(128,),	 'layer1.0.bn1.weight',	(128,),
 'block1.0.conv_bn1.bn.bias',	(128,),	 'layer1.0.bn1.bias',	(128,),
 'block1.0.conv_bn1.bn.running_mean',	(128,),	 'layer1.0.bn1.running_mean',	(128,),
 'block1.0.conv_bn1.bn.running_var',	(128,),	 'layer1.0.bn1.running_var',	(128,),
 'block1.0.conv_bn2.conv.weight',	(128, 4, 3, 3),	 'layer1.0.conv2.weight',	(128, 4, 3, 3),
 'block1.0.conv_bn2.bn.weight',	(128,),	 'layer1.0.bn2.weight',	(128,),
 'block1.0.conv_bn2.bn.bias',	(128,),	 'layer1.0.bn2.bias',	(128,),
 'block1.0.conv_bn2.bn.running_mean',	(128,),	 'layer1.0.bn2.running_mean',	(128,),
 'block1.0.conv_bn2.bn.running_var',	(128,),	 'layer1.0.bn2.running_var',	(128,),
 'block1.0.conv_bn3.conv.weight',	(256, 128, 1, 1),	 'layer1.0.conv3.weight',	(256, 128, 1, 1),
 'block1.0.conv_bn3.bn.weight',	(256,),	 'layer1.0.bn3.weight',	(256,),
 'block1.0.conv_bn3.bn.bias',	(256,),	 'layer1.0.bn3.bias',	(256,),
 'block1.0.conv_bn3.bn.running_mean',	(256,),	 'layer1.0.bn3.running_mean',	(256,),
 'block1.0.conv_bn3.bn.running_var',	(256,),	 'layer1.0.bn3.running_var',	(256,),
 'block1.0.scale.fc1.weight',	(16, 256, 1, 1),	 'layer1.0.se_module.fc1.weight',	(16, 256, 1, 1),
 'block1.0.scale.fc1.bias',	(16,),	 'layer1.0.se_module.fc1.bias',	(16,),
 'block1.0.scale.fc2.weight',	(256, 16, 1, 1),	 'layer1.0.se_module.fc2.weight',	(256, 16, 1, 1),
 'block1.0.scale.fc2.bias',	(256,),	 'layer1.0.se_module.fc2.bias',	(256,),
 'block1.0.shortcut.conv.weight',	(256, 64, 1, 1),	 'layer1.0.downsample.0.weight',	(256, 64, 1, 1),
 'block1.0.shortcut.bn.weight',	(256,),	 'layer1.0.downsample.1.weight',	(256,),
 'block1.0.shortcut.bn.bias',	(256,),	 'layer1.0.downsample.1.bias',	(256,),
 'block1.0.shortcut.bn.running_mean',	(256,),	 'layer1.0.downsample.1.running_mean',	(256,),
 'block1.0.shortcut.bn.running_var',	(256,),	 'layer1.0.downsample.1.running_var',	(256,),
 'block1.1.conv_bn1.conv.weight',	(128, 256, 1, 1),	 'layer1.1.conv1.weight',	(128, 256, 1, 1),
 'block1.1.conv_bn1.bn.weight',	(128,),	 'layer1.1.bn1.weight',	(128,),
 'block1.1.conv_bn1.bn.bias',	(128,),	 'layer1.1.bn1.bias',	(128,),
 'block1.1.conv_bn1.bn.running_mean',	(128,),	 'layer1.1.bn1.running_mean',	(128,),
 'block1.1.conv_bn1.bn.running_var',	(128,),	 'layer1.1.bn1.running_var',	(128,),
 'block1.1.conv_bn2.conv.weight',	(128, 4, 3, 3),	 'layer1.1.conv2.weight',	(128, 4, 3, 3),
 'block1.1.conv_bn2.bn.weight',	(128,),	 'layer1.1.bn2.weight',	(128,),
 'block1.1.conv_bn2.bn.bias',	(128,),	 'layer1.1.bn2.bias',	(128,),
 'block1.1.conv_bn2.bn.running_mean',	(128,),	 'layer1.1.bn2.running_mean',	(128,),
 'block1.1.conv_bn2.bn.running_var',	(128,),	 'layer1.1.bn2.running_var',	(128,),
 'block1.1.conv_bn3.conv.weight',	(256, 128, 1, 1),	 'layer1.1.conv3.weight',	(256, 128, 1, 1),
 'block1.1.conv_bn3.bn.weight',	(256,),	 'layer1.1.bn3.weight',	(256,),
 'block1.1.conv_bn3.bn.bias',	(256,),	 'layer1.1.bn3.bias',	(256,),
 'block1.1.conv_bn3.bn.running_mean',	(256,),	 'layer1.1.bn3.running_mean',	(256,),
 'block1.1.conv_bn3.bn.running_var',	(256,),	 'layer1.1.bn3.running_var',	(256,),
 'block1.1.scale.fc1.weight',	(16, 256, 1, 1),	 'layer1.1.se_module.fc1.weight',	(16, 256, 1, 1),
 'block1.1.scale.fc1.bias',	(16,),	 'layer1.1.se_module.fc1.bias',	(16,),
 'block1.1.scale.fc2.weight',	(256, 16, 1, 1),	 'layer1.1.se_module.fc2.weight',	(256, 16, 1, 1),
 'block1.1.scale.fc2.bias',	(256,),	 'layer1.1.se_module.fc2.bias',	(256,),
 'block1.2.conv_bn1.conv.weight',	(128, 256, 1, 1),	 'layer1.2.conv1.weight',	(128, 256, 1, 1),
 'block1.2.conv_bn1.bn.weight',	(128,),	 'layer1.2.bn1.weight',	(128,),
 'block1.2.conv_bn1.bn.bias',	(128,),	 'layer1.2.bn1.bias',	(128,),
 'block1.2.conv_bn1.bn.running_mean',	(128,),	 'layer1.2.bn1.running_mean',	(128,),
 'block1.2.conv_bn1.bn.running_var',	(128,),	 'layer1.2.bn1.running_var',	(128,),
 'block1.2.conv_bn2.conv.weight',	(128, 4, 3, 3),	 'layer1.2.conv2.weight',	(128, 4, 3, 3),
 'block1.2.conv_bn2.bn.weight',	(128,),	 'layer1.2.bn2.weight',	(128,),
 'block1.2.conv_bn2.bn.bias',	(128,),	 'layer1.2.bn2.bias',	(128,),
 'block1.2.conv_bn2.bn.running_mean',	(128,),	 'layer1.2.bn2.running_mean',	(128,),
 'block1.2.conv_bn2.bn.running_var',	(128,),	 'layer1.2.bn2.running_var',	(128,),
 'block1.2.conv_bn3.conv.weight',	(256, 128, 1, 1),	 'layer1.2.conv3.weight',	(256, 128, 1, 1),
 'block1.2.conv_bn3.bn.weight',	(256,),	 'layer1.2.bn3.weight',	(256,),
 'block1.2.conv_bn3.bn.bias',	(256,),	 'layer1.2.bn3.bias',	(256,),
 'block1.2.conv_bn3.bn.running_mean',	(256,),	 'layer1.2.bn3.running_mean',	(256,),
 'block1.2.conv_bn3.bn.running_var',	(256,),	 'layer1.2.bn3.running_var',	(256,),
 'block1.2.scale.fc1.weight',	(16, 256, 1, 1),	 'layer1.2.se_module.fc1.weight',	(16, 256, 1, 1),
 'block1.2.scale.fc1.bias',	(16,),	 'layer1.2.se_module.fc1.bias',	(16,),
 'block1.2.scale.fc2.weight',	(256, 16, 1, 1),	 'layer1.2.se_module.fc2.weight',	(256, 16, 1, 1),
 'block1.2.scale.fc2.bias',	(256,),	 'layer1.2.se_module.fc2.bias',	(256,),
 'block2.0.conv_bn1.conv.weight',	(256, 256, 1, 1),	 'layer2.0.conv1.weight',	(256, 256, 1, 1),
 'block2.0.conv_bn1.bn.weight',	(256,),	 'layer2.0.bn1.weight',	(256,),
 'block2.0.conv_bn1.bn.bias',	(256,),	 'layer2.0.bn1.bias',	(256,),
 'block2.0.conv_bn1.bn.running_mean',	(256,),	 'layer2.0.bn1.running_mean',	(256,),
 'block2.0.conv_bn1.bn.running_var',	(256,),	 'layer2.0.bn1.running_var',	(256,),
 'block2.0.conv_bn2.conv.weight',	(256, 8, 3, 3),	 'layer2.0.conv2.weight',	(256, 8, 3, 3),
 'block2.0.conv_bn2.bn.weight',	(256,),	 'layer2.0.bn2.weight',	(256,),
 'block2.0.conv_bn2.bn.bias',	(256,),	 'layer2.0.bn2.bias',	(256,),
 'block2.0.conv_bn2.bn.running_mean',	(256,),	 'layer2.0.bn2.running_mean',	(256,),
 'block2.0.conv_bn2.bn.running_var',	(256,),	 'layer2.0.bn2.running_var',	(256,),
 'block2.0.conv_bn3.conv.weight',	(512, 256, 1, 1),	 'layer2.0.conv3.weight',	(512, 256, 1, 1),
 'block2.0.conv_bn3.bn.weight',	(512,),	 'layer2.0.bn3.weight',	(512,),
 'block2.0.conv_bn3.bn.bias',	(512,),	 'layer2.0.bn3.bias',	(512,),
 'block2.0.conv_bn3.bn.running_mean',	(512,),	 'layer2.0.bn3.running_mean',	(512,),
 'block2.0.conv_bn3.bn.running_var',	(512,),	 'layer2.0.bn3.running_var',	(512,),
 'block2.0.scale.fc1.weight',	(32, 512, 1, 1),	 'layer2.0.se_module.fc1.weight',	(32, 512, 1, 1),
 'block2.0.scale.fc1.bias',	(32,),	 'layer2.0.se_module.fc1.bias',	(32,),
 'block2.0.scale.fc2.weight',	(512, 32, 1, 1),	 'layer2.0.se_module.fc2.weight',	(512, 32, 1, 1),
 'block2.0.scale.fc2.bias',	(512,),	 'layer2.0.se_module.fc2.bias',	(512,),
 'block2.0.shortcut.conv.weight',	(512, 256, 1, 1),	 'layer2.0.downsample.0.weight',	(512, 256, 1, 1),
 'block2.0.shortcut.bn.weight',	(512,),	 'layer2.0.downsample.1.weight',	(512,),
 'block2.0.shortcut.bn.bias',	(512,),	 'layer2.0.downsample.1.bias',	(512,),
 'block2.0.shortcut.bn.running_mean',	(512,),	 'layer2.0.downsample.1.running_mean',	(512,),
 'block2.0.shortcut.bn.running_var',	(512,),	 'layer2.0.downsample.1.running_var',	(512,),
 'block2.1.conv_bn1.conv.weight',	(256, 512, 1, 1),	 'layer2.1.conv1.weight',	(256, 512, 1, 1),
 'block2.1.conv_bn1.bn.weight',	(256,),	 'layer2.1.bn1.weight',	(256,),
 'block2.1.conv_bn1.bn.bias',	(256,),	 'layer2.1.bn1.bias',	(256,),
 'block2.1.conv_bn1.bn.running_mean',	(256,),	 'layer2.1.bn1.running_mean',	(256,),
 'block2.1.conv_bn1.bn.running_var',	(256,),	 'layer2.1.bn1.running_var',	(256,),
 'block2.1.conv_bn2.conv.weight',	(256, 8, 3, 3),	 'layer2.1.conv2.weight',	(256, 8, 3, 3),
 'block2.1.conv_bn2.bn.weight',	(256,),	 'layer2.1.bn2.weight',	(256,),
 'block2.1.conv_bn2.bn.bias',	(256,),	 'layer2.1.bn2.bias',	(256,),
 'block2.1.conv_bn2.bn.running_mean',	(256,),	 'layer2.1.bn2.running_mean',	(256,),
 'block2.1.conv_bn2.bn.running_var',	(256,),	 'layer2.1.bn2.running_var',	(256,),
 'block2.1.conv_bn3.conv.weight',	(512, 256, 1, 1),	 'layer2.1.conv3.weight',	(512, 256, 1, 1),
 'block2.1.conv_bn3.bn.weight',	(512,),	 'layer2.1.bn3.weight',	(512,),
 'block2.1.conv_bn3.bn.bias',	(512,),	 'layer2.1.bn3.bias',	(512,),
 'block2.1.conv_bn3.bn.running_mean',	(512,),	 'layer2.1.bn3.running_mean',	(512,),
 'block2.1.conv_bn3.bn.running_var',	(512,),	 'layer2.1.bn3.running_var',	(512,),
 'block2.1.scale.fc1.weight',	(32, 512, 1, 1),	 'layer2.1.se_module.fc1.weight',	(32, 512, 1, 1),
 'block2.1.scale.fc1.bias',	(32,),	 'layer2.1.se_module.fc1.bias',	(32,),
 'block2.1.scale.fc2.weight',	(512, 32, 1, 1),	 'layer2.1.se_module.fc2.weight',	(512, 32, 1, 1),
 'block2.1.scale.fc2.bias',	(512,),	 'layer2.1.se_module.fc2.bias',	(512,),
 'block2.2.conv_bn1.conv.weight',	(256, 512, 1, 1),	 'layer2.2.conv1.weight',	(256, 512, 1, 1),
 'block2.2.conv_bn1.bn.weight',	(256,),	 'layer2.2.bn1.weight',	(256,),
 'block2.2.conv_bn1.bn.bias',	(256,),	 'layer2.2.bn1.bias',	(256,),
 'block2.2.conv_bn1.bn.running_mean',	(256,),	 'layer2.2.bn1.running_mean',	(256,),
 'block2.2.conv_bn1.bn.running_var',	(256,),	 'layer2.2.bn1.running_var',	(256,),
 'block2.2.conv_bn2.conv.weight',	(256, 8, 3, 3),	 'layer2.2.conv2.weight',	(256, 8, 3, 3),
 'block2.2.conv_bn2.bn.weight',	(256,),	 'layer2.2.bn2.weight',	(256,),
 'block2.2.conv_bn2.bn.bias',	(256,),	 'layer2.2.bn2.bias',	(256,),
 'block2.2.conv_bn2.bn.running_mean',	(256,),	 'layer2.2.bn2.running_mean',	(256,),
 'block2.2.conv_bn2.bn.running_var',	(256,),	 'layer2.2.bn2.running_var',	(256,),
 'block2.2.conv_bn3.conv.weight',	(512, 256, 1, 1),	 'layer2.2.conv3.weight',	(512, 256, 1, 1),
 'block2.2.conv_bn3.bn.weight',	(512,),	 'layer2.2.bn3.weight',	(512,),
 'block2.2.conv_bn3.bn.bias',	(512,),	 'layer2.2.bn3.bias',	(512,),
 'block2.2.conv_bn3.bn.running_mean',	(512,),	 'layer2.2.bn3.running_mean',	(512,),
 'block2.2.conv_bn3.bn.running_var',	(512,),	 'layer2.2.bn3.running_var',	(512,),
 'block2.2.scale.fc1.weight',	(32, 512, 1, 1),	 'layer2.2.se_module.fc1.weight',	(32, 512, 1, 1),
 'block2.2.scale.fc1.bias',	(32,),	 'layer2.2.se_module.fc1.bias',	(32,),
 'block2.2.scale.fc2.weight',	(512, 32, 1, 1),	 'layer2.2.se_module.fc2.weight',	(512, 32, 1, 1),
 'block2.2.scale.fc2.bias',	(512,),	 'layer2.2.se_module.fc2.bias',	(512,),
 'block2.3.conv_bn1.conv.weight',	(256, 512, 1, 1),	 'layer2.3.conv1.weight',	(256, 512, 1, 1),
 'block2.3.conv_bn1.bn.weight',	(256,),	 'layer2.3.bn1.weight',	(256,),
 'block2.3.conv_bn1.bn.bias',	(256,),	 'layer2.3.bn1.bias',	(256,),
 'block2.3.conv_bn1.bn.running_mean',	(256,),	 'layer2.3.bn1.running_mean',	(256,),
 'block2.3.conv_bn1.bn.running_var',	(256,),	 'layer2.3.bn1.running_var',	(256,),
 'block2.3.conv_bn2.conv.weight',	(256, 8, 3, 3),	 'layer2.3.conv2.weight',	(256, 8, 3, 3),
 'block2.3.conv_bn2.bn.weight',	(256,),	 'layer2.3.bn2.weight',	(256,),
 'block2.3.conv_bn2.bn.bias',	(256,),	 'layer2.3.bn2.bias',	(256,),
 'block2.3.conv_bn2.bn.running_mean',	(256,),	 'layer2.3.bn2.running_mean',	(256,),
 'block2.3.conv_bn2.bn.running_var',	(256,),	 'layer2.3.bn2.running_var',	(256,),
 'block2.3.conv_bn3.conv.weight',	(512, 256, 1, 1),	 'layer2.3.conv3.weight',	(512, 256, 1, 1),
 'block2.3.conv_bn3.bn.weight',	(512,),	 'layer2.3.bn3.weight',	(512,),
 'block2.3.conv_bn3.bn.bias',	(512,),	 'layer2.3.bn3.bias',	(512,),
 'block2.3.conv_bn3.bn.running_mean',	(512,),	 'layer2.3.bn3.running_mean',	(512,),
 'block2.3.conv_bn3.bn.running_var',	(512,),	 'layer2.3.bn3.running_var',	(512,),
 'block2.3.scale.fc1.weight',	(32, 512, 1, 1),	 'layer2.3.se_module.fc1.weight',	(32, 512, 1, 1),
 'block2.3.scale.fc1.bias',	(32,),	 'layer2.3.se_module.fc1.bias',	(32,),
 'block2.3.scale.fc2.weight',	(512, 32, 1, 1),	 'layer2.3.se_module.fc2.weight',	(512, 32, 1, 1),
 'block2.3.scale.fc2.bias',	(512,),	 'layer2.3.se_module.fc2.bias',	(512,),
 'block3.0.conv_bn1.conv.weight',	(512, 512, 1, 1),	 'layer3.0.conv1.weight',	(512, 512, 1, 1),
 'block3.0.conv_bn1.bn.weight',	(512,),	 'layer3.0.bn1.weight',	(512,),
 'block3.0.conv_bn1.bn.bias',	(512,),	 'layer3.0.bn1.bias',	(512,),
 'block3.0.conv_bn1.bn.running_mean',	(512,),	 'layer3.0.bn1.running_mean',	(512,),
 'block3.0.conv_bn1.bn.running_var',	(512,),	 'layer3.0.bn1.running_var',	(512,),
 'block3.0.conv_bn2.conv.weight',	(512, 16, 3, 3),	 'layer3.0.conv2.weight',	(512, 16, 3, 3),
 'block3.0.conv_bn2.bn.weight',	(512,),	 'layer3.0.bn2.weight',	(512,),
 'block3.0.conv_bn2.bn.bias',	(512,),	 'layer3.0.bn2.bias',	(512,),
 'block3.0.conv_bn2.bn.running_mean',	(512,),	 'layer3.0.bn2.running_mean',	(512,),
 'block3.0.conv_bn2.bn.running_var',	(512,),	 'layer3.0.bn2.running_var',	(512,),
 'block3.0.conv_bn3.conv.weight',	(1024, 512, 1, 1),	 'layer3.0.conv3.weight',	(1024, 512, 1, 1),
 'block3.0.conv_bn3.bn.weight',	(1024,),	 'layer3.0.bn3.weight',	(1024,),
 'block3.0.conv_bn3.bn.bias',	(1024,),	 'layer3.0.bn3.bias',	(1024,),
 'block3.0.conv_bn3.bn.running_mean',	(1024,),	 'layer3.0.bn3.running_mean',	(1024,),
 'block3.0.conv_bn3.bn.running_var',	(1024,),	 'layer3.0.bn3.running_var',	(1024,),
 'block3.0.scale.fc1.weight',	(64, 1024, 1, 1),	 'layer3.0.se_module.fc1.weight',	(64, 1024, 1, 1),
 'block3.0.scale.fc1.bias',	(64,),	 'layer3.0.se_module.fc1.bias',	(64,),
 'block3.0.scale.fc2.weight',	(1024, 64, 1, 1),	 'layer3.0.se_module.fc2.weight',	(1024, 64, 1, 1),
 'block3.0.scale.fc2.bias',	(1024,),	 'layer3.0.se_module.fc2.bias',	(1024,),
 'block3.0.shortcut.conv.weight',	(1024, 512, 1, 1),	 'layer3.0.downsample.0.weight',	(1024, 512, 1, 1),
 'block3.0.shortcut.bn.weight',	(1024,),	 'layer3.0.downsample.1.weight',	(1024,),
 'block3.0.shortcut.bn.bias',	(1024,),	 'layer3.0.downsample.1.bias',	(1024,),
 'block3.0.shortcut.bn.running_mean',	(1024,),	 'layer3.0.downsample.1.running_mean',	(1024,),
 'block3.0.shortcut.bn.running_var',	(1024,),	 'layer3.0.downsample.1.running_var',	(1024,),
 'block3.1.conv_bn1.conv.weight',	(512, 1024, 1, 1),	 'layer3.1.conv1.weight',	(512, 1024, 1, 1),
 'block3.1.conv_bn1.bn.weight',	(512,),	 'layer3.1.bn1.weight',	(512,),
 'block3.1.conv_bn1.bn.bias',	(512,),	 'layer3.1.bn1.bias',	(512,),
 'block3.1.conv_bn1.bn.running_mean',	(512,),	 'layer3.1.bn1.running_mean',	(512,),
 'block3.1.conv_bn1.bn.running_var',	(512,),	 'layer3.1.bn1.running_var',	(512,),
 'block3.1.conv_bn2.conv.weight',	(512, 16, 3, 3),	 'layer3.1.conv2.weight',	(512, 16, 3, 3),
 'block3.1.conv_bn2.bn.weight',	(512,),	 'layer3.1.bn2.weight',	(512,),
 'block3.1.conv_bn2.bn.bias',	(512,),	 'layer3.1.bn2.bias',	(512,),
 'block3.1.conv_bn2.bn.running_mean',	(512,),	 'layer3.1.bn2.running_mean',	(512,),
 'block3.1.conv_bn2.bn.running_var',	(512,),	 'layer3.1.bn2.running_var',	(512,),
 'block3.1.conv_bn3.conv.weight',	(1024, 512, 1, 1),	 'layer3.1.conv3.weight',	(1024, 512, 1, 1),
 'block3.1.conv_bn3.bn.weight',	(1024,),	 'layer3.1.bn3.weight',	(1024,),
 'block3.1.conv_bn3.bn.bias',	(1024,),	 'layer3.1.bn3.bias',	(1024,),
 'block3.1.conv_bn3.bn.running_mean',	(1024,),	 'layer3.1.bn3.running_mean',	(1024,),
 'block3.1.conv_bn3.bn.running_var',	(1024,),	 'layer3.1.bn3.running_var',	(1024,),
 'block3.1.scale.fc1.weight',	(64, 1024, 1, 1),	 'layer3.1.se_module.fc1.weight',	(64, 1024, 1, 1),
 'block3.1.scale.fc1.bias',	(64,),	 'layer3.1.se_module.fc1.bias',	(64,),
 'block3.1.scale.fc2.weight',	(1024, 64, 1, 1),	 'layer3.1.se_module.fc2.weight',	(1024, 64, 1, 1),
 'block3.1.scale.fc2.bias',	(1024,),	 'layer3.1.se_module.fc2.bias',	(1024,),
 'block3.2.conv_bn1.conv.weight',	(512, 1024, 1, 1),	 'layer3.2.conv1.weight',	(512, 1024, 1, 1),
 'block3.2.conv_bn1.bn.weight',	(512,),	 'layer3.2.bn1.weight',	(512,),
 'block3.2.conv_bn1.bn.bias',	(512,),	 'layer3.2.bn1.bias',	(512,),
 'block3.2.conv_bn1.bn.running_mean',	(512,),	 'layer3.2.bn1.running_mean',	(512,),
 'block3.2.conv_bn1.bn.running_var',	(512,),	 'layer3.2.bn1.running_var',	(512,),
 'block3.2.conv_bn2.conv.weight',	(512, 16, 3, 3),	 'layer3.2.conv2.weight',	(512, 16, 3, 3),
 'block3.2.conv_bn2.bn.weight',	(512,),	 'layer3.2.bn2.weight',	(512,),
 'block3.2.conv_bn2.bn.bias',	(512,),	 'layer3.2.bn2.bias',	(512,),
 'block3.2.conv_bn2.bn.running_mean',	(512,),	 'layer3.2.bn2.running_mean',	(512,),
 'block3.2.conv_bn2.bn.running_var',	(512,),	 'layer3.2.bn2.running_var',	(512,),
 'block3.2.conv_bn3.conv.weight',	(1024, 512, 1, 1),	 'layer3.2.conv3.weight',	(1024, 512, 1, 1),
 'block3.2.conv_bn3.bn.weight',	(1024,),	 'layer3.2.bn3.weight',	(1024,),
 'block3.2.conv_bn3.bn.bias',	(1024,),	 'layer3.2.bn3.bias',	(1024,),
 'block3.2.conv_bn3.bn.running_mean',	(1024,),	 'layer3.2.bn3.running_mean',	(1024,),
 'block3.2.conv_bn3.bn.running_var',	(1024,),	 'layer3.2.bn3.running_var',	(1024,),
 'block3.2.scale.fc1.weight',	(64, 1024, 1, 1),	 'layer3.2.se_module.fc1.weight',	(64, 1024, 1, 1),
 'block3.2.scale.fc1.bias',	(64,),	 'layer3.2.se_module.fc1.bias',	(64,),
 'block3.2.scale.fc2.weight',	(1024, 64, 1, 1),	 'layer3.2.se_module.fc2.weight',	(1024, 64, 1, 1),
 'block3.2.scale.fc2.bias',	(1024,),	 'layer3.2.se_module.fc2.bias',	(1024,),
 'block3.3.conv_bn1.conv.weight',	(512, 1024, 1, 1),	 'layer3.3.conv1.weight',	(512, 1024, 1, 1),
 'block3.3.conv_bn1.bn.weight',	(512,),	 'layer3.3.bn1.weight',	(512,),
 'block3.3.conv_bn1.bn.bias',	(512,),	 'layer3.3.bn1.bias',	(512,),
 'block3.3.conv_bn1.bn.running_mean',	(512,),	 'layer3.3.bn1.running_mean',	(512,),
 'block3.3.conv_bn1.bn.running_var',	(512,),	 'layer3.3.bn1.running_var',	(512,),
 'block3.3.conv_bn2.conv.weight',	(512, 16, 3, 3),	 'layer3.3.conv2.weight',	(512, 16, 3, 3),
 'block3.3.conv_bn2.bn.weight',	(512,),	 'layer3.3.bn2.weight',	(512,),
 'block3.3.conv_bn2.bn.bias',	(512,),	 'layer3.3.bn2.bias',	(512,),
 'block3.3.conv_bn2.bn.running_mean',	(512,),	 'layer3.3.bn2.running_mean',	(512,),
 'block3.3.conv_bn2.bn.running_var',	(512,),	 'layer3.3.bn2.running_var',	(512,),
 'block3.3.conv_bn3.conv.weight',	(1024, 512, 1, 1),	 'layer3.3.conv3.weight',	(1024, 512, 1, 1),
 'block3.3.conv_bn3.bn.weight',	(1024,),	 'layer3.3.bn3.weight',	(1024,),
 'block3.3.conv_bn3.bn.bias',	(1024,),	 'layer3.3.bn3.bias',	(1024,),
 'block3.3.conv_bn3.bn.running_mean',	(1024,),	 'layer3.3.bn3.running_mean',	(1024,),
 'block3.3.conv_bn3.bn.running_var',	(1024,),	 'layer3.3.bn3.running_var',	(1024,),
 'block3.3.scale.fc1.weight',	(64, 1024, 1, 1),	 'layer3.3.se_module.fc1.weight',	(64, 1024, 1, 1),
 'block3.3.scale.fc1.bias',	(64,),	 'layer3.3.se_module.fc1.bias',	(64,),
 'block3.3.scale.fc2.weight',	(1024, 64, 1, 1),	 'layer3.3.se_module.fc2.weight',	(1024, 64, 1, 1),
 'block3.3.scale.fc2.bias',	(1024,),	 'layer3.3.se_module.fc2.bias',	(1024,),
 'block3.4.conv_bn1.conv.weight',	(512, 1024, 1, 1),	 'layer3.4.conv1.weight',	(512, 1024, 1, 1),
 'block3.4.conv_bn1.bn.weight',	(512,),	 'layer3.4.bn1.weight',	(512,),
 'block3.4.conv_bn1.bn.bias',	(512,),	 'layer3.4.bn1.bias',	(512,),
 'block3.4.conv_bn1.bn.running_mean',	(512,),	 'layer3.4.bn1.running_mean',	(512,),
 'block3.4.conv_bn1.bn.running_var',	(512,),	 'layer3.4.bn1.running_var',	(512,),
 'block3.4.conv_bn2.conv.weight',	(512, 16, 3, 3),	 'layer3.4.conv2.weight',	(512, 16, 3, 3),
 'block3.4.conv_bn2.bn.weight',	(512,),	 'layer3.4.bn2.weight',	(512,),
 'block3.4.conv_bn2.bn.bias',	(512,),	 'layer3.4.bn2.bias',	(512,),
 'block3.4.conv_bn2.bn.running_mean',	(512,),	 'layer3.4.bn2.running_mean',	(512,),
 'block3.4.conv_bn2.bn.running_var',	(512,),	 'layer3.4.bn2.running_var',	(512,),
 'block3.4.conv_bn3.conv.weight',	(1024, 512, 1, 1),	 'layer3.4.conv3.weight',	(1024, 512, 1, 1),
 'block3.4.conv_bn3.bn.weight',	(1024,),	 'layer3.4.bn3.weight',	(1024,),
 'block3.4.conv_bn3.bn.bias',	(1024,),	 'layer3.4.bn3.bias',	(1024,),
 'block3.4.conv_bn3.bn.running_mean',	(1024,),	 'layer3.4.bn3.running_mean',	(1024,),
 'block3.4.conv_bn3.bn.running_var',	(1024,),	 'layer3.4.bn3.running_var',	(1024,),
 'block3.4.scale.fc1.weight',	(64, 1024, 1, 1),	 'layer3.4.se_module.fc1.weight',	(64, 1024, 1, 1),
 'block3.4.scale.fc1.bias',	(64,),	 'layer3.4.se_module.fc1.bias',	(64,),
 'block3.4.scale.fc2.weight',	(1024, 64, 1, 1),	 'layer3.4.se_module.fc2.weight',	(1024, 64, 1, 1),
 'block3.4.scale.fc2.bias',	(1024,),	 'layer3.4.se_module.fc2.bias',	(1024,),
 'block3.5.conv_bn1.conv.weight',	(512, 1024, 1, 1),	 'layer3.5.conv1.weight',	(512, 1024, 1, 1),
 'block3.5.conv_bn1.bn.weight',	(512,),	 'layer3.5.bn1.weight',	(512,),
 'block3.5.conv_bn1.bn.bias',	(512,),	 'layer3.5.bn1.bias',	(512,),
 'block3.5.conv_bn1.bn.running_mean',	(512,),	 'layer3.5.bn1.running_mean',	(512,),
 'block3.5.conv_bn1.bn.running_var',	(512,),	 'layer3.5.bn1.running_var',	(512,),
 'block3.5.conv_bn2.conv.weight',	(512, 16, 3, 3),	 'layer3.5.conv2.weight',	(512, 16, 3, 3),
 'block3.5.conv_bn2.bn.weight',	(512,),	 'layer3.5.bn2.weight',	(512,),
 'block3.5.conv_bn2.bn.bias',	(512,),	 'layer3.5.bn2.bias',	(512,),
 'block3.5.conv_bn2.bn.running_mean',	(512,),	 'layer3.5.bn2.running_mean',	(512,),
 'block3.5.conv_bn2.bn.running_var',	(512,),	 'layer3.5.bn2.running_var',	(512,),
 'block3.5.conv_bn3.conv.weight',	(1024, 512, 1, 1),	 'layer3.5.conv3.weight',	(1024, 512, 1, 1),
 'block3.5.conv_bn3.bn.weight',	(1024,),	 'layer3.5.bn3.weight',	(1024,),
 'block3.5.conv_bn3.bn.bias',	(1024,),	 'layer3.5.bn3.bias',	(1024,),
 'block3.5.conv_bn3.bn.running_mean',	(1024,),	 'layer3.5.bn3.running_mean',	(1024,),
 'block3.5.conv_bn3.bn.running_var',	(1024,),	 'layer3.5.bn3.running_var',	(1024,),
 'block3.5.scale.fc1.weight',	(64, 1024, 1, 1),	 'layer3.5.se_module.fc1.weight',	(64, 1024, 1, 1),
 'block3.5.scale.fc1.bias',	(64,),	 'layer3.5.se_module.fc1.bias',	(64,),
 'block3.5.scale.fc2.weight',	(1024, 64, 1, 1),	 'layer3.5.se_module.fc2.weight',	(1024, 64, 1, 1),
 'block3.5.scale.fc2.bias',	(1024,),	 'layer3.5.se_module.fc2.bias',	(1024,),
 'block4.0.conv_bn1.conv.weight',	(1024, 1024, 1, 1),	 'layer4.0.conv1.weight',	(1024, 1024, 1, 1),
 'block4.0.conv_bn1.bn.weight',	(1024,),	 'layer4.0.bn1.weight',	(1024,),
 'block4.0.conv_bn1.bn.bias',	(1024,),	 'layer4.0.bn1.bias',	(1024,),
 'block4.0.conv_bn1.bn.running_mean',	(1024,),	 'layer4.0.bn1.running_mean',	(1024,),
 'block4.0.conv_bn1.bn.running_var',	(1024,),	 'layer4.0.bn1.running_var',	(1024,),
 'block4.0.conv_bn2.conv.weight',	(1024, 32, 3, 3),	 'layer4.0.conv2.weight',	(1024, 32, 3, 3),
 'block4.0.conv_bn2.bn.weight',	(1024,),	 'layer4.0.bn2.weight',	(1024,),
 'block4.0.conv_bn2.bn.bias',	(1024,),	 'layer4.0.bn2.bias',	(1024,),
 'block4.0.conv_bn2.bn.running_mean',	(1024,),	 'layer4.0.bn2.running_mean',	(1024,),
 'block4.0.conv_bn2.bn.running_var',	(1024,),	 'layer4.0.bn2.running_var',	(1024,),
 'block4.0.conv_bn3.conv.weight',	(2048, 1024, 1, 1),	 'layer4.0.conv3.weight',	(2048, 1024, 1, 1),
 'block4.0.conv_bn3.bn.weight',	(2048,),	 'layer4.0.bn3.weight',	(2048,),
 'block4.0.conv_bn3.bn.bias',	(2048,),	 'layer4.0.bn3.bias',	(2048,),
 'block4.0.conv_bn3.bn.running_mean',	(2048,),	 'layer4.0.bn3.running_mean',	(2048,),
 'block4.0.conv_bn3.bn.running_var',	(2048,),	 'layer4.0.bn3.running_var',	(2048,),
 'block4.0.scale.fc1.weight',	(128, 2048, 1, 1),	 'layer4.0.se_module.fc1.weight',	(128, 2048, 1, 1),
 'block4.0.scale.fc1.bias',	(128,),	 'layer4.0.se_module.fc1.bias',	(128,),
 'block4.0.scale.fc2.weight',	(2048, 128, 1, 1),	 'layer4.0.se_module.fc2.weight',	(2048, 128, 1, 1),
 'block4.0.scale.fc2.bias',	(2048,),	 'layer4.0.se_module.fc2.bias',	(2048,),
 'block4.0.shortcut.conv.weight',	(2048, 1024, 1, 1),	 'layer4.0.downsample.0.weight',	(2048, 1024, 1, 1),
 'block4.0.shortcut.bn.weight',	(2048,),	 'layer4.0.downsample.1.weight',	(2048,),
 'block4.0.shortcut.bn.bias',	(2048,),	 'layer4.0.downsample.1.bias',	(2048,),
 'block4.0.shortcut.bn.running_mean',	(2048,),	 'layer4.0.downsample.1.running_mean',	(2048,),
 'block4.0.shortcut.bn.running_var',	(2048,),	 'layer4.0.downsample.1.running_var',	(2048,),
 'block4.1.conv_bn1.conv.weight',	(1024, 2048, 1, 1),	 'layer4.1.conv1.weight',	(1024, 2048, 1, 1),
 'block4.1.conv_bn1.bn.weight',	(1024,),	 'layer4.1.bn1.weight',	(1024,),
 'block4.1.conv_bn1.bn.bias',	(1024,),	 'layer4.1.bn1.bias',	(1024,),
 'block4.1.conv_bn1.bn.running_mean',	(1024,),	 'layer4.1.bn1.running_mean',	(1024,),
 'block4.1.conv_bn1.bn.running_var',	(1024,),	 'layer4.1.bn1.running_var',	(1024,),
 'block4.1.conv_bn2.conv.weight',	(1024, 32, 3, 3),	 'layer4.1.conv2.weight',	(1024, 32, 3, 3),
 'block4.1.conv_bn2.bn.weight',	(1024,),	 'layer4.1.bn2.weight',	(1024,),
 'block4.1.conv_bn2.bn.bias',	(1024,),	 'layer4.1.bn2.bias',	(1024,),
 'block4.1.conv_bn2.bn.running_mean',	(1024,),	 'layer4.1.bn2.running_mean',	(1024,),
 'block4.1.conv_bn2.bn.running_var',	(1024,),	 'layer4.1.bn2.running_var',	(1024,),
 'block4.1.conv_bn3.conv.weight',	(2048, 1024, 1, 1),	 'layer4.1.conv3.weight',	(2048, 1024, 1, 1),
 'block4.1.conv_bn3.bn.weight',	(2048,),	 'layer4.1.bn3.weight',	(2048,),
 'block4.1.conv_bn3.bn.bias',	(2048,),	 'layer4.1.bn3.bias',	(2048,),
 'block4.1.conv_bn3.bn.running_mean',	(2048,),	 'layer4.1.bn3.running_mean',	(2048,),
 'block4.1.conv_bn3.bn.running_var',	(2048,),	 'layer4.1.bn3.running_var',	(2048,),
 'block4.1.scale.fc1.weight',	(128, 2048, 1, 1),	 'layer4.1.se_module.fc1.weight',	(128, 2048, 1, 1),
 'block4.1.scale.fc1.bias',	(128,),	 'layer4.1.se_module.fc1.bias',	(128,),
 'block4.1.scale.fc2.weight',	(2048, 128, 1, 1),	 'layer4.1.se_module.fc2.weight',	(2048, 128, 1, 1),
 'block4.1.scale.fc2.bias',	(2048,),	 'layer4.1.se_module.fc2.bias',	(2048,),
 'block4.2.conv_bn1.conv.weight',	(1024, 2048, 1, 1),	 'layer4.2.conv1.weight',	(1024, 2048, 1, 1),
 'block4.2.conv_bn1.bn.weight',	(1024,),	 'layer4.2.bn1.weight',	(1024,),
 'block4.2.conv_bn1.bn.bias',	(1024,),	 'layer4.2.bn1.bias',	(1024,),
 'block4.2.conv_bn1.bn.running_mean',	(1024,),	 'layer4.2.bn1.running_mean',	(1024,),
 'block4.2.conv_bn1.bn.running_var',	(1024,),	 'layer4.2.bn1.running_var',	(1024,),
 'block4.2.conv_bn2.conv.weight',	(1024, 32, 3, 3),	 'layer4.2.conv2.weight',	(1024, 32, 3, 3),
 'block4.2.conv_bn2.bn.weight',	(1024,),	 'layer4.2.bn2.weight',	(1024,),
 'block4.2.conv_bn2.bn.bias',	(1024,),	 'layer4.2.bn2.bias',	(1024,),
 'block4.2.conv_bn2.bn.running_mean',	(1024,),	 'layer4.2.bn2.running_mean',	(1024,),
 'block4.2.conv_bn2.bn.running_var',	(1024,),	 'layer4.2.bn2.running_var',	(1024,),
 'block4.2.conv_bn3.conv.weight',	(2048, 1024, 1, 1),	 'layer4.2.conv3.weight',	(2048, 1024, 1, 1),
 'block4.2.conv_bn3.bn.weight',	(2048,),	 'layer4.2.bn3.weight',	(2048,),
 'block4.2.conv_bn3.bn.bias',	(2048,),	 'layer4.2.bn3.bias',	(2048,),
 'block4.2.conv_bn3.bn.running_mean',	(2048,),	 'layer4.2.bn3.running_mean',	(2048,),
 'block4.2.conv_bn3.bn.running_var',	(2048,),	 'layer4.2.bn3.running_var',	(2048,),
 'block4.2.scale.fc1.weight',	(128, 2048, 1, 1),	 'layer4.2.se_module.fc1.weight',	(128, 2048, 1, 1),
 'block4.2.scale.fc1.bias',	(128,),	 'layer4.2.se_module.fc1.bias',	(128,),
 'block4.2.scale.fc2.weight',	(2048, 128, 1, 1),	 'layer4.2.se_module.fc2.weight',	(2048, 128, 1, 1),
 'block4.2.scale.fc2.bias',	(2048,),	 'layer4.2.se_module.fc2.bias',	(2048,),
 'logit.weight',	(1000, 1280),	 'last_linear.weight',	(1000, 2048),
 'logit.bias',	(1000,),	 'last_linear.bias',	(1000,),
]

def load_pretrain(net, skip=[], pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=True):

    #raise NotImplementedError
    print('\tload pretrain_file: %s'%pretrain_file)

    #pretrain_state_dict = torch.load(pretrain_file)
    pretrain_state_dict = torch.load(pretrain_file, map_location=lambda storage, loc: storage)
    state_dict = net.state_dict()

    i = 0
    conversion = np.array(CONVERSION).reshape(-1,4)
    for key,_,pretrain_key,_ in conversion:
        if any(s in key for s in
            ['.num_batches_tracked',]+skip):
            continue

        #print('\t\t',key)
        if is_print:
            print('\t\t','%-48s  %-24s  <---  %-32s  %-24s'%(
                key, str(state_dict[key].shape),
                pretrain_key, str(pretrain_state_dict[pretrain_key].shape),
            ))
        i = i+1

        state_dict[key] = pretrain_state_dict[pretrain_key]


    net.load_state_dict(state_dict)
    print('')
    print('len(pretrain_state_dict.keys()) = %d'%len(pretrain_state_dict.keys()))
    print('len(state_dict.keys())          = %d'%len(state_dict.keys()))
    print('loaded    = %d'%i)
    print('')


def load_modi_file(net, pretrain_file=PRETRAIN_FILE):
    print('\tload pretrain_file: %s' % pretrain_file)
    pretrain_state_dict = torch.load(pretrain_file)
    net.load_state_dict(pretrain_state_dict)



class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1


class RGB(nn.Module):
    def __init__(self, ):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('std', torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x


class ConvBn2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_channel, reduction=4, ):
        super(SqueezeExcite, self).__init__()

        self.fc1 = nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1, padding=0)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.fc1(s)
        s = F.relu(s, inplace=True)
        s = self.fc2(s)
        x = x * torch.sigmoid(s)
        return x


#############  resnext50 pyramid feature net #######################################
# https://github.com/Hsuxu/ResNeXt/blob/master/models.py
# https://github.com/D-X-Y/ResNeXt-DenseNet/blob/master/models/resnext.py
# https://github.com/miraclewkf/ResNeXt-PyTorch/blob/master/resnext.py


# bottleneck type C
class SENextBottleneck(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride=1, group=32,
                 reduction=16, pool=None, is_shortcut=False):
        super(SENextBottleneck, self).__init__()

        self.conv_bn1 = ConvBn2d(in_channel, channel[0], kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(channel[0], channel[1], kernel_size=3, padding=1, stride=1, groups=group)
        self.conv_bn3 = ConvBn2d(channel[1], out_channel, kernel_size=1, padding=0, stride=1)
        self.scale = SqueezeExcite(out_channel, reduction)

        # ---
        self.is_shortcut = is_shortcut
        self.stride = stride
        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1)

        if stride == 2:
            if pool == 'max': self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            if pool == 'avg': self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        z = F.relu(self.conv_bn1(x), inplace=True)
        z = F.relu(self.conv_bn2(z), inplace=True)
        if self.stride == 2:
            z = self.pool(z)

        z = self.scale(self.conv_bn3(z))
        if self.is_shortcut:
            if self.stride == 2:
                x = F.avg_pool2d(x, 2, 2)  # avg_pool2d
            x = self.shortcut(x)

        # z += x
        z = torch.max(z, x)
        z = F.relu(z, inplace=True)
        return z


class Identity(nn.Module):
    def forward(self, x):
        return x


# resnext50_32x4d
class ResNext50(nn.Module):

    def __init__(self, num_class=1000):
        super(ResNext50, self).__init__()
        self.rgb = RGB()

        self.block0 = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False), #bias=0
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, padding=0, stride=2, ceil_mode=True),
            # nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
            # Identity(),

            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # bias=0
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),  # bias=0
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),  # bias=0
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
            Identity(),
        )
        self.block1 = nn.Sequential(
            SENextBottleneck(64, [128, 128], 256, stride=2, is_shortcut=True, pool='max', ),
            *[SENextBottleneck(256, [128, 128], 256, stride=1, is_shortcut=False, ) for i in range(1, 3)],
        )
        self.block2 = nn.Sequential(
            SENextBottleneck(256, [256, 256], 512, stride=2, is_shortcut=True, pool='max', ),
            *[SENextBottleneck(512, [256, 256], 512, stride=1, is_shortcut=False, ) for i in range(1, 4)],
        )
        self.block3 = nn.Sequential(
            SENextBottleneck(512, [512, 512], 1024, stride=2, is_shortcut=True, pool='max', ),
            *[SENextBottleneck(1024, [512, 512], 1024, stride=1, is_shortcut=False, ) for i in range(1, 6)],
        )
        self.block4 = nn.Sequential(
            SENextBottleneck(1024, [1024, 1024], 2048, stride=2, is_shortcut=True, pool='avg', ),
            *[SENextBottleneck(2048, [1024, 1024], 2048, stride=1, is_shortcut=False) for i in range(1, 3)],
        )

        self.logit = nn.Linear(2048, num_class)

    def forward(self, x):
        batch_size = len(x)
        # x = self.rgb(x)

        x = self.block0(x)
        # x = F.max_pool2d(x,kernel_size=2,stride=2)
        x = self.block1(x)
        # x = F.max_pool2d(x,kernel_size=2,stride=2)
        x = self.block2(x)
        # x = F.max_pool2d(x,kernel_size=2,stride=2)
        x = self.block3(x)
        # x = F.max_pool2d(x,kernel_size=2,stride=2)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        logit = self.logit(x)
        return logit


class Net(nn.Module):
    def load_pretrain(self, skip=['block0.', 'logit.'], is_print=True):
    #def load_pretrain(self, skip=[], is_print=True):
        #load_modi_file(self, pretrain_file=PRETRAIN_FILE)
        load_pretrain(self, skip, pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=is_print)

    def __init__(self, num_class=NUM_CLASS):
        super(Net, self).__init__()
        e = ResNext50()

        self.block0 = e.block0
        self.block1 = e.block1
        self.block2 = e.block2
        self.block3 = e.block3
        self.block4 = e.block4
        e = None  # dropped

        # self.dropblock0 = DropBlock2D(drop_prob=0.2, block_size=32)
        # self.dropblock1 = DropBlock2D(drop_prob=0.2, block_size=16)
        # self.dropblock2 = DropBlock2D(drop_prob=0.2, block_size=8)
        # self.dropblock3 = DropBlock2D(drop_prob=0.2, block_size=4)
        # self.dropblock4 = DropBlock2D(drop_prob=0.2, block_size=2)

        self.logit = nn.ModuleList(
            [nn.Linear(2048, c) for c in num_class]
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        # if (H,W) !=(64,112):
        #     x = F.interpolate(x,size=(64,112), mode='bilinear',align_corners=False)

        x = x.repeat(1, 3, 1, 1)
        x = self.block0(x)
        # x = self.dropblock0(x)
        x = self.block1(x)
        # x = self.dropblock1(x)
        x = self.block2(x)
        # x = self.dropblock2(x)
        x = self.block3(x)
        # x = self.dropblock3(x)
        x = self.block4(x)
        # x = self.dropblock4(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = F.dropout(x, 0.2, self.training)

        # feature = None
        logit = [l(x) for l in self.logit]
        return logit[0], logit[1], logit[2], logit[3]

model = Net()
model.load_pretrain(is_print=False)
model.to(device)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, gr, v, c, g, alpha):
    indices = torch.randperm(data.size(0))
    # shuffled_data = data[indices]
    shuffled_gr = gr[indices]
    shuffled_v = v[indices]
    shuffled_c = c[indices]
    shuffled_g = g[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = [gr, v, c, g]
    shuffled_targets = [shuffled_gr, shuffled_v, shuffled_c, shuffled_g]

    return data, targets, shuffled_targets, lam


def mixup(data, gr, v, c, g, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_gr = gr[indices]
    shuffled_v = v[indices]
    shuffled_c = c[indices]
    shuffled_g = g[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = [gr, v, c, g]
    shuffled_targets = [shuffled_gr, shuffled_v, shuffled_c, shuffled_g]

    return data, targets, shuffled_targets, lam


def shuffled_loss_fn(preds, targets, shuffled_targets, lam):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    loss = lam * criterion(preds[0], targets[0]) + (1 - lam) * criterion(preds[0], shuffled_targets[0]) \
           + lam * criterion(preds[1], targets[1]) + (1 - lam) * criterion(preds[1], shuffled_targets[1]) \
           + lam * criterion(preds[2], targets[2]) + (1 - lam) * criterion(preds[2], shuffled_targets[2]) \
           + lam * criterion(preds[3], targets[3]) + (1 - lam) * criterion(preds[3], shuffled_targets[3])
    return loss / 4


def macro_recall(pred_y, y, n_grapheme_root=168, n_vowel=11, n_consonant=7, n_grapheme=1295):
    pred_y = torch.split(pred_y, [n_grapheme_root, n_vowel, n_consonant, n_grapheme], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]

    y = y.cpu().numpy()

    pred_g = pd.Series(pred_labels[3])
    pred_gr = pred_g.map(gr_map)
    pred_v = pred_g.map(v_map)
    pred_c = pred_g.map(c_map)

    # for i in range(len(pred_labels[3])):
    #    pred_labels[0][i] = grapheme_map.loc[pred_labels[3][i], 'grapheme_root']
    #    pred_labels[1][i] = grapheme_map.loc[pred_labels[3][i], 'vowel_diacritic']
    #    pred_labels[2][i] = grapheme_map.loc[pred_labels[3][i], 'consonant_diacritic']

    recall_gr = recall_score(y[:, 0], pred_gr, average='macro')
    recall_v = recall_score(y[:, 1], pred_v, average='macro')
    recall_c = recall_score(y[:, 2], pred_c, average='macro')
    recall_g = recall_score(y[:, 3], pred_g, average='macro')
    recall_tot = np.average([recall_gr, recall_v, recall_c], weights=[2, 1, 1])
    # print(f'recall: grapheme {recall_g:.5f}, vowel {recall_v:.5f}, consonant {recall_c:.5f}, 'f'total {recall_tot:.5f}')

    return recall_gr, recall_v, recall_c, recall_g, recall_tot


def loss_fn(outputs, targets):
    o1, o2, o3, o4 = outputs
    t1, t2, t3, t4 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    l4 = nn.CrossEntropyLoss()(o4, t4)

    return (l1 + l2 + l3 + l4) / 4


def run_train(data_loader, model, optimizer, steps_per_epoch):
    model.train()
    final_loss = 0
    counter = 0
    final_outputs = []
    final_targets = []

    is_plot = False

    for bi, (original, image, gr, v, c, g) in tqdm(enumerate(data_loader), total=steps_per_epoch):
        counter = counter + 1

        image = image.to(device, dtype=torch.float)
        gr = gr.to(device, dtype=torch.long)
        v = v.to(device, dtype=torch.long)
        c = c.to(device, dtype=torch.long)
        g = g.to(device, dtype=torch.long)

        optimizer.zero_grad()

        # if is_plot:
        #    is_plot = False
        #    plot_images(original, image, title='cutmix')

        regularization_decision = np.random.rand()
        if regularization_decision < 0.25:  # MIXUP
            MIXUP_ALPHA = 0.4
            image, targets, shuffled_targets, lam = mixup(original, gr, v, c, g, MIXUP_ALPHA)
            outputs = model(image)
            loss = shuffled_loss_fn(outputs, targets, shuffled_targets, lam)
        elif regularization_decision < 0.50:  # CUTMIX
            CUTMIX_ALPHA = 1.0
            image, targets, shuffled_targets, lam = cutmix(original, gr, v, c, g, CUTMIX_ALPHA)
            outputs = model(image)
            loss = shuffled_loss_fn(outputs, targets, shuffled_targets, lam)
        elif regularization_decision < 0.75:  # CUTOUT
            image, _, _ = do_random_batch_cutout(original, None)
            outputs = model(image)
            targets = (gr, v, c, g)
            loss = loss_fn(outputs, targets)
        else:  # normal aug
            outputs = model(image)
            targets = (gr, v, c, g)
            loss = loss_fn(outputs, targets)

        # apex
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        # loss.backward()
        optimizer.step()

        final_loss += loss.item()

        o1, o2, o3, o4 = outputs
        t1, t2, t3, t4 = targets
        final_outputs.append(torch.cat((o1, o2, o3, o4), dim=1))
        final_targets.append(torch.stack((t1, t2, t3, t4), dim=1))

        # if bi % 10 == 0:
        #    break
    final_outputs = torch.cat(final_outputs)
    final_targets = torch.cat(final_targets)

    # print("=================Train=================")
    recall_gr, recall_v, recall_c, recall_g, recall_tot = macro_recall(final_outputs, final_targets)

    return final_loss / counter, recall_gr, recall_v, recall_c, recall_g, recall_tot


def run_evaluate(data_loader, model):
    with torch.no_grad():
        model.eval()
        final_loss = 0
        counter = 0
        final_outputs = []
        final_targets = []
        for bi, (original, image, gr, v, c, g) in enumerate(data_loader):
            counter = counter + 1

            image = image.to(device, dtype=torch.float)
            gr = gr.to(device, dtype=torch.long)
            v = v.to(device, dtype=torch.long)
            c = c.to(device, dtype=torch.long)
            g = g.to(device, dtype=torch.long)

            outputs = model(image)
            targets = (gr, v, c, g)
            loss = loss_fn(outputs, targets)
            final_loss += loss.item()

            o1, o2, o3, o4 = outputs
            t1, t2, t3, t4 = targets
            # print(t1.shape)
            final_outputs.append(torch.cat((o1, o2, o3, o4), dim=1))
            final_targets.append(torch.stack((t1, t2, t3, t4), dim=1))

        final_outputs = torch.cat(final_outputs)
        final_targets = torch.cat(final_targets)

        # print("=================Valid=================")
        recall_gr, recall_v, recall_c, recall_g, recall_tot = macro_recall(final_outputs, final_targets)

    return final_loss / counter, recall_gr, recall_v, recall_c, recall_g, recall_tot


train_idx = train[train['fold']!=val_fold].index
val_idx = train[train['fold']==val_fold].index

train_data_size = 200 if debug else len(train_idx)
valid_data_size = 100 if debug else len(val_idx)
steps_per_epoch = math.ceil(train_data_size/batch_size)

train_dataset = BengaliAIDataset(train_images, train_labels, transform=train_augment, indices=train_idx)
train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True, num_workers=num_workers)

valid_dataset = BengaliAIDataset(train_images, train_labels, transform=valid_augment, indices=val_idx)
valid_loader = DataLoader(dataset=valid_dataset, batch_size= batch_size, shuffle=True, num_workers=num_workers//2)
print('train_dataset', len(train_dataset), 'valid_dataset', len(valid_dataset), 'steps_per_epoch', steps_per_epoch)

lr = 0.00075*torch.cuda.device_count()*batch_size/(batch_per_cuda*torch.cuda.device_count())
print(f'initial lr : {lr}')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='max',
                                                       patience=5,
                                                       factor=0.5, verbose=True)

# Initialization
opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    #os.environ['MASTER_ADDR'] = '127.0.0.1'
    #os.environ['MASTER_PORT'] = '29500'
    #torch.distributed.init_process_group(backend="gloo", group_name="main")
    #model = apex.parallel.DistributedDataParallel(model)

    #local_rank = 0
    #os.environ['MASTER_ADDR'] = '127.0.0.1'
    #os.environ['MASTER_PORT'] = '29500'
    #os.environ['RANK'] = f'{local_rank}'
    #os.environ['WORLD_SIZE'] = '1'
    #torch.cuda.set_device(local_rank)
    #torch.distributed.init_process_group(backend='nccl', init_method='env://')
    #model = DistributedDataParallel(model, delay_allreduce=True)

clear_cache()
best_score = -1
histories = []

for i in range(num_epochs):
    start = time.time()
    epoch = i + 1

    print(f'Epoch: {epoch}')

    history = {
        'epoch': epoch,
    }

    train_loss, tr_recall_gr, tr_recall_v, tr_recall_c, tr_recall_g, tr_recall_tot = run_train(train_loader, model,
                                                                                               optimizer,
                                                                                               steps_per_epoch)
    val_loss, val_recall_gr, val_recall_v, val_recall_c, val_recall_g, val_recall_tot = run_evaluate(valid_loader,
                                                                                                     model)

    lr = scheduler.optimizer.param_groups[0]['lr']
    scheduler.step(val_recall_tot)

    if val_recall_tot > best_score:
        best_score = val_recall_tot
        file_name = modeldir / f'model_{experi_num}_epoch_{epoch:03d}_fold_{val_fold}_recall_{val_recall_tot:.4f}.pt'
        torch.save(model.state_dict(), file_name)
        print('save max accuracy model: ', val_recall_tot)

    elapsed = time.time() - start

    history['lr'] = lr
    history['train/loss'] = train_loss
    history['train/recall_gr'] = tr_recall_gr
    history['train/recall_v'] = tr_recall_v
    history['train/recall_c'] = tr_recall_c
    history['train/recall_g'] = tr_recall_g
    history['train/recall'] = tr_recall_tot
    history['val/loss'] = val_loss
    history['val/recall_gr'] = val_recall_gr
    history['val/recall_v'] = val_recall_v
    history['val/recall_c'] = val_recall_c
    history['val/recall_g'] = val_recall_g
    history['val/recall'] = val_recall_tot
    history['elapsed_time'] = elapsed
    histories.append(history)

    pd.DataFrame(histories).to_csv(logdir / f'log_{experi_num}.csv', index=False)

    epoch_len = len(str(num_epochs))
    print_msg = (
            f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
            f'lr: {lr:.5f}, ' +
            f'train_loss: {train_loss:.5f}, ' +
            f'train_recall/gr: {tr_recall_gr:.5f}, ' +
            f'train_recall/v: {tr_recall_v:.5f}, ' +
            f'train_recall/c: {tr_recall_c:.5f}, ' +
            f'train_recall/g: {tr_recall_g:.5f}, ' +
            f'train_recall/tot: {tr_recall_tot:.5f} \n          ' +
            f'valid_loss: {val_loss:.5f}, ' +
            f'valid_recall/gr: {val_recall_gr:.5f}, ' +
            f'valid_recall/v: {val_recall_v:.5f}, ' +
            f'valid_recall/c: {val_recall_c:.5f}, ' +
            f'valid_recall/g: {val_recall_g:.5f}, ' +
            f'valid_recall/tot: {val_recall_tot:.5f}, ' +
            f'elasped: {elapsed}'
    )

    print(print_msg)
    print('-' * 100)
