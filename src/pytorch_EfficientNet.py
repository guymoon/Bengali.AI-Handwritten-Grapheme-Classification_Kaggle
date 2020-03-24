import warnings
warnings.filterwarnings("ignore")

import gc
import os
from pathlib import Path
import random
import time
import multiprocessing
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.metrics import recall_score

from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

import apex
from apex import amp
from apex.parallel import DistributedDataParallel

import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as AF
import cv2
from PIL import Image, ImageEnhance, ImageOps

from torchvision import transforms
import pretrainedmodels
#from torchsummary import summary

###### utils

import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.
        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.
        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, image_size=None, num_classes=1000):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


url_map = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
}


url_map_advprop = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth',
    'efficientnet-b8': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth',
}


def load_pretrained_weights(model, model_name, load_fc=True, advprop=False, ch=1):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    # AutoAugment or Advprop (different preprocessing)
    url_map_ = url_map_advprop if advprop else url_map
    # with url
    #state_dict = model_zoo.load_url(url_map_[model_name])

    # with local weight
    state_dict = torch.load('/workspace/inyong/OFEDGENet/model/model_iy-eb6-02_epoch_147_fold_0_recall_0.9939.pt')

    if load_fc:
        #if ch == 1:
        #    conv1_weight = state_dict['_conv_stem.weight']
        #    state_dict['_conv_stem.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        model.load_state_dict(state_dict)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        #if ch == 1:
        #    conv1_weight = state_dict['_conv_stem.weight']
        #    state_dict['_conv_stem.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias', '_fc_1.weight', '_fc_1.bias', '_fc_2.weight', '_fc_2.bias', '_fc_3.weight', '_fc_3.bias']), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(model_name))



############


###### model

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

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        # add yong
        self._fc_1 = nn.Linear(out_channels, 168)
        self._fc_2 = nn.Linear(out_channels, 11)
        self._fc_3 = nn.Linear(out_channels, 7)

        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        gr = self._fc_1(x)
        v = self._fc_2(x)
        c = self._fc_3(x)
        g = self._fc(x)

        return gr, v, c, g

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})

        # others
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1295), advprop=advprop)
        # if use pre-trained model such as imagenet
        #load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        #if in_channels != 3:
        #    Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
        #    out_channels = round_filters(32, model._global_params)
        #    model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b' + str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

############

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

#######
debug=False
submission=False
batch_size=156*torch.cuda.device_count()
device='cuda'
out='.'
image_height=137
image_width=236
model_name='efficientnet-b6'
num_epochs = 150
experi_num = 'iy-eb6-03'
num_workers=3
SEED = 42
print(f'batch_size:{batch_size}, num_workers:{num_workers}')

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
def prepare_image(datadir, featherdir, data_type='train', submission=False, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
    if submission:
        image_df_list = [pd.read_parquet(datadir / f'{data_type}_image_data_{i}.parquet')
                         for i in indices]
    else:
        image_df_list = [
            feather.read_dataframe(featherdir / f'{data_type}_image_data_{i}_{image_height}_{image_width}.feather')
            for i in indices]
        #image_df_list = [pd.read_feather(featherdir / f'{data_type}_image_data_{i}_{image_height}_{image_width}.feather')
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
        self.train = labels is not None
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        i = self.indices[i]
        image = self.images[i]
        # image = cv2.resize(image,(image_size,image_size))
        # image = np.stack((image, image, image), axis=-1)
        #image = cv2.cvtColor(image ,cv2.COLOR_GRAY2RGB)
        image = 1 - (image / 255.0).astype(np.float32)

        # if self.transform:
        #    image = self.transform(image=image)['image']
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        if self.transform:
            aug_image = self.transform(image.copy())
            image = image[:, :, np.newaxis]
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            aug_image = aug_image[:, :, np.newaxis]
            aug_image = np.transpose(aug_image, (2, 0, 1)).astype(np.float32)
            if self.train:
                y = self.labels[i]
                return image, aug_image, y[0], y[1], y[2], y[3]
            else:
                return image, aug_image
        else:
            image = image[:, :, np.newaxis]
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            if self.train:
                y = self.labels[i]
                return image, y[0], y[1], y[2], y[3]
            else:
                return image

# helper --
class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                        int(i * grid_h): int(i * grid_h + grid_h / 2),
                        int(j * grid_w): int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                            int(i * grid_h + grid_h / 2): int(i * grid_h + grid_h),
                            int(j * grid_w + grid_w / 2): int(j * grid_w + grid_w)
                            ] = self.fill_value

                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = AF.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:, :, np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h + h, rand_w:rand_w + w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')

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
class_cols = ['grapheme_root','vowel_diacritic','consonant_diacritic','grapheme_enc']
NUM_CLASS = [train[cc].nunique() for cc in class_cols]
print('NUM_CLASS:', NUM_CLASS)

def ohem_loss(rate, cls_pred, cls_target):
    batch_size = cls_pred.size(0)
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size * rate))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss


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


def shuffled_loss_fn(preds, targets, shuffled_targets, lam, rate=0.7):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # criterion = ohem_loss
    loss = lam * criterion(preds[0], targets[0]) + (1 - lam) * criterion(preds[0], shuffled_targets[0]) \
           + lam * criterion(preds[1], targets[1]) + (1 - lam) * criterion(preds[1], shuffled_targets[1]) \
           + lam * criterion(preds[2], targets[2]) + (1 - lam) * criterion(preds[2], shuffled_targets[2]) \
           + lam * criterion(preds[3], targets[3]) + (1 - lam) * criterion(preds[3], shuffled_targets[3])
    return loss / 4


def macro_recall(pred_y, y, n_grapheme_root=NUM_CLASS[0], n_vowel=NUM_CLASS[1], n_consonant=NUM_CLASS[2],
                 n_grapheme=NUM_CLASS[3]):
    pred_y = torch.split(pred_y, [n_grapheme_root, n_vowel, n_consonant, n_grapheme], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]

    y = y.cpu().numpy()

    pred_g = pd.Series(pred_labels[3])
    pred_g_gr = pred_g.map(gr_map)
    pred_g_v = pred_g.map(v_map)
    pred_g_c = pred_g.map(c_map)

    recall_gr = recall_score(y[:, 0], pred_labels[0], average='macro')
    recall_v = recall_score(y[:, 1], pred_labels[1], average='macro')
    recall_c = recall_score(y[:, 2], pred_labels[2], average='macro')
    recall_g = recall_score(y[:, 3], pred_labels[3], average='macro')
    recall_tot = np.average([recall_gr, recall_v, recall_c], weights=[2, 1, 1])

    recall_g_gr = recall_score(y[:, 0], pred_g_gr, average='macro')
    recall_g_v = recall_score(y[:, 1], pred_g_v, average='macro')
    recall_g_c = recall_score(y[:, 2], pred_g_c, average='macro')
    recall_g_tot = np.average([recall_g_gr, recall_g_v, recall_g_c], weights=[2, 1, 1])

    metric = {}

    metric['recall_gr'] = recall_gr
    metric['recall_v'] = recall_v
    metric['recall_c'] = recall_c
    metric['recall_g'] = recall_g
    metric['recall_tot'] = recall_tot

    metric['recall_g_gr'] = recall_g_gr
    metric['recall_g_v'] = recall_g_v
    metric['recall_g_c'] = recall_g_c
    metric['recall_g_tot'] = recall_g_tot

    return metric


def loss_fn(outputs, targets, rate=0.7):
    o1, o2, o3, o4 = outputs
    t1, t2, t3, t4 = targets

    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    l4 = nn.CrossEntropyLoss()(o4, t4)

    # l1 = ohem_loss(rate, o1, t1)
    # l2 = ohem_loss(rate, o2, t2)
    # l3 = ohem_loss(rate, o3, t3)
    # l4 = ohem_loss(rate, o4, t4)

    return (l1 + l2 + l3 + l4) / 4


def run_train(data_loader, model, optimizer, steps_per_epoch):
    model.train()
    final_loss = 0
    counter = 0
    final_outputs = []
    final_targets = []

    is_plot = True

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
        if regularization_decision < 0.4:  # CUTMIX 0.4
            CUTMIX_ALPHA = 1.0
            image, targets, shuffled_targets, lam = cutmix(original, gr, v, c, g, CUTMIX_ALPHA)
            outputs = model(image)
            loss = shuffled_loss_fn(outputs, targets, shuffled_targets, lam)
        elif regularization_decision < 0.7:  # MIXUP 0.3
            MIXUP_ALPHA = 0.4
            image, targets, shuffled_targets, lam = mixup(original, gr, v, c, g, MIXUP_ALPHA)
            outputs = model(image)
            loss = shuffled_loss_fn(outputs, targets, shuffled_targets, lam)
        else:  # gridmask 0.2, normal aug 0.1
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
    metric = macro_recall(final_outputs, final_targets)
    metric['loss'] = final_loss / counter

    return metric


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
        metric = macro_recall(final_outputs, final_targets)
        metric['loss'] = final_loss / counter

    return metric

val_fold = 0

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

model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=1295, in_channels=1).to(device)

#summary(model, input_size=(3, image_height, image_width))
#model = EfficientNet.from_name('efficientnet-b4')
#print(model)


lr = 1e-4*torch.cuda.device_count()
print(f'initial lr : {lr}')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='max',
                                                       patience=10,
                                                       factor=0.5, verbose=True)

# Initialization
opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

clear_cache()
best_score = -1
histories = []

for i in range(num_epochs):
    start = time.time()
    epoch = i + 1

    print(f'Epoch: {epoch}')

    history = {
        'epoch': epoch,
        'steps_per_epoch': steps_per_epoch,
        'batch_size': batch_size,
    }

    trn_metric = run_train(train_loader, model, optimizer, steps_per_epoch)

    val_metric = run_evaluate(valid_loader, model)

    lr = scheduler.optimizer.param_groups[0]['lr']
    scheduler.step(val_metric['recall_g_tot'])

    if val_metric['recall_g_tot'] > best_score:
        # optimizer.swap_swa_sgd()
        best_score = val_metric['recall_g_tot']
        file_name = modeldir / f'model_{experi_num}_epoch_{epoch:03d}_fold_{val_fold}_recall_{best_score:.4f}.pt'
        torch.save(model.module.state_dict(), file_name)
        print('save max accuracy model: ', best_score)

        # optimizer.swap_swa_sgd()

    # optimizer.update_swa()

    elapsed = time.time() - start

    epoch_len = len(str(num_epochs))
    print_msg = (
            f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
            f'steps_per_epoch: {steps_per_epoch}, ' +
            f'batch_size: {batch_size}, ' +
            f'lr: {lr}, ' +
            f'trn_loss: {trn_metric["loss"]:.5f}, ' +
            f'trn_recall/gr: {trn_metric["recall_gr"]:.5f}, ' +
            f'trn_recall/v: {trn_metric["recall_v"]:.5f}, ' +
            f'trn_recall/c: {trn_metric["recall_c"]:.5f}, ' +
            f'trn_recall/g: {trn_metric["recall_g"]:.5f}, ' +
            f'trn_recall/tot: {trn_metric["recall_tot"]:.5f} ' +
            f'trn_recall/g_gr: {trn_metric["recall_g_gr"]:.5f}, ' +
            f'trn_recall/g_v: {trn_metric["recall_g_v"]:.5f}, ' +
            f'trn_recall/g_c: {trn_metric["recall_g_c"]:.5f}, ' +
            f'trn_recall/g_tot: {trn_metric["recall_g_tot"]:.5f} \n' +
            f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
            f'steps_per_epoch: {steps_per_epoch}, ' +
            f'batch_size: {batch_size}, ' +
            f'lr: {lr}, ' +
            f'val_loss: {val_metric["loss"]:.5f}, ' +
            f'val_recall/gr: {val_metric["recall_gr"]:.5f}, ' +
            f'val_recall/v: {val_metric["recall_v"]:.5f}, ' +
            f'val_recall/c: {val_metric["recall_c"]:.5f}, ' +
            f'val_recall/g: {val_metric["recall_g"]:.5f}, ' +
            f'val_recall/tot: {val_metric["recall_tot"]:.5f}, ' +
            f'val_recall/g_gr: {val_metric["recall_g_gr"]:.5f}, ' +
            f'val_recall/g_v: {val_metric["recall_g_v"]:.5f}, ' +
            f'val_recall/g_c: {val_metric["recall_g_c"]:.5f}, ' +
            f'val_recall/g_tot: {val_metric["recall_g_tot"]:.5f}, \n' +
            f'elasped: {elapsed}'
    )

    print(print_msg)
    print('-' * 100)

    history['lr'] = lr
    trn_metric = {f'trn/{k}': v for k, v in trn_metric.items()}
    history.update(trn_metric)
    val_metric = {f'val/{k}': v for k, v in val_metric.items()}
    history.update(val_metric)
    history['elapsed_time'] = elapsed
    histories.append(history)

    pd.DataFrame(histories).to_csv(logdir / f'log_{experi_num}.csv', index=False)

# optimizer.swap_swa_sgd()
