# -*- coding: utf-8 -*-

import torchvision
import os
from PIL import Image
import sys
import torch
import math
sys.path.append('..')
from common.libs import utilities

TMPDIR = 'tmp'
os.makedirs(TMPDIR, exist_ok=True)

def fpath_to_tensor(imgpath):
    totensor = torchvision.transforms.ToTensor()
    pilimg = Image.open(imgpath).convert('RGB')
    return totensor(pilimg)

def to_smallest_type(tensor, integers=False):
    '''
    Convert a tensor to the smallest dtype with no loss
    '''
    minval = tensor.min()
    maxval = tensor.max()
    if integers:
        if maxval <= 255 and minval >= 0:
            tensor = tensor.byte()
        elif maxval <= 32767 and minval >= 0:
            tensor = tensor.short()
        else:
            raise NotImplementedError('to_smallest_type: min={}, max={}'.format(
                    minval, maxval))
    else:
        raise NotImplementedError('to_smallest_type with integers=False')
    return tensor

def bits_per_value(tensor):
    '''
    returns the maximum of bits needed to encode one value from given pt/np tensor (with no compression)
    '''
    minval = tensor.min()
    maxval = tensor.max()
    if minval >= 0 and maxval <= 0:
        return 0
    elif minval >= 0:
        return math.floor(math.log2(maxval)+1)
    # if minval >= 0 and maxval <= 1:
    #     return 1
    # elif minval >= 0 and maxval <= 3:
    #     return 2
    # elif minval >= 0 and maxval <= 7:
    #     return 3
    # elif minval >= 0 and maxval <= 15:
    #     return 4
    # elif minval >= 0 and maxval <= 31:
    #     return 5
    # elif minval >= 0 and maxval <= 63:
    #     return 6
    # elif minval >= 0 and maxval <= 127:
    #     return 7
    # elif minval >= 0 and maxval <= 255:
    #     return 8
    # elif minval >= 0 and maxval <= 511:
    #     return 9
    else:
        raise NotImplementedError('bits_per_value w/ min={}, max={}'.format(minval, maxval))


def get_num_bits(tensor, integers=False, compression='lzma', try_png=True):
    '''Compress a tensor and get the number of bits used to do so (smallest
    with or without compression)
    TODO compression=None'''
    if tensor.min() == 0 and tensor.max() == 0:
        return 0
    tensor = to_smallest_type(tensor, integers=integers)
    ext = 'tar.xz' if compression == 'lzma' else 'bin'
    tmp_fpath = os.path.join(TMPDIR, str(os.getpid())+'.'+ext)
    tensor = tensor.numpy()
    tensor.tofile(tmp_fpath)
    num_bits = (os.stat(tmp_fpath).st_size)*8
    utilities.compress_lzma(tmp_fpath, tmp_fpath)
    num_bits = min(num_bits, (os.stat(tmp_fpath).st_size)*8)
    png_success = try_png and utilities.compress_png(tensor, outfpath=tmp_fpath+'.png')
    if png_success:
        num_bits = min(num_bits, (os.stat(tmp_fpath+'.png').st_size)*8)
    if integers:
        num_bits = min(num_bits, tensor.size*bits_per_value(tensor))
    return num_bits

def get_device(device_n=None):
    """get device given index (-1 = CPU)"""
    if isinstance(device_n, torch.device):
        return device_n
    elif isinstance(device_n, str):
        if device_n == 'cpu':
            return torch.device('cpu')
        device_n = int(device_n)
    if device_n is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print('get_device: cuda not available; defaulting to cpu')
            return torch.device("cpu")
    elif torch.cuda.is_available() and device_n >= 0:
        return torch.device("cuda:%i" % device_n)
    elif device_n >= 0:
        print('get_device: cuda not available')
    return torch.device('cpu')

def tensor_to_imgfile(tensor, path):
    if tensor.dtype == torch.float32:
        return torchvision.utils.save_image(tensor, path)
    tensor = tensor.permute(1, 2, 0).to(torch.uint8).numpy()
    pilimg = Image.fromarray(tensor)
    pilimg.save(path)

torch_cuda_synchronize = torch.cuda.synchronize if torch.cuda.is_available() else utilities.noop
