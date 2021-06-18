import torch
import cv2
from PIL import Image
import torchvision
import numpy as np
import os
import math
import sys
sys.path.append('..')
from common.libs import np_imgops
from common.libs import pt_losses
from common.libs import utilities

TMPDIR = 'tmp'
os.makedirs(TMPDIR, exist_ok=True)

def fpath_to_tensor(img_fpath, device=torch.device(type='cpu'), batch=False):
    #totensor = torchvision.transforms.ToTensor()
    #pilimg = Image.open(imgpath).convert('RGB')
    #return totensor(pilimg)  # replaced w/ opencv to handle >8bits
    tensor = torch.tensor(np_imgops.img_path_to_np_flt(img_fpath), device=device)
    if batch:
        tensor = tensor.unsqueeze(0)
    return tensor

def tensor_to_imgfile(tensor, path):
    if tensor.dtype == torch.float32:
        if path[-4:].lower() in ['.jpg', 'jpeg']:  # 8-bit
            return torchvision.utils.save_image(tensor.clip(0,1), path)
        elif path[-4:].lower() in ['.png', '.tif', 'tiff']:  # 16-bit?
            if math.floor(tensor.max()) > 1 and tensor.max() <= 255: # 8 bit expressed as 0-255. should have been uint8 but handle it here anyway.
                print('tensor_to_imgfile: warning: float tensor interpreted as uint8')
                nptensor = tensor.round().cpu().numpy().astype(np.uint8).transpose(1,2,0)
            else:  # 16 bit
                nptensor = (tensor.clip(0,1)*65535).round().cpu().numpy().astype(np.uint16).transpose(1,2,0)
            nptensor = cv2.cvtColor(nptensor, cv2.COLOR_RGB2BGR)
            outflags = None
            if path.endswith('tif') or path.endswith('tiff'):
                outflags = ((cv2.IMWRITE_TIFF_COMPRESSION, 34925))  # lzma2
            cv2.imwrite(path, nptensor, outflags)
        else:
            raise NotImplementedError(f'Extension in {path}')
    elif tensor.dtype == torch.uint8:
        tensor = tensor.permute(1, 2, 0).to(torch.uint8).numpy()
        pilimg = Image.fromarray(tensor)
        pilimg.save(path)
    else:
        raise NotImplementedError(tensor.dtype)
    
def get_losses(img1_fpath, img2_fpath):
    img1 = fpath_to_tensor(img1_fpath).unsqueeze(0)
    img2 = fpath_to_tensor(img2_fpath).unsqueeze(0)
    assert img1.shape == img2.shape, f'{img1.shape=}, {img2.shape=}'
    res = dict()
    res['mse'] = torch.nn.functional.mse_loss(img1, img2).item()
    res['ssim'] = pt_losses.SSIM_loss()(img1, img2).item()
    res['msssim'] = pt_losses.MS_SSIM_loss()(img1, img2).item()
    return res

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
    
torch_cuda_synchronize = torch.cuda.synchronize if torch.cuda.is_available() else utilities.noop

