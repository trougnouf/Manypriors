# -*- coding: utf-8 -*-

import subprocess
from typing import Union
from PIL import Image
import unittest
import os
from typing import Optional
import tensorflow as tf

import sys
sys.path.append('..')
try:
    from common.libs import tf_helpers
except ModuleNotFoundError as e:
    print('libimganalysis: warning: {}'.format(e))

VALID_IMG_EXT = ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'ppm', 'j2k', 'webp']

def ipx_psnr(img1path: str, img2path: str):
    """
    intoPIX psnr_compare implementation
    multithread-safe
    """
    args = ['psnr_compare', img1path, img2path]
    res = subprocess.run(args, capture_output=True)
    try:
        return min(float(res.stdout.decode().split('Psnr Global:\t')[1].split(' dB')[0]), 100)
    except IndexError as e:
        raise ValueError('ipx_psnr: unable to get psnr for %s, %s: %s' % (img1path, img2path, e))

def tf_psnr(img1: Union[str, tf.Tensor], img2: Union[str, tf.Tensor]):
    """
    Tensorflow PSNR implementation
    img1 and img2 can be paths or tf tensors.
    not multithread-safe"""
    img1 = tf_helpers.tf_open_img(img1)
    img2 = tf_helpers.tf_open_img(img2)
    return max(100, float(tf.image.psnr(img1, img2, 255)))

def tf_msssim(img1: Union[str, tf.Tensor], img2: Union[str, tf.Tensor]):
    """
    Tensorflow MSSSIM implementation
    img1 and img2 can be paths or tf tensors.
    not multithread-safe"""
    #with tf.device("/cpu:0"):

    img1 = tf_helpers.tf_open_img(img1)
    img2 = tf_helpers.tf_open_img(img2)
    return float(tf.image.ssim_multiscale(img1, img2, 255))

def pil_get_resolution(imgpath):
    return Image.open(imgpath).size

def tf_psnr_cpu(img1: Union[str, tf.Tensor], img2: Union[str, tf.Tensor]):
    """not multithread-safe"""
    with tf.device("/cpu:0"):
        return tf_psnr(img1, img2)

def tf_msssim_cpu(img1: Union[str, tf.Tensor], img2: Union[str, tf.Tensor]):
    """not multithread-safe"""
    with tf.device("/cpu:0"):
        return tf_msssim(img1, img2)

psnr = tf_psnr
msssim = tf_msssim

def is_valid_img(img_fpath, open_img=False, save_img=False, clean=False):
    Image.MAX_IMAGE_PIXELS = 15000**2
    ext_is_valid = img_fpath.split('.')[-1].lower() in VALID_IMG_EXT
    if not open_img and not save_img:
        return ext_is_valid
    try:
        img = Image.open(img_fpath)
        if save_img:
            img = img.resize((128,128))
            img.save('tmp.png')
        else:
            img.verify()
        return True
    except OSError as e:
        print(e)
        if clean:
            os.remove(img_fpath)
            print('rm {}'.format(img_fpath))
        return False
    except Image.DecompressionBombError as e:
        print(e)
        if clean:
            os.remove(img_fpath)
            print('rm {}'.format(img_fpath))
        return False

def tf_compare(img1, img2, bitstream: Optional[str] = None, ignore_border: int = 4):
    '''img1 and img2 can be fpath strings or tf tensors. bitstream is an optional fpath'''
    res = dict()
    if isinstance(img1, str):
        img1 = tf_helpers.tff_open_img(img1)
    if isinstance(img2, str):
        img2 = tf_helpers.tff_open_img(img2)
    npixels = img1.shape[0]*img1.shape[1]
    assert  img1.shape == img2.shape
    if ignore_border > 0:
        img1 = img1[ignore_border:-ignore_border, ignore_border:-ignore_border, :]
        img2 = img2[ignore_border:-ignore_border, ignore_border:-ignore_border, :]
    if bitstream is not None:
        res['bytes'] = os.path.getsize(bitstream)
        res['bpp'] = (res['bytes']*8)/npixels
    res['psnr'] = float(tf.image.psnr(img1, img2, 1))
    res['msssim'] = float(tf.image.ssim_multiscale(img1, img2, 1))
    res['ssim'] = float(tf.image.ssim(img1, img2, 1))
    res['mse'] = float(tf.reduce_mean(tf.keras.losses.MeanSquaredError()(img1, img2)))
    return res

class Test_libimganalysis(unittest.TestCase):
    """
    [common]$ python -m unittest discover libs/ -p *.py
    """

    DSROOTPATH = os.path.join('..', '..', 'datasets')  # running from common/
    TESTIMG = os.path.join(
            "FeaturedPictures",
            "Isaac Lake during golden hour, moments before a storm (DSCF2631).jpg")
    def setUp(self):

        from common.libs.libimgops import bpg_encdec
        self.inimgpath = os.path.join(self.DSROOTPATH, self.TESTIMG)
        self.tmpimgpath = "test_libimganalysis.png"
        bpg_encdec(self.inimgpath, self.tmpimgpath, qp=29)
    def tearDown(self):
        os.remove(self.tmpimgpath)
    def test_tf_msssim(self):
        with tf.device("/cpu:0"):
            score = tf_msssim(self.inimgpath, self.tmpimgpath)
            print('msssim score: {}'.format(score))
        self.assertGreater(score, 0.5)
    def test_tf_psnr(self):
        with tf.device("/cpu:0"):
            score = tf_psnr(self.inimgpath, self.tmpimgpath)
        self.assertGreater(score, 10)

if __name__ == '__main__':
    unittest.main()