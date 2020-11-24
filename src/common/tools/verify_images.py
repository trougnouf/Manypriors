# -*- coding: utf-8 -*-
"""
Typical usage: python verify_images.py imgdirpath [--save_img]
"""

import sys
import os
from tqdm import tqdm
sys.path.append('..')
from common.libs import libimganalysis
from common.libs import utilities
imagedir = sys.argv[1]

def is_valid_img_mtrunner(img_fn):
    return libimganalysis.is_valid_img(os.path.join(imagedir, img_fn),
                                       open_img=True,
                                       save_img='--save_img' in sys.argv, #  --save_img is more thorough
                                       clean=True)
utilities.mt_runner(is_valid_img_mtrunner, os.listdir(imagedir))
