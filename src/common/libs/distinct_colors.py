"""Generate a palette of n distinct colors (useful eg for segmentation)"""

import json
import os
import sys
sys.path += ['..', '.']
from common.extlibs.glasbey import Glasbey


def _gen_color_palette(num):
    """Generate a new color palette"""
    gl = Glasbey()
    pa = gl.generate_palette(size=num)
    return gl.convert_palette_to_rgb(pa)


def get_color_palette(num):
    """Get a color palette with n distinct colors"""
    palette_dpath = os.path.join('..', 'common', 'cfg', 'color_palettes')
    palette_fpath = os.path.join(palette_dpath, "%u.json" % num)
    if not os.path.isfile(palette_fpath):
        palette = _gen_color_palette(num)
        os.makedirs(palette_dpath, exist_ok=True)
        with open(palette_fpath, 'w') as fp:
            json.dump(palette, fp)
    else:
        with open(palette_fpath, 'r') as fp:
            palette = json.load(fp)
    return palette
