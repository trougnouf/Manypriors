PyTorch reference implementation of
"End-to-end optimized image compression with competition of prior distributions"
by Trougnouf / Benoit Brummer ( https://github.com/trougnouf/Manypriors )

Forked from PyTorch implementation of
"Variational image compression with a scale hyperprior"
by Jiaheng Liu ( https://github.com/liujiaheng/compression )

This code is experimental.

## Requirements

TODO torchac should be switched to the standalone release on https://github.com/fab-jul/torchac (which was not yet released at the time of writing this code)

### Arch
```
pacaur -S python-tqdm python-pytorch-torchac python-configargparse python-yaml python-ptflops python-colorspacious python-pypng python-pytorch-piqa-git
```

### Ubuntu / Slurm cluster / misc:

```
TMPDIR=tmp pip3 install --user torch==1.7.0+cu92 torchvision==0.8.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html
TMPDIR=tmp pip3 install --user tqdm matplotlib tensorboardX scipy scikit-image scikit-video ConfigArgParse pyyaml h5py ptflops colorspacious pypng piqa

```
torchac must be compiled and installed per https://github.com/trougnouf/L3C-PyTorch/tree/master/src/torchac
```
torchac $ COMPILE_CUDA=auto python3 setup.py build
torchac $ python3 setup.py install --optimize=1 --skip-build
```
or (untested)
```
torchac $ pip install .
```

Once Ubuntu updates PyTorch then tensorboardX won't be required

## Dataset gathering
Copy the kodak dataset into datasets/test/kodak

```
cd ../common
python tools/wikidownloader.py --category "Category:Featured pictures on Wikimedia Commons"
python tools/wikidownloader.py --category "Category:Formerly featured pictures on Wikimedia Commons"
python tools/wikidownloader.py --category "Category:Photographs taken on Ektachrome and Elite Chrome film"
mv "../../datasets/Category:Featured pictures on Wikimedia Commons" ../../datasets/FeaturedPictures
mv "../../datasets/Category:Formerly featured pictures on Wikimedia Commons" ../../datasets/Formerly_featured_pictures_on_Wikimedia_Commons
mv "../../datasets/Category:Photographs taken on Ektachrome and Elite Chrome film" ../../datasets/Photographs_taken_on_Ektachrome_and_Elite_Chrome_film
python tools/verify_images.py ../../datasets/FeaturedPictures/
python tools/verify_images.py ../../datasets/Formerly_featured_pictures_on_Wikimedia_Commons/
python tools/verify_images.py ../../datasets/Photographs_taken_on_Ektachrome_and_Elite_Chrome_film/

# TODO make a list of train/test img automatically s.t. images don't have to be copied over the network
```
Crop images to 1024*1024. from src/common: (in python)

```
import os
from libs import libdsops
for ads in ['Formerly_featured_pictures_on_Wikimedia_Commons', 'Photographs_taken_on_Ektachrome_and_Elite_Chrome_film', 'FeaturedPictures']:
    libdsops.split_traintest(ads)
    libdsops.crop_ds_dpath(ads, 1024, root_ds_dpath=os.path.join(libdsops.ROOT_DS_DPATH, 'train'), num_threads=os.cpu_count()//2)

#verify crops
python3 tools/verify_images.py ../../datasets/train/resized/1024/FeaturedPictures/
python3 tools/verify_images.py ../../datasets/train/resized/1024/Formerly_featured_pictures_on_Wikimedia_Commons/
python3 tools/verify_images.py ../../datasets/train/resized/1024/Photographs_taken_on_Ektachrome_and_Elite_Chrome_film/
# use the --save_img flag at the end of verify_images.py commands if training fails after the simple verification
```
Move a small subset of the training cropped images to a matching test directory and use it as args.val_dpath

JPEG/BPG compression of the Commons Test Images is done with common/tools/bpg_jpeg_compress_commons.py and comp/tools/bpg_jpeg_test_commons.py

## Loading

Loading a model: provide all necessary (non-default) parameters s.a. arch, num_distributions, etc.
Saved yaml can be used iff the ConfigArgParse patch from https://github.com/trougnouf/ConfigArgParse is applied,
otherwise unset values are overwritten with the "None" string.

## Training

Train a base model (given arch and num_distributions) for 6M steps at train_lambda=4096, fine-tune for 4M steps with lower train_lambda and/or msssim lossf
Set arch to Manypriors for this work, use num_distributions 1 for Balle2017, or set arch to Balle2018PTTFExp for Balle2018 (hyperprior)
egrun:

```
python train.py --num_distributions 64 --arch ManyPriors --train_lambda 4096 --expname mse_4096_manypriors_64_CLI
# and/or
python train.py --config configs/mse_4096_manypriors_64pr.yaml
# and/or
python train.py --config configs/mse_2048_manypriors_64pr.yaml --pretrain mse_4096_manypriors_64pr --reset_lr --reset_global_step # --reset_optimizer
# and/or
python train.py --config configs/mse_4096_hyperprior.yaml
```

--passthrough_ae is now activated by default. It was not used in the paper, but should result in better rate-distortion. To turn it off, change config/defaults.yaml or use --no_passthrough_ae
## Tests

egruns:
Test complexity:

```
python tests.py --complexity --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64
```
Test timing:

```
python tests.py --timing "../../datasets/test/Commons_Test_Photographs" --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64
```
Segment the images in commons_test_dpath by distribution index:

```
python tests.py --segmentation --commons_test_dpath "../../datasets/test/Commons_Test_Photographs" --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64
```
Visualize cumulative distribution functions:

```
python tests.py --plot --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64
```
Test on kodak images:

```
python tests.py --encdec_kodak --test_dpath "../../datasets/test/kodak/" --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64
```
Test on commons images (larger, uses CPU):

```
python tests.py --encdec_commons --test_commons_dpath "../../datasets/test/Commons_Test_Photographs/" --pretrain checkpoints/mse_4096_manypriors_64pr/saved_models/checkpoint.pth --arch ManyPriors --num_distributions 64
```
Encode an image:

```
python tests.py --encode "../../datasets/test/Commons_Test_Photographs/Garden_snail_moving_down_the_Vennbahn_in_disputed_territory_(DSCF5879).png" --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64 --device -1
```
Decode that image:

```
python tests.py --decode "checkpoints/mse_4096_manypriors_64pr/encoded/Garden_snail_moving_down_the_Vennbahn_in_disputed_territory_(DSCF5879).png" --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64 --device -1
```