#from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
# from data_loader.datasets import Dataset
import torch
import sys
sys.path.append('..')
try:
    from siren import dataio
except ModuleNotFoundError:
    print('datasets.py: warning: could not load siren dataset format')
from common.libs import utilities
from common.libs import pt_ops


class Datasets(Dataset):
    def __init__(self, data_dpaths, image_size=256):
        #self.data_dir = data_dir
        if isinstance(data_dpaths, str):
            data_dpaths = [data_dpaths]
        self.image_size = image_size
        self.image_paths = []
        for data_dir in data_dpaths:
            if not os.path.exists(data_dir):
                raise Exception(f"[!] {data_dir} not exitd")
            self.image_paths.extend(sorted(glob(os.path.join(data_dir, "*.*"))))

    def __getitem__(self, item):
        image_ori = self.image_paths[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_paths)

class Datasets_Img_Coords(Datasets):
    def __init__(self, data_dpaths, image_size=256):
        super().__init__(data_dpaths, image_size)
        assert isinstance(image_size, int)  # TODO test dif shape s.a. 768*512
        self.mgrid = dataio.get_mgrid((image_size, image_size))

    def __getitem__(self, item):
        img = super().__getitem__(item)
        return img



def get_loader(train_data_dir, test_data_dir, image_size, batch_size):
    train_dataset = Datasets(train_data_dir, image_size)
    test_dataset = Datasets(test_data_dir, image_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


def get_train_loader(train_data_dir, image_size, batch_size):
    train_dataset = Datasets(train_data_dir, image_size)
    torch.manual_seed(3334)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True)
    return train_dataset, train_loader

class TestDirDataset(Dataset):
    def __init__(self, data_dir, data_dir_2=None, resize=None, verbose=False, crop_to_multiple=None):
        self.data_dir = data_dir
        self.data_dir_2 = data_dir_2
        if resize is not None:
            resize = int(resize)
            raise NotImplementedError('resize')
        self.resize = resize
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        elif os.path.isdir(data_dir):
            self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))
        elif os.path.isfile(data_dir):
            self.image_path = [data_dir]
        if data_dir_2 is not None:
            if not os.path.exists(data_dir_2):
                raise ValueError(f'data_dir_2={data_dir_2} does not exist')
                #raise ValueError(f'{data_dir_2=} does not exist')  # FIXME restore this (req modern python3)
            self.image_path_2 = sorted(glob(os.path.join(data_dir_2, "*.png")))
            if len(self.image_path_2) == 0:
                self.image_path_2 = sorted(glob(os.path.join(data_dir_2, "*.jpg")))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.verbose = verbose
        self.crop_to_multiple = crop_to_multiple

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        if self.verbose:
            print('{}: loading {}'.format(self, image_ori))
        image = Image.open(image_ori).convert('RGB')
        image = self.transform(image)
        ch, h, w = image.shape
        #image = image[:, :h-h%64, :w-w%64] # ensure divisible by 16, actually no longer necessary bc taken care of in preprocessing
        if hasattr(self, 'image_path_2'):
            #print(self.image_path_2)
            #print(item)
            image_ori2 = self.image_path_2[item]
            image2 = Image.open(image_ori2).convert('RGB')
            image2 = self.transform(image2)
            if os.path.exists(image_ori2+'.bpg'):
                size = utilities.filesize(image_ori2+'.bpg')
            else:
                size = utilities.filesize(image_ori2)
            if self.crop_to_multiple is not None:
                return pt_ops.crop_to_multiple(image, self.crop_to_multiple), pt_ops.crop_to_multiple(image2, self.crop_to_multiple), size
            return image, image2, size
        if self.crop_to_multiple is not None:
            return pt_ops.crop_to_multiple(image, self.crop_to_multiple)
        return image

    def __len__(self):
        return len(self.image_path)

TestKodakDataset = TestDirDataset

def get_val_test_loaders(val_dpath, test_dpath):
    if test_dpath is None:
        test_loader = None
    else:
        test_dataset = TestDirDataset(data_dir=test_dpath)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)
    if val_dpath is None:
        val_loader = None
    else:
        val_dataset = TestDirDataset(data_dir=val_dpath)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)
    return val_loader, test_loader