from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import nibabel as nib
import torch
import pydicom as pyd
import SimpleITK as sitk
import random


class BatchDataset(Dataset):

    def init(self, datasetfile, split, iterations, batch_size, res, seed):
        df = pd.read_csv(datasetfile, index_col=0)
        if type(split) is list:
            selection = np.any([df.split == x for x in split], axis=0)
        else:
            selection = df.split == split

        self.df = df.loc[selection]
        self.df = self.df.reset_index()

        if res is not None:
            if type(res) is list:
                if 'scanner' in self.df.columns:
                    selection = np.any([self.df.scanner == x for x in res], axis=0)
                else:
                    selection = np.any([self.df.Scanner == x for x in res], axis=0)
            else:
                if 'scanner' in self.df.columns:
                    selection = self.df.scanner == res
                else:
                    selection = self.df.Scanner == res

            self.df = self.df.loc[selection]
            self.df = self.df.reset_index()

        if iterations is not None:
            self.df = self.df.sample(iterations * batch_size, replace=True, random_state=seed)
            self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)



class CardiacBatch(BatchDataset):

    def __init__(self, datasetfile, split=['base'], iterations=None, batch_size=None, res=None, seed=None):
        super(BatchDataset, self).__init__()
        self.init(datasetfile, split, iterations, batch_size, res, seed)
        self.outsize = (240, 192)

        print('init cardiac batch with datasetfile', datasetfile)

    def crop_center_or_pad(self, img, cropx, cropy):
        x, y = img.shape

        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)

        if startx < 0:
            outimg = np.zeros(self.outsize)
            startx *= -1
            outimg[startx:self.outsize[0] - startx, :] = img[:, starty:starty + cropy]
            return outimg

        return img[startx:startx + cropx, starty:starty + cropy]

    def load_image(self, elem):
        # print("Loading image from:", elem.slicepath)
        try:
            img = np.load(elem.slicepath)
            mask = np.load(elem.slicepath[:-4] + '_gt.npy')
        except FileNotFoundError:
            # Handle missing file gracefully, e.g., print a message and skip the entry
            print(f"File not found: {elem.slicepath}")
            return None, None

        if img.shape != self.outsize:
            img = self.crop_center_or_pad(img, self.outsize[0], self.outsize[1])
            mask = self.crop_center_or_pad(mask, self.outsize[0], self.outsize[1])

        return img[None, :, :], mask

    def __getitem__(self, index):
        elem = self.df.iloc[index]
        img, mask = self.load_image(elem)
        return torch.as_tensor(img, dtype=torch.float32), torch.as_tensor(mask,
                                                                          dtype=torch.long), elem.scanner, elem.slicepath
