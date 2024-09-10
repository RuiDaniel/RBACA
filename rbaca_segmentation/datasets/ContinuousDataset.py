from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import nibabel as nib
import torch
import pydicom as pyd
import SimpleITK as sitk
import random

class ContinuousDataset(Dataset):
    def init(self, datasetfile, transition_phase_after, order, seed, random=False):
        df = pd.read_csv(datasetfile)
        assert set(['train']).issubset(df.split.unique())
        np.random.seed(seed)

        if random:
            df = df.loc[df.split == 'train']
            df = df.sample(frac=1, random_state=seed)
            self.df = df
        else:
            res_dfs = []
            for r in order:
                res_df = df.loc[(df.scanner == r) & (df.split == 'train')]
                res_df = res_df.sample(frac=1, random_state=seed)
                res_dfs.append(res_df.reset_index(drop=True))

            combds = []  # Initialize the combds list

            new_idx = 0
            for j in range(len(res_dfs) - 1):
                old = res_dfs[j]
                new = res_dfs[j + 1]

                old_end = int((len(old) - new_idx) * transition_phase_after) + new_idx
                combds.append(old.iloc[new_idx:old_end].copy())

                old_idx = old_end
                old_max = len(old) - 1
                new_idx = 0
                i = 0

                while old_idx <= old_max and (i / ((old_max - old_end) * 2) < 1):
                    take_newclass = np.random.binomial(1, min(i / ((old_max - old_end) * 2), 1))
                    if take_newclass:
                        if new_idx < len(new):
                            combds.append(new.iloc[new_idx:new_idx + 1].copy())
                            new_idx += 1
                    else:
                        combds.append(old.iloc[old_idx:old_idx + 1].copy())
                        old_idx += 1
                    i += 1

                combds.append(old.iloc[old_idx:].copy())

            combds.append(new.iloc[new_idx:].copy())
            self.df = pd.concat(combds, ignore_index=True)

    def __len__(self):
        return len(self.df)

class CardiacContinuous(ContinuousDataset):

    def __init__(self, datasetfile, transition_phase_after=.8, order=['Siemens', 'GE', 'Philips', 'Canon'], seed=None, random=False):
        super(ContinuousDataset, self).__init__()
        self.init(datasetfile, transition_phase_after, order, seed, random)

        self.outsize = (240, 192)

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
        img = np.load(elem.slicepath)
        mask = np.load(elem.slicepath[:-4] + '_gt.npy')

        if img.shape != self.outsize:
           img = self.crop_center_or_pad(img, self.outsize[0], self.outsize[1])
           mask = self.crop_center_or_pad(mask, self.outsize[0], self.outsize[1])

        return img[None, :, :], mask

    def __getitem__(self, index):
        elem = self.df.iloc[index]
        img, mask = self.load_image(elem)
        return torch.as_tensor(img, dtype=torch.float32), torch.as_tensor(mask,
                                                                          dtype=torch.long), elem.scanner, elem.slicepath