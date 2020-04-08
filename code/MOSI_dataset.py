import re
from collections import defaultdict

import torch
import torch.utils.data as Data
import numpy as np
from consts import global_consts as gc
import h5py
import sys

if gc.SDK_PATH is None:
    print("SDK path is not specified! Please specify first in constants/paths.py")
    exit(0)
else:
    print("Added gc.SDK_PATH")
    import os

    print(os.getcwd())
    sys.path.append(gc.SDK_PATH)

import mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI.cmu_mosi_std_folds as std_folds

from mmsdk import mmdatasdk as md

from consts import global_consts as gc

DATASET = md.cmu_mosi

# obtain the train/dev/test splits - these splits are based on video IDs
trainvid = std_folds.standard_train_fold
testvid = std_folds.standard_test_fold
validvid = std_folds.standard_valid_fold


def mid(a):
    return (a[0] + a[1]) / 2.0


class MOSISubdata():
    def __init__(self, name="train"):
        self.name = name
        self.X = np.empty(0)
        self.y = np.empty(0)


class MOSIDataset(Data.Dataset):
    trainset = MOSISubdata("train")
    testset = MOSISubdata("test")
    validset = MOSISubdata("valid")

    def __init__(self, root, cls="train", src="csd", save=False):
        self.root = root
        self.cls = cls
        if len(MOSIDataset.trainset.X) != 0 and cls != "train":
            print("Data has been previously loaded, fetching from previous lists.")
        else:
            self.load_data()

        if self.cls == "train":
            self.dataset = MOSIDataset.trainset
        elif self.cls == "test":
            self.dataset = MOSIDataset.testset
        elif self.cls == "valid":
            self.dataset = MOSIDataset.validset

        self.X = torch.tensor(self.dataset.X, dtype=torch.float32)
        self.y = torch.tensor(self.dataset.y)


    def load_data(self):
        MOSIDataset.trainset.X = h5py.File(os.path.join(gc.data_path, 'X_train.h5'), 'r')['data']
        MOSIDataset.trainset.y = h5py.File(os.path.join(gc.data_path, 'y_train.h5'), 'r')['data']

        MOSIDataset.validset.X = h5py.File(os.path.join(gc.data_path, 'X_valid.h5'), 'r')['data']
        MOSIDataset.validset.y = h5py.File(os.path.join(gc.data_path, 'y_valid.h5'), 'r')['data']

        MOSIDataset.testset.X = h5py.File(os.path.join(gc.data_path, 'X_test.h5'), 'r')['data']
        MOSIDataset.testset.y = h5py.File(os.path.join(gc.data_path, 'y_test.h5'), 'r')['data']

    def __getitem__(self, index):
        inputLen = len(self.X[index])
        return self.X[index, :, :gc.dim_l], \
               self.X[index, :, gc.dim_l: gc.dim_l + gc.dim_a], \
               self.X[index, :, gc.dim_l + gc.dim_a:], \
               inputLen, self.y[index].squeeze()

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":
    dataset = MOSIDataset(gc.data_path, src="csd", save=False)
