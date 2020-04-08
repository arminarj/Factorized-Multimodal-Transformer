import os
import pickle

import numpy as np
import torch
import torch.utils.data as Data

from Multimodal_dataset import MultimodalSubdata
from consts import global_consts as gc


class PomDataset(Data.Dataset):
    trainset = MultimodalSubdata("train")
    testset = MultimodalSubdata("test")
    validset = MultimodalSubdata("valid")

    def __init__(self, root, cls="train"):
        self.root = root
        self.cls = cls
        if len(PomDataset.trainset.y) != 0 and cls != "train":
            print("Data has been previously loaded, fetching from previous lists.")
        else:
            self.load_data()

        if self.cls == "train":
            self.dataset = PomDataset.trainset
        elif self.cls == "test":
            self.dataset = PomDataset.testset
        elif self.cls == "valid":
            self.dataset = PomDataset.validset

        self.text = self.dataset.text
        self.audio = self.dataset.audio
        self.vision = self.dataset.vision
        self.y = self.dataset.y


    def load_data(self):
        data_path = os.path.join(gc.data_path, 'pom')
        for ds, split_type in [(PomDataset.trainset, 'train'), (PomDataset.validset, 'valid'),
                               (PomDataset.testset, 'test')]:
            ds.text = torch.tensor(pickle.load(open(os.path.join(data_path, 'text_%s.p' % split_type), 'rb'),
                                               encoding='latin1').astype(np.float32)).cpu().detach()
            ds.audio = torch.tensor(pickle.load(open(os.path.join(data_path, 'covarep_%s.p' % split_type), 'rb'),
                                                encoding='latin1').astype(np.float32))
            ds.audio[ds.audio == -np.inf] = 0
            ds.audio = ds.audio.clone().cpu().detach()
            ds.vision = torch.tensor(pickle.load(open(os.path.join(data_path, 'facet_%s.p' % split_type), 'rb'),
                                                 encoding='latin1').astype(np.float32)).cpu().detach()
            ds.y = torch.tensor(pickle.load(open(os.path.join(data_path, 'y_%s.p' % split_type), 'rb'),
                                            encoding='latin1').astype(np.float32)).cpu().detach()
        gc.padding_len = PomDataset.trainset.text.shape[1]
        gc.dim_l = PomDataset.trainset.text.shape[2]
        gc.dim_a = PomDataset.trainset.audio.shape[2]
        gc.dim_v = PomDataset.trainset.vision.shape[2]

    def __getitem__(self, index):
        inputLen = len(self.text[index])
        return self.text[index], self.audio[index], self.vision[index], \
               inputLen, self.y[index].squeeze()

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":
    dataset = PomDataset(gc.data_path)
