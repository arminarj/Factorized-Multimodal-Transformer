import os
import pickle

import numpy as np
import torch
import torch.utils.data as Data

from Multimodal_dataset import MultimodalSubdata
from consts import global_consts as gc

text_dim = 300
vision_dim = 47
audio_dim = 74


class MosiNewDataset(Data.Dataset):
    trainset = MultimodalSubdata("train")
    testset = MultimodalSubdata("test")
    validset = MultimodalSubdata("valid")

    def __init__(self, root, cls="train"):
        self.root = root
        self.cls = cls
        if len(MosiNewDataset.trainset.y) != 0 and cls != "train":
            print("Data has been previously loaded, fetching from previous lists.")
        else:
            self.load_data()

        if self.cls == "train":
            self.dataset = MosiNewDataset.trainset
        elif self.cls == "test":
            self.dataset = MosiNewDataset.testset
        elif self.cls == "valid":
            self.dataset = MosiNewDataset.validset

        self.text = self.dataset.text
        self.audio = self.dataset.audio
        self.vision = self.dataset.vision
        self.y = self.dataset.y


    def load_data(self):
        data_path = os.path.join(gc.data_path, gc.dataset)
        for ds, split_type in [(MosiNewDataset.trainset, 'train'), (MosiNewDataset.validset, 'valid'),
                               (MosiNewDataset.testset, 'test')]:
            x = np.load(os.path.join(data_path, 'x_%s.npy' % split_type)).astype(np.float32)
            ds.text = torch.tensor(x[:, :, :text_dim]).cpu().detach()
            ds.vision = torch.tensor(x[:, :, text_dim:text_dim+vision_dim]).cpu().detach()
            ds.audio = torch.tensor(x[:, :, text_dim+vision_dim:])
            ds.audio[ds.audio == -np.inf] = 0
            ds.audio = ds.audio.clone().cpu().detach()
            y = np.load(os.path.join(data_path, 'y_%s.npy' % split_type)).astype(np.float32)
            ds.y = torch.tensor(y).cpu().detach()
        gc.padding_len = MosiNewDataset.trainset.text.shape[1]
        gc.dim_l = MosiNewDataset.trainset.text.shape[2]
        gc.dim_a = MosiNewDataset.trainset.audio.shape[2]
        gc.dim_v = MosiNewDataset.trainset.vision.shape[2]

    def __getitem__(self, index):
        inputLen = len(self.text[index])
        return self.text[index], self.audio[index], self.vision[index], \
               inputLen, self.y[index].squeeze()

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":
    dataset = MosiNewDataset(gc.data_path)
