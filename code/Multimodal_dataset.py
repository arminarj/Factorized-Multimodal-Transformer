import os
import pickle

import numpy as np
import torch
import torch.utils.data as Data

from consts import global_consts as gc


class MultimodalSubdata():
    def __init__(self, name="train"):
        self.name = name
        self.text = torch.empty(0)
        self.audio = torch.empty(0)
        self.vision = torch.empty(0)
        self.y = torch.empty(0)


class MultimodalDataset(Data.Dataset):
    trainset = MultimodalSubdata("train")
    testset = MultimodalSubdata("test")
    validset = MultimodalSubdata("valid")

    def __init__(self, root, cls="train"):
        self.root = root
        self.cls = cls
        if len(MultimodalDataset.trainset.y) != 0 and cls != "train":
            print("Data has been previously loaded, fetching from previous lists.")
        else:
            self.load_data()

        if self.cls == "train":
            self.dataset = MultimodalDataset.trainset
        elif self.cls == "test":
            self.dataset = MultimodalDataset.testset
        elif self.cls == "valid":
            self.dataset = MultimodalDataset.validset

        self.text = self.dataset.text
        self.audio = self.dataset.audio
        self.vision = self.dataset.vision
        self.y = self.dataset.y


    def load_data(self):
        dataset_path = os.path.join(gc.data_path, gc.dataset + '.dt')
        dataset = pickle.load(open(dataset_path, 'rb'))

        _vision_1 = 'OpenFace_2.0'
        _vision_2 = 'FACET 4.2'
        _audio_1 = 'COAVAREP' 
        _audio_2 = 'OpenSMILE'
        _text = 'glove_vectors' 
        _labels = 'All Labels'

        gc.padding_len = dataset['test'][_text].shape[1]
        gc.dim_l = dataset['test'][_text].shape[2]
        gc.dim_a = dataset['test'][_audio_1].shape[2]
        gc.dim_v = dataset['test'][_vision_1].shape[2]

        for ds, split_type in [(MultimodalDataset.trainset, 'train'), (MultimodalDataset.validset, 'valid'),
                               (MultimodalDataset.testset, 'test')]:
            ds.text = dataset[split_type][_text].clone().float().cpu().detach()
            ds.audio = dataset[split_type][_audio_1].float()
            ds.audio[self.audio == -float("Inf")] = 0
            ds.audio = ds.audio.clone().cpu().detach()
            ds.vision = dataset[split_type][_vision_1].float().clone().cpu().detach()
            if gc.dataset == 'iemocap':
                ds.y = torch.tensor(dataset[split_type]['labels'].astype(np.long)).cpu().detach()[:,:,1]
            else:
                ds.y = dataset[split_type][_labels].float().cpu().detach()

    def __getitem__(self, index):
        inputLen = len(self.text[index])
        return self.text[index], self.audio[index], self.vision[index], \
               inputLen, self.y[index].squeeze()

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":
    dataset = MultimodalDataset(gc.data_path)
