import os
import cv2
import numpy as np
import pandas as pd
# from PIL import Image

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None, phase='train'):
        if not isTest:
            self.isTest = False
            self.data_path = data_path

            # data_path = '/HDD/dataset/imagenet/ILSVRC/*
            if phase.lower() == 'train':
                self.data = pd.read_csv(
                    os.path.join(data_path, 'ImageSets/CLS-LOC/train_cls.txt'), 
                    sep=' ', names=['path', 'index']
                    )
                self.label_map = pd.read_csv(
                    os.path.join(data_path, 'ImageSets/CLS-LOC/map_clsloc.txt'), 
                    sep=' ', names=['code', 'index', 'names']
                    )
            elif phase.lower() == 'valid':
                self.data = pd.read_csv(
                    os.path.join(data_path, 'ImageSets/CLS-LOC/val.txt'), 
                    sep=' ', names=['path', 'index']
                    )
            elif phase.lower() == 'test':
                self.data = pd.read_csv(
                    os.path.join(data_path, 'ImageSets/CLS-LOC/test.txt'), 
                    sep=' ', names=['path', 'index']
                    )
            else:
                raise Exception("phase value must be in ['train', 'valid', 'test']")

        self.num_data = len(self.data)
        self.transform = transform

    def __getitem__(self, index):
        image = cv2.imread(self.img_path[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Image Augmentation
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # Return Value
        if not self.isTest:
            label = self.label[index]
            return torch.tensor(image, dtype=torch.float), label
        else:
            img_id = self.img_id[index]
            return torch.tensor(image, dtype=torch.float), img_id

    def __len__(self):
        return self.num_data