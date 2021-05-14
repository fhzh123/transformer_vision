import os
import pandas as pd
from PIL import Image
# Import PyTorch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None, phase='train'):
        self.phase = phase.lower()
        self.data_path = data_path

        # data_path = '/HDD/dataset/imagenet/ILSVRC/*'
        if self.phase == 'train':
            self.data = pd.read_csv(
                os.path.join(data_path, 'ImageSets/CLS-LOC/train_cls.txt'), 
                sep=' ', names=['path', 'index']
                )
            self.data['path'] = os.path.join(data_path, 'Data/CLS-LOC/train/') + \
                                self.data['path'] + '.JPEG'
            self.data['label_code'] = self.data['path'].apply(lambda x: x.split('/')[-2])
            label_map = pd.read_csv(
                os.path.join(data_path, 'ImageSets/CLS-LOC/map_clsloc.txt'), 
                sep=' ', names=['code', 'index', 'names']
                )
            self.label = self.data['label_code'].map(label_map.set_index('code')['index']-1).tolist()
        elif self.phase == 'valid':
            self.data = pd.read_csv(
                os.path.join(data_path, 'ImageSets/CLS-LOC/val.txt'), 
                sep=' ', names=['path', 'index']
                )
            self.data['path'] = os.path.join(data_path, 'Data/CLS-LOC/val/') + \
                                self.data['path'] + '.JPEG'
            self.label_dat = pd.read_csv(
                os.path.join(data_path, 'ImageSets/CLS-LOC/LOC_val_solution.csv'), 
                )
            self.label_dat = self.label_dat.sort_values(by='ImageId')
            self.label_dat['label_code'] = self.label_dat['PredictionString'].apply(lambda x: x.split()[0])
            label_map = pd.read_csv(
                os.path.join(data_path, 'ImageSets/CLS-LOC/map_clsloc.txt'), 
                sep=' ', names=['code', 'index', 'names']
                )
            self.label = self.label_dat['label_code'].map(label_map.set_index('code')['index']-1).tolist()
        elif self.phase == 'test':
            self.data = pd.read_csv(
                os.path.join(data_path, 'ImageSets/CLS-LOC/test.txt'), 
                sep=' ', names=['path', 'index']
                )
        else:
            raise Exception("phase value must be in ['train', 'valid', 'test']")

        self.num_data = len(self.data)
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.data['path'][index]).convert('RGB')
        # Image Augmentation
        if self.transform is not None:
            image = self.transform(image)
        else:
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        # Return Value
        if self.phase == 'test':
            img_id = index+1
            return image, img_id
        else:
            label = self.label[index]
            return image, label

    def __len__(self):
        return self.num_data