import os
import random
import pandas as pd
from PIL import Image
from einops import rearrange
# Import PyTorch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None, phase='train',
                 image_drop=False, img_size=224, patch_size=16, drop_ratio=0.1):
        self.phase = phase.lower()
        self.image_drop = image_drop
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

        # Hyper-parameter setting
        self.num_data = len(self.data)
        self.transform = transform
        self.patch_size = patch_size
        self.img_size = img_size
        self.drop_ratio = drop_ratio

    def __getitem__(self, index):
        image = Image.open(self.data['path'][index]).convert('RGB')
        # Image Augmentation
        if self.transform is not None:
            image = self.transform(image)
        else:
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        # Image patch random drop
        if self.img_drop:
            patch_image = self.to_tensor(image).unsqueeze(0)
            patch_image = rearrange(patch_image, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', 
                                s1=self.patch_size, s2=self.patch_size)
            patch_size = patch_image.size(1)
            drop_patch = random.sample(list(range(patch_size)),
                                       int(patch_size * self.drop_ratio))
            for d_ix in drop_patch:
                patch_image[d_ix] = 0
        # Return Value
        if self.phase == 'test':
            img_id = index+1
            return image, img_id
        else:
            label = self.label[index]
            return original_image, patch_image, label

    def __len__(self):
        return self.num_data