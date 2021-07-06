import os
import json
from PIL import Image
from glob import glob
# Import PyTorch
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, data_path, spm_model, transform=None, phase='train', min_len=4, max_len=300):

        self.phase = phase.lower()
        self.spm_model = spm_model

        # Training dataset
        if self.phase == 'train':

            self.data = list()

            with open(os.path.join(data_path, 'annotations/captions_train2017.json'), 'r') as f:
                c_data = json.load(f)['annotations']

            for c in c_data:

                # Image pre-processing
                img_path_id = str(c['image_id']).zfill(12)
                img_path = os.path.join(data_path, 'train2017/' + img_path_id + '.jpg')

                # Caption pre-processing
                caption_encode = [self.spm_model.bos_id()] + \
                                  self.spm_model.EncodeAsIds(c['caption']) + \
                                 [self.spm_model.eos_id()]
                # zero pad 
                if min_len <= len(caption_encode) <= max_len:
                    caption_ = torch.zeros(max_len, dtype=torch.long)
                    caption_[:len(caption_encode)] = torch.tensor(caption_encode, dtype=torch.long)

                    # Append
                    self.data.append({
                        'path': img_path,
                        'caption': caption_
                    })

        # Validation dataset
        elif self.phase == 'valid':

            self.data = list()

            with open(os.path.join(data_path, 'annotations/captions_val2017.json'), 'r') as f:
                c_data = json.load(f)['annotations']

            for c in c_data:

                # Image pre-processing
                img_path_id = str(c['image_id']).zfill(12)
                img_path = os.path.join(data_path, 'val2017/' + img_path_id + '.jpg')

                # Caption pre-processing
                caption_encode = [self.spm_model.bos_id()] + \
                                  self.spm_model.EncodeAsIds(c['caption']) + \
                                 [self.spm_model.eos_id()]
                                 
                if min_len <= len(caption_encode) <= max_len:
                    caption_ = torch.zeros(max_len, dtype=torch.long)
                    caption_[:len(caption_encode)] = torch.tensor(caption_encode, dtype=torch.long)

                    # Append
                    self.data.append({
                        'path': img_path,
                        'caption': caption_
                    })

        # Testing dataset
        elif self.phase == 'test':
            self.data = list()
            for path in glob(os.path.join(data_path, 'test2017/*.jpg')):
                self.data.append({
                    'path': path,
                    'id': path.split('/')[-1][:-4]
                })

        # Raise error when phase does not exist
        else:
            raise Exception("phase value must be in ['train', 'valid', 'test']")

        self.data = tuple(self.data)
        self.num_data = len(self.data)
        self.transform = transform

    def __getitem__(self, index):

        # Open image
        image = Image.open(self.data[index]['path']).convert('RGB')

        # Image Augmentation
        if self.transform is not None:
            image = self.transform(image)
        else:
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)

        # Return Value
        if self.phase == 'test':
            img_id = self.data[index]['id']
            return image, img_id
        else:
            caption = self.data[index]['caption']
            return image, caption

    def __len__(self):
        return self.num_data