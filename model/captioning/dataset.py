import os
import json
from PIL import Image
from glob import glob
# Import PyTorch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, data_path, spm_model, transform=None, phase='train'):

        # Pre-setting
        self.phase = phase.lower()
        self.spm_model = spm_model

        # Training dataset
        if self.phase == 'train':

            with open(os.path.join(data_path, 'annotations/captions_train2017.json'), 'r') as f:
                self.data = json.load(f)['annotations']

            for c in self.data:
                img_path_id = str(c['image_id']).zfill(12)
                c['path'] = os.path.join(data_path, 'train2017/' + img_path_id + '.jpg')

        # Validation dataset
        elif self.phase == 'valid':

            with open(os.path.join(data_path, 'annotations/captions_valid2017.json'), 'r') as f:
                self.data = json.load(f)['annotations']

            for c in self.data:
                img_path_id = str(c['image_id']).zfill(12)
                c['path'] = os.path.join(data_path, 'valid2017/' + img_path_id + '.jpg')

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

        self.num_data = len(self.data)
        self.transform = transform

    def __getitem__(self, index):

        # Open image & caption
        image = Image.open(self.data[index]['path']).convert('RGB')
        caption = self.data[index]['caption']

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
            caption = [self.spm_model.bos_id] + \
                self.spm_model.EncodeAsIds(caption) + [self.spm_model.eos_id]
            return image, caption

    def __len__(self):
        return self.num_data