from pathlib import Path

import torch
import torch.utils.data
import torchvision
import torch.utils.data as data

import model.detection.datasets.transform as T


class CocoDetection(torchvision.datasets.CocoDetection):

    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)

        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):

        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target} 
        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        """
        Return example 

        Image size : torch.Size([3, 576, 638])

        Target : 
        {'boxes': tensor([[0.6240, 0.2875, 0.1698, 0.3400],
                        [0.2351, 0.5314, 0.4701, 0.9371],
                        [0.4722, 0.6277, 0.2816, 0.7445],
                        [0.7453, 0.5528, 0.5094, 0.8945],
                        [0.6128, 0.6995, 0.1498, 0.0647],
                        [0.4731, 0.4819, 0.1561, 0.0233],
                        [0.9721, 0.2031, 0.0187, 0.1447],
                        [0.9882, 0.1968, 0.0236, 0.1408],
                        [0.9960, 0.6344, 0.0080, 0.4890],
                        [0.1050, 0.2647, 0.2100, 0.4400],
                        [0.9948, 0.4351, 0.0104, 0.0984],
                        [0.9401, 0.1980, 0.0253, 0.1481]]), 
        'labels': tensor([ 1,  1,  1,  1, 51, 48, 50, 50, 79,  1, 51, 50]), 
        'area': tensor([ 21221.3789, 161893.6094,  77055.0859, 167438.7031,   3563.4573,
                        1337.9399,    993.6324,   1218.7225,   1434.2280,  33958.5820,
                        374.9960,   1375.6669]), 
        'iscrowd': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
        'orig_size': tensor([426, 640]), 
        'size': tensor([576, 638])}
        """

        # print("Image shape :", img.shape)
        return img, target


class ConvertCocoPolysToMask(object):

    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):

    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    mode = 'instances'

    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    
    return dataset
