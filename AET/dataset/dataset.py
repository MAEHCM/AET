from typing import Optional, Callable, Any

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from transformers import LayoutLMv3Tokenizer

from Process import Layoutlmv3FeatureExtractor
from randaugment import RandomAugment
from utils import GaussianBlur

tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    img = cv2.imread(path, cv2.IMREAD_LOAD_GDAL)
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return img.convert("RGB")



class image_folder(DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
        self.loader=loader
        self.tokenizer=tokenizer
        self.max_seq_len = 512


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path,target=self.imgs[index]

        sample = self.loader(path)
        #PIL image
        feature_extractor = Layoutlmv3FeatureExtractor()
        features=feature_extractor(sample,return_tensors='pt')

        encoded_inputs=self.tokenizer(
            text=features['words'],
            boxes=features['boxes'],
        )
        token=encoded_inputs['input_ids'][0]
        bbox=encoded_inputs['bbox'][0]
        attention_mask=encoded_inputs['attention_mask'][0]

        max_seq_len=self.max_seq_len

        if len(token) > max_seq_len:
            token = token[: max_seq_len]
            bbox = bbox[: max_seq_len]
            attention_mask=attention_mask[:max_seq_len]

        #padding on right
        padding_length=max_seq_len-len(token)

        token += [1] * padding_length

        for i in range(padding_length):
            bbox.append([0,0,0,0])

        attention_mask += [0] * padding_length

        assert len(token) == max_seq_len
        assert len(bbox) == max_seq_len
        assert len(attention_mask) == max_seq_len

        tokens = torch.tensor(token, dtype=torch.long)
        bboxes = torch.tensor(bbox, dtype=torch.long)
        attention_masks = torch.tensor(attention_mask, dtype=torch.long)

        images=features.pop('pixel_values').squeeze(0)
        visual_images=features.pop('visual_token_transform').squeeze(0)
        mask_position_generator=features.pop('mask_position_generator').squeeze(0)

        return(
               images,
               visual_images,
               mask_position_generator,
               tokens,
               bboxes,
               attention_masks
        )


def create_dataset(dataset, config):
    'Pretrain config'
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # jinyu: add augmentation
    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    return dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = True
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders