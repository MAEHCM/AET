# torch
import os
from typing import Optional, Callable, Any

import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from tqdm import trange
from transformers import LayoutLMv3Tokenizer


from Process import Layoutlmv3FeatureExtractor
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS


# constants
# data
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
        #然后拿去检测ocr坐标哇
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

            token[-1]=tokenizer.sep_token_id
            bbox[-1]=[0,0,0,0]
            attention_mask[-1]=1


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

        return(
               path,
               tokens,
               bboxes,
               attention_masks
        )

class InputExample(object):

    def __init__(self, root, tokens, bboxes, attention_masks):

        self.root = root
        self.tokens = tokens
        self.bboxes = bboxes
        self.attention_masks = attention_masks

def read_examples_from_file(data):
    examples=[]

    root=data[0]
    tokens=data[1]
    bboxes=data[2]
    attention_masks=data[3]

    examples.append(
        InputExample(
            root=root,
            tokens=tokens,
            bboxes=bboxes,
            attention_masks=attention_masks
        )
    )
    return examples


ds = image_folder('/mnt//disk2/CORD/train')


for i in trange(len(ds)):
    temp_root=ds[i][0].split('/')[-1]
    temp_root=temp_root[:-4]
    cached_features_file = os.path.join(
        '/mnt/disk2//cord_pretrain',
        "cached_{}".format(
            temp_root,
        ),
    )
    if os.path.exists(cached_features_file):
        continue
    else:
        features = read_examples_from_file(ds[i])
        features = torch.save(features,cached_features_file)






