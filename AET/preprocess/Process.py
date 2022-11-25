from typing import List,Optional,Union
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from transformers.feature_extraction_utils import BatchFeature,FeatureExtractionMixin
from transformers.image_utils import IMAGENET_STANDARD_MEAN,IMAGENET_STANDARD_STD,ImageFeatureExtractionMixin,is_torch_tensor
from transformers.utils import TensorType,requires_backends

import pytesseract


ImageInput = Union[
    Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]  # noqa
]

#boxes归一化到[0,1000]

def normalize_box(box,width,height):
    return [
        int(1000*(box[0]/width)),
        int(1000*(box[1]/height)),
        int(1000*(box[2]/width)),
        int(1000*(box[3]/height)),
    ]

def apply_tesseract(image:Image,lang:Optional[str]):
    data=pytesseract.image_to_data(image,lang=lang,output_type='dict')
    words,left,top,width,height=data['text'],data['left'],data['top'],data['width'],data['height']


    #过滤掉空字符+它的bbox
    irrelevant_indices=[idx for idx , word in enumerate(words) if not word.strip()]

    words=[word for idx,word in enumerate(words) if idx not in irrelevant_indices]
    left=[coord for idx,coord in enumerate(left) if idx not in irrelevant_indices]
    top=[coord for idx,coord in enumerate(top) if idx not in irrelevant_indices]
    width=[coord for idx,coord in enumerate(width) if idx not in irrelevant_indices]
    height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]

    #[左上x,y  右下x,y]
    actual_boxes=[]
    for x,y,w,h in zip(left,top,width,height):
        actual_box=[x,y,x+w,y+h]
        actual_boxes.append(actual_box)

    image_width,image_height=image.size

    #归一化boxes
    normalized_boxes=[]
    for box in actual_boxes:
        normalized_boxes.append(normalize_box(box,image_width,image_height))

    return words,normalized_boxes


class Layoutlmv3FeatureExtractor(FeatureExtractionMixin,ImageFeatureExtractionMixin):
    model_input_names=['pixes_values']

    def __init__(self,
                 do_resize=True,
                 size=224,
                 resample=Image.BILINEAR,
                 do_normalize=True,
                 image_mean=None,
                 image_std=None,
                 apply_ocr=True,
                 ocr_lang=None,
                 **kwargs):
        super(Layoutlmv3FeatureExtractor, self).__init__()
        self.do_resize=do_resize
        self.size=size##########################resize 的图片大小
        self.resample=resample
        self.do_normalize=do_normalize
        self.image_mean=image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std=image_std if image_std is not None else IMAGENET_STANDARD_STD

        self.apply_ocr=apply_ocr
        self.ocr_lang=ocr_lang


    def __call__(self, images: ImageInput, return_tensors: Optional[Union[str, TensorType]] = None, **kwargs
                 ) -> BatchFeature:

        valid_images=False

        #########################################################
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples), "
                f"but is of type {type(images)}."
            )
        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]
        ########################################################

        if self.apply_ocr:
            requires_backends(self,'pytesseract')
            words_batch=[]
            boxes_batch=[]

            for image in images:
                words,boxes=apply_tesseract(self.to_pil_image(image),self.ocr_lang)
                words_batch.append(words)
                boxes_batch.append(boxes)

        #resize

        if self.apply_ocr:
            data={
                'words':words_batch,
                'boxes':boxes_batch
            }

        return data


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
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











