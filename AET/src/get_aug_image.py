from torchvision.transforms import transforms

from masking_generator import MaskingGenerator
from transforms import RandomResizedCropAndInterpolationWithTwoPic

from dall_e.utils import map_pixels
import cv2
import torch
from torchvision import transforms
from PIL import Image
from randaugment import RandomAugment
from utils import GaussianBlur


IMAGENET_DEFAULT_MEAN = [0.5, 0.5, 0.5]
IMAGENET_DEFAULT_STD = [0.5, 0.5, 0.5]


mean = IMAGENET_DEFAULT_MEAN
std = IMAGENET_DEFAULT_STD

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    img = cv2.imread(path, cv2.IMREAD_LOAD_GDAL)
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return img.convert("RGB")

t = []
t.append(RandomResizedCropAndInterpolationWithTwoPic(
        size=224,#224
        second_size=112,#128
        interpolation='bicubic',#'bicubic'
        second_interpolation='lanczos',#'lanczos'
    ))  # to maintain same ratio w.r.t. 224 images

common_transform = transforms.Compose(t)

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
# jinyu: add augmentation
pretrain_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
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
masked_position_generator = MaskingGenerator(
            (14,14), num_masking_patches=75,
            max_num_patches=None,
            min_num_patches=16,
        )

patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                map_pixels,
            ])

def get_image(roots,pretrain=False):

    batch_image1_transform=[]
    batch_image2_transform = []
    batch_mask=[]
    batch_patch_transform=[]
    batch_visual_token_transform=[]

    if type(roots)==str:
        img = pil_loader(roots)
        res_image1_transform= pretrain_transform(img).tolist()
        res_image2_transform = pretrain_transform(img).tolist()

    else:
        for root in roots:
            img=pil_loader(root)


            image1 = pretrain_transform(img).unsqueeze(0)
            image2 = pretrain_transform(img).unsqueeze(0)

            for_patches, for_visual_tokens = common_transform(img)

            patch_transforms=patch_transform(for_patches).unsqueeze(0)
            visual_token_transforms=visual_token_transform(for_visual_tokens).unsqueeze(0)
            mask=torch.from_numpy(masked_position_generator()).unsqueeze(0)

            batch_image1_transform.append(image1)
            batch_image2_transform.append(image2)

            batch_patch_transform.append(patch_transforms)
            batch_visual_token_transform.append(visual_token_transforms)

            batch_mask.append(mask)



        res_image1_transform=torch.concat(batch_image1_transform,dim=0)
        res_image2_transform=torch.concat(batch_image2_transform,dim=0)

        res_patch_transform=torch.concat(batch_patch_transform,dim=0)
        res_visual_token_transform = torch.concat(batch_visual_token_transform, dim=0)

        res_mask=torch.concat(batch_mask,dim=0)

    if pretrain:
        return res_patch_transform, res_visual_token_transform,res_mask

    else:
        return res_image1_transform,res_image2_transform


