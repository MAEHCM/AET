from PIL import  Image,ImageDraw,ImageFont

import argparse
import os

import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import torch.distributed as dist
from torch.utils.data.dataset import Dataset
import ruamel.yaml as yaml

from DVAE import DiscreteVAE
# from create_pretrain_datset import CDIP_dataset
from TClayout_model import ALBEF
from vit import interpolate_pos_embed
from transformers import BertTokenizer, AutoConfig, LayoutLMv3Processor, AutoProcessor
from Layoutlmv3model import LayoutLMv3ForPretrain

import util
from dataset import create_dataset, create_sampler, create_loader
from scheduler_factory import create_scheduler
from optim_factory import create_optimizer
from transformers import LayoutLMv3Config
from get_aug_image import get_image

################################################################
data_dir="/mnt/disk2//CORD/data"
img_path="/mnt/disk2//CORD/train/image"
label_path='/mnt/disk2//CORD/label.txt'


def get_labels(path):
    with open(path, 'r') as f:
        labels = f.read().splitlines()
    if 'O' not in labels:
        labels = ["O"] + labels
    return labels


labeles = get_labels(label_path)
label2idx = {label: i for i, label in enumerate(labeles)}
idx2label = {i: label for i, label in enumerate(labeles)}


def read_examples_from_file(data_dir, mode='train'):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))

    box_file_path = os.path.join(data_dir, "{}_box.txt".format(mode))

    image_file_path = os.path.join(data_dir, "{}_image.txt".format(mode))

    image_path = os.path.join(data_dir, "{}_image_path.txt".format(mode))

    guid_index = 1

    word = []
    box = []
    label = []
    actual_box = []
    ##########
    words = []
    boxes = []
    images = []
    labels = []
    actual_boxes = []

    images_path = []

    with open(file_path, encoding='utf-8') as f, \
            open(box_file_path, encoding='utf-8') as fb, \
            open(image_file_path, encoding='utf') as fi, \
            open(image_path, encoding='utf8') as fm:

        for line in fm:
            line = line.rstrip()
            img = Image.open(line).convert("RGB")
            images.append(img)
            images_path.append(line)

        for line, bline, iline in zip(f, fb, fi):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if word:
                    words.append(word)
                    labels.append(label)
                    boxes.append(box)
                    actual_boxes.append(actual_box)
                    # 重置，更新
                    guid_index += 1
                    word = []
                    box = []
                    label = []
                    actual_box = []
            else:
                splits = line.split("\t")  # ['R&D', 'O\n']
                bsplits = bline.split("\t")  # ['R&D', '383 91 493 175\n']
                isplits = iline.split("\t")  # ['R&D', '292 91 376 175', '762 1000', '0000971160.png\n']
                assert len(splits) == 2
                assert len(bsplits) == 2
                assert len(isplits) == 4
                assert splits[0] == bsplits[0]

                word.append(splits[0])

                if len(splits) > 1:
                    t = splits[-1].replace("\n", "")
                    label.append(int(label2idx[t]))

                    bo = bsplits[-1].replace("\n", "")
                    bo = [int(b) for b in bo.split()]
                    box.append(bo)

                    actual = [int(b) for b in isplits[1].split()]
                    actual_box.append(actual)
        if word:
            words.append(word)
            labels.append(label)
            boxes.append(box)
            actual_boxes.extend(actual_box)
    # 151
    return words, labels, boxes, images, actual_boxes, images_path


test_words, test_labels, test_boxes, test_images, test_actual_boxes, test_images_path = read_examples_from_file(
    data_dir, mode='test')

def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]



processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)


def Process(images, words, boxes, labels,actual_boxes):
    encoded_inputs = processor(images, words, boxes=boxes, word_labels=labels, padding="max_length", truncation=True)
    encoded_inputs_2=processor(images, words, boxes=actual_boxes, word_labels=labels, padding="max_length", truncation=True)

    encoded_inputs['input_ids'] = torch.tensor(encoded_inputs['input_ids'])
    encoded_inputs['attention_mask'] = torch.tensor(encoded_inputs['attention_mask'])
    encoded_inputs['bbox'] = torch.tensor(encoded_inputs['bbox'])
    encoded_inputs['labels'] = torch.tensor(encoded_inputs['labels'])

    encoded_inputs['actual_box'] = torch.tensor(encoded_inputs_2['bbox'])

    return encoded_inputs


from torch.utils.data import Dataset


class V2Dataset(Dataset):
    def __init__(self, encoded_inputs):
        self.all_images = test_images_path
        self.all_input_ids = encoded_inputs['input_ids']
        self.all_attention_masks = encoded_inputs['attention_mask']
        self.all_bboxes = encoded_inputs['bbox']
        self.all_labels = encoded_inputs['labels']
        self.all_actual_boxes=encoded_inputs['actual_box']

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, index):
        return (
            self.all_images[index],
            self.all_input_ids[index],
            self.all_attention_masks[index],
            self.all_bboxes[index],
            self.all_labels[index],
            self.all_actual_boxes[index]
        )

test_dataset = V2Dataset(Process(test_images, test_words, test_boxes, test_labels,test_actual_boxes))



def main(args, config):
    device = torch.device(args.device)

    #### Dataset ####
    print("Creating dataset")

    datasets_test = test_dataset

    data_test_loader = DataLoader(datasets_test, batch_size=1, num_workers=0)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    #### Model ####
    print("Creating model")

    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)

    model = model.to(device)


    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                     model.visual_encoder_m)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

        model.load_state_dict(state_dict)

        print('load checkpoint from %s' % args.checkpoint)

    preds = None
    out_label_ids = None
    ocr_actual=None
    ocr=None
    image_paths=None
    model.eval()

    for j, (image_path,
            tokens,
            attention_masks,
            bboxes,
            labels,actual_box) in enumerate(data_test_loader):

        with torch.no_grad():

            images, images_aug = get_image(image_path)

            image = images.to(device, non_blocking=True)
            image_aug = images_aug.to(device, non_blocking=True)

            # torch.Size([6, 3, 224, 224])

            tokens = tokens.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            labels=labels.to(device, non_blocking=True)

            attention_masks = attention_masks.to(device, non_blocking=True)

            _, logits= model(text=tokens, bbox=bboxes, attention_mask=attention_masks,
                                                         image=image,
                                                         image_aug=image_aug, labels=labels)

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
            ocr_boxes=bboxes.detach().cpu().numpy()
            ocr_actual_boxes = actual_box.detach().cpu().numpy()
            image_paths=image_path

        else:

            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
            ocr_boxes=np.append(ocr_boxes, bboxes.detach().cpu().numpy(), axis=0)
            ocr_actual_boxes=np.append(ocr_actual_boxes, actual_box.detach().cpu().numpy(), axis=0)
            image_paths=np.append(image_paths,image_path,axis=0)

    preds = np.argmax(preds, axis=2)


    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    ocr_list=[[]for _ in range(out_label_ids.shape[0])]
    actual_ocr_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100:

                out_label_list[i].append(idx2label[out_label_ids[i][j]])
                preds_list[i].append(idx2label[preds[i][j]])
                ocr_list[i].append(ocr_boxes[i][j])
                actual_ocr_list[i].append(ocr_actual_boxes[i][j])

    return preds_list,out_label_list,ocr_list,actual_ocr_list,image_paths


def iob_to_label(label):
    if label !="O":
        return label[2:]
    else:
        return "OTHER"




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./Pretrain.yaml')
    parser.add_argument('--checkpoint', default='/mnt/disk2//checkpoint_13.pth')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='roberta-base')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    print(config)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    preds_list,out_label_list,ocr_list,actual_ocr_list,image_paths=main(args, config)


    label2color = {'OTHER':(0,225,255),
                   'MENU.SUB_ETC': (0,225,255),
                   'MENU.ITEMSUBTOTAL': (0,225,255),
                   'MENU.PRICE': (0,225,255),
                   'MENU.UNITPRICE': (0,225,255),
                   'TOTAL.TOTAL_PRICE': (0,225,255),
                   'SUB_TOTAL.TAX_PRICE': (0,225,255),
                   'MENU.SUB_NM': (0,225,255),
                   'MENU.ETC': (0,225,255),
                   'TOTAL.MENUQTY_CNT': (0,225,255),
                   'MENU.NM': (0,225,255),
                   'MENU.SUB_UNITPRICE': (0,225,255),
                   'TOTAL.EMONEYPRICE': (0,225,255),
                   'TOTAL.CREDITCARDPRICE': (0,225,255),
                   'MENU.VATYN': (0,225,255),
                   'VOID_MENU.PRICE': (0,225,255),
                   'MENU.SUB_CNT': (0,225,255),
                   'MENU.DISCOUNTPRICE': (0,225,255),
                   'SUB_TOTAL.OTHERSVC_PRICE': (0,225,255),
                   'SUB_TOTAL.DISCOUNT_PRICE': (0,225,255),
                   'SUB_TOTAL.ETC': (0,225,255),
                   'TOTAL.CASHPRICE': (0,225,255),
                   'TOTAL.CHANGEPRICE': (0,225,255),
                   'TOTAL.TOTAL_ETC': (0,225,255),
                   'TOTAL.MENUTYPE_CNT': (0,225,255),
                   'MENU.CNT': (0,225,255),
                   'SUB_TOTAL.SUBTOTAL_PRICE': (0,225,255),
                   'SUB_TOTAL.SERVICE_PRICE': (0,225,255),
                   'MENU.SUB_PRICE': (0,225,255),
                   'VOID_MENU.NM': (0,225,255),
                   'MENU.NUM': (0,225,255),}

    for idx,image_p in enumerate(image_paths):
        image=Image.open(image_p).convert('RGB')

        width,length=image.size

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()


        for prediction, box in zip(preds_list[idx], ocr_list[idx]):

            #prediction_label = iob_to_label(label_map[prediction]).lower()
            box=unnormalize_box(box,length,width)
            prediction_label= iob_to_label(prediction)

            print(label2color[prediction_label])
            # 画
            draw.rectangle(box, outline=label2color[prediction_label])
            draw.text((box[0] + 10, box[1] - 10), text=prediction_label, fill=label2color[prediction_label], font=font)

        image.save('/mnt/disk2/hjb/{}.png'.format(idx))
        print('finish {}'.format(image_p))
        break

##############################################################








