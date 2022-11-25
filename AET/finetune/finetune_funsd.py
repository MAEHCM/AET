import argparse
import os

import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
from PIL import Image
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import ruamel.yaml as yaml
from AET_funsd import AET
from vit import interpolate_pos_embed
from transformers import AutoProcessor

import util
from dataset import create_sampler,create_loader
from scheduler_factory import create_scheduler
from optim_factory import create_optimizer
from get_aug_image import get_image


def get_labels(path):
    with open(path,'r') as f:
        labels=f.read().splitlines()
    if 'O' not in labels:
        labels=["O"]+labels
    return labels

labeles=get_labels(label_path)
label2idx={label:i for i,label in enumerate(labeles)}
idx2label={i:label for i,label in enumerate(labeles)}


def read_examples_from_file(data_dir,mode='train'):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    box_file_path = os.path.join(data_dir, "{}_box.txt".format(mode))
    image_file_path = os.path.join(data_dir, "{}_image.txt".format(mode))
    image_path=os.path.join(data_dir,"{}_image_path.txt".format(mode))

    guid_index=1

    word = []
    box = []
    label = []
    actual_box = []
    words = []
    boxes = []
    images = []
    labels = []
    actual_boxes = []

    images_path=[]

    with open(file_path,encoding='utf-8') as f,\
        open(box_file_path,encoding='utf-8') as fb,\
        open(image_file_path,encoding='utf') as fi,\
        open(image_path,encoding='utf8') as fm:

        for line in fm:
            line=line.rstrip()
            img = Image.open(line).convert("RGB")
            images.append(img)
            images_path.append(line)

        for line,bline,iline in zip(f,fb,fi):
            if line.startswith("-DOCSTART-") or line =="" or line =="\n":
                if word:
                    words.append(word)
                    labels.append(label)
                    boxes.append(box)
                    actual_boxes.append(actual_box)
                    guid_index+=1
                    word=[]
                    box=[]
                    label=[]
                    actual_box=[]
            else:
                splits=line.split("\t")
                bsplits=bline.split("\t")
                isplits=iline.split("\t")
                assert len(splits)==2
                assert len(bsplits)==2
                assert len(isplits)==4
                assert splits[0]==bsplits[0]

                word.append(splits[0])

                if len(splits)>1:
                    t=splits[-1].replace("\n","")
                    label.append(int(label2idx[t]))

                    bo=bsplits[-1].replace("\n","")
                    bo=[int(b) for b in bo.split()]
                    box.append(bo)

                    actual=[int(b) for b in isplits[1].split()]
                    actual_box.append(actual)
        if word:
            words.append(word)
            labels.append(label)
            boxes.append(box)
            actual_boxes.extend(actual_box)
    return words,labels,boxes,images,actual_boxes,images_path

train_words,train_labels,train_boxes,train_images,train_actual_boxes,train_images_path=read_examples_from_file(data_dir,mode='train')
test_words,test_labels,test_boxes,test_images,test_actual_boxes,test_images_path=read_examples_from_file(data_dir,mode='test')
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

def Process(images,words,boxes,labels):
    encoded_inputs=processor(images,words,boxes=boxes,word_labels=labels,padding="max_length",truncation=True)
    encoded_inputs['input_ids']=torch.tensor(encoded_inputs['input_ids'])
    encoded_inputs['attention_mask']=torch.tensor(encoded_inputs['attention_mask'])
    encoded_inputs['bbox']=torch.tensor(encoded_inputs['bbox'])
    encoded_inputs['labels']=torch.tensor(encoded_inputs['labels'])
    return encoded_inputs


from torch.utils.data import Dataset, DataLoader


class V2Dataset(Dataset):
    def __init__(self,encoded_inputs):
        self.all_images=train_images_path
        self.all_input_ids=encoded_inputs['input_ids']
        self.all_attention_masks=encoded_inputs['attention_mask']
        self.all_bboxes=encoded_inputs['bbox']
        self.all_labels=encoded_inputs['labels']
    def __len__(self):
        return len(self.all_labels)
    def __getitem__(self, index):
        return (
            self.all_images[index],
            self.all_input_ids[index],
            self.all_attention_masks[index],
            self.all_bboxes[index],
            self.all_labels[index]
        )

train_dataset=V2Dataset(Process(train_images,train_words,train_boxes,train_labels))
test_dataset=V2Dataset(Process(test_images,test_words,test_boxes,test_labels))


from seqeval.metrics import(
    f1_score,
    precision_score,
    recall_score,
)



def train(model,data_loader,optimizer, epoch, warmup_steps, device, scheduler):
    model.train()

    metric_logger = util.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', util.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', util.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image_path,
               tokens,
               attention_masks,
               bboxes,
               labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()

        images,images_aug=get_image(image_path)

        image = images.to(device, non_blocking=True)
        image_aug = images_aug.to(device, non_blocking=True)

        labels=labels.to(device,non_blocking=True)
        tokens=tokens.to(device,non_blocking=True)
        bboxes=bboxes.to(device,non_blocking=True)
        attention_masks=attention_masks.to(device,non_blocking=True)

        loss_ita,loss_cls,_,loss_lol=model(text=tokens,bbox=bboxes,attention_mask=attention_masks,image=image,image_aug=image_aug,labels=labels)


        loss=loss_cls+loss_ita+loss_lol

        loss.backward()

        optimizer.step()
	
        metric_logger.update(loss=loss.item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
	
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config):
    util.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + util.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    start_epoch = 0

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    print("Creating dataset")
    datasets = [train_dataset]
    datasets_test=test_dataset

    if args.distributed:
        num_tasks = util.get_world_size()
        global_rank = util.get_rank()
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)
    else:
        samplers = [None]



    data_loader = \
    create_loader(datasets, samplers, batch_size=[4], num_workers=[0], is_trains=[True],
                  collate_fns=[None])[0]

    data_test_loader=DataLoader(datasets_test, batch_size=4)

    print("Creating model")

    model = AET(config=config, text_encoder=args.text_encoder, init_deit=True)

    model = model.to(device)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    arg_opt = util.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)

    arg_sche = util.AttrDict(config['schedular'])

    lr_scheduler= create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                         model.visual_encoder_m)
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
        model.load_state_dict(state_dict)
        print('load checkpoint from %s' % args.checkpoint)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train(model,data_loader,data_test_loader, optimizer, epoch, warmup_steps, device, lr_scheduler, config)
        if util.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            #torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()

        preds = None
        out_label_ids = None
        model.eval()

        for j, (image_path,
                tokens,
                attention_masks,
                bboxes,
                labels) in enumerate(data_test_loader, ):

            with torch.no_grad():
                optimizer.zero_grad()

                images, images_aug = get_image(image_path)

                image = images.to(device, non_blocking=True)
                image_aug = images_aug.to(device, non_blocking=True)


                tokens = tokens.to(device, non_blocking=True)
                bboxes = bboxes.to(device, non_blocking=True)
                attention_masks = attention_masks.to(device, non_blocking=True)

                loss_ita, loss_cls, logits, loss_lol = model(text=tokens, bbox=bboxes, attention_mask=attention_masks,
                                                             image=image,
                                                             image_aug=image_aug, labels=labels)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:

                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=2)

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != -100:
                    out_label_list[i].append(idx2label[out_label_ids[i][j]])
                    preds_list[i].append(idx2label[preds[i][j]])

        results = {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }
        print(results)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./Pretrain.yaml')
    parser.add_argument('--checkpoint', default='')
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

    main(args, config)