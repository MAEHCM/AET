# torch
import os
import shutil

import torch
from PIL import Image
import cv2
from tqdm import trange, tqdm
from transformers import LayoutLMv3Tokenizer


from Process import Layoutlmv3FeatureExtractor

tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")

def pil_loader(path):
    img = cv2.imread(path, cv2.IMREAD_LOAD_GDAL)
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return img.convert("RGB")


class InputExample(object):

    def __init__(self, root, tokens, bboxes, attention_masks,labels):

        self.root = root
        self.tokens = tokens
        self.bboxes = bboxes
        self.attention_masks = attention_masks
        self.labels=labels


label_file='/mnt/disk2//archive/labels/train.txt'
'''with open(label_file,'r') as lines:
    for idx,line in tqdm(enumerate(lines)):

        image_path=line.split(' ')[0]
        label=line.split(' ')[1]
        image_res_path=os.path.join('/mnt/disk2//archive/images',image_path)

        cached_features_file = os.path.join(
            '/mnt/disk2/hjb/for_train',
            "cached_{}".format(
                image_res_path.split('/')[-1].split('.')[-2],
            ),
        )

        if os.path.exists(cached_features_file):
            continue
        else:
            try:
                sample = pil_loader(image_res_path)
                # PIL image
                feature_extractor = Layoutlmv3FeatureExtractor()
                features = feature_extractor(sample, return_tensors='pt')

                encoded_inputs = tokenizer(
                    text=features['words'],
                    boxes=features['boxes'],
                )

                token = encoded_inputs['input_ids'][0]
                bbox = encoded_inputs['bbox'][0]
                attention_mask = encoded_inputs['attention_mask'][0]

                max_seq_len = 512

                if len(token) > max_seq_len:
                    token = token[: max_seq_len]
                    bbox = bbox[: max_seq_len]
                    attention_mask = attention_mask[:max_seq_len]

                    token[-1] = tokenizer.sep_token_id
                    bbox[-1] = [0, 0, 0, 0]
                    attention_mask[-1] = 1

                # padding on right
                padding_length = max_seq_len - len(token)

                token += [1] * padding_length

                for i in range(padding_length):
                    bbox.append([0, 0, 0, 0])

                attention_mask += [0] * padding_length

                assert len(token) == max_seq_len
                assert len(bbox) == max_seq_len
                assert len(attention_mask) == max_seq_len

                tokens = torch.tensor(token, dtype=torch.long)
                bboxes = torch.tensor(bbox, dtype=torch.long)
                attention_masks = torch.tensor(attention_mask, dtype=torch.long)

                examples=[]
                examples.append(
                    InputExample(
                        root=image_res_path,
                        tokens=tokens,
                        bboxes=bboxes,
                        attention_masks=attention_masks,
                        labels=label,
                    )
                )
                features = torch.save(examples, cached_features_file)
            except:
                print('Error picture or Empty picture')
                continue
'''

data_dir=os.listdir('/mnt/disk2//for_train')

def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        print(srcfile)
        print(dstpath + fname)
        #shutil.move(srcfile, dstpath + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + fname))


for cache in data_dir:
    cache_name=os.path.join('/mnt/disk2//for_train',cache)

    file=torch.load(cache_name)

    label_file=file[0].labels
    file_name = os.path.join('/mnt/disk2//', str(label_file))
    print(file_name)
    mycopyfile(cache_name,file_name)






