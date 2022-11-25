from functools import partial

import einops

from vit import VisionTransformer, interpolate_pos_embed
from transformers import RobertaForMaskedLM, ViTModel
#from roberta import RobertaForMaskedLM
import torch
import torch.nn.functional as F
from torch import nn
from Layoutlmv3 import LayoutLMv3ForSequenceClassification
import numpy as np
import random

label_path='/mnt/disk2//DocVQA/label.txt'

def get_labels(path):
    with open(path,'r') as f:
        labels=f.read().splitlines()
    return labels

labeles=get_labels(label_path)
label2idx={label:i for i,label in enumerate(labeles)}
idx2label={i:label for i,label in enumerate(labeles)}

print(idx2label)

class ClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks. Reference: RobertaClassificationHead
    """
    def __init__(self, config, class_label):
        super().__init__()
        self.dense = nn.Linear(768, 768)

        classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(768, 7)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def check_bbox(ocr_bbox):
    #1. 不完全包括的token不用预测标记为-100 flag=0
    #2. 完全包括的token查看patch是否为True，flag=1 未对齐
    #3. flag=2 对齐
    x0, y0, x1, y1 = ocr_bbox

    d1=(x0/72).long()
    d2=(x1/72).long()

    d3=(y0/72).long()
    d4=(y1/72).long()

    if d1 != d2 or d3 != d4:
        return -100
    else:
        return d3*14+d1#需要计算的部分



class AET(nn.Module):
    def __init__(self,
                 text_encoder=None,

                 config=None,
                 temp=0.07,
                 init_deit=True
                 ):
        super().__init__()


        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']

        self.visual_encoder = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        #img_sizw=256*256

        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print(msg)

        #self.visual_encoder=ViTModel.from_pretrained('google/vit-large-patch16-224')

        vision_width = config['vision_width']
        #bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = RobertaForMaskedLM.from_pretrained(text_encoder)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)# [768,256]
        self.text_proj = nn.Linear(text_width, embed_dim)#[768,256]

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        ############################################温度系数：0.07
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']

        #下面是一模一样的动量模型，但是不跟新梯度
        self.visual_encoder_m = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.vision_proj_m = nn.Linear(vision_width, embed_dim)


        ##############################################################also for mlm
        self.text_encoder_m = RobertaForMaskedLM.from_pretrained(text_encoder)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        #########################################################################
        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]
        #模型对，动量更新
        #########################################################################

        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))#[256,65536]
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))#[256,65536]
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        #随机初始化，并对256维度做归一化

        self.fusion_model=LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base",
                                                                           label2id=label2idx,
                                                                           id2label=idx2label,)
        self.ce=nn.CrossEntropyLoss(ignore_index=-100)


    def forward(
            self,
            text=None,
            bbox=None,
            attention_mask=None,
            image=None,
            image_aug=None,
            labels=None,
            alpha=0.0,
    ):
        '''
        :param image: [b,3,256,256]
        :param image_aug: [b,3,256,256]
        :param text: [b,10(max_seq)]
        :param alpha:0.0
        :return:
        '''
        #tensor(0.0700, device='cuda:0', requires_grad=True)
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        #torch.Size([6, 3, 224, 224])
        #with torch.no_grad():
        image_embeds = self.visual_encoder(image)
        #torch.Size([6, 197, 768])

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        #torch.Size([6,197])
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        #cls头 [1,768]->[1,256],并沿着最后一层做归一化

        text_output = self.text_encoder.roberta(text,attention_mask=attention_mask,return_dict=True)
        ##########################0,12层，比较0，6层
        #直接用预训练好的bert作为encoder

        text_embeds = text_output.last_hidden_state
        #[6,512,768]

        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        #cls头 [1,768]->[1,256],并且沿着最后一层做归一化

        #########################################################################loc-loc LOSS
        batch_size = text_embeds.shape[0]
        text_length= text_embeds.shape[1]

        label_of_patch = torch.ones(batch_size * text_length).reshape(batch_size, text_length).to(text.device)

        for i in range(batch_size):
            # 对于每个batch里token的bbox去找属于自己的patch idx，对应的label
            for j in range(bbox[i].shape[0]):
                if attention_mask[i][j] == True:
                    label_of_patch[i][j] = check_bbox(bbox[i][j])
                else:
                    label_of_patch[i][j] = -100

        label_of_patch = label_of_patch.unsqueeze(1).repeat((1, 196, 1)).to(text.device)

        mask_batch = (label_of_patch == torch.arange(0, 196,device=text.device)
                      .view((196, 1))).long().unsqueeze(-1).to(text.device)

        # 2,1,4
        token_rep = text_embeds.unsqueeze(1).repeat((1, 196, 1, 1)).to(text.device)

        cnt = torch.sum(mask_batch, dim=2)
        token_patch_embedding = torch.sum(mask_batch * token_rep, dim=2) / torch.max(torch.ones_like(cnt,device=text.device), cnt)

        #torch.Size([4, 196, 768])
        lol_gt = torch.arange(196).repeat(batch_size).to(text.device)

        image_patch_embedding=self.visual_encoder(image)[:, 1:, :]


        lol_logits_1=torch.matmul(image_patch_embedding,token_patch_embedding.transpose(1,2))
        lol_logits_2 = torch.matmul(token_patch_embedding, image_patch_embedding.transpose(1, 2))

        lol_logits_1=einops.rearrange(lol_logits_1, 'b n C -> (b n) C')
        lol_logits_2 = einops.rearrange(lol_logits_2, 'b n C -> (b n) C')

        loss_lol_1=self.ce(lol_logits_1,lol_gt)
        loss_lol_2 = self.ce(lol_logits_2, lol_gt)

        loss_lol=(loss_lol_1+loss_lol_2)/2
        #########################################################################

        #维护动量队列
        # get momentum features
        with torch.no_grad():
            #先动量更新梯度
            self._momentum_update()

            #然后放入增广后的图像，得到增广后的图像向量，并把cls放到image队列的尾巴
            image_embeds_m = self.visual_encoder_m(image_aug)

            #torch.Size([6, 197, 768])
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            #归一化后的动量值：cls[6,768]->[6,256]
            image_feat_m_l = F.normalize(self.vision_proj_m(image_embeds_m[:, 1:, :]), dim=-1)
            #torch.Size([6, 196, 256])

            image_feat_m_l = self.patch_pooling(image_feat_m_l)  # pooling for image patches
            #torch.Size([6, 9, 256])

            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            #torch.Size([256, 65537]),加到队列中1+65536=65537

            #放入增广后的文本，得到增广后的文本向量，并把cls放到text队列的尾巴
            text_output_m = self.text_encoder_m.roberta(text,attention_mask=attention_mask,return_dict=True)
            #torch.Size([1, 10, 768])
            # text cls
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            #[1,256]
            # text patch
            text_feat_m_l = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 1:, :]), dim=-1)
            #[1,511,256]
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
            #torch.Size([256, 65537]),加到队列中1+65536=65537

            #全局图像文本对比
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp
            #torch.Size([1, 65537])

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            #tensor([[0., 0., 0., ..., 0., 0., 0.]], device='cuda:0')
            sim_targets.fill_diagonal_(1)
            #tensor([[1., 0., 0., ..., 0., 0., 0.]], device='cuda:0')

            #利用动量单模态编码器来计算 image-text 的相似性；
            #一开始alpha为0，表示直接用真值，后面会采用伪标签
            #动量文本，图像 去和 动量图像，文本队列 对比得到的结果需要一个置信度和真值融合得到的伪标签
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        #[1,256]*[256,65537]->[1,65537]
        sim_t2i = text_feat @ image_feat_all / self.temp
        #tensor([[-12.9957,  -0.0000,  -0.0000,  ...,  -0.0000,  -0.0000,  -0.0000]])


        #文本，图像 去和 动量图像，文本队列 对比·······················CMA
        #动量得到的值充当真值标签
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        #为什么要这样做？本文的另外一个贡献是 Momentum Distillation：
        #对于 ITC learning，一张图像的 negative text 可能也会匹配上图像的内容；
        #对于 MLM，可能存在其他的单词不同于 annotation，但是依然很好地描述了图像中的内容
        #然而，ITC 和 MLM 的 one-hot labels 惩罚了所有的 negative prediction，却没有考虑到其正确性。
        #为了解决上述问题，作者提出使用动量模型的方法，从伪真值上进行学习。动量模型时一种连续进化的教师，包含了 单模态和多模态编码器的指数级移动平均版本。特别的，对于 ITC 来说，作者首先利用动量单模态编码器来计算 image-text 的相似性；


        # jinyu: add inMod g2l loss······························LMI
        #单一模态内计算
        #动量text_batch : text_batch
        loss_t2t_inMod_l = self.in_batch_g2l_loss(text_feat_m_l, text_feat, self.temp, attention_mask[:, 1:])
        #每个patch都和该batch中的cls互为正对，因此对于一个batch计算[6,512,1,1]
        #每个patch都和所有batch中的
        loss_i2i_inMod_l = self.in_batch_g2l_loss(image_feat_m_l, image_feat, self.temp)

        # jinyu: add in-modality g2g loss························IMC
        sim_i2i = image_feat @ image_feat_all / self.temp
        #[1,256]*[256, 65537]->[1,65537]
        sim_t2t = text_feat @ text_feat_all / self.temp

        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_targets, dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_targets, dim=1).mean()

        loss_ita = (loss_t2t_inMod_l + loss_i2i_inMod_l + loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 6

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###=================================###layoutlmv3

        outputs=self.fusion_model(
            input_ids=text,
            bbox=bbox,
            attention_mask=attention_mask,
            pixel_values=image,
            align_text=text_embeds,
            align_image=image_embeds,
            labels=labels
        )
        #torch.Size([4, 512, 2])

        # forward the positve image-text pair
        #return loss_ita,outputs.loss,outputs.logits
        return outputs.loss,outputs.logits

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        #cls feature,ptr表示指针，进出队列
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)

        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


    #图像块的池化以减少计算并扩大感受野
    def patch_pooling(self, x):
        #torch.Size([6, 256, 256])
        #[6,196,256]
        batch_size, seq_length, dim = x.size()
        b1 = int(np.sqrt(seq_length))
        #14
        x = x.reshape(batch_size, b1, b1, dim)
        #[6,14,14,256]
        x = x.permute(0, 3, 1, 2)
        #[6,256,14,14]
        c1 = int(np.sqrt(b1))
        #3
        x = F.avg_pool2d(x, 5, stride=4)
        #torch.Size([1, 256, 4, 4])

        x = x.permute(0, 2, 3, 1).reshape(batch_size, c1 * c1, dim)
        #[1,3,3,256]->[1,9,256]
        return x

    # jinyu: in-batch g2l loss
    def in_batch_g2l_loss(self, l, m, temp, attention_mask=None):
        #一个batch里来自相同文本的视为正样本，不同batch的视为负样本
        '''
        :param l: [6,511,256]
        :param m: [6,256]
        :param temp: 0.07
        :param attention_mask:torch.Size([6, 511])
        :return:
        '''

        m = m.unsqueeze(1)
        #torch.Size([6, 1, 256])

        N, n_locals, dim = l.size()

        l_n = l.reshape(-1, dim)  # (N * n_locals) * d
        #[6*511,256]
        m_n = m.reshape(-1, dim)  # N * d
        #[6*1,256]

        # Inner product for positive samples. Outer product for negative. We need to do it this way
        #正样本对
        u_p = torch.matmul(l, m.permute(0, 2, 1)).unsqueeze(2) / temp  # N * n_locals * 1 * 1
        #[6,511,256] * [6,256,1]->[6,511,1]
        #torch.Size([6, 511, 1, 1])

        # if l comes from text, then attention_mask is not None
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(2).unsqueeze(3)
            #[6,511,1,1]

            u_p = (temp_mask * u_p) + (10000. * (1 - temp_mask))
            #文本被mask掉的地方用10000去代替

        #不同batch的文本patch视为负样本
        u_n = torch.mm(m_n, l_n.t()) / temp
        #[6*1,256] * [256,6*511]
        #torch.Size([6*1, 6*511])
        u_n = u_n.reshape(N, 1, N, n_locals).permute(0, 2, 3, 1)  #[6,1,6,511]
        #torch.Size([6, 6, 511, 1])


        # We need to mask the diagonal part of the negative tensor.
        mask = torch.eye(N)[:, :, None, None].to(l.device)  # N*N*1*1
        #[6,6,1,1],相同batch的不计算，只计算不同batch的视为负样本对

        n_mask = 1 - mask
        # Masking is done by shifting the diagonal before exp.
        u_n = (n_mask * u_n) - (10000. * (1 - n_mask))  # mask out "self" examples

        #对角线的值，相同batch的用-10000代替，并且文本被mask掉的地方，也用-10000去代替
        # if l comes from test, we mask out the padding tokens
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(0).unsqueeze(3).expand(N, -1, -1, -1)
            u_n = (temp_mask * u_n) - (10000. * (1 - temp_mask))

        #torch.Size([6, 6, 511, 1])
        u_n = u_n.reshape(N, N * n_locals, 1).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)
        #[6,6*511,1]->[6,511,511*6,1]


        # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
        #构建正负样本对完毕！
        #[3,11,1,1]，[3,11,33,1]

        pred_lgt = torch.cat([u_p, u_n], dim=2)
        #[3,11,34,1]
        pred_log = F.log_softmax(pred_lgt, dim=2)
        #torch.Size([3, 11, 34, 1])


        # The positive score is the first element of the log softmax.
        #正样本对齐始终在第一个位置,此类为2分类
        #torch.Size([3, 11, 1])
        if attention_mask is not None:
            loss = (torch.sum(-pred_log[:, :, 0].squeeze(), dim=1) / torch.sum(attention_mask, dim=1)).mean()
        else:
            loss = -pred_log[:, :, 0].mean()

        return loss


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output