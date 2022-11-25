from functools import partial

import einops

from vit import VisionTransformer, interpolate_pos_embed
#from transformers import RobertaForMaskedLM, ViTModel
from roberta import RobertaForMaskedLM
import torch
import torch.nn.functional as F
from torch import nn
from Layoutlmv3 import LayoutLMv3ForTokenClassification
import numpy as np

data_dir=""
img_path=""
label_path=''

def get_labels(path):
    with open(path,'r') as f:
        labels=f.read().splitlines()
    if 'O' not in labels:
        labels=["O"]+labels
    return labels

labeles=get_labels(label_path)
label2idx={label:i for i,label in enumerate(labeles)}
idx2label={i:label for i,label in enumerate(labeles)}

print(idx2label)

def check_bbox(ocr_bbox):
    x0, y0, x1, y1 = ocr_bbox

    d1=(x0/72).long()
    d2=(x1/72).long()

    d3=(y0/72).long()
    d4=(y1/72).long()

    if d1 != d2 or d3 != d4:
        return -100
    else:
        return d3*14+d1



class AET(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 config=None,
                 init_deit=True
                 ):
        super().__init__()

        embed_dim = config['embed_dim']

        self.visual_encoder = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print(msg)

        vision_width = config['vision_width']
     
        self.text_encoder = RobertaForMaskedLM.from_pretrained(text_encoder)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])

        self.queue_size = config['queue_size']
        self.momentum = config['momentum']

        self.visual_encoder_m = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.vision_proj_m = nn.Linear(vision_width, embed_dim)

        self.text_encoder_m = RobertaForMaskedLM.from_pretrained(text_encoder)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]
        self.copy_params()
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.fusion_model=LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base",
                                                                           label2id=label2idx,
                                                                           id2label=idx2label,
                                                                           )
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
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds = self.visual_encoder(image)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)


        text_output = self.text_encoder.roberta(text,attention_mask=attention_mask,return_dict=True)
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        import csv
        with open('./output.tsv', 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['0', image_feat])
            tsv_writer.writerow(['1', text_feat])

        batch_size = text_embeds.shape[0]
        text_length= text_embeds.shape[1]

        label_of_patch = torch.ones(batch_size * text_length).reshape(batch_size, text_length).to(text.device)

        for i in range(batch_size):
            for j in range(bbox[i].shape[0]):
                if attention_mask[i][j] == True:
                    label_of_patch[i][j] = check_bbox(bbox[i][j])
                else:
                    label_of_patch[i][j] = -100

        label_of_patch = label_of_patch.unsqueeze(1).repeat((1, 196, 1)).to(text.device)

        mask_batch = (label_of_patch == torch.arange(0, 196,device=text.device)
                      .view((196, 1))).long().unsqueeze(-1).to(text.device)

        token_rep = text_embeds.unsqueeze(1).repeat((1, 196, 1, 1)).to(text.device)

        cnt = torch.sum(mask_batch, dim=2)
        token_patch_embedding = torch.sum(mask_batch * token_rep, dim=2) / torch.max(torch.ones_like(cnt,device=text.device), cnt)

        lol_gt = torch.arange(196).repeat(batch_size).to(text.device)
        image_patch_embedding=self.visual_encoder(image)[:, 1:, :]


        lol_logits_1=torch.matmul(image_patch_embedding,token_patch_embedding.transpose(1,2))
        lol_logits_2 = torch.matmul(token_patch_embedding, image_patch_embedding.transpose(1, 2))

        lol_logits_1=einops.rearrange(lol_logits_1, 'b n C -> (b n) C')
        lol_logits_2 = einops.rearrange(lol_logits_2, 'b n C -> (b n) C')

        loss_lol_1=self.ce(lol_logits_1,lol_gt)
        loss_lol_2 = self.ce(lol_logits_2, lol_gt)

        loss_lol=(loss_lol_1+loss_lol_2)/2

        with torch.no_grad():
            self._momentum_update()

            image_embeds_m = self.visual_encoder_m(image_aug)

            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_m_l = F.normalize(self.vision_proj_m(image_embeds_m[:, 1:, :]), dim=-1)
            image_feat_m_l = self.patch_pooling(image_feat_m_l)

            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_output_m = self.text_encoder_m.roberta(text,attention_mask=attention_mask,return_dict=True)
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_m_l = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 1:, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_t2t_inMod_l = self.in_batch_g2l_loss(text_feat_m_l, text_feat, self.temp, attention_mask[:, 1:])
        loss_i2i_inMod_l = self.in_batch_g2l_loss(image_feat_m_l, image_feat, self.temp)

        sim_i2i = image_feat @ image_feat_all / self.temp
        sim_t2t = text_feat @ text_feat_all / self.temp

        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_targets, dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_targets, dim=1).mean()

        loss_ita = (loss_t2t_inMod_l+loss_i2i_inMod_l+loss_i2t+loss_t2i+loss_i2i + loss_t2t) / 6
       
        self._dequeue_and_enqueue(image_feat_m, text_feat_m)



        outputs=self.fusion_model(
            input_ids=text,
            bbox=bbox,
            attention_mask=attention_mask,
            pixel_values=image,
            align_text=text_embeds,
            align_image=image_embeds,
            labels=labels
        )

        return loss_ita,outputs.loss,outputs.logits,loss_lol

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)

        assert self.queue_size % batch_size == 0


        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def patch_pooling(self, x):
        batch_size, seq_length, dim = x.size()
        b1 = int(np.sqrt(seq_length))
        x = x.reshape(batch_size, b1, b1, dim)
        x = x.permute(0, 3, 1, 2)
        c1 = int(np.sqrt(b1))
        x = F.avg_pool2d(x, 5, stride=4)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, c1 * c1, dim)
        return x

    def in_batch_g2l_loss(self, l, m, temp, attention_mask=None):
        m = m.unsqueeze(1)
        N, n_locals, dim = l.size()
        l_n = l.reshape(-1, dim)
        m_n = m.reshape(-1, dim)
        u_p = torch.matmul(l, m.permute(0, 2, 1)).unsqueeze(2) / temp

        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(2).unsqueeze(3)


            u_p = (temp_mask * u_p) + (10000. * (1 - temp_mask))



        u_n = torch.mm(m_n, l_n.t()) / temp

        u_n = u_n.reshape(N, 1, N, n_locals).permute(0, 2, 3, 1)

        mask = torch.eye(N)[:, :, None, None].to(l.device)


        n_mask = 1 - mask

        u_n = (n_mask * u_n) - (10000. * (1 - n_mask))


        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(0).unsqueeze(3).expand(N, -1, -1, -1)
            u_n = (temp_mask * u_n) - (10000. * (1 - temp_mask))


        u_n = u_n.reshape(N, N * n_locals, 1).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

        pred_lgt = torch.cat([u_p, u_n], dim=2)
        pred_log = F.log_softmax(pred_lgt, dim=2)

        if attention_mask is not None:
            loss = (torch.sum(-pred_log[:, :, 0].squeeze(), dim=1) / torch.sum(attention_mask, dim=1)).mean()
        else:
            loss = -pred_log[:, :, 0].mean()

        return loss

@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output