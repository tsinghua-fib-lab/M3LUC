import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vit_l_16, ViT_L_16_Weights
import open_clip
from transformers import BertModel
import pdb
import numpy as np

class landuse_net_att(nn.Module):
    def __init__(self, num_classes,dropout = 0.2):
        super(landuse_net_att, self).__init__()

        self.vit_rs, _, self.preprocess = open_clip.create_model_and_transforms("ViT-L-14")
        ckpt = torch.load("/data3/lisibo/euluc/codes/bert-cache/RemoteCLIP-ViT-L-14.pt", map_location="cpu")
        message = self.vit_rs.load_state_dict(ckpt)
        print("vit-rs",message)

        self.vit_other = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
        print("vit-other")
        self.poi_encoder = BertModel.from_pretrained('/data3/lisibo/euluc/codes/bert-cache')
        print("poi-encoder")
        
        self.avg_popu_emb = nn.Embedding(64, 768)

        self.att_layer = nn.TransformerEncoderLayer(d_model=32, nhead=8, dropout=dropout,batch_first=True,
                                                    )
        self.att_encoder = nn.TransformerEncoder(self.att_layer, num_layers=2)
        self.att_decoder_layer = nn.TransformerDecoderLayer(d_model=32, nhead=8, dropout=dropout,batch_first=True,
                                                            )
        self.att_decoder = nn.TransformerDecoder(self.att_decoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(8416, 256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def get_embedding(self, poi_names, mask, img, img_mask, img_building):
        with torch.no_grad():
            img_emb = self.vit_rs.encode_image(img)
            img_emb_mask = self.vit_other(img_mask)
            img_emb_building = self.vit_other(img_building)

            poi_names = poi_names.squeeze(1)
            poi_embed = self.poi_encoder(input_ids=poi_names, attention_mask=mask, return_dict=True)
            poi_embed = torch.mean(poi_embed.last_hidden_state, dim=1)

        x = torch.cat([img_emb, poi_embed, img_emb_mask, img_emb_building], dim=1)
        return x
    
    def forward_emb(self, emb, avg_popu=0):
        bs = emb.shape[0]
        popu_emb = self.avg_popu_emb(avg_popu)
        emb_all = torch.cat([emb, popu_emb], dim=1)

        emb_all = torch.cat([emb_all, torch.zeros(bs,16).to(emb_all.device)], dim=1)
    
        emb_all = emb_all.reshape(bs,-1,32)

        is_train = self.training
        ## Modality Dropout
        if is_train:
            for modal in range(8):
                rand = np.random.random()
                if rand < 0.2:
                    emb_all[:,modal*32:(modal+1)*32] = torch.zeros(bs,32,32).to(emb_all.device)

        att_encoder_out = self.att_encoder(emb_all)
        att_decoder_out = self.att_decoder(emb_all, att_encoder_out)
        att_decoder_out = att_decoder_out.reshape(bs,-1)

        x = self.classifier(att_decoder_out)

        return x
        