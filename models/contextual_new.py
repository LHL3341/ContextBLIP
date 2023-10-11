'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
from models.med import BertConfig, BertModel
import transformers
transformers.logging.set_verbosity_error()

import torch
from torch import nn
import torch.nn.functional as F

from models.blip import BLIP_Base
"""
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class MultiLevelAdapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(MultiLevelAdapter, self).__init__()
        self.adapt_layer = range(12)
        self.down = nn.ModuleList([DownSampler(c_in) for i in self.adapt_layer])
        self.up = UpSampler(c_in,reduction=len(self.adapt_layer)//reduction)

    def forward(self,x, hidden):
        latent_features = []
        for i,layer in enumerate(self.adapt_layer):
            latent = self.down[i](hidden[layer-1])
            latent_features.append(latent)
        latent_features = torch.cat(latent_features,dim=2)
        x = x + self.up(latent_features)
        return x

class DownSampler(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(DownSampler, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class UpSampler(nn.Module):
    def __init__(self, c_in, reduction=1):
        super(UpSampler, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in * reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
"""
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class MultiLevelAdapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(MultiLevelAdapter, self).__init__()
        self.adapt_layer = [3,6,9,12]
        layer_num=len(self.adapt_layer)
        self.down = nn.ModuleList([DownSampler(c_in,reduction) for i in self.adapt_layer])
        self.up = UpSampler(c_in,reduction=reduction/layer_num)

    def forward(self,x, hidden):
        latent_features = []
        for i,layer in enumerate(self.adapt_layer):
            latent = self.down[i](hidden[layer-1])
            latent_features.append(latent)
        latent_features = torch.cat(latent_features,dim=2)
        x = x + self.up(latent_features)
        return x

class DownSampler(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(DownSampler, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class UpSampler(nn.Module):
    def __init__(self, c_in, reduction=1):
        super(UpSampler, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(int(c_in / reduction), c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

    
    
class Adapter_BLIP(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/bert_config.json',  
                 blip_path = 'model_base_14M.pth',
                 reduction = 2,
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                    
                 embed_dim = 256,     
                 queue_size = 57600,
                 momentum = 0.995,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.pretrained_blip = BLIP_Base(med_config,image_size=image_size)
        vision_width = self.pretrained_blip.visual_encoder.embed_dim

        self.vision_adapter = MultiLevelAdapter(vision_width,reduction)
        
    def forward(self, image, caption,output_attentions = False):
    
        image_embeds,hidden = self.pretrained_blip.visual_encoder(image,output_hidden_states=True) # [b,197,768]
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)            
            
        text = self.pretrained_blip.tokenizer(caption, padding='max_length', truncation=True, max_length=30, 
                                return_tensors="pt").to(image.device)  # [b,30]
             
        #image_embeds = image_embeds + self.vision_adapter(image_embeds)
        image_embeds = self.vision_adapter(image_embeds, hidden)
        ###============== Image-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:,0] = self.pretrained_blip.tokenizer.enc_token_id
        
        # forward the positve image-text pair
        bs = image.size(0)
        output_pos = self.pretrained_blip.text_encoder(encoder_input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,
                                        output_attentions = True,
                                       return_dict = True,
                                      ) # [b,30,768]      
        if output_attentions:
            attention_map = torch.stack(output_pos['cross_attentions'],dim=1)
            avg_attention_map = attention_map.mean(dim=1).mean(dim=1).detach()
            return output_pos.last_hidden_state, avg_attention_map    
        return output_pos.last_hidden_state
    
    def inference(self,image,caption):
        with torch.no_grad():
        
            image_embeds = self.pretrained_blip.visual_encoder(image) # [b,197,768]
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
            image_feat = F.normalize(self.pretrained_blip.vision_proj(image_embeds[:,0,:]),dim=-1) # [b,256]         
            
            text = self.pretrained_blip.tokenizer(caption, padding='max_length', truncation=True, max_length=30, 
                                return_tensors="pt").to(image.device)  # [b,30]
            text_output = self.pretrained_blip.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')  # [b,30,768]      
            text_feat = F.normalize(self.pretrained_blip.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)# [b,256]                 
             
            image_embeds = image_embeds + self.vision_adapter(image_embeds)

            ###============== Image-text Matching ===================###
            encoder_input_ids = text.input_ids.clone()
            encoder_input_ids[:,0] = self.pretrained_blip.tokenizer.enc_token_id
            
            # forward the positve image-text pair
            bs = image.size(0)
            output_pos = self.pretrained_blip.text_encoder(encoder_input_ids,
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,
                                            output_attentions = True,
                                        return_dict = True,
                                        ) # [b,30,768] 
            logits = self.pretrained_blip.itm_head(output_pos)
            return logits
