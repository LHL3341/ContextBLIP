'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
import transformers
transformers.logging.set_verbosity_error()

import torch
from torch import nn
import torch.nn.functional as F

from models.blip import create_vit, init_tokenizer, load_checkpoint,BLIP_Base
from mask import mask_image,mask_text
from time import time

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
    
class Adapter_BLIP(nn.Module):
    def __init__(self,     
                mask_rate,
                 prompt_length,            
                 med_config = 'configs/bert_config.json',  
                 blip_path = './model_base_14M.pth',
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
        self.pretrained_blip = BLIP_Base(med_config)
        self.pretrained_blip,msg=load_checkpoint(self.pretrained_blip,blip_path)

        vision_width = self.pretrained_blip.visual_encoder.embed_dim
        text_width = vision_width
        # create the decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = vision_width        
        decoder_config.num_hidden_layers = 4
        decoder_config.num_attention_heads = 8
        #self.text_decoder.resize_token_embeddings(len(self.tokenizer)) 
        #tie_encoder_decoder_weights(self.text_encoder,self.text_decoder,'','/attention')

        self.visual_decoder = BertModel(config=decoder_config, add_pooling_layer=False)
        self.decoder_pos_embed = nn.Embedding(196, vision_width)
        self.recon_head = nn.Linear(vision_width,768)
        self.pred_head = nn.Linear(text_width,len(self.pretrained_blip.tokenizer)-2)
        self.mask_embed = nn.Parameter(torch.randn(vision_width))
        prompt_length = 5
        self.prompt = nn.Parameter(torch.randn(17,prompt_length,text_width))

        self.visual_decoder.embeddings.word_embeddings=None
        self.visual_decoder.embeddings.position_embeddings=None
        self.visual_decoder.embeddings.LayerNorm=None

        self.vision_adapter = Adapter(vision_width)
        self.text_adapter = Adapter(text_width)
        self.text_decoder = self.pretrained_blip.text_encoder

        for param in self.pretrained_blip.parameters():
            param.requires_grad = False
        
    def forward(self, image, caption, alpha):
        #t1= time()
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
        with torch.no_grad():       
            sim_i2t = image_feat @ text_feat.t()
            sim_t2i = text_feat @ image_feat.t()
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)+1e-4 
            weights_t2i.fill_diagonal_(0)            
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)+1e-4 
            weights_i2t.fill_diagonal_(0)   
            
        # select a negative image for each text
        image_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0) # [b,197,768]

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg],dim=0)    # [2b,30]
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0) # [2b,30]    

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0) # [2b,197,768]
        image_atts_all = torch.cat([image_atts,image_atts],dim=0) # [2b,197]

        output_neg = self.pretrained_blip.text_encoder(text_ids_all,
                                       attention_mask = text_atts_all,
                                       encoder_hidden_states = image_embeds_all,
                                       encoder_attention_mask = image_atts_all,     
                                       return_dict = True,
                                      )                            

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        # [3b,768]
        vl_output = self.pretrained_blip.itm_head(vl_embeddings)  #[3b,2]          

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)  

        attention_map = torch.stack(output_pos['cross_attentions'],dim=1)
        avg_attention_map = attention_map.mean(dim=1).mean(dim=1).detach()

        ##================= MLM ========================##     
        #t2= time()
        masked_ids, concept_type = mask_text(caption,self.pretrained_blip.tokenizer)
        #masked_ids, concept_type = text.input_ids, [0,0]
        #t3= time()
        B = masked_ids.shape[0]
        prompt_length = self.prompt.shape[1]
        #decoder_input_ids = text.input_ids.clone()      
        #decoder_input_ids[:,0] = self.tokenizer.bos_token_id
        #decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100) 
        masked_ids = masked_ids.to(image.device)
        prompt = torch.stack([self.prompt[i] for i in concept_type])
        decoder_output = self.text_decoder(masked_ids, 
                                           attention_mask = torch.ones([B,prompt_length+30]).to(image.device), 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,
                                           output_attentions = True,
                                           return_dict = True,
                                           prompt = prompt
                                          )
        predict = self.pred_head(decoder_output.last_hidden_state[:,prompt_length:][masked_ids==103])
        loss_mlm = nn.CrossEntropyLoss()(predict,text.input_ids[masked_ids==103])

        #attention_map_mask = torch.stack(decoder_output['cross_attentions'],dim=1)
        #avg_attention_map_mask = attention_map_mask.mean(dim=1).mean(dim=1).detach()

        #mlm_attn = avg_attention_map_mask[text_mask==1]/avg_attention_map_mask[text_mask==1].norm(dim=1,keepdim=True)
        #itm_attn = avg_attention_map[text_mask==1]/avg_attention_map[text_mask==1].norm(dim=1,keepdim=True)
        #loss_attn = torch.zeros([])#nn.MSELoss(reduction='sum')(mlm_attn, itm_attn)

        ##================= MIM ========================##   
        
        with torch.no_grad():
            image_patches,masked_idx,unmasked_idx = mask_image(image,avg_attention_map[...,1:])
            
            unmasked_patches = torch.stack([image_patches[i,idx] for i,idx in enumerate(unmasked_idx)],dim=0)

            self.pretrained_blip.visual_encoder.patch_embed.img_size = (16,16)
            unmask_tokens = self.pretrained_blip.visual_encoder.patch_embed(unmasked_patches.view(-1,3,16,16)).view(B,-1,768)
            self.pretrained_blip.visual_encoder.patch_embed.img_size = (224,224)
            cls_tokens = self.pretrained_blip.visual_encoder.cls_token.expand(B, -1, -1)
            unmask_tokens = torch.cat((cls_tokens, unmask_tokens), dim=1)
            pos_embed = torch.cat([self.pretrained_blip.visual_encoder.pos_embed[:,0:1,:].repeat(B,1,1),
                                self.pretrained_blip.visual_encoder.pos_embed[:,unmasked_idx+1,:].squeeze(0)],dim=1)
            unmask_tokens = unmask_tokens + pos_embed
            unmask_tokens = self.pretrained_blip.visual_encoder.pos_drop(unmask_tokens)
            
            for i,blk in enumerate(self.pretrained_blip.visual_encoder.blocks):
                unmask_tokens = blk(unmask_tokens)
            unmask_tokens = self.pretrained_blip.visual_encoder.norm(unmask_tokens)
        
        unmask_tokens = unmask_tokens + self.vision_adapter(unmask_tokens)
        text_embeds = output_pos.last_hidden_state + self.text_adapter(output_pos.last_hidden_state)

        masked_tokens = self.mask_embed[None, None, :].repeat(B, masked_idx.shape[1], 1)
        masked_tokens += self.decoder_pos_embed(masked_idx)
        concat_tokens = torch.cat([masked_tokens, unmask_tokens], dim=1)

        ids = torch.cat([masked_idx,torch.zeros(B,1).cuda()-1,unmasked_idx],dim=1)
        sorted_id = ids.argsort()
        dec_input_tokens = torch.stack([concat_tokens[i,id] for i,id in enumerate(sorted_id)],dim=0)

        recon_image_embeds = self.visual_decoder(encoder_embeds=dec_input_tokens,
                                                 attention_mask = image_atts,
                                                 encoder_hidden_states = text_embeds,
                                                 encoder_attention_mask = text.attention_mask,
                                                 return_dict = True,  
                                                )
        recon_image = self.recon_head(torch.stack([recon_image_embeds.last_hidden_state[i,idx+1] 
                                                   for i,idx in enumerate(masked_idx)],dim=0))
        masked_patches = torch.stack([image_patches[i,idx] for i,idx in enumerate(masked_idx)],dim=0)
        loss_mim = nn.MSELoss()(recon_image,masked_patches)
        #t4= time()
        #print(t4-t1,t3-t2)
        return torch.zeros([]), loss_itm, loss_mlm, loss_mim
    
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
            return output_pos


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


def blip_pretrain(**kwargs):
    model = Adapter_BLIP(**kwargs)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    #if args.distributed == False:
    #return tensor
    
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output     


from typing import List
def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias                
            print(module_name+' is tied')    
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)  
