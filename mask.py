import random
import torch
from transformers import BertTokenizer
import time
import json
from data.utils import pre_caption


pos_list = ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON',
            'PROPN','PUNCT','SCONJ','SYM','VERB','X']
pos ={}
with open('pos_tags.json') as f:
    json_file = json.load(f)
for key, value in json_file.items():
    pos[pre_caption(key,30)] = value
print("captions:",len(pos))

def mask_text_(input_ids):
    mask_map = torch.zeros_like(input_ids)
    mask_ids = input_ids.clone()
    for b, input_id in enumerate(input_ids):
        text_length = input_id.shape[0]-(input_id==0).sum()
        
        mask = random.randint(0,text_length-1)
        mask_ids[b,mask] = 103
        mask_map[b,mask] = 1


    return mask_ids, mask_map


def mask_image(image,avg_attention_map,num_patches=196, mask_rate=0.5):
    patch_h = 16
    patch_w = 16
    b,c,h,w = image.shape
    num_patches = (h // patch_h) * (w // patch_w)
    # (b, c=3, h, w)->(b, n_patches, patch_size**2 * c)
    patches = image.view(
        b, c,
        h // patch_h, patch_h, 
        w // patch_w, patch_w
    ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)
    batch_size = image.shape[0]
    mask_num = int(mask_rate * num_patches)

    masked_token = random.randint(0,avg_attention_map.shape[1]-1)

    mask = avg_attention_map[:,masked_token].clone()
    mask /= mask.norm(dim=-1).unsqueeze(1)
    noise = torch.rand(batch_size,num_patches)*0.02

    mask = mask+noise.cuda()
    idx = mask.argsort(descending=True)
    masked_idx = idx[:,:mask_num]
    unmasked_idx = idx[:,mask_num:]

    return patches,masked_idx,unmasked_idx

def mask_text(texts,tokenizer):
    
    masked_ids = []
    poses = []
    for text in texts:

        tags = pos[text]
        masked_id = random.randint(0,len(tags)-1)
        masked_pos = tags[masked_id][1]

        prefix = pre_caption(' '.join([word[0] for word in tags[:masked_id]]),30)
        suffix = pre_caption(' '.join([word[0] for word in tags[masked_id+1:]]),30)

        masked_word = ' '.join(["[MASK]" for _ in tokenizer(tags[masked_id][0]).input_ids[1:-1]])
        
        masked_text = tokenizer(' '.join([prefix, masked_word, suffix]),padding='max_length', truncation=True, max_length=30, 
                                return_tensors="pt")

        masked_ids.append(masked_text.input_ids)
        poses.append(pos_list.index(masked_pos))

    masked_ids = torch.stack(masked_ids,dim=0).squeeze()
    return masked_ids,poses