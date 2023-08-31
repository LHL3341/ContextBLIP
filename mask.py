import random
import torch
def mask_text(input_ids):
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

