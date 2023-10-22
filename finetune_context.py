# inspired from: https://github.com/openai/CLIP/issues/83
# https://github.com/openai/CLIP/issues/83
import warnings
warnings.filterwarnings("ignore")
import json
import os
import random

import torch
from torch import autograd
import tqdm
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from collections import defaultdict
import sys
import torchvision.transforms as transforms
from volta_src.config import BertConfig
from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel 
from volta_src.embeddings import BertLayerNorm
from volta_src.encoders import GeLU
from extras_ import convert_sents_to_features, BertLayer
import argparse
from urllib.parse import urlparse
from timm.models.hub import download_cached_file

from transform.randaugment import RandomAugment
from torchvision.transforms.functional import InterpolationMode
from data.dataset import ImageCoDeDataset
from finetune_notf import ContextualBLIP as Adapter_BLIP
from torch.utils.data import DataLoader 
from utils import pre_caption




#改变工作目录
os.chdir(sys.path[0])


class ContextualBLIP(torch.nn.Module):
    def __init__(self, bert_config, args=None,test=False):
        super(ContextualBLIP, self).__init__()

        bert_config = json.load(open('vilbert-and-bert-config.json', 'r'))
        self.blip = Adapter_BLIP(bert_config, args,pretrain=False).cuda()
        if test==False:
            checkpoint = torch.load(args.finetuned_checkpoint_path)
            msg = self.blip.load_state_dict(checkpoint['model_state_dict'],strict= False)
            print(msg)
        
        config = BertConfig.from_dict(bert_config)
        config.hidden_size = 768
        config.num_attention_heads = 8
        self.transformer = nn.ModuleList([BertLayer(config) for _ in range(2)])
        self.positional_emb = torch.nn.Embedding(10, config.hidden_size)

        print('tf params:',sum(p.numel() for p in self.transformer.parameters() if p.requires_grad))
        for param in self.blip.parameters():
            param.requires_grad = False
        

    def forward(self, images, text, pos_mask,output_attn=False):
        with torch.no_grad():
            batchsize = images.shape[0]
            if output_attn:
                features,attn_map = self.blip.blip(images,text)
            else:
                features = self.blip.blip(images,text)
            features = features[:, 0, :].squeeze()
            features = features / features.norm(dim=-1, keepdim=True)

        embs = self.positional_emb(torch.arange(batchsize).cuda())
        embs = embs * pos_mask
        x_pos = features + embs
        x_pos = x_pos.unsqueeze(dim=0)

        attention_mask = torch.ones((1, 1, 1, batchsize)).cuda()
        x = self.transformer[0](x_pos, attention_mask)
        for layer_module in self.transformer[1:]:
            x = layer_module(x, attention_mask)

        x = x + features
        preds = self.blip.prediction_layer(x)
        if output_attn:
            return preds, attn_map
        return preds

if __name__ == "__main__":
    import wandb
    wandb.init(project='blip-adapter-finetune-temporal', settings=wandb.Settings(start_method="thread"))
    config = wandb.config
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batchsize", type=int, default=36)
    parser.add_argument("--wd", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("-m", "--model", type=str, default='ViT-B/16')
    parser.add_argument("-a", "--activation", default='relu')
    parser.add_argument("-s", "--logit_scale", default=1000)
    parser.add_argument("--frozen_blip", action="store_true",default=False)
    parser.add_argument("--finetuned_checkpoint_path", default='4_0.25_0.3719_17.pt')
    parser.add_argument("--add_input", action="store_true",default=True)
    parser.add_argument("--positional", action="store_true",default=True)
    parser.add_argument("--scheduler", default= 0.95, type=float)
    parser.add_argument("--transformer_layers", default=2, type=int)
    parser.add_argument("--augmentation", default=0, type=int)
    parser.add_argument("--all_pos", action="store_true",default=False)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='output/finetune_2/0.5_0')
    parser.add_argument('--valid_descr_path', type=str, default='./dataset/valid_data.json')
    parser.add_argument('--train_descr_path', type=str, default='./dataset/train_data.json')
    parser.add_argument('--train_split_path', type=str, default='./dataset/train_split.json')
    parser.add_argument('--valid_split_path', type=str, default='./dataset/valid_split.json')
    parser.add_argument('--imgs_path', type=str, default='./dataset/image-sets')
    parser.add_argument("--random_seed",  type=int)
    parser.add_argument("--job_id")

    args = parser.parse_args()
    assert args.activation in ['leaky-relu', 'relu', 'gelu']
    wandb.config.update(args)
    print("random_seed:",args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    trainset = ImageCoDeDataset('dataset',split='train',image_transform=args.augmentation)
    validset = ImageCoDeDataset('dataset',split='valid')
    train_loader = DataLoader(trainset,36,shuffle=True,num_workers=4)
    valid_loader = DataLoader(validset,36,shuffle=False,num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'DEVICE USED: {device}')


    bert_config = json.load(open('vilbert-and-bert-config.json', 'r'))
    contextual_blip = ContextualBLIP(bert_config, args).cuda()
    #clip.model.convert_weights(contextual_blip)
    #contextual_blip.blip.float()

    config = wandb.config
    wandb.watch(contextual_blip)
    # if device == "cpu":
    #     contextual_blip.float()
    # else:
    #     clip.model.convert_weights(
    #         contextual_blip)  # Actually this line is unnecessary since clip by default already on float16

    MAX_EPOCHS = 40
    loss_mix = nn.CrossEntropyLoss()
    

    optimizer = optim.Adam(params=contextual_blip.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=config.wd)

    lambda1 = lambda epoch: args.scheduler ** epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    best_val = 0

    for i in range(args.epochs):
        save_model = False
        # EVALUATE
        if i != 0:
            preprocess = transforms.Compose([
            transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])  
            correct = 0
            total = 0
            video_correct = 0
            video_total = 0
            img_correct = 0
            img_total = 0
            ranks = defaultdict(int)
            contextual_blip.eval()
            for img, txt, img_index,is_video in tqdm.tqdm(valid_loader):
            # 36,10,3,224,224
            # 36
                img = img.to(device)
                B = img.shape[0]
                for b in range(B):
                    img_idx = int(img_index[b])
                    text = [pre_caption(txt[b])]
                    image = img[b]
                    if not is_video[b]:
                        pos_mask = torch.zeros((10, 1)).cuda()
                    else:
                        pos_mask = torch.ones((10, 1)).cuda()
                    if args.all_pos:
                        pos_mask = torch.ones((10, 1)).cuda()
                    logits = contextual_blip(image, text, pos_mask)
                    pred = torch.argmax(logits).squeeze()
                    if img_idx == pred:
                        correct += 1
                    if not is_video[b]:
                        img_total += 1
                        if img_idx == pred:
                            img_correct += 1
                    else:
                        video_total += 1
                        if img_idx == pred:
                            video_correct += 1        
                    total += 1
            print(len(validset))
            print(ranks)
            acc = round(correct / total, 4)
            print(acc)
            video_acc = round(video_correct / video_total, 4)
            img_acc = round(img_correct / img_total, 4)
            wandb.log({'acc': acc})
            wandb.log({'video_acc': video_acc})
            wandb.log({'img_acc': img_acc})
            if acc > best_val:
                best_val = acc
                save_model = True
                string = ''
                for key, val in list(vars(args).items()):
                    if 'path' not in key:
                        string += f'_{val}'
                string += f'_{best_val}_{i}'
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                torch.save({
                    'epoch': i,
                    'model_state_dict': contextual_blip.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{args.output_dir}/CONTEXTUAL_blip_best_{string.replace('/', '')}.pt")
            print('------------------------------')

        print(f'EPOCH: {i}')
        step = 0

        contextual_blip.train()
        correct = 0
        total = 0
        acc =0
        """
        preprocess = transforms.Compose([
            transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])  
        """
        preprocess = transforms.Compose([                        
                transforms.RandomResizedCrop(224,scale=(0.2, 1.0),interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        wandb.log({'lr':optimizer.state_dict()['param_groups'][0]['lr']})
        for img, txt, img_index,is_video in tqdm.tqdm(train_loader):
            # 36,10,3,224,224
            # 36
            img = img.to(device)
            B = img.shape[0]
            for b in range(B):
                img_idx = int(img_index[b])
                text = [pre_caption(txt[b])]
                image = img[b]
                if not is_video[b]:
                    pos_mask = torch.zeros((10, 1)).cuda()
                else:
                    pos_mask = torch.ones((10, 1)).cuda()
                if args.all_pos:
                    pos_mask = torch.ones((10, 1)).cuda()
                logits = contextual_blip(image, text, pos_mask)
                logits = logits.squeeze()
                logits = logits.unsqueeze(dim=0)
                ground_truth = torch.tensor([img_idx]).long().to(device)  # the index of the correct one
                loss = loss_mix(logits, ground_truth)
                pred = torch.argmax(logits).squeeze()
                if img_idx == pred:
                    correct += 1
                total += 1

                loss.backward()
            
            #print(f'TOTAL LOSS: {loss}')
            #print('STEP: ' + str(step))
            wandb.log({'loss': loss})
            
            #convert_models_to_fp32(contextual_blip)
            optimizer.step()
            #clip.model.convert_weights(contextual_blip)
            #contextual_blip.blip.float()
            optimizer.zero_grad()
        acc = round(correct / total, 4)
        wandb.log({'train_acc': acc})
        
        scheduler.step()
