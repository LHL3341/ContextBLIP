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
#导入blip
from blip_finetune import ContextualBLIP as BLIP
from utils import pre_caption

random.seed(10)
torch.manual_seed(10)


#改变工作目录
os.chdir(sys.path[0])


def find_best_matches(text_features, photo_features):
    similarities = (photo_features @ text_features.T).squeeze(1)
    best_photo_idx = (-similarities).argsort()
    similarities = -similarities
    similarities.sort()
    return best_photo_idx, similarities


def convert_models_to_fp32(model):
    for p in model.parameters():
        if p.grad is not None:
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()


class ContextualBLIP(torch.nn.Module):
    def __init__(self, bert_config, args=None,pretrain = True,test=False):
        super(ContextualBLIP, self).__init__()
        bert_config = json.load(open('vilbert-and-bert-config.json', 'r'))
        self.blip = BLIP(bert_config, args,pretrain=False).cuda()
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

    def forward(self, images, text, pos_mask):
        with torch.no_grad():
            batchsize = images.shape[0]
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
        return preds

if __name__ == "__main__":
    import wandb
    wandb.init(project='blip_context', settings=wandb.Settings(start_method="thread"))
    config = wandb.config
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batchsize", type=int, default=36)
    parser.add_argument("--wd", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("-m", "--model", type=str, default='ViT-B/16')
    parser.add_argument("-a", "--activation", default='relu')
    parser.add_argument("-s", "--logit_scale", default=1000)
    parser.add_argument("--frozen_blip", action="store_true",default=False)
    parser.add_argument("--finetuned_checkpoint_path", default='blip_0.3623_19.pt')
    parser.add_argument("--add_input", action="store_true",default=True)
    parser.add_argument("--positional", action="store_true",default=True)
    parser.add_argument("--scheduler", default= 0.95, type=float)
    parser.add_argument("--transformer_layers", default=2, type=int)
    parser.add_argument("--augmentation", default=0, type=int)
    parser.add_argument("--all_pos", action="store_true",default=False)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='output/finetune_2/0.5_0')
    parser.add_argument('--valid_descr_path', type=str, default='./dataset/valid_data.json')
    parser.add_argument('--train_descr_path', type=str, default='./dataset/train_data.json')
    parser.add_argument('--train_split_path', type=str, default='./dataset/train_split.json')
    parser.add_argument('--valid_split_path', type=str, default='./dataset/valid_split.json')
    parser.add_argument('--imgs_path', type=str, default='./dataset/image-sets')
    parser.add_argument("--job_id")


    args = parser.parse_args()
    assert args.activation in ['leaky-relu', 'relu', 'gelu']
    wandb.config.update(args)

    img_dirs = args.imgs_path
    valid_data = json.load(open(args.valid_descr_path, 'r'))
    train_data = json.load(open(args.train_descr_path, 'r'))
    train = []
    for img_dir, data in train_data.items():
        for img_idx, text in data.items():
            train.append((img_dir, img_idx, text))
    valid = []
    for img_dir, data in valid_data.items():
        for img_idx, text in data.items():
            valid.append((img_dir, img_idx, text))
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
            correct = 0
            total = 0
            video_correct = 0
            video_total = 0
            img_correct = 0
            img_total = 0
            ranks = defaultdict(int)
            contextual_blip.eval()
            for img_dir, img_idx, text in tqdm.tqdm(valid):
                text = [pre_caption(text)]
                img_idx = int(img_idx)
                img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
                img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
                images = [Image.open(photo_file).convert("RGB") for photo_file in img_files]
                #把图片转成tensor
                preprocess = transforms.Compose([
                    transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])  
                images = [preprocess(image) for image in images]
                image = torch.stack(images).to(device)
                if "open-images" in str(img_dir):
                    pos_mask = torch.zeros((10, 1)).cuda()
                else:
                    pos_mask = torch.ones((10, 1)).cuda()
                with torch.no_grad():
                    logits = contextual_blip(image, text, pos_mask).squeeze()
                pred = torch.argmax(logits).squeeze()
                if img_idx == pred:
                    correct += 1
                if 'open-images' in img_dir:
                    img_total += 1
                    if img_idx == pred:
                        img_correct += 1
                else:
                    video_total += 1
                    if img_idx == pred:
                        video_correct += 1        
                total += 1
            print(len(valid))
            print(ranks)
            acc = round(correct / total, 4)
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
        random.shuffle(train)
        contextual_blip.train()
        correct = 0
        video_correct = 0
        img_correct = 0
        video_total = 0
        img_total = 0
        total = 0
        for img_dir, img_idx, text in train:
            step += 1
            text = [pre_caption(text)]
            img_idx = int(img_idx)
            img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            images = [Image.open(photo_file).convert("RGB") for photo_file in img_files]
            #把图片转成tensor
            preprocess = transforms.Compose([                        
                transforms.RandomResizedCrop(224,scale=(0.2, 1.0),interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            images = [preprocess(image) for image in images]
            image = torch.stack(images).to(device)
            if "open-images" in str(img_dir):
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
            loss.backward()
            pred = torch.argmax(logits).squeeze()
            if img_idx == pred:
                correct += 1
            if "open-images" in str(img_dir):
                img_total += 1
                if img_idx == pred:
                    img_correct += 1
            else:
                video_total += 1
                if img_idx == pred:
                    video_correct += 1  
            total += 1
            if step % config.batchsize == 0:
                print(f'TOTAL LOSS: {loss}')
                print('STEP: ' + str(step))
                wandb.log({'loss': loss})
                #convert_models_to_fp32(contextual_blip)
                optimizer.step()
                #clip.model.convert_weights(contextual_blip)
                #contextual_blip.blip.float()
                optimizer.zero_grad()
        acc = round(correct / total, 4)
        wandb.log({'train_acc': acc})
        vacc = round(video_correct / video_total, 4)
        sacc = round(img_correct / img_total, 4)
        wandb.log({'train_vacc': vacc})
        wandb.log({'train_sacc': sacc})
        scheduler.step()
