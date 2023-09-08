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

from models.contextual import Adapter_BLIP

random.seed(10)
torch.manual_seed(10)

parser = argparse.ArgumentParser()
parser.add_argument("--finetuned_checkpoint_path", default='output/Pretrain/checkpoint_00.pth')

parser.add_argument('--valid_descr_path', type=str, default='./data/valid_data.json')
parser.add_argument('--train_descr_path', type=str, default='./data/train_data.json')
parser.add_argument('--imgs_path', type=str, default='../data/image-sets')

args = parser.parse_args()


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
model = Adapter_BLIP()
checkpoint = torch.load(args.finetuned_checkpoint_path)
model.load_state_dict(checkpoint['model'],strict= False)


correct = 0
total = 0
video_correct = 0
video_total = 0
img_correct = 0
img_total = 0
ranks = defaultdict(int)
model.cuda()
model.eval()
for img_dir, img_idx, text in tqdm.tqdm(valid):
    text = [text]
    img_idx = int(img_idx)
    img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
    img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
    images = [Image.open(photo_file).convert("RGB") for photo_file in img_files]
    preprocess = transforms.Compose([
        transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])  
    images = [preprocess(image) for image in images]
    image = torch.stack(images).to(device)
    with torch.no_grad():
        output = model(image, text).squeeze()
        logits = model.pretrained_blip.itm_head(output[:,0,:])
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
print("TOTAL ACC:",acc)
print("STATIC ACC:",img_acc)
print("VIDEO ACC:",video_acc)

