import warnings
warnings.filterwarnings("ignore")
import json
import os
import random
import torch
import tqdm
from PIL import Image
from pathlib import Path
from collections import defaultdict
import sys
sys.path[0] = os.path.abspath('.')
import torchvision.transforms as transforms
import argparse
from torchvision.transforms.functional import InterpolationMode
from models.contextual import Adapter_BLIP
from extras_ import convert_sents_to_features, BertLayer
from finetune_notf import ContextualBLIP as ContextualBLIP2
from finetune_context import ContextualBLIP
from finetune_36 import ContextualBLIP as ContextualBLIP1

from volta_src.config import BertConfig
from utils import pre_caption
import yaml
import torch.nn as nn


random.seed(10)
torch.manual_seed(10)

parser = argparse.ArgumentParser()
parser.add_argument("--finetuned_checkpoint_path", default='0.3849_context.pt')

parser.add_argument('--descr_path', type=str, default='video/2/')
parser.add_argument("--add_input", action="store_true",default=True)
parser.add_argument("--positional", action="store_true",default=True)
parser.add_argument("--transformer_layers", default=2, type=int)
parser.add_argument("--all_pos", action="store_true",default=False)
parser.add_argument("-a", "--activation", default='relu')
parser.add_argument("-s", "--logit_scale", default=1000)
parser.add_argument("--frozen_blip", action="store_true",default=True)

args = parser.parse_args()
    
img_dirs = args.descr_path


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'DEVICE USED: {device}')

bert_config = json.load(open('vilbert-and-bert-config.json', 'r'))
model = ContextualBLIP(bert_config, args,pretrain=False).cuda()
checkpoint = torch.load(args.finetuned_checkpoint_path)
msg = model.load_state_dict(checkpoint['model_state_dict'],strict= False)
print(msg)



correct = 0
total = 0
video_correct = 0
video_total = 0
img_correct = 0
img_total = 0
ranks = defaultdict(int)
model.cuda()
model.eval()
for img_dir in os.listdir(img_dirs):
    if len(img_dir) ==3:
        with open(img_dirs+img_dir+'/data.json','r')as f:
            data = json.load(f)
        img_dir_, img_idx, text = data["source"],data["target"],data["question"]
        text = [pre_caption(text)]
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
        if "open-images" in str(img_dir_):
            pos_mask = torch.zeros((4, 1)).cuda()
        else:
            pos_mask = torch.ones((4, 1)).cuda()
        with torch.no_grad():
            logits = model(image, text,pos_mask).squeeze()
        pred = torch.argmax(logits).squeeze()
        if img_idx == pred:
            correct += 1
       
        total += 1

print(correct/total)