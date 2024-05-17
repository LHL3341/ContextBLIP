# inspired from: https://github.com/openai/CLIP/issues/83
# https://github.com/openai/CLIP/issues/83
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

sys.path[0] = os.path.abspath(".")
import torchvision.transforms as transforms
import argparse
from torchvision.transforms.functional import InterpolationMode
from models.contextual import Adapter_BLIP
from extras_ import convert_sents_to_features, BertLayer

from finetune_context import ContextualBLIP

# from finetune_notf import ContextualBLIP
from volta_src.config import BertConfig
from utils import pre_caption
import yaml
import torch.nn as nn

random.seed(10)
torch.manual_seed(10)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--finetuned_checkpoint_path",
    default="Models/finetune/0.25_2d_42/10/0.3849_context.pt",
)

parser.add_argument("--valid_descr_path", type=str, default="dataset/valid_data.json")
parser.add_argument("--train_descr_path", type=str, default="dataset/train_data.json")
parser.add_argument("--imgs_path", type=str, default="dataset/image-sets")
parser.add_argument("--add_input", action="store_true", default=True)
parser.add_argument("--positional", action="store_true", default=True)
parser.add_argument("--transformer_layers", default=2, type=int)
parser.add_argument("--all_pos", action="store_true", default=False)
parser.add_argument("-a", "--activation", default="relu")
parser.add_argument("-s", "--logit_scale", default=1000)
parser.add_argument("--frozen_blip", action="store_true", default=True)

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE USED: {device}")

bert_config = json.load(open("vilbert-and-bert-config.json", "r"))
model = ContextualBLIP(bert_config, args).cuda()
checkpoint = torch.load(args.finetuned_checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)

data_path = "static/1/"
dataset = json.load(open(data_path + "data.json", "r"))

correct = 0
total = 0
video_correct = 0
video_total = 0
img_correct = 0
img_total = 0
ranks = defaultdict(int)
model.cuda()
model.eval()
img_dirs = data_path
for img_dir, img_idx, text in tqdm.tqdm(dataset):
    text = [pre_caption(text)]
    img_idx = int(img_idx)
    img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
    img_files = sorted(
        img_files, key=lambda x: int(str(x).split("/")[-1].split(".")[0][3:])
    )
    images = [Image.open(photo_file).convert("RGB") for photo_file in img_files]
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    images = [preprocess(image) for image in images]
    image = torch.stack(images).to(device)
    if "open-images" in str(img_dir):
        pos_mask = torch.zeros((10, 1)).cuda()
    else:
        pos_mask = torch.ones((10, 1)).cuda()
    with torch.no_grad():
        logits = model(image, text, pos_mask).squeeze()
    pred = torch.argmax(logits).squeeze()
    if img_idx == pred:
        correct += 1
    if "open-images" in img_dir:
        img_total += 1
        if img_idx == pred:
            img_correct += 1
    else:
        video_total += 1
        if img_idx == pred:
            video_correct += 1
    total += 1
print(ranks)
acc = round(correct / total, 4)
if img_total == 0:
    video_acc = round(video_correct / video_total, 4)
    print("VIDEO ACC:", video_acc)
else:
    img_acc = round(img_correct / img_total, 4)
    print("STATIC ACC:", img_acc)
print("TOTAL ACC:", acc)