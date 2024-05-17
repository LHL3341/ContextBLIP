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


def count_sentences_by_clause_number(json_file_path):
    # 使用defaultdict初始化一个计数器，以便自动为新键分配默认值0
    clause_counts = defaultdict(int)

    # 读取JSON文件
    with open(json_file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    # 遍历JSON数据，统计每个复合句分解成的子句数量
    for sentences in json_data.values():
        clause_counts[len(sentences)] += 1

    return dict(clause_counts)


# 调用函数并打印结果
sentence_counts = count_sentences_by_clause_number("dataset/valid_split.json")
print(sentence_counts)

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
with open("analysis/manual_annotation_valid.yaml") as f:
    annotations = yaml.load(f.read(), Loader=yaml.FullLoader)
phenomenons = defaultdict(int)
correct_cases = defaultdict(int)
acc_each_case = defaultdict(float)
whole_name = {
    "s": "Spatial Relations(空间关系)",
    "col": "Colors(颜色)",
    "q": "Quantities(量词)",
    "n": "Nuances(细微、偏僻部分)",
    "con": "Context(上下文)",
    "img": "Unknown Type",
    "neg": "Negation(否定)",
    "v": "Visibility(不可见、遮挡)",
    "cor": "Co-reference(指代)",
    "t": "Temporal(时序)",
    "": "Others",
}
for key, value in annotations.items():
    for k, v in value.items():
        annotation = v["annotation"].split(",")
        for case in annotation:
            phenomenons[case] += 1
print("phenomenons:", phenomenons)


img_dirs = args.imgs_path
split_data = json.load(open("dataset/valid_split.json", "r"))
valid_data = json.load(open(args.valid_descr_path, "r"))
valid = []
for img_dir, data in valid_data.items():
    for img_idx, text in data.items():
        length = len(split_data[text])
        valid.append((img_dir, img_idx, text, length))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE USED: {device}")


bert_config = json.load(open("vilbert-and-bert-config.json", "r"))
model = ContextualBLIP(bert_config, args).cuda()
checkpoint = torch.load(args.finetuned_checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)


correct = 0
total = 0
video_correct = 0
video_total = 0
img_correct = 0
img_total = 0
correct_of_lengths = defaultdict(int)
model.cuda()
model.eval()
for img_dir, img_idx, text, length in tqdm.tqdm(valid):
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
        correct_of_lengths[length] += 1
        if img_dir in annotations.keys():
            if str(img_idx) in annotations[img_dir].keys():
                phenomenon = annotations[img_dir][str(img_idx)]["annotation"].split(",")
                for case in phenomenon:
                    correct_cases[case] += 1
    if "open-images" in img_dir:
        img_total += 1
        if img_idx == pred:
            img_correct += 1
    else:
        video_total += 1
        if img_idx == pred:
            video_correct += 1
    total += 1
print(len(valid))
print(correct_of_lengths)
acc = round(correct / total, 4)
video_acc = round(video_correct / video_total, 4)
img_acc = round(img_correct / img_total, 4)
print("TOTAL ACC:", acc)
print("STATIC ACC:", img_acc)
print("VIDEO ACC:", video_acc)
for case in correct_cases.keys():
    acc_each_case[whole_name[case]] = round(correct_cases[case] / phenomenons[case], 4)
print("acc in different cases:", acc_each_case)
