import csv
import os
from PIL import Image
import torch
from clip import load
import clip
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import csv
import json
from utils import pre_caption
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
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
# 导入blip
from models.blip import load_checkpoint, BLIP_Base
from utils import pre_caption
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
# os.chdir(os.path.abspath('..'))
from finetune_context import ContextualBLIP
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from utils import pre_caption


def benchmark_model(model_name, benchmark_dir, device="cpu"):
    # model, preprocess = load(model_name, device=device)
    bert_config = json.load(open("vilbert-and-bert-config.json", "r"))
    model = ContextualBLIP(bert_config, args, test=True).cuda()
    checkpoint = torch.load("Models/finetune/0.25_2d_42/10/0.3849_context.pt")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
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
    image_dir = os.path.join(benchmark_dir, "MLLM_VLM Images")
    csv_file = os.path.join(benchmark_dir, "Questions.csv")

    csv_outfile = open("output.csv", "w", newline="")
    csv_writer = csv.writer(csv_outfile)
    csv_writer.writerow(
        ["qid1", "qid2", "pred1", "pred2", "gt1", "gt2", "q1score", "q2score"]
    )  # header

    categories = [
        "Orientation and Direction",
        "Presence of Specific Features",
        "State and Condition",
        "Quantity and Count",
        "Positional and Relational Context",
        "Color and Appearance",
        "Structural Characteristics",
        "Texts",
        "Viewpoint and Perspective",
    ]

    pair_accuracies = {category: 0 for category in categories}
    num_pairs = 0

    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in enumerate(reader):
            qid1, qtype1, statement1 = row

            # Get next row for the pair
            row = next(reader, None)
            if not row:
                break
            qid2, qtype2, statement2 = row

            qid1, qid2 = int(qid1), int(qid2)

            img1 = Image.open(os.path.join(image_dir, qtype1, f"{qid1}.jpg"))
            img2 = Image.open(os.path.join(image_dir, qtype1, f"{qid2}.jpg"))

            # text1 = 'a photo of ' + statement1
            # text2 = 'a photo of ' + statement2

            # text1 = clip.tokenize([text1]).to(device)
            # text2 = clip.tokenize([text2]).to(device)
            text1 = ['a photo of ' + statement1]
            text2 = ['a photo of ' + statement2]

            img1 = preprocess(img1).unsqueeze(0).to(device)
            img2 = preprocess(img2).unsqueeze(0).to(device)
            imgs = torch.cat((img1, img2), dim=0)

            with torch.no_grad():
                pos_mask = torch.zeros((2, 1)).cuda()
                itm_score1 = model(imgs, text1, pos_mask).squeeze()
                # pred1 = torch.argmax(itm_score1).squeeze()

                itm_score2 = model(imgs, text2, pos_mask).squeeze()
                # pred2 = torch.argmax(itm_score2).squeeze()

                # logits_per_image1, logits_per_text1 = model(imgs, text1)
                # logits_per_image2, logits_per_text2 = model(imgs, text2)

                # probs1 = logits_per_text1.softmax(dim=-1).cpu().numpy()
                # probs2 = logits_per_text2.softmax(dim=-1).cpu().numpy()

            img1_score1 = itm_score1[0]  # probs1[0][0]
            img1_score2 = itm_score2[0]

            pred1 = (
                "img1" if img1_score1 > itm_score1[1] else "img2"
            )  # if img1_score1 > 0.5 else "img2"
            pred2 = (
                "img1" if img1_score2 > itm_score2[1] else "img2"
            )  # if img1_score2 > 0.5 else "img2"

            gt1 = "img1" if qid1 % 2 == 1 else "img2"
            gt2 = "img1" if qid2 % 2 == 1 else "img2"

            csv_writer.writerow(
                [qid1, qid2, pred1, pred2, gt1, gt2, img1_score1, img1_score2]
            )

            current_category = categories[num_pairs // 15]
            if pred1 == gt1 and pred2 == gt2:
                pair_accuracies[current_category] += 1
            num_pairs += 1

        csv_outfile.close()

    # Calculate percentage accuracies
    total_acc = 0
    for category in pair_accuracies:
        pair_accuracies[category] = (
            pair_accuracies[category] / (num_pairs // len(categories))
        ) * 100
        total_acc += pair_accuracies[category]
    pair_accuracies["Average"] = total_acc / 9
    return pair_accuracies


parser = argparse.ArgumentParser(description="Process a directory path.")

# Adding an argument for the directory path
parser.add_argument("--directory", type=str, default="MMVP/MMVP_VLM")

# Parsing the arguments
args = parser.parse_args()

# OpenAI models
models = ["ContextBLIP"]

results_openai = {
    f"openai-{model}": benchmark_model(model, args.directory,device='cuda') for model in models
}


# Merge results
results = {**results_openai}

# Convert results to format suitable for star plot
categories = results[list(results.keys())[0]].keys()
data = {"Categories": list(categories)}
for model in list(results_openai.keys()):
    data[model] = [results[model][category] for category in categories]

print(results)
