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


def benchmark_model(model_name, benchmark_dir, device="cpu"):
    # model, preprocess = load(model_name, device=device)
    bert_config = json.load(open('vilbert-and-bert-config.json', 'r'))
    model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain",
                                                                       device=device, is_eval=True)
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
            text1 = [text_processors["eval"](statement1)]
            text2 = [text_processors["eval"](statement2)]

            img1 = vis_processors["eval"](img1).unsqueeze(0).to(device)
            img2 = vis_processors["eval"](img2).unsqueeze(0).to(device)
            imgs = torch.cat((img1, img2), dim=0)

            with torch.no_grad():
                # pred1 = torch.argmax(itm_score1).squeeze()
                itm_output11 = model({"image": img1, "text_input": text1}, match_head="itm")
                itm_scores11 = torch.nn.functional.softmax(itm_output11, dim=1)
                itm_output21 = model({"image": img2, "text_input": text1}, match_head="itm")
                itm_scores21 = torch.nn.functional.softmax(itm_output21, dim=1)
                # pred2 = torch.argmax(itm_score2).squeeze()
                itm_output12 = model({"image": img1, "text_input": text2}, match_head="itm")
                itm_scores12 = torch.nn.functional.softmax(itm_output12, dim=1)
                itm_output22 = model({"image": img2, "text_input": text2}, match_head="itm")
                itm_scores22 = torch.nn.functional.softmax(itm_output22, dim=1)
                # logits_per_image1, logits_per_text1 = model(imgs, text1)
                # logits_per_image2, logits_per_text2 = model(imgs, text2)

                # probs1 = logits_per_text1.softmax(dim=-1).cpu().numpy()
                # probs2 = logits_per_text2.softmax(dim=-1).cpu().numpy()

            # img1_score1 = itm_scores1[0]  # probs1[0][0]
            # img1_score2 = itm_scores2[0]

            pred1 = (
                "img1" if itm_scores11[:, 1].item() > itm_scores12[:, 1].item() else "img2"
            )  # if img1_score1 > 0.5 else "img2"
            pred2 = (
                "img1" if itm_scores21[:, 1].item() > itm_scores22[:, 1].item() else "img2"
            )  # if img1_score2 > 0.5 else "img2"

            gt1 = "img1" if qid1 % 2 == 1 else "img2"
            gt2 = "img1" if qid2 % 2 == 1 else "img2"

            csv_writer.writerow(
                [qid1, qid2, pred1, pred2, gt1, gt2]
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
