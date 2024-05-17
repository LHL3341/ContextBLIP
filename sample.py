import numpy as np
import json
import random
import shutil
import os


def process(data_path="video/5/"):
    with open(data_path + "data.json", "r") as f:
        data_dict = json.load(f)
        i = 0
        for item in data_dict:
            info = {
                "source": item[0],
                "question": item[2],
                "target": item[1],
                "result": "",
            }
            pic_dict = os.path.join(data_path, item[0])
            target_pic = item[1]
            for dirpath, dirnames, filenames in os.walk(pic_dict):
                new_path = os.path.join(data_path, "processed", str(i).zfill(3))
                os.makedirs(new_path, exist_ok=True)
                filenames.sort(key=natural_sort_key)
                if target_pic == "0" or target_pic == "1":
                    for j in range(0, 4):
                        source_path = os.path.join(dirpath, filenames[j])
                        destination_path = os.path.join(new_path, str(j))
                        shutil.copy(source_path, new_path)
                elif target_pic == "9":
                    for j in range(6, 10):
                        source_path = os.path.join(dirpath, filenames[j])
                        destination_path = os.path.join(new_path, str(j))
                        shutil.copy(source_path, new_path)
                        info["target"] = "3"
                else:
                    for j in range(int(target_pic) - 2, int(target_pic) + 2):
                        source_path = os.path.join(dirpath, filenames[j])
                        destination_path = os.path.join(new_path, str(j))
                        shutil.copy(source_path, new_path)
                        info["target"] = "2"
                with open(
                    os.path.join(new_path, "data.json"), "w", encoding="utf-8"
                ) as info_file:
                    json.dump(info, info_file, ensure_ascii=False, indent=4)
                i = i + 1


def natural_sort_key(s):
    """Sort strings containing numbers correctly."""
    import re

    # 从文件名中提取数字
    match = re.search("img(\d+)", s)
    if match:
        return int(match.group(1))
    return 0


valid_data = json.load(open("dataset/valid_data.json", "r"))
static = []
video = []
for img_dir, data in valid_data.items():
    for img_idx, text in data.items():
        if "open-images" in img_dir:
            static.append((img_dir, img_idx, text))
        else:
            video.append((img_dir, img_idx, text))

id = 1  # 1:1 2:10 3:100
random.seed(id)
random.shuffle(static)
random.shuffle(video)
max_num = 50


for img_dir, img_idx, text in static[:max_num]:
    if not os.path.exists(f"static/{id}/{img_dir}"):
        shutil.copytree(f"dataset/image-sets/{img_dir}", f"static/{id}/{img_dir}")


with open(f"static/{id}/data.json", "w") as f:
    json.dump(static[:max_num], f)


for img_dir, img_idx, text in video[:max_num]:
    if not os.path.exists(f"video/{id}/{img_dir}"):
        shutil.copytree(f"dataset/image-sets/{img_dir}", f"video/{id}/{img_dir}")


with open(f"video/{id}/data.json", "w") as f:
    json.dump(video[:max_num], f)

process(f"static/{id}/")
process(f"video/{id}/")
