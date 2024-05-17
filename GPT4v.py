import base64
import requests
from PIL import Image
from io import BytesIO
import os
import json
from openai import OpenAI

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 指定文件夹路径
folder_path = "./static/1"
# JSON 结果
result = {}


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        image = Image.open(image_file)
        image = image.resize((224, 224))  # 缩放图像到指定大小
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


# 遍历文件夹中的每一个子文件夹
for subdir in os.listdir(folder_path):
    subdir_path = os.path.join(folder_path, subdir)
    if os.path.isdir(subdir_path):
        # 图像结果列表
        encoded_images = []
        # 遍历子文件夹中的每一张图像
        for i in range(10):
            image_filename = f"img{i}.jpg"  # 图像文件名
            image_path = os.path.join(subdir_path, image_filename)
            # 调用处理图像的函数
            image_result = encode_image(
                image_path
            )  # 这里假设处理函数返回一个结果字符串
            encoded_images.append(image_result)

        client = OpenAI(api_key="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")
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Choose which image matches the description best:A picture of a chicken with dark feathers.  You can only see one eye of the chicken and it is facing to the left of the image which has a grey background.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_images[0]}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_images[1]}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_images[2]}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_images[3]}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_images[4]}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_images[5]}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_images[6]}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_images[7]}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_images[8]}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_images[9]}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        # print(response.choices[0])

        # 提取 response.choices[0] 的内容
        choice = response.choices[0]
        result_dict = {
            "message": {"role": choice.message.role, "content": choice.message.content},
            "finish_reason": choice.finish_reason,
            "index": choice.index,
        }

        # 将字典写入 JSON 文件
        with open("result.json", "w") as json_file:
            json.dump(result_dict, json_file, indent=4)

        print("JSON 文件已生成。")
