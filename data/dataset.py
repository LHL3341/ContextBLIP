import json
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import Pad, Resize, ToTensor, Compose
from torchvision.transforms.functional import InterpolationMode
from transform.randaugment import RandomAugment
# from transformers import BertTokenizerFast
from PIL import Image
import json

def default_image_transform(img, img_size=224):
    img = img.convert('RGB')
    w, h = img.size
    img = Compose([
        Pad([0, (w-h)//2] if w>h else [(h-w)//2, 0]), 
        Resize([img_size, img_size]), 
        ToTensor()
    ])(img)
    return img


def default_text_transform(text, tokenizer, max_length=77):
    inputs = tokenizer(
        text,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors='np'
    )
    return inputs

transform_train = transforms.Compose([                        
        transforms.RandomResizedCrop(224,scale=(0.2, 1.0),interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]) 
transform_1 = transforms.Compose([                        
        transforms.RandomResizedCrop(224,scale=(0.2, 1.0),interpolation=InterpolationMode.BICUBIC),
        RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]) 
transform_2 = transforms.Compose([                        
        transforms.RandomResizedCrop(224,scale=(0.5, 1.0),interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]) 

transform_3 = transforms.Compose([                        
        transforms.RandomResizedCrop(224,scale=(0.2, 1.0),interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2,10,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]) 

transform_test = transforms.Compose([
            transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])  
class ImageCoDeDataset(Dataset):

    def __init__(self, data_dir, split, split_text=False,image_transform=None, text_transform=None, video_only=False):
        super().__init__()
        assert split in ['train', 'valid']
    
        if image_transform is not None:
            self.image_transform = image_transform
        else:
            self.image_transform = default_image_transform
        if split == 'train':
            self.image_transform = transform_train
            if image_transform == 1:
                self.image_transform = transform_1
            if image_transform == 2:
                self.image_transform = transform_2
            if image_transform == 3:
                self.image_transform = transform_3
            
            
        if split == 'valid':
            self.image_transform = transform_test
        
        # if text_transform is not None:
        self.text_transform = text_transform
        # else:
        #     self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        #     self.text_transform = partial(default_text_transform, tokenizer=self.tokenizer)

        self.data = self.load_data(Path(data_dir),Path(data_dir,'image-sets'), split, video_only,split_text=split_text)

    @staticmethod
    def load_data(data_dir, img_path, split, video_only=False,split_text=False):
        if split_text:
            with open(data_dir / f'{split}_split.json') as f:
                split_data = json.load(f)

        with open(data_dir / f'{split}_data.json') as f:
            json_file = json.load(f)

        dataset = []
        for img_dir, data in json_file.items():
            img_files = list((Path(f'{img_path}/{img_dir}')).glob('*.jpg'))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for img_idx, text in data.items():
                static = 'open-images' in img_dir
                if video_only:
                    if not static:
                        dataset.append((img_dir, img_files, int(img_idx), text))
                        if split_text and text in split_data:
                            dataset += [(img_dir, img_files,int(img_idx), sentence) for sentence in split_data[text]]
                else:
                    dataset.append((img_dir, img_files, int(img_idx), text))
                    if split_text and text in split_data:
                        dataset += [(img_dir, img_files,int(img_idx), sentence) for sentence in split_data[text]]
        return dataset
    
    def __getitem__(self, idx):
        img_dir, img_files, img_idx, text = self.data[idx]
        
        images = [self.image_transform(Image.open(img_file).convert('RGB')) for img_file in img_files]
        img = torch.stack(images, dim=0)
        
        txt = text#self.text_transform(text)
        is_video = torch.tensor(1 if 'open-images' not in img_dir else 0)
        
        return img, txt, img_idx, is_video
    
    def __len__(self):
        return len(self.data)
    