from data.dataset import ImageCoDeDataset
import argparse
from torchvision.transforms import transforms
import yaml
from transform.randaugment import RandomAugment
from torchvision.transforms.functional import InterpolationMode
from data import create_dataset
from models.mask_adapter import Adapter_BLIP
from torch.utils.data import DataLoader
import json
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='configs/pretrain.yaml')
parser.add_argument('--output_dir', default='../output/Pretrain')  
parser.add_argument('--checkpoint', default='')    
parser.add_argument('--evaluate', action='store_true',default=False)    
parser.add_argument('--device', default='cuda')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--distributed', default=True, type=bool)
args = parser.parse_args()
config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
transform_train = transforms.Compose([                        
        transforms.RandomResizedCrop(config['image_size'],scale=(0.5, 1.0),interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                          'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
        transforms.ToTensor(),
        normalize,
    ])        
checkpoint_path = ''
dataset_valid = create_dataset('imagecode', config)
data_loader = DataLoader(dataset_valid,batch_size=1,num_workers=4)
model = Adapter_BLIP()
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint,strict=False)

total = 0
img_total = 0
vid_total = 0
correct = 0
img_correct =0 
vid_correct =0 

for i, image, text, img_idx, is_video in enumerate(data_loader):
    logits = model.inference(image,text)
    prediction = logits.argmax()

    total += 1
    correct += (prediction == img_idx)

    if not is_video:
        img_total += 1
        if prediction == img_idx:
            img_correct += 1
    else:
        vid_total += 1
        if prediction == img_idx:
            vid_correct += 1       

print('OVERALL ACC: ' + str(round(correct/total,4)))
print('VIDEO ACC: ' + str(round(vid_correct/vid_total,4)))
print('IMG ACC: ' + str(round(img_correct/img_total,4)))
