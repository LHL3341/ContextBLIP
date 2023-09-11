import json
import os

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from data.utils import pre_caption
import os,glob

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, laion_path, transform): 

        self.ann_pretrain = []
        for f in ann_file:
            print('loading '+f)
            ann = json.load(open(f,'r'))
            self.ann_pretrain += ann
        #self.ann_pretrain = self.ann_pretrain[:100]
        
        self.laion_path = laion_path
        if self.laion_path:
            self.laion_files = glob.glob(os.path.join(laion_path,'*.json'))

            print('loading '+self.laion_files[0])
            with open(self.laion_files[0],'r') as f:
                self.ann_laion = json.load(f)  

            self.annotation = self.ann_pretrain + self.ann_laion
        else:
            self.annotation = self.ann_pretrain
            
        self.transform = transform

    def reload_laion(self, epoch):
        n = epoch%len(self.laion_files)
        print('loading '+self.laion_files[n])
        with open(self.laion_files[n],'r') as f:
            self.ann_laion = json.load(f)      
        
        self.annotation = self.ann_pretrain + self.ann_laion    
        
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]   
        if 'visual-genome' in ann['image']:
            img_path = 'pretrain_data/vl_pair/visual-genome/VG_100K/'+ann['image'].removeprefix('/export/share/datasets/vision/visual-genome/image/')
            if not os.path.exists(img_path):
                img_path = 'pretrain_data/vl_pair/visual-genome/VG_100K_2/'+ann['image'].removeprefix('/export/share/datasets/vision/visual-genome/image/')
        else:
            img_path = 'pretrain_data/vl_pair/coco/'+ann['image'].removeprefix('/export/share/datasets/vision/coco/images/')

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        caption = pre_caption(ann['caption'],30)
        
        return image, caption