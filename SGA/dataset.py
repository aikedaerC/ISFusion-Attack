import json 
import os
import re

import torch 
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import copy
import cv2
from torchvision import transforms

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

class paired_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.ori_size = (0,0)
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []

        self.txt2img = {}
        self.img2txt = {}
        self.model = YOLO("yolov8n.pt")

        txt_id = 0
        for i, ann in enumerate(self.ann):
            self.img2txt[i] = []
            self.image.append(ann['image']) 
            for j, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.txt2img[txt_id] = i
                self.img2txt[i].append(txt_id)
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])

        cv2image = cv2.imread(image_path)
        ori_image = copy.deepcopy(cv2image)
        R = copy.deepcopy(cv2image[:,:,2])
        G = copy.deepcopy(cv2image[:,:,1])
        B = copy.deepcopy(cv2image[:,:,0])
        # 1. BGR -> GRB
        cv2image[:,:,0] = G 
        cv2image[:,:,1] = R 
        cv2image[:,:,2] = B 
        
        image = Image.fromarray(cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB))
        self.ori_size = image.size
        image = self.transform(image)

        obj_patch = {"patch": [], "patch_size": []}
        results = self.model.predict(cv2image, show=False)
        boxes = results[0].boxes.xyxy.cpu().tolist()
        topil = transforms.ToPILImage()
        if boxes is not None:
            for box in boxes: 
                p = cv2image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                patch_tensor = self.transform(topil(p))
                obj_patch["patch"].append(patch_tensor)
                obj_patch["patch_size"].append(p.shape[:-1])
        

        text_ids =  self.img2txt[index]
        texts = [self.text[i] for i in self.img2txt[index]]
        fname = self.image[index]
        return image, ori_image, texts, index, text_ids, fname, self.ori_size, boxes, obj_patch

    def collate_fn(self, batch):
        imgs, ori_image, txt_groups, img_ids, text_ids_groups, fname, ori_size, boxes, obj_patch = list(zip(*batch))        
        imgs = torch.stack(imgs, 0)
        return imgs, ori_image, txt_groups, list(img_ids), text_ids_groups, fname, ori_size, boxes, obj_patch

import pandas as pd

class paired_dataset2(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = pd.read_csv(ann_file)
        # import pdb;pdb.set_trace()
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = self.ann['caption']
        self.image = self.ann['image']


    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        texts = pre_caption(self.text[index], self.max_words)
        text_ids = None
        return image, texts, index, text_ids

    def collate_fn(self, batch):
        imgs, txt_groups, img_ids, text_ids_groups = list(zip(*batch))        
        imgs = torch.stack(imgs, 0)
        return imgs, txt_groups, list(img_ids), text_ids_groups


if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from PIL import Image
    config = {
        "image_res": 299,
        "test_file": "/workspace/data/Flickr8k/captions.txt",
        "image_root": "/workspace/data/Flickr8k/Images/",
    }
    s_test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),        
    ])

    test_dataset = paired_dataset2(config['test_file'], s_test_transform, config['image_root'])
    test_loader = DataLoader(test_dataset, batch_size=8,
                             num_workers=0, collate_fn=test_dataset.collate_fn)
    for batch_idx, (images, texts_group, images_ids, text_ids_groups) in enumerate(test_loader):
        print(images_ids)
        break
