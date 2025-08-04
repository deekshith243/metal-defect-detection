# src/dataset.py

import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image

class MetalDefectDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.image_infos = {img['id']: img for img in self.coco_data['images']}
        self.annotations = {img['id']: [] for img in self.coco_data['images']}
        for anno in self.coco_data['annotations']:
            self.annotations[anno['image_id']].append(anno)
            
        self.image_ids = list(self.image_infos.keys())

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_infos[image_id]
        file_name = image_info['file_name']
        img_path = os.path.join(self.images_dir, file_name)
        
        img = Image.open(img_path).convert("RGB")
        
        annos = self.annotations.get(image_id, [])
        
        boxes = []
        labels = []
        
        for anno in annos:
            x, y, w, h = anno['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(anno['category_id'])
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        image_id = torch.tensor([image_id])
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, target

    def __len__(self):
        return len(self.image_ids)

def get_transform():
    # We can add more transforms like RandomHorizontalFlip, etc.
    # For simplicity, we just convert the image to tensor here.
    return F.to_tensor