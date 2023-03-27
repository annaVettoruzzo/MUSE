import torch, os, random, numpy as np
import torchvision.datasets as datasets
from torchvision import transforms
from collections import defaultdict
from utils import DEVICE

'''
    Download file from: https://github.com/yaoyao-liu/mini-imagenet-tools.
    Use the same split used by Ravi & Larochelle (2017).  
'''
class MiniImageNet:
    def __init__(self, split="train", resize=None):
        self.resize = resize
        
        self.root_dir = f"./dataset/data/miniimagenet/{split}/"            
        if resize is not None:
            trs = transforms.Compose([ transforms.Resize((resize)), transforms.ToTensor() ])
        else: 
            trs = transforms.ToTensor()
        data = datasets.ImageFolder(self.root_dir, transform=trs)
        print(len(data))
        
        # Group images by their class
        self.ds_dict = defaultdict(list)
        for img, c in data:
            self.ds_dict[c].append(img.numpy())
                 
    def __len__(self):
        """Returns number of examples in the dataset"""
        return len(self.data) 
    