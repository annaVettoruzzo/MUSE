import torch, random
import torchvision as tv
from torchvision import transforms
import numpy as np
from collections import defaultdict


class OmniglotDataset:
    def __init__(self, split="train", resize=None):
        self.resize = resize
        
        if split=="train": background = True
        else: background = False
        
        # Download Omniglot. background=True (or False) loads the set of training (or evaluation) alphabets
        if resize is not None:
            trs = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1) ), transforms.Resize((resize))])
        else: 
            trs = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
                                     
        self.data = tv.datasets.Omniglot("./dataset/data", background, transform=trs, download=True) 
        ds = list(self.data)
        print(len(ds))
        
        # Group images by their class
        self.ds_dict = defaultdict(list)
        for img, c in ds:
            self.ds_dict[c].append(img.numpy())
    
    def __len__(self):
        """Returns number of examples in the dataset"""
        return len(self.data) 