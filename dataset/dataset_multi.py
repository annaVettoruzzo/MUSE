import torch, os, random, numpy as np
import torchvision.datasets as datasets
from torchvision import transforms
from collections import defaultdict
from utils import DEVICE
        
class MultiDataset:
    def __init__(self, name, split="train", resize=None, print_output=False):
        self.dataset_name =  name
        self.split = split
        self.root_dir = './dataset/data/meta-dataset/{}/{}/'.format(self.dataset_name, split)
        
        if resize is not None:
            trs = transforms.Compose([ transforms.Resize((resize)), transforms.ToTensor() ])
        else: 
            trs = transforms.ToTensor()
            
        self.data = datasets.ImageFolder(self.root_dir, transform=trs)
        
        # Group images by their class
        self.ds_dict = defaultdict(list)
        for img, c in self.data:
            self.ds_dict[c].append(img.numpy())
       
        self.classes = self.data.classes
        self.num_classes = len(self.classes)
        
        if print_output: 
            print("Dataset Info")
            print(f"Number of classes: {self.num_classes}")
            print(f"Image size : {img.shape}")
                 
    def __len__(self):
        """Returns number of examples in the dataset"""
        return len(self.data) 
    