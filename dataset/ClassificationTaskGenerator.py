import torch, numpy as np, random
import torchvision.datasets as datasets
from torchvision import transforms
from collections import defaultdict
from .dataset_multi import MultiDataset
from .omniglot import OmniglotDataset
from .cifarFS import CIFARFS
from .miniimagenet import MiniImageNet
from utils import DEVICE

'''
    Download meta-dataset from https://github.com/Carbonaraa/TSA-MAML.
'''
class ClassificationTaskGenerator:
    def __init__(self, n, k, q, modes, split="train", resize=None):
        self.n = n          # n_ways: number of classes
        self.k = k          # k_shot: number of examples in the support set
        self.q = q          # Number of examples in the query set
        self.resize = resize
        self.modes = modes
        
        # self.all_datasets = [MultiDataset("FGVC_Aircraft", split, self.resize), MultiDataset("FGVCx_Fungi", split, self.resize), MultiDataset("CUB_Bird", split, self.resize)]
        self.all_datasets={}
        for mode_nbr in self.modes:
            if mode_nbr == 0: self.all_datasets[mode_nbr] = MultiDataset("FGVC_Aircraft", split, self.resize)
            elif mode_nbr == 1: self.all_datasets[mode_nbr] = MultiDataset("FGVCx_Fungi", split, self.resize)
            elif mode_nbr == 2: self.all_datasets[mode_nbr] = MultiDataset("CUB_Bird", split, self.resize)
            elif mode_nbr == 3: self.all_datasets[mode_nbr] = OmniglotDataset(split, self.resize)
            elif mode_nbr == 4: self.all_datasets[mode_nbr] = CIFARFS(split, self.resize)
            elif mode_nbr == 5: self.all_datasets[mode_nbr] = MiniImageNet(split, self.resize)

    # -------------------------------------------------------------------
    # Sample a support set (n*k examples) and a query set (n*q examples) from the same task
    def batch(self):
        # Randomly select the sub-dataset
        task_mode_label = random.choice(list(self.all_datasets.keys()))
        data = self.all_datasets[task_mode_label]
        
        # Randomly select the classes
        classes = list(data.ds_dict.keys())       # All possible classes
        classes = random.sample(classes, self.n)  # Randomly select n classes
        
        # Randomly map each selected class to a label in {0, ..., n-1}
        labels = random.sample(range(self.n), self.n)
        label_map = dict(zip(classes, labels))
        # Randomly select k support examples and q query examples from each of the selected classes
        X_sp, y_sp, X_qr, y_qr = [], [], [], []
        for c in classes:
            images = random.sample(data.ds_dict[c], self.k + self.q)
            X_sp += images[:self.k]
            y_sp += [label_map[c] for _ in range(self.k)]
            X_qr += images[self.k:]
            y_qr += [label_map[c] for _ in range(self.q)]
        
        # Transform these lists to appropriate tensors and return them
        X_sp, y_sp, X_qr, y_qr = [torch.from_numpy(np.array(lst)).to(DEVICE).float() for lst in [X_sp, y_sp, X_qr, y_qr]]
        y_sp, y_qr = y_sp.long(), y_qr.long()
        
        return X_sp, y_sp, X_qr, y_qr, task_mode_label
    
    # -------------------------------------------------------------------
    # Generate a specific task mode
    def batch_task(self, task_mode_label):
        data = self.all_datasets[task_mode_label]
            
        # Randomly select the classes
        classes = list(data.ds_dict.keys())       # All possible classes
        classes = random.sample(classes, self.n)  # Randomly select n classes
        
        # Randomly map each selected class to a label in {0, ..., n-1}
        labels = random.sample(range(self.n), self.n)
        label_map = dict(zip(classes, labels))
        # Randomly select k support examples and q query examples from each of the selected classes
        X_sp, y_sp, X_qr, y_qr = [], [], [], []
        for c in classes:
            images = random.sample(data.ds_dict[c], self.k + self.q)
            X_sp += images[:self.k]
            y_sp += [label_map[c] for _ in range(self.k)]
            X_qr += images[self.k:]
            y_qr += [label_map[c] for _ in range(self.q)]
        
        # Transform these lists to appropriate tensors and return them
        X_sp, y_sp, X_qr, y_qr = [torch.from_numpy(np.array(lst)).to(DEVICE).float() for lst in [X_sp, y_sp, X_qr, y_qr]]
        y_sp, y_qr = y_sp.long(), y_qr.long()
        
        return X_sp, y_sp, X_qr, y_qr, task_mode_label
  