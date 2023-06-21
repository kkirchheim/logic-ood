from torch.utils.data import Dataset, DataLoader
from os.path import join
from os import listdir
import pandas as pd 
from PIL import Image
from torch.optim import SGD
import matplotlib.pyplot as plt 


class AwA(Dataset):
    """
    Animals with Attributes 2 dataset 
    """
    img_dir = "JPEGImages"
    
    def __init__(self, root, transform=None, full=True):
        self.root = join(root, "Animals_with_Attributes2") 
        self.tansform = transform 
        self.full = full
        
        with open(join(self.root, "classes.txt"), "r") as f:
            self.class_name_to_idx = {line.strip().split()[1]: n for n, line in enumerate(f.readlines(), 0)} 
            self.class_idx_to_name = {v: k for k, v in self.class_name_to_idx.items()}
            
        with open(join(self.root, "predicates.txt"), "r") as f:
            self.att_name_to_idx = {line.strip().split()[1]: n for n, line in enumerate(f.readlines(), 0)} 
            self.att_idx_to_name = {v: k for k, v in self.att_name_to_idx.items()}
            
        self.images = []
        self.labels = []
        
        for class_name in self.class_name_to_idx.keys():
            dir_name = join(self.root, AwA.img_dir, class_name)
            label = self.class_name_to_idx[class_name]
            tmp = [join(dir_name, f) for f in listdir(dir_name)]
            self.images += tmp
            self.labels += [label] * len(tmp)
            
        self.predicates = pd.read_csv(join(self.root, "predicate-matrix-binary.txt"), delimiter="\s+", header=None, engine="python").astype(float)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        y = self.labels[index]
        z = self.predicates.iloc[y].values
        
        x = Image.open(self.images[index])
        
        if self.tansform:
            x = self.tansform(x)
        
        if self.full:
            return x, y, z
        else:
            return x, y 
        
