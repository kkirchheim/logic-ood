from torchvision.datasets import VisionDataset
from os.path import join
from os import listdir
from PIL import Image
import torch 

class PrimateNet(VisionDataset):
            
    data = {
            'n02494079': ('squirrel_monkey', 0, 0, 0, 0, 0, 0, 1, 1),
            'n02492660': ('howler_monkey', 1, 0, 0, 0, 0, 0, 1, 1),
            'n02486410': ('baboon', 2, 0, 0, 0, 0, 1, 0, 1),
            'n02488702': ('colobus', 3, 0, 0, 0, 0, 1, 0, 1),
            'n02481823': ('chimpanzee', 4, 1, 0, 1, 0, 0, 0, 0),
            'n02492035': ('capuchin', 5, 0, 0, 0, 0, 0, 1, 1),
            'n02493509': ('titi', 6, 0, 0, 0, 0, 0, 1, 1),
            'n02493793': ('spider_monkey', 7, 0, 0, 0, 0, 0, 1, 1),
            'n02483362': ('gibbon', 8, 0, 1, 1, 0, 0, 0, 0),
            'n02488291': ('langur', 9, 0, 0, 0, 0, 1, 0, 1),
            'n02484975': ('guenon', 10, 0, 0, 0, 0, 1, 0, 1),
            'n02497673': ('madagascar_cat', 11, 0, 0, 0, 1, 0, 0, 0),
            'n02487347': ('macaque', 12, 0, 0, 0, 0, 1, 0, 1),
            'n02490219': ('marmoset', 13, 0, 0, 0, 0, 0, 1, 1),
            'n02480495': ('orangutan', 14, 1, 0, 1, 0, 0, 0, 0),
            'n02483708': ('siamang', 15, 0, 1, 1, 0, 0, 0, 0)
    }
    
    attributes = ["class", "great_ape", "lesser_ape", "ape", "lemur", "old_world_monkey", "new_world_monkey", "monkey"]
        
    def __init__(self, root, train=True, transform=None, target_transform=None):
        if train:
            root = join(root, "train")
        else:
            root = join(root, "val")
            
        super().__init__(root, transform=transform, target_transform=target_transform)


        self.imgs = []
        self.labels = []
        self._load()
    
    def __len__(self):
        return len(self.imgs)
    
    def _load(self) -> None:
        for l, d in enumerate(self.data.keys()):
            try:
                p = join(self.root, d)
                # print(f"{d} ({l}) {self.data[d][0]}  -> {len(listdir(p))} {self.data[d][1:]}")
                self.imgs += [join(p, i) for i in listdir(p)]
                self.labels += [torch.tensor(self.data[d][1:]) for i in listdir(p)]
            except FileNotFoundError:
                print(f"Could not find directory {p}")
                
        self.labels = torch.stack(self.labels)
        return 
    
    def __getitem__(self, index: int):
        x = self.imgs[index]
        y = self.labels[index]
        
        x = Image.open(x).convert("RGB")
        
        if self.transform:
            x = self.transform(x)
            
        if self.target_transform:
            y = self.target_transform(y)
    
        return x, y 
    
    

class PrimateNetOOD(VisionDataset):
    """
    Randomly selected OOD data from imagenet 
    """
            
    classes = {
        "n03126707": "crane",
        "n01883070": "wombat",
        "n01843383": "toucan",
        "n04070727": "refridgerator",
        "n04540053": "volleyball",
        "n01855672": "goose",
        "n01910747": "jellyfish",
        "n02317335": "starfish",
        "n04335435": "trolley",
        "n03045698": "cloak",
    }
        
    def __init__(self, root, train=True, transform=None, target_transform=None):
        if train:
            root = join(root, "train")
        else:
            root = join(root, "val")
            
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.imgs = []
        self.labels = []
        self._load()
    
    def __len__(self):
        return len(self.imgs)
    
    def _load(self) -> None:
        for l, d in enumerate(self.classes.keys()):
            try:
                p = join(self.root, d)
                # print(f"{d} ({l}) {self.classes[d]}  -> {len(listdir(p))}")
                self.imgs += [join(p, i) for i in listdir(p)]
                self.labels += [torch.tensor(-1) for i in listdir(p)]
            except FileNotFoundError:
                print(f"Could not find directory {p}")
                
        self.labels = torch.stack(self.labels)
        return 
    
    def __getitem__(self, index: int):
        x = self.imgs[index]
        y = self.labels[index]
        
        x = Image.open(x).convert("RGB")
        
        if self.transform:
            x = self.transform(x)
            
        if self.target_transform:
            y = self.target_transform(y)
    
        return x, y 
    
    
