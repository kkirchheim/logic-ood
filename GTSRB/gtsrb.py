"""
GTSRB dataset as downloaded from kaggle, with an additional labels.txt
"""
from torch.utils.data import Dataset, DataLoader
from os.path import join
import pandas as pd
from PIL import Image
from torch.optim import SGD
import seaborn as sb
import numpy as np


class GTSRB(Dataset):
    """ """

    shape_to_name = {0: "triangle", 1: "circle", 2: "square", 3: "octagon", 4: "inv-triange"}

    color_to_name = {
        0: "red",
        1: "blue",
        2: "yellow",
        3: "white",
    }

    def __init__(self, root, train=True, transforms=None, target_transform=None, transform=None):
        self.root = join(root, "GTSRB")
        self.meta_csv = pd.read_csv(join(self.root, "Meta.csv"))
        self.class_to_color = {}
        self.class_to_shape = {}
        for idx, (clazz, shape, color) in self.meta_csv[
            ["ClassId", "ShapeId", "ColorId"]
        ].iterrows():
            self.class_to_color[clazz] = color
            self.class_to_shape[clazz] = shape

        self.data = pd.read_csv(join(self.root, f"{'Train' if train else 'Test'}.csv"))
        self.paths = list(self.data["Path"])
        self.labels = list(self.data["ClassId"])
        self.transform = transform
        self.transforms = transforms
        self.target_transform = target_transform

        with open(join(self.root, "labels.txt"), "r") as f:
            label_lines = [l.strip().replace("'", "") for l in f.readlines()]

        self.class_to_name = {n: name for n, name in enumerate(label_lines, 0)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        color = self.class_to_color[label]
        shape = self.class_to_shape[label]
        path = join(self.root, self.paths[index])

        y = np.array([label, color, shape])
        x = Image.open(path).convert("RGB")

        if self.transforms:
            x = self.transforms(x)

        if self.target_transform:
            y = self.target_transform(y)

        if self.transform:
            x, y = self.transform(x, y)

        return x, y
