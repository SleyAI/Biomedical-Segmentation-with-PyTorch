import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random


from .utils import *


class BreastCancerDataset(Dataset):
    def __init__(self, imagepath=None, labelpath=None, data_dir=None, transform=None):
        #  make sure label match with image
        self.data_dir = data_dir
        self.transform = transform
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)
        self.image = []
        self.label = []
        with open(imagepath, 'r') as f:
            for line in f:
                self.image.append(line.strip())
        with open(labelpath, 'r') as f:
            for line in f:
                self.label.append(line.strip())

        """make sure that images and labels are at the same indices"""
        self.image.sort()
        self.label.sort()

        """Shuffle images and labels equally"""
        c = list(zip(self.image, self.label))
        random.shuffle(c)
        self.image, self.label = zip(*c)

    def __getitem__(self, index):
        filename = self.image[index]
        filenameGt = self.label[index]

        with open(os.path.join(self.data_dir, filename), 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(os.path.join(self.data_dir, filenameGt), 'rb') as f:
            label = Image.open(f).convert('L')

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms

        if self.transform is not None:
            image = self.transform(image)

        random.seed(seed)  # apply this seed to target tranfsorms
        if self.transform is not None:
            label = self.transform(label)

        image = resize(image)
        label = resize(label)
        return image, label

    def __len__(self):
        return len(self.image)

