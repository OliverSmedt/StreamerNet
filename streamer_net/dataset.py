from operator import is_none

from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import numpy as np
import cv2
from torchvision.transforms import ToTensor

class PictureDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform = None):
        """

            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if is_none(transform):
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.root_dir = root_dir
        self.n_categories = sum(1 for line in open(os.path.join(root_dir, "categories.txt")))
        category_sums = np.zeros(self.n_categories + 1)
        for category in range(self.n_categories):
            category_sums[category+1] = len(os.listdir(self.root_dir + '/' + str(category)))
        category_sums = np.cumsum(category_sums)
        self.category_sums = category_sums.astype(int)

    def __len__(self):
        return self.category_sums[-1]

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        category = int(np.max(np.argwhere((self.category_sums-1<idx))))
        number = int(idx - self.category_sums[category])

        img_name = os.path.join(self.root_dir, str(category), str(number)+".png")
        image = cv2.imread(img_name)

        if self.transform:
            image = self.transform(image)

        sample = (image, category)


        return sample

if __name__ == '__main__':
    dataset = PictureDataset("emote_training_data")
    print(dataset.category_sums)