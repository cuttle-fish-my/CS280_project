import os
import pickle
import random

import torch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


class COCOAllDataset(Dataset):
    def __init__(self, resolution, root, filename_pickle, train=True):
        """
        :param resolution: resolution of the image
        :param root: root directory of the dataset
        :param category: category name
        :param category_id: category id
        :param filename_pickle: pickle file containing dict mapping category name to list of image ids
        :param num_support: number of support images per category
        :param train: if True, use training set, else use validation set
        """
        super().__init__()
        self.resolution = resolution
        self.root = root
        self.train = train
        self.all_categories = pickle.load(open(filename_pickle, 'rb'))
        self.all_filenames = []
        for k in self.all_categories:
            self.all_filenames += self.all_categories[k]
            # print(len(self.all_filename))

    def __len__(self):
        return len(self.all_filenames)

    def __getitem__(self, idx):
        img, label = self._load_img_label(self.all_filenames[idx])
        # print(np.max(label+1))
        return {
            "img": img,
            "label": label,
            "idx": idx,
        }

    def _load_img_label(self, path):
        img = Image.open(os.path.join(self.root, "train2017" if self.train else "val2017", path))
        label = Image.open(os.path.join(self.root,
                                        "annotations",
                                        "train2017" if self.train else "val2017",
                                        path.replace(".jpg", ".png")))
        img = img.resize((self.resolution, self.resolution))
        img = img.convert("RGB")
        label = label.resize((self.resolution, self.resolution), resample=Image.NEAREST)
        img = np.array(img).astype(np.float32) / 127.5 - 1
        label = np.array(label).astype(np.uint8)
        label += 1
        # label = (label == self.category_id).astype(np.float32)

        return img.transpose([2, 0, 1]), label



# if __name__ == "__main__":
#     file_list = pickle.load(open("../datasets/COCO/train_filenames_sieved.pkl", 'rb'))
#     coco = COCOCategoryLoaderDataset(128, "../datasets/COCO", "../datasets/COCO/train_filenames_sieved.pkl",
#                                      "../datasets/COCO/categories.pkl", batch_size=10, shuffle=True, num_workers=0,
#                                      num_support=20)
#     dataloader = COCOCategoryLoaderDataLoader(dataset=coco, num_workers=0)
#     for batch in dataloader:
#         print(batch['category'])
#         print(batch['idx'])
#     # print(1)
