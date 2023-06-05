import os
import pickle
import random

import torch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


class COCOCategoryLoaderDataset(Dataset):
    def __init__(self,
                 resolution,
                 root,
                 filename_pickle,
                 label_id_map,
                 num_support=5,
                 train=True,
                 batch_size=1,
                 shuffle=True,
                 num_workers=1,
                 drop_last=True
                 ):
        """
        :param resolution: resolution of the image
        :param root: root directory of the dataset
        :param filename_pickle: pickle file containing dict mapping category name to list of image ids
        :param label_id_map: pickle file containing dict mapping category name to category id
        :param num_support: number of support images per category
        :param train: if True, use training set, else use validation set

        The following parameters are for DataLoader of each category (NOT FOR THIS DATASET !!!!!!)

        :param batch_size: batch size for each category
        :param shuffle: if True, shuffle the dataset for each category
        :param num_workers: number of workers for data loading
        :param drop_last: if True, drop the last batch if it is not full
        """
        label_id_map = pickle.load(open(label_id_map, 'rb'))
        self.dataLoader = []
        self.weights = []
        for key, value in tqdm(label_id_map.items(), desc="Loading dataset"):
            file_list = pickle.load(open(filename_pickle, 'rb'))[key]
            if len(file_list) < num_support + batch_size:
                continue
            # if key != "person":
            #     continue
            dataset = COCODataset(resolution, root, key, value, filename_pickle, num_support=num_support, train=train)
            self.weights.append(len(dataset))
            self.dataLoader.append(
                COCODataLoader(dataset=dataset,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               num_workers=num_workers,
                               drop_last=drop_last)
            )
        self.weights = np.array(self.weights) / np.sum(self.weights)

    def __len__(self):
        return len(self.dataLoader)

    def __getitem__(self, idx):
        return self.dataLoader[idx]


class COCOCategoryLoaderDataLoader(DataLoader):
    def __init__(self, **kwargs):
        """
        Use weighted random sampler to sample from each category.
        Forcing batch_size to be 1 and shuffle to be False since its mutually exclusive with sampler.
        """
        weight = torch.from_numpy(kwargs['dataset'].weights)
        kwargs['sampler'] = WeightedRandomSampler(weight, 1, False)
        kwargs['batch_size'] = 1
        kwargs['shuffle'] = False
        super().__init__(**kwargs)
        assert isinstance(self.dataset, COCOCategoryLoaderDataset), "dataset must be COCOCategoryLoaderDataset"

    def __iter__(self):
        while True:
            idx = self.sampler.__iter__().__next__()
            loader = self.dataset[idx]
            batch = loader.__iter__().__next__()
            yield batch


class COCODataset(Dataset):
    def __init__(self, resolution, root, category, category_id, filename_pickle, num_support=5, train=True):
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
        self.num_support = num_support
        self.root = root
        self.category = category
        self.category_id = category_id
        self.train = train
        self.filename_list = pickle.load(open(filename_pickle, 'rb'))[category]

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        img, label = self._load_img_label(self.filename_list[idx])
        return {
            "img": img,
            "label": label,
            "idx": idx,
            "category": self.category
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
        label = (label == self.category_id).astype(np.float32)

        return img.transpose([2, 0, 1]), label

    def support_set(self, selected_idx):
        idx = set(range(self.__len__()))
        idx.difference_update(set(selected_idx))
        support_idx = random.choices(list(idx), k=self.num_support)
        support_filenames = [self.filename_list[i] for i in support_idx]
        support_img = []
        support_label = []
        for filename in support_filenames:
            img, label = self._load_img_label(filename)
            support_img.append(img)
            support_label.append(label)
        support_img = np.stack(support_img, axis=0)
        support_label = np.stack(support_label, axis=0)
        return support_img, support_label.astype(np.float32)


class COCODataLoader(DataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self.dataset, COCODataset), "dataset must be COCODataset"

    def __iter__(self):
        for batch in super().__iter__():
            batch['support_img'], batch['support_label'] = self.support_set(batch['idx'])
            # batch.pop('idx')
            yield batch

    def support_set(self, selected_idx):
        self.dataset: COCODataset
        support_img, support_label = self.dataset.support_set(selected_idx)
        support_img = torch.from_numpy(support_img).float()
        support_label = torch.from_numpy(support_label).float()
        return support_img, support_label


if __name__ == "__main__":
    file_list = pickle.load(open("../datasets/COCO/train_filenames_sieved.pkl", 'rb'))
    coco = COCOCategoryLoaderDataset(128, "../datasets/COCO", "../datasets/COCO/train_filenames_sieved.pkl",
                                     "../datasets/COCO/categories.pkl", batch_size=10, shuffle=True, num_workers=0,
                                     num_support=20)
    dataloader = COCOCategoryLoaderDataLoader(dataset=coco, num_workers=0)
    for batch in dataloader:
        print(batch['category'])
        print(batch['idx'])
    # print(1)
