import os
import pickle
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2


def load_data(
        *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y: crop_y + self.resolution, crop_x: crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


class COCOCategoryDataset(Dataset):
    def __init__(self, resolution, root, filename_pickle, label_id_map, num_support=5, train=True):
        label_id_map = pickle.load(open(label_id_map, 'rb'))
        self.datasets = []
        for key, value in label_id_map.items():
            self.datasets.append(
                COCODataset(resolution, root, key, value, filename_pickle, num_support=num_support, train=train)
            )

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]


class COCODataset(Dataset):
    def __init__(self, resolution, root, category, category_id, filename_pickle, num_support=5, train=True):
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
            "idx": idx
        }

    def _load_img_label(self, path):
        img = Image.open(os.path.join(self.root, "train2017" if self.train else "val2017", path))
        label = Image.open(os.path.join(self.root,
                                        "stuffthingmaps_trainval2017",
                                        "train2017" if self.train else "val2017",
                                        path.replace(".jpg", ".png")))
        img = img.resize((self.resolution, self.resolution))
        label = label.resize((self.resolution, self.resolution), resample=Image.NEAREST)
        img = np.array(img).astype(np.float32) / 127.5 - 1
        label = np.array(label).astype(np.uint8)
        label = (label == self.category_id).astype(np.int64)

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
        return support_img, support_label


if __name__ == "__main__":
    # a = [1, 4, 6, 7, 9]
    # b = [1, 7]
    # a = set(a)
    # b = set(b)
    # a.difference_update(b)
    # print(list(a))

    # coco = COCODataset(128, "../datasets/COCO", "person", 1, "../datasets/COCO/train_filenames.pkl")
    # data = coco[0]
    # print(1)
    coco = COCOCategoryDataset(128, "../datasets/COCO", "../datasets/COCO/train_filenames.pkl",
                               "../datasets/COCO/categories.pkl")
    a = coco[0]
    loader = DataLoader(a, batch_size=2, shuffle=True)
    for b in loader:
        s_img, s_label = a.support_set(b['idx'])
