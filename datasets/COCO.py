import os.path
import string

from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import cv2
import pickle

from tqdm import tqdm


def labelIdMap(label_list_dir, names):
    label_id_map = {}
    with open(label_list_dir, "r") as f:
        while True:
            l = f.readline().split('\n')[0]
            if not l:
                break
            id = l.split(':')[0]
            category = l.split(':')[1][1:]
            if category in names:
                label_id_map[category] = int(id) - 1
    return label_id_map


def SieveImage(coco, category, root, mode, threshold):
    """
    :param coco: COCO object
    :param root: root directory of the dataset
    :param mode: train or val
    :param threshold: threshold for the number of pixels of the mask
    :return: list of image ids
    """
    catIds = coco.getCatIds(catNms=[category])[0] - 1
    imgIds = coco.getImgIds(catIds=catIds + 1)
    imgs = coco.loadImgs(imgIds)
    sieved_imgs = []
    for img in tqdm(imgs, desc=f"{category}", position=1, leave=False):
        label = Image.open(os.path.join(root, f"{mode}2017", img['file_name'][:-4] + ".png"))
        label = np.array(label)
        ratio = (label == catIds).sum() / (label.shape[0] * label.shape[1])
        if ratio > threshold:
            sieved_imgs.append(img)
    return sieved_imgs


def CategoryImageFileNamePickle(thing_json_dir, stuff_json_dir, label_list_dir):
    mode = os.path.basename(stuff_json_dir).split(".")[0].split("_")[1][:-4]
    stuff_coco = COCO(stuff_json_dir)
    thing_coco = COCO(thing_json_dir)
    stuff_cats = stuff_coco.loadCats(stuff_coco.getCatIds())
    thing_cats = thing_coco.loadCats(thing_coco.getCatIds())
    thing_names = [cat['name'] for cat in thing_cats]
    stuff_names = [cat['name'] for cat in stuff_cats]
    names = thing_names + stuff_names
    names.remove("other")
    label_id_map = labelIdMap(label_list_dir, names)
    pickle.dump(label_id_map, open("COCO/categories.pkl", "wb"))
    filename_pickle = {}
    for name in tqdm(names, desc="CategoryLoop", position=0):
        if name in thing_names:
            imgs = SieveImage(thing_coco, name, os.path.dirname(thing_json_dir), mode, 0.1)
        elif name in stuff_names:
            imgs = SieveImage(stuff_coco, name, os.path.dirname(stuff_json_dir), mode, 0.1)
        else:
            raise ValueError
        filename_pickle[name] = [img['file_name'] for img in imgs]
    pickle.dump(filename_pickle, open(f"COCO/{mode}_filenames.pkl", "wb"))


if __name__ == "__main__":
    CategoryImageFileNamePickle("COCO/annotations/thing_train2017.json",
                                "COCO/annotations/stuff_train2017.json",
                                "COCO/labels.txt")
    CategoryImageFileNamePickle("COCO/annotations/thing_val2017.json",
                                "COCO/annotations/stuff_val2017.json",
                                "COCO/labels.txt")
