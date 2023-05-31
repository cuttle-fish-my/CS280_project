import os.path
import string

from pycocotools.coco import COCO
import numpy as np
import cv2
import pickle


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
            else:
                print(f"{category} not found!")
    return label_id_map

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
    for name in names:
        if name in thing_names:
            catIds = thing_coco.getCatIds(catNms=[name])
            imgIds = thing_coco.getImgIds(catIds=catIds)
            imgs = thing_coco.loadImgs(imgIds)
        elif name in stuff_names:
            catIds = stuff_coco.getCatIds(catNms=[name])
            imgIds = stuff_coco.getImgIds(catIds=catIds)
            imgs = stuff_coco.loadImgs(imgIds)
        else:
            raise ValueError
        filename_pickle[name] = [img['file_name'] for img in imgs]
    pickle.dump(filename_pickle, open(f"COCO/{mode}_filenames.pkl", "wb"))


if __name__ == "__main__":
    CategoryImageFileNamePickle("COCO/thing_annotations/instances_train2017.json",
                                "COCO/stuff_annotations/stuff_train2017.json",
                                "COCO/labels.txt")
    CategoryImageFileNamePickle("COCO/thing_annotations/instances_val2017.json",
                                "COCO/stuff_annotations/stuff_val2017.json",
                                "COCO/labels.txt")
