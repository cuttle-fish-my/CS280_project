import os.path
import string

from pycocotools.coco import COCO
import numpy as np
import cv2
import pickle


def CategoryImageFileNamePickle(json_path):
    mode = os.path.basename(json_path).split(".")[0].split("_")[1][:-4]
    coco = COCO(json_path)
    cats = coco.loadCats(coco.getCatIds())
    names = [cat['name'] for cat in cats]
    pickle.dump(names, open("COCO/categories.pkl", "wb"))
    filename_pickle = {}
    for name in names:
        catIds = coco.getCatIds(catNms=[name])
        imgIds = coco.getImgIds(catIds=catIds)
        imgs = coco.loadImgs(imgIds)
        filename_pickle[name] = [img['file_name'] for img in imgs]
    pickle.dump(filename_pickle, open(f"COCO/{mode}_filenames.pkl", "wb"))


if __name__ == "__main__":
    CategoryImageFileNamePickle("COCO/annotations/stuff_train2017.json")
    CategoryImageFileNamePickle("COCO/annotations/stuff_val2017.json")

