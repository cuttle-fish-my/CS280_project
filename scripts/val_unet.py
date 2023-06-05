import argparse
import pickle

import torch
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import COCODataset as Dataset
from naive_unet.unet_model import UNet
from PIL import Image


def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_unet(args)
    model.load_state_dict(torch.load(args.load_dir))
    model = model.to(device)
    id = pickle.load(open(args.category_pickle, "rb"))[args.ft_category]
    dataset = Dataset(resolution=args.image_size,
                      root=args.data_dir,
                      category=args.ft_category,
                      category_id=id,
                      filename_pickle=args.filename_pickle,
                      num_support=args.num_support,
                      train=False)
    iou = 0
    if not os.path.isdir(os.path.join(args.save_dir, args.ft_category)):
        os.makedirs(os.path.join(args.save_dir, args.ft_category))
    for i in tqdm(range(len(dataset))):
        batch = dataset[i]
        img = batch["img"][None]
        label = batch["label"][None]
        support_img, support_label = dataset.support_set([batch["idx"]])
        support_img = support_img
        support_label = support_label
        img = torch.tensor(img, dtype=torch.float32).to(device)
        label = torch.tensor(label, dtype=torch.float32).to(device)
        support_img = torch.tensor(support_img, dtype=torch.float32).to(device)
        support_label = torch.tensor(support_label, dtype=torch.float32).to(device)
        pred = model(img)
        iou += IoU(pred[:,id+1,:,:], label)
        pred = pred > args.threshold
        pred = pred[0].cpu().numpy()
        pred = pred.astype("uint8")[0]
        pred = Image.fromarray(pred * 255)
        pred.save(os.path.join(args.save_dir, args.ft_category, batch["name"]))
    iou /= len(dataset)
    print("IoU: ", iou.item())


def IoU(pred, label):
    pred = pred > 0.5
    label = label > 0.5
    return (pred & label).sum() / (pred | label).sum()


def create_unet(args):
    model = UNet(3, args.num_categories)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--category_pickle", type=str, required=True, help="Path to category pickle")
    parser.add_argument("--filename_pickle", type=str, required=True, help="Path to filename pickle")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save model")

    parser.add_argument("--load_dir", type=str, required=False, help="Path to load model weight")

    parser.add_argument("--image_size", type=int, default=64)

    parser.add_argument("--num_support", type=int, default=5, help="cardinality of support set")
    parser.add_argument("--ft_category", type=str, default="person", help="category to finetune")
    parser.add_argument("--num_categories", type=int, default=183, help="Number of categories")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for IoU")

    opts = parser.parse_args()
    main(opts)
