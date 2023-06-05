import argparse
import pickle

import torch
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import COCODataset as Dataset
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model, create_segmentor,
)
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from improved_diffusion.unet import UNetAndDecoder
from PIL import Image


def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    model = load_pretrained_ddpm(args)
    id = pickle.load(open(args.category_pickle, "rb"))[args.ft_category]
    dataset = Dataset(resolution=args.image_size,
                      root=args.data_dir,
                      category=args.ft_category,
                      category_id=id,
                      filename_pickle=args.filename_pickle,
                      num_support=args.num_support,
                      train=False)
    iou = 0
    if not os.path.exists(os.path.join(args.save_dir, args.ft_category)):
        os.makedirs(args.save_dir)
    for i in tqdm(range(len(dataset))):
        batch = dataset[i]
        img = batch["img"][None]
        label = batch["label"][None]
        support_img, support_label = dataset.support_set([batch["idx"]])
        support_img = support_img
        support_label = support_label
        img = torch.tensor(img, dtype=torch.float32).to(dev())
        label = torch.tensor(label, dtype=torch.float32).to(dev())
        support_img = torch.tensor(support_img, dtype=torch.float32).to(dev())
        support_label = torch.tensor(support_label, dtype=torch.float32).to(dev())
        pred = model(img, support_img, support_label)
        iou += IoU(pred, label)
        pred = pred > 0.5
        pred = pred[0].cpu().numpy()
        pred = pred.astype("uint8")
        pred = Image.fromarray(pred * 255)
        pred.save(os.path.join(args.save_dir, args.ft_category, batch["name"]))
    iou /= len(dataset)
    print("IoU: ", iou.item())


def IoU(pred, label):
    pred = pred > 0.5
    label = label > 0.5
    return (pred & label).sum() / (pred | label).sum()


def load_pretrained_ddpm(args):
    DDPM_args = model_and_diffusion_defaults()
    DDPM_args.update({
        "image_size": 64,
        "num_channels": 128,
        "num_res_blocks": 3,
        "learn_sigma": True,
    })
    unet = create_model(**DDPM_args)
    segmentor = create_segmentor(num_support=args.num_support, **DDPM_args)
    unet.load_state_dict(torch.load(os.path.join(args.DDPM_dir), map_location="cpu"))
    segmentor.load_state_dict(torch.load(os.path.join(args.segmentor_dir), map_location="cpu"))
    model = UNetAndDecoder(unet, segmentor)
    model.eval()
    model.to(dev())
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--category_pickle", type=str, required=True, help="Path to category pickle")
    parser.add_argument("--filename_pickle", type=str, required=True, help="Path to filename pickle")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save model")

    parser.add_argument("--segmentor_dir", type=str, required=False, help="Path to segmentor directory")
    parser.add_argument("--DDPM_dir", type=str, required=True, help="Path to pretrained DDPM")

    parser.add_argument("--image_size", type=int, default=64)

    parser.add_argument("--num_support", type=int, default=5, help="cardinality of support set")
    parser.add_argument("--ft_category", type=str, default="person", help="category to finetune")
    opts = parser.parse_args()
    main(opts)
