import argparse
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import torch

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import COCOCategoryLoaderDataset as Dataset
from improved_diffusion.image_datasets import COCOCategoryLoaderDataLoader as DataLoader
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)


def main(args):
    dist_util.setup_dist()
    logger.configure(args.save_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = load_pretrained_ddpm(args)
    dataset = Dataset(resolution=args.image_size,
                      root=args.data_dir,
                      filename_pickle=args.filename_pickle,
                      label_id_map=args.category_pickle,
                      num_support=args.num_support,
                      train=True,
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=args.num_workers,
                      drop_last=True)

    dataloader = DataLoader(dataset=dataset,
                            num_workers=0)

    for epoch in range(args.epochs):
        for batch in dataloader:
            img = batch['img']
            label = batch['label']
            support_img = batch['support_img']
            support_label = batch['support_label']
            print(batch["category"], batch["idx"])


def load_pretrained_ddpm(args):
    DDPM_args = model_and_diffusion_defaults()
    DDPM_args.update({
        "image_size": 64,
        "num_channels": 128,
        "num_res_blocks": 3,
        "learn_sigma": True,
        "diffusion_steps": 4000,
        "noise_schedule": "cosine"
    })
    model, diffusion = create_model_and_diffusion(**DDPM_args)
    model.load_state_dict(torch.load(args.DDPM_dir, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()
    return model, diffusion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--category_pickle", type=str, required=True, help="Path to category pickle")
    parser.add_argument("--filename_pickle", type=str, required=True, help="Path to filename pickle")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save model")

    parser.add_argument("--model_dir", type=str, required=False, help="Path to model directory")
    parser.add_argument("--DDPM_dir", type=str, required=True, help="Path to pretrained DDPM")

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--num_support", type=int, default=5, help="cardinality of support set")

    opts = parser.parse_args()
    main(opts)
