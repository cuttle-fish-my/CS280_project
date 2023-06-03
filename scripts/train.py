import argparse
import torch
import os
import sys

local_rank = int(os.environ.get("LOCAL_RANK", 0))
sys.path.append(os.path.dirname(sys.path[0]))

import torch.distributed as dist
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import COCOCategoryLoaderDataset as Dataset
from improved_diffusion.image_datasets import COCOCategoryLoaderDataLoader as DataLoader
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_unet_and_segmentor,
)
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from improved_diffusion.unet import UNetandDecoder

def main(args):
    logger.configure(args.save_dir)

    logger.log("creating model and diffusion...")
    model = load_pretrained_ddpm(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False) if torch.cuda.is_available() else model
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
        for i, batch in enumerate(dataloader):
            # pred = model(batch["img"], timesteps=torch.tensor([0] * args.batch_size))
            img = batch["img"]
            label = batch["label"]
            support_img = batch["support_img"]
            support_label = batch["support_label"]
            pred = model(img, support_img, support_label, timestep=torch.tensor([0] * args.batch_size))
            loss = torch.nn.BCELoss()(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch {epoch}, iter {i}, loss {loss.item()}")


def load_pretrained_ddpm(args):
    DDPM_args = model_and_diffusion_defaults()
    DDPM_args.update({
        "image_size": 64,
        "num_channels": 128,
        "num_res_blocks": 3,
        "learn_sigma": True,
    })
    unet, segmentor = create_unet_and_segmentor(**DDPM_args)
    dist_util.load_checkpoint(args.DDPM_dir, unet)
    dist_util.load_checkpoint(args.segmentor_dir, segmentor)
    unet.requires_grad_(False)
    unet.eval()
    model = UNetandDecoder(unet, segmentor)
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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--num_support", type=int, default=5, help="cardinality of support set")

    opts = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", str(local_rank))
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://")
    main(opts)
