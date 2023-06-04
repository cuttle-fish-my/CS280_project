import argparse
import torch
import os
import sys

local_rank = int(os.environ.get("LOCAL_RANK", 0))
sys.path.append(os.path.dirname(sys.path[0]))

from improved_diffusion import dist_util, logger
from naive_unet.image_datasets import COCOAllDataset as Dataset
from naive_unet.unet_model import UNet
from torch.utils.data import DataLoader
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if local_rank == 0:
        logger.configure(args.save_dir)
        logger.log("creating unet model...")
    model = create_unet(args)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataset = Dataset(resolution=args.image_size,
                      root=args.data_dir,
                      filename_pickle=args.filename_pickle,
                      train=True)
    dataloader = DataLoader(dataset=dataset,batch_size=args.batch_size, shuffle=True,num_workers=0, drop_last=True)
    iteration = 0 if args.load_dir is None else int(args.load_dir.split("_")[-1].split(".")[0])
    # loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    loss_fn = torch.nn.CrossEntropyLoss()


    for i, batch in enumerate(dataloader):
        # pred = model(batch["img"], timesteps=torch.tensor([0] * args.batch_size))
        optimizer.zero_grad()
        img = batch["img"]
        label = batch["label"]
        pred = model(img.to(device)).squeeze()
        # print(pred.shape, pred[0,:,0,0].sum())
        loss = loss_fn(pred, label.to(device).long())
        loss.backward()
        optimizer.step()
        if iteration % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"unet_{iteration}.pt"))
        if iteration % args.log_interval == 0:
            if local_rank == 0:
                logger.logkv("iteration", iteration)
                logger.logkv_mean("loss", loss.item())
                logger.dumpkvs()
        iteration += 1
        if iteration > args.iteration:
            break
        anneal_lr(optimizer, iteration, args.iteration)



def anneal_lr(optimizer, iteration, total_iteration):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (1 - iteration / total_iteration)


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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=5e6)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=1)

    parser.add_argument("--num_support", type=int, default=5, help="cardinality of support set")
    parser.add_argument("--num_categories", type=int, default=183, help="Number of categories")

    opts = parser.parse_args()

    main(opts)
