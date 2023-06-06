import argparse
import torch
import os
import sys
import pickle

local_rank = int(os.environ.get("LOCAL_RANK", 0))
sys.path.append(os.path.dirname(sys.path[0]))

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import COCODataset as Dataset
from naive_unet.unet_model import UNet
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if local_rank == 0:
        logger.configure(args.save_dir)
        logger.log("creating unet model...")
    model = create_unet(args)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    id = pickle.load(open(args.category_pickle, "rb"))[args.ft_category]
    dataset = Dataset(resolution=args.image_size,
                      root=args.data_dir,
                      category=args.ft_category,
                      category_id=id,
                      filename_pickle=args.filename_pickle,
                      num_support=args.num_support,
                      train=True)
    # loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    loss_fn = torch.nn.BCELoss()
    writer = SummaryWriter()
    iteration = 0 if args.load_dir is None else int(args.load_dir.split("_")[-1].split(".")[0])
    for epoch in range(args.epochs):
        print("Epoch: {}".format(epoch))
        iou = 0
        for i in tqdm(range(len(dataset))):
            optimizer.zero_grad()
            batch = dataset[i]
            img = batch["img"][None]
            label = batch["label"][None]
            img = torch.tensor(img, dtype=torch.float32).to(device)
            label = torch.tensor(label, dtype=torch.float32).to(device)
            pred = model(img).squeeze(-3)
            # print(img.shape, label.shape, pred.shape)
            # print(label.type(), pred.type())
            # print(label, pred)
            iou += IoU(pred, label, args.threshold)
            # pred = pred > args.threshold
            # pred = pred[0].cpu().numpy()
            # pred = pred.astype("uint8")[0]
            # pred = Image.fromarray(pred * 255)
            # pred.save(os.path.join(args.save_dir, args.ft_category, batch["name"]))
            loss = loss_fn(pred, label.to(device))
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss.item(), iteration)
            if iteration % args.log_interval == 0:
                grid_img = make_grid(img, nrow=2, normalize=True, scale_each=True)
                writer.add_image("Images/img", grid_img, iteration)
                grid_label = make_grid(label.float(), nrow=2, normalize=True, scale_each=True)
                writer.add_image("Images/label", grid_label, iteration)
                grid_pred = make_grid(pred, nrow=2, normalize=True, scale_each=True)
                writer.add_image("Images/pred", grid_pred, iteration)
            iteration += 1
        iou /= len(dataset)
        print("IoU: ", iou.item())
        if local_rank == 0:
            logger.logkv("iteration", iteration)
            logger.logkv_mean("loss", loss.item())
            logger.dumpkvs()
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"unet_{epoch}.pt"))
    writer.close()



def anneal_lr(optimizer, iteration, total_iteration):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (1 - iteration / total_iteration)

def IoU(pred, label, threshold=0.5):
    pred = pred > threshold
    label = label > threshold
    return (pred & label).sum() / (pred | label).sum()

def create_unet(args):
    model = UNet(3, 1)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--category_pickle", type=str, required=True, help="Path to category pickle")
    parser.add_argument("--filename_pickle", type=str, required=True, help="Path to filename pickle")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save model")
    parser.add_argument("--load_dir", type=str, required=False, help="Path to load model weight")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    
    parser.add_argument("--ft_category", type=str, default="person", help="category to finetune")
    parser.add_argument("--num_support", type=int, default=5, help="cardinality of support set")
    parser.add_argument("--num_categories", type=int, default=183, help="Number of categories")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for IoU")


    opts = parser.parse_args()

    main(opts)
