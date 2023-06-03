"""
Helpers for distributed training.
"""
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8


def load_checkpoint(path, model):
    """
    Load a checkpoint from disk.
    """
    if path is None:
        print(f"Warning: No checkpoint loaded for model {type(model)}")
    if dist.get_rank() == 0 and path is not None:
        checkpoint = th.load(path, map_location="cpu")
        model.load_state_dict(checkpoint)
    model.to(dev())
    sync_params(model.parameters())


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{dist.get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def save_checkpoint(model, path):
    """
    Save a checkpoint to disk.
    """
    if dist.get_rank() == 0:
        if isinstance(model, DDP):
            model = model.module
        th.save(model.decoder.state_dict(), path)
