import os
import subprocess

import torch
import torch.distributed as dist


def setup_distributed(backend=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    if backend is None:
        # NCCL is unavailable on Windows; fall back to gloo there
        backend = "nccl" if dist.is_nccl_available() else "gloo"

    num_gpus = torch.cuda.device_count()

    # torch>=2.4 Windows builds lack libuv support for TCPStore
    if os.name == "nt":
        os.environ.setdefault("USE_LIBUV", "0")

    # allow plain "python train_xxx.py" single-process launches
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    init_kwargs = dict(backend=backend, world_size=world_size, rank=rank)

    if os.name == "nt":
        # Windows TCPStore is broken in some torch builds; use a FileStore instead
        import tempfile
        if world_size == 1:
            store_file = os.path.join(tempfile.gettempdir(), "realnet_dist_{}".format(os.getpid()))
        else:
            store_file = os.path.join(
                tempfile.gettempdir(), "realnet_dist_{}".format(os.environ["MASTER_PORT"]))
        init_kwargs["init_method"] = "file:///{}".format(store_file.replace("\\", "/"))

    dist.init_process_group(**init_kwargs)

    return rank, world_size
