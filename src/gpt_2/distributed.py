import torch
import os
from torch.distributed import init_process_group, destroy_process_group
from dataclasses import dataclass


@dataclass
class DDPInfo:
    ddp: bool
    ddp_rank: int
    ddp_local_rank: int
    ddp_world_size: int
    device: str

    @property
    def master_process(self):
        return self.ddp_rank == 0

    @property
    def device_type(self):
        return "cuda" if self.device.startswith("cuda") else "cpu"


def setup_ddp() -> DDPInfo:

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("cuda is required for ddp")
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
    else:
        # vanilla
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

    if ddp_rank == 0:
        print("used device: ", device)

    return DDPInfo(ddp, ddp_rank, ddp_local_rank, ddp_world_size, device)


def destroy_ddp(ddp_info: DDPInfo) -> None:
    if ddp_info.ddp:
        destroy_process_group()
