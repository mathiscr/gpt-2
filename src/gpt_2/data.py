from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import torch
import os


class TokenDataset(Dataset):
    def __init__(self, T, split, data_root):
        self.T = T
        files_dir = os.listdir(data_root)
        files_dir = [file for file in files_dir if split in file]
        files_dir = sorted(files_dir)
        if not files_dir:
            raise FileNotFoundError(f"No {split} found in {data_root}")
        files_dir = [os.path.join(data_root, file) for file in files_dir]

        self.shards = [np.load(s, mmap_mode="r") for s in files_dir]
        # number of windows of size T+1  for each shards
        self.window_sizes = [(len(s) - 1) // T for s in self.shards]
        # information about the number of windows processed after each shards
        self.cumulative = np.cumsum([0] + self.window_sizes)

    def __len__(self):
        return int(self.cumulative[-1])

    def __getitem__(self, index):
        current_shard = int(np.searchsorted(self.cumulative[1:], index, side="right"))
        local_index = int(index - self.cumulative[current_shard])
        buf_size = self.T
        buf = self.shards[current_shard][
            local_index * buf_size : local_index * buf_size + buf_size + 1
        ].astype(np.int64)
        buf = torch.from_numpy(buf)
        x = buf[:-1]
        y = buf[1:]
        return x, y


def build_loader(B, T, split, world_size, rank, num_workers, data_root):
    token_dataset = TokenDataset(T, split, data_root)
    shuffle = split == "train"
    sampler = (
        DistributedSampler(
            token_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )
        if world_size > 1
        else None
    )
    loader = DataLoader(
        token_dataset,
        batch_size=B,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return loader


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch
