import tiktoken
from tqdm import tqdm
import datasets
import numpy as np
import multiprocessing as mp
import os

enc = tiktoken.get_encoding("gpt2")
eot = enc.encode_single_token("<|endoftext|>")
shard_size = int(1e8)
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
max_shards = 100  # approx 10B tokens
fw = datasets.load_dataset(
    "HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True
)

# create the cache for the local dir
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)


def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


nprocs = max(1, os.cpu_count() // 2)

with mp.Pool(nprocs) as pool:
    shard_index = 0
    tokens_count = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        if tokens_count + len(tokens) < shard_size:
            all_tokens_np[tokens_count : tokens_count + len(tokens)] = tokens
            tokens_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(
                    total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                )
            progress_bar.update(len(tokens))
        else:
            reminder = shard_size - tokens_count
            if progress_bar is None:
                progress_bar = tqdm(
                    total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                )
            progress_bar.update(reminder)
            all_tokens_np[tokens_count:] = tokens[:reminder]
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
            )
            write_datafile(filename, all_tokens_np)
            progress_bar.close()
            progress_bar = None
            shard_index += 1

            if max_shards is not None and shard_index >= max_shards:
                tokens_count = 0
                break

            tokens_count = len(tokens) - reminder
            all_tokens_np[:tokens_count] = tokens[reminder:]

    if tokens_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:tokens_count])

print(f"{shard_index} shards wrote in {DATA_CACHE_DIR}")
