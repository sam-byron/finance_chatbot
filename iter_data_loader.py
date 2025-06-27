import math
# from torch.utils.data import Dataset
import os
import random
import glob
from functools import partial
from utils_mp import load_chunk, load_chunk_safe
from multiprocessing import Pool
from itertools import chain
import torch
from torch.utils.data import DataLoader, Dataset
from collator import Collator
from concurrent.futures import ThreadPoolExecutor
import time
from datasets import concatenate_datasets
from transformers import AutoTokenizer, get_scheduler
import argparse
import json
import multiprocessing as mp
from itertools import islice
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from torch.utils.data import IterableDataset

class ChunkedIterableDataset(IterableDataset):
    def __init__(self, chunk_paths, block_size, dtype=torch.uint16):
        self.chunk_paths = chunk_paths
        self.block_size = block_size
        self.dtype = dtype

    def __iter__(self):
        # optional shuffle of chunk order
        random.shuffle(self.chunk_paths)
        for path in self.chunk_paths:
            # load chunk (this yields a list of lists of ints)
            sequences = torch.load(path, map_location="cpu")
            for seq in sequences:
                # cast and split into fixed‐size blocks on the fly
                ids = torch.tensor(seq, dtype=self.dtype)
                for i in range(0, len(ids), self.block_size):
                    chunk = ids[i : i + self.block_size]
                    if len(chunk) == self.block_size:
                        yield chunk
    
# def iter_data_loader(config, tokenizer, cache_path):

#     block_size = config["block_size"]
#     batch_size = config["batch_size"]
#     chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))

#     print(f"Found {len(chunk_paths)} cached chunks.")
#     if not chunk_paths:
#         raise RuntimeError(f"No cached chunks found in {cache_path}")

#     if len(chunk_paths) == 0:
#         raise RuntimeError(f"No cached chunks found in {cache_path}. Please run the tokenization step first.")

#     max_workers = min(len(chunk_paths), 96)

#     # Split chunks files into train and test sets
#     split = 0.9
#     chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))
#     split_index = int(split * len(chunk_paths))
#     train_chunk_paths = chunk_paths[:split_index]
#     test_chunk_paths = chunk_paths[split_index:]

#     # def __init__(self, chunk_paths, block_size, dtype=torch.int16):
#     train_dataset = ChunkedIterableDataset(train_chunk_paths, block_size=block_size)
#     test_dataset = ChunkedIterableDataset(test_chunk_paths, block_size=block_size)

#     pad_id = tokenizer.pad_token_id
#     collate_fn = Collator(pad_id)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         num_workers=6,
#         pin_memory=True,
#         collate_fn=collate_fn,
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         num_workers=6,
#         pin_memory=True,
#         collate_fn=collate_fn,
#     )

#     print("Data preparation complete.")

#     return train_loader, test_loader, collate_fn

def _count_blocks_in_chunk(args):
    path, block_size = args
    seqs = torch.load(path, map_location="cpu")
    return sum(len(seq)//block_size for seq in seqs)

def iter_data_loader(config, tokenizer, cache_path):
    block_size = config["block_size"]
    batch_size = config["batch_size"]

    # read all chunk paths
    chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))
    print(f"Found {len(chunk_paths)} cached chunks.")
    if not chunk_paths:
        raise RuntimeError(f"No cached chunks found in {cache_path}")

    # fraction splits (you can also pull these from config)
    train_frac = config.get("train_frac", 0.85)
    val_frac   = config.get("val_frac",   0.05)
    assert train_frac + val_frac < 1.0, "train_frac + val_frac must be < 1.0"
    N = len(chunk_paths)
    idx1 = int(train_frac * N)
    idx2 = int((train_frac + val_frac) * N)

    train_paths = chunk_paths[:idx1]
    val_paths   = chunk_paths[idx1:idx2]
    test_paths  = chunk_paths[idx2:]

    # create IterableDatasets
    train_ds = ChunkedIterableDataset(train_paths, block_size=block_size)
    val_ds   = ChunkedIterableDataset(val_paths,   block_size=block_size)
    test_ds  = ChunkedIterableDataset(test_paths,  block_size=block_size)

    pad_id     = tokenizer.pad_token_id
    collate_fn = Collator(pad_id)

    # build loaders (no shuffle flag on IterableDataset)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              num_workers=6, pin_memory=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              num_workers=6, pin_memory=True,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              num_workers=6, pin_memory=True,
                              collate_fn=collate_fn)

    print(f"Data preparation complete. "
          f"Train files: {len(train_paths)}, "
          f"Val files: {len(val_paths)}, "
          f"Test files: {len(test_paths)}")
    
    # count how many real samples in train_paths
    print("Counting total blocks in train paths...")
    total_train_blocks = 0
    with ProcessPoolExecutor() as exe:
        counts = exe.map(_count_blocks_in_chunk,
                             ((p, block_size) for p in train_paths))
        total_train_blocks = sum(counts)
    print(f"Total blocks in train paths: {total_train_blocks}")

    return train_loader, val_loader, test_loader, collate_fn, total_train_blocks