from datetime import datetime
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
import pickle

from torch.utils.data import IterableDataset, get_worker_info

class ChunkedIterableDataset(IterableDataset):
    def __init__(self, chunk_paths, block_size, dtype=torch.uint16):
        self.chunk_paths = chunk_paths
        self.block_size = block_size
        self.dtype = dtype

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single‐process
            paths = self.chunk_paths
        else:
            # split paths across workers
            idx, num = worker_info.id, worker_info.num_workers
            paths = self.chunk_paths[idx::num]
        for path in paths:
            # load chunk (this yields a list of lists of ints)
            sequences = torch.load(path, map_location="cpu")
            for seq in sequences:
                # cast and split into fixed‐size blocks on the fly
                ids = torch.tensor(seq, dtype=self.dtype)
                if len(ids) < self.block_size:
                    yield ids.tolist()  # yield the whole sequence if it's shorter than block_size
                else:
                    for i in range(0, len(ids), self.block_size):
                        chunk = ids[i : i + self.block_size]
                        yield chunk


def count_batches_in_chunk(args):
    path, block_size, batch_size = args
    seqs = torch.load(path, map_location="cpu")
    # count how many sequences have length <= block_size
    total = 0
    total += sum(1 for seq in seqs if len(seq) < block_size)
    total += sum(len(seq) // block_size for seq in seqs if len(seq) >= block_size)
    return total // batch_size
    # return sum(len(seq)//block_size for seq in seqs)


def create_and_cache_splits(config):
    """Create train/val/test splits once and cache them."""
    
    cache_path = config["cache_path"]
    # Create splits directory
    splits_dir = Path(cache_path) / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    # Check if splits already exist
    splits_file = splits_dir / "dataset_splits.json"
    if splits_file.exists():
        print("Dataset splits already exist, loading cached splits...")
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        return splits['train_paths'], splits['val_paths'], splits['test_paths']
    
    # Create new splits
    print("Creating new dataset splits...")
    chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))
    
    # Shuffle once and save the order
    random.shuffle(chunk_paths)
    
    train_frac = config.get("train_frac", 0.89)
    val_frac   = config.get("val_frac", 0.01)
    
    N = len(chunk_paths)
    idx1 = int(train_frac * N)
    idx2 = int((train_frac + val_frac) * N)

    train_paths = chunk_paths[:idx1]
    val_paths   = chunk_paths[idx1:idx2]
    test_paths  = chunk_paths[idx2:]
    
    # Cache the splits
    splits = {
        'train_paths': train_paths,
        'val_paths': val_paths,
        'test_paths': test_paths,
        'created_at': str(datetime.now()),
        'config': {
            'train_frac': train_frac,
            'val_frac': val_frac,
            'total_chunks': N
        }
    }
    
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Cached dataset splits to {splits_file}")
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    return train_paths, val_paths, test_paths

def iter_data_loader(config, tokenizer, cache_path):
    block_size = config["block_size"]
    batch_size = config["batch_size"]

    # Load or create cached splits
    train_paths, val_paths, test_paths = create_and_cache_splits(config)

    # create IterableDatasets
    train_ds = ChunkedIterableDataset(train_paths, block_size=block_size)
    val_ds   = ChunkedIterableDataset(val_paths,   block_size=block_size)
    test_ds  = ChunkedIterableDataset(test_paths,  block_size=block_size)

    pad_id     = tokenizer.pad_token_id
    collate_fn = Collator(pad_id)

    # build loaders (no shuffle flag on IterableDataset)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              num_workers=8, pin_memory=True,
                              collate_fn=collate_fn, prefetch_factor=3, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              num_workers=8, pin_memory=True,
                              collate_fn=collate_fn, prefetch_factor=3, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              num_workers=8, pin_memory=True,
                              collate_fn=collate_fn, prefetch_factor=3, drop_last=True)

    print(f"Data preparation complete. "
          f"Train files: {len(train_paths)}, "
          f"Val files: {len(val_paths)}, "
          f"Test files: {len(test_paths)}")
    
    # Cache batch counts too
    counts_file = Path(cache_path) / "splits" / "batch_counts.json"
    if counts_file.exists():
        with open(counts_file, 'r') as f:
            counts = json.load(f)
        total_train_batches = counts['train_batches']
        total_val_batches = counts['val_batches']
        total_test_batches = counts['test_batches']
        print(
            f"Loaded cached batch counts:\n"
            f"    train={total_train_batches},\n"
            f"    val={total_val_batches},\n"
            f"    test={total_test_batches}"
        )
    else:
        # count how many real samples in train_paths
        print("Counting total batches in train paths...")
        with ProcessPoolExecutor() as exe:
            counts = exe.map(count_batches_in_chunk,
                                ((p, block_size, batch_size) for p in train_paths))
            total_train_batches = sum(counts)
        # for p in train_paths:
        #     total_train_blocks += _count_blocks_in_chunk((p, block_size))
        print(f"Total batches in train paths: {total_train_batches}")

        # count how many real samples in val_paths
        print("Counting total batches in val paths...")
        with ProcessPoolExecutor() as exe:
            counts = exe.map(count_batches_in_chunk,
                                ((p, block_size, batch_size) for p in val_paths))
            total_val_batches = sum(counts)
        print(f"Total batches in val paths: {total_val_batches}")

        # count how many real samples in test_paths
        print("Counting total batches in test paths...")
        with ProcessPoolExecutor() as exe:
            counts = exe.map(count_batches_in_chunk,
                                ((p, block_size, batch_size) for p in test_paths))
            total_test_batches = sum(counts)
        print(f"Total batches in test paths: {total_test_batches}")

        # Cache the counts
        counts = {
            'train_batches': total_train_batches,
            'val_batches': total_val_batches,
            'test_batches': total_test_batches,
            'computed_at': str(datetime.now())
        }
        with open(counts_file, 'w') as f:
            json.dump(counts, f, indent=2)

    return train_loader, val_loader, test_loader, collate_fn, total_train_batches, total_val_batches, total_test_batches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iter Data Loader Script")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader, test_loader, collate_fn, total_train_batches, total_val_batches, total_test_batches = iter_data_loader(config, tokenizer, config["cache_path"])
    
    print(f"Train loader: {total_train_batches} batches")
    print(f"Val loader: {total_val_batches} batches")
    print(f"Test loader: {total_test_batches} batches")