import math
# from torch.utils.data import Dataset
import os
import random
import glob
from datasets import load_dataset, Dataset
from functools import partial
from utils_mp import load_chunk, load_chunk_safe
from multiprocessing import Pool
from itertools import chain
import torch
from torch.utils.data import DataLoader
from collator import Collator
from concurrent.futures import ThreadPoolExecutor
import time
from datasets import concatenate_datasets
from transformers import AutoTokenizer, get_scheduler
import argparse
import json
import multiprocessing as mp
from itertools import islice

from torch.utils.data import IterableDataset

class ChunkedIterableDataset(IterableDataset):
    def __init__(self, chunk_paths, block_size, dtype=torch.int16):
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


class ChatDataset(Dataset):
    def __init__(self, tokenized_texts, block_size=256):
        """
        Initialize the dataset with tokenized texts and block size.
        Precompute all chunks for index-based access.
        """
        self.tokenized_texts = tokenized_texts
        self.block_size = block_size
        self.chunks = self._create_chunks()

    def _create_chunks(self):
        """
        Precompute all chunks from the tokenized texts.
        Each chunk is of size `block_size`.
        """
        chunks = []
        for ids in self.tokenized_texts:
            for i in range(0, len(ids), self.block_size):
                chunk = ids[i : i + self.block_size]
                if len(chunk) == self.block_size:
                    chunks.append(chunk)
        return chunks

    def __len__(self):
        """
        Return the total number of chunks.
        """
        return len(self.chunks)

    def __getitem__(self, idx):
        """
        Return the chunk at the specified index.
        """
        return self.chunks[idx]
    
def data_loader(config, tokenizer, cache_path):

    block_size = config["block_size"]
    batch_size = config["batch_size"]
    chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))

    print(f"Found {len(chunk_paths)} cached chunks.")
    if not chunk_paths:
        raise RuntimeError(f"No cached chunks found in {cache_path}")

    if len(chunk_paths) == 0:
        raise RuntimeError(f"No cached chunks found in {cache_path}. Please run the tokenization step first.")

    max_workers = min(len(chunk_paths), 96)
    tokenized_texts_chunks = []

    # Use processes instead of threads so we bypass the GIL
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tokenized_texts_chunks = list(
            executor.map(load_chunk_safe, chunk_paths)
        )

    # drop any empty results
    tokenized_texts_chunks = [c for c in tokenized_texts_chunks if c]

    # tokenized_texts_chunks = list(map(load_chunk, chunk_paths))
    
    print(f"Loaded {len(tokenized_texts_chunks)} chunks.")
    tokenized_texts = list(chain.from_iterable(tokenized_texts_chunks))
    tokenized_texts = list(map(torch.tensor, tokenized_texts))
    print(f"Loaded {len(tokenized_texts)} samples from cache.")

    print(f"Finished parallel loading: {len(tokenized_texts_chunks)} chunks", flush=True)
    print("Shuffling tokenized texts...")
    random.shuffle(tokenized_texts)
    split_index = int(0.75 * len(tokenized_texts))
    train_texts = tokenized_texts[:split_index]
    test_texts = tokenized_texts[split_index:]

    print(f"Train texts: {len(train_texts)}, Test texts: {len(test_texts)}")

    train_dataset = ChatDataset(train_texts, block_size=block_size)
    test_dataset = ChatDataset(test_texts, block_size=block_size)

    pad_id = tokenizer.pad_token_id
    collate_fn = Collator(pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print("Data preparation complete.")

    return train_loader, test_loader, test_texts, collate_fn