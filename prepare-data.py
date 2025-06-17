import math
# from torch.utils.data import Dataset
import os
import random
import glob
from datasets import load_dataset, Dataset
from functools import partial
from utils import tokenize_sample, load_chunk, process_and_save_chunk
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
    
def chunked(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            return
        yield batch

def prepare_data(config, tokenizer, cache_path):
    nw = min(os.cpu_count(), 8)

    # if accelerator.is_main_process:
    # Only rank 0 loads & tokenizes the raw data
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    num_samples = config["num_samples"]
    chunk_size = config["chunk_size"]
    num_chunks = math.ceil(num_samples / chunk_size)
        
        # nw = 60
        # how many workers we’ll use for loading chunks later
        
    # Ensure the cache folder exists
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        print(f"Created cache folder: {cache_path}")

    # Check if chunk files exist
    file_path = "/home/sam-byron/.cache/huggingface/hub/datasets--Skylion007--openwebtext/snapshots/f3808c30e817981b845ec549c43e82bb467d8144/subsets/*"
    #Count the number of files in file_path
    # num_files = len(sorted(glob.glob(file_path)))
    chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))

    data_files = sorted(glob.glob(file_path))
    print(f"Found {len(data_files)} data files: {data_files}")
    
    tokenize_with_tokenizer = partial(tokenize_sample, tokenizer=tokenizer)
    # while len(chunk_paths) < 25:
    # while len(chunk_paths) < num_chunks:  
        # max_samples = 0
        # for index, file in enumerate(data_files[num_files-1:], start=num_files-1):
    for file in enumerate(data_files[len(chunk_paths):]):
        print(f"Loading dataset from file: {file}")
        # ds_list = []
        stream = load_dataset(
            "Skylion007/openwebtext",
            # "json",
            data_files={"train":[file[1]]},
            trust_remote_code=True,
            # num_proc=64,
            streaming=True,
            split="train",
        )
        chunk_idx = len(chunk_paths)
        chunks = chunked(stream, chunk_size)
        # for chunk in enumerate(chunked(stream, chunk_size)):
        
        # for chunk in enumerate(chunks):
        while len(chunk_paths) < num_chunks:
            # Extract the next chhunk in chunks
            chunk = next(chunks)
            print(f"Processing chunk {chunk_idx}, got {len(chunk[1])} examples")
            chunk_arg = (chunk, chunk_idx, cache_path, tokenize_with_tokenizer)
            process_and_save_chunk(chunk_arg, tokenizer)
            chunk_idx += 1
            chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))
            if len(chunk_paths) <= num_chunks:
                break
            # break

        # chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk_*.pt")))

        # accelerator.wait_for_everyone()
        chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))
        
        print(f"Found {len(chunk_paths)} cached chunks.")
        if not chunk_paths:
            raise RuntimeError(f"No cached chunks found in {cache_path}")

        if len(chunk_paths) == 0:
            raise RuntimeError(f"No cached chunks found in {cache_path}. Please run the tokenization step first.")

    max_workers = min(len(chunk_paths), 32)
    tokenized_texts_chunks = []
    from multiprocessing.dummy import Pool as ThreadPool  # uses threads, not processes
    with ThreadPool(processes=max_workers) as pool:
        tokenized_texts_chunks = pool.map(load_chunk, chunk_paths)

    
    # with Pool(processes=max_workers) as pool:
    #     # this will spin up worker processes and call load_chunk in parallel
    #     tokenized_texts_chunks = pool.map(load_chunk, chunk_paths)
    # use imap_unordered with chunksize=1 so each load_chunk runs in parallel
    # with Pool(processes=max_workers) as pool:
    #     for path, chunk in zip(
    #         chunk_paths,
    #         pool.imap_unordered(load_chunk, chunk_paths, chunksize=1)
    #     ):
    #         print(f"Loaded chunk from {path}", flush=True)
    #         tokenized_texts_chunks.append(chunk)
    # with Pool(processes=max_workers) as pool:
    #     for path, chunk in pool.imap_unordered(load_chunk, chunk_paths, chunksize=1):
    #         print(f"Loaded chunk from {path}", flush=True)
    #         tokenized_texts_chunks.append(chunk)

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

def main():
    parser = argparse.ArgumentParser(description="Chatbot Training Script with Accelerate")    
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, test_loader, test_texts, collate_fn = prepare_data(
        config, tokenizer, config["cache_path"]
    )

if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    main()