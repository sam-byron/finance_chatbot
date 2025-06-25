import os
import glob
from datasets import load_dataset, Dataset
from functools import partial
from utils_mp import tokenize_sample, load_chunk, process_and_save_chunk
from multiprocessing import Pool
from itertools import chain
import torch
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
import time
from datasets import concatenate_datasets
from transformers import AutoTokenizer, get_scheduler
import argparse
import json
import multiprocessing as mp
from itertools import islice

os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]       = "1"
os.environ["OPENBLAS_NUM_THREADS"]  = "1"
os.environ["NUMEXPR_NUM_THREADS"]   = "1"
# disable HuggingFace tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# limit PyTorch to 1 thread per process as well
torch.set_num_threads(1)
    
def chunked(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            return None
        yield batch

def prepare_data_mp(file_idx_pair, config, tokenizer, cache_path):

    chunk_size = config["chunk_size"]

    file, file_idx = file_idx_pair

    tokenize_with_tokenizer = partial(tokenize_sample, tokenizer=tokenizer)

    # Save index of file processed
    with open(os.path.join(cache_path, "processed_files.txt"), "w") as f:
        print(f"Loading dataset from file: {file}")
        # ds_list = []
        stream = load_dataset(
            "Skylion007/openwebtext",
            # "json",
            data_files={"train":[file]},
            trust_remote_code=True,
            # num_proc=64,
            streaming=True,
            split="train",
        )
        # chunk_idx = len(chunk_paths)
        chunk_idx = 0
        chunks = chunked(stream, chunk_size)
      
        while chunks:
            # Extract the next chunk in chunks
            chunk = next(chunks)
            print(f"Processing chunk {chunk_idx} for file_idx {file_idx}, got {len(chunk[1])} examples")
            chunk_arg = (file_idx, chunk, chunk_idx, cache_path, tokenize_with_tokenizer)
            process_and_save_chunk(chunk_arg, tokenizer)
            chunk_idx += 1
    

def prepare_data(config, tokenizer, cache_path):

    # Ensure the cache folder exists
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        print(f"Created cache folder: {cache_path}")

    # Check if chunk files exist
    file_path = "/home/sam-byron/.cache/huggingface/hub/datasets--Skylion007--openwebtext/snapshots/f3808c30e817981b845ec549c43e82bb467d8144/subsets/*"
    # chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))

    data_files = sorted(glob.glob(file_path))
    num_files = len(data_files)
    print(f"Found {num_files} data files: {data_files}")

    file_indices = []
    for file_idx, file in enumerate(data_files):
        file_indices.append(file_idx)

    file_idx_pair = list(zip(data_files, file_indices))
    prepare_data_mp_partial = partial(prepare_data_mp, config=config, tokenizer=tokenizer, cache_path=cache_path)
    with Pool(processes=min(mp.cpu_count(), num_files)) as pool:
        # Use the pool to process each file in parallel
        pool.map(prepare_data_mp_partial, file_idx_pair)

    return 1

def main():
    parser = argparse.ArgumentParser(description="Chatbot Training Script with Accelerate")    
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    prepare_data(
        config, tokenizer, config["cache_path"]
    )

if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    main()