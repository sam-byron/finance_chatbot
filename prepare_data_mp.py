import os
import glob
import pickle
from datasets import load_dataset, Dataset
from functools import partial
from utils_mp import tokenize_sample, load_chunk, process_and_save_chunk
from multiprocessing import Pool
from itertools import chain
import torch
import argparse
import json
import multiprocessing as mp
from itertools import islice
import gc
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# os.environ["OMP_NUM_THREADS"]       = "1"
# os.environ["MKL_NUM_THREADS"]       = "1"
# os.environ["OPENBLAS_NUM_THREADS"]  = "1"
# os.environ["NUMEXPR_NUM_THREADS"]   = "1"
# disable HuggingFace tokenizer parallelism
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# limit PyTorch to 1 thread per process as well
# torch.set_num_threads(1)
    
def chunked(iterable, size):
    it = iter(iterable)
    while True:
        try:
            # Attempt to get the next item from the iterator
            batch = list(islice(it, size))
            yield batch
        except StopIteration:
            # If StopIteration is raised, it means the iterator is exhausted
            break

def prepare_data(config, tokenizer, cache_path):

    # Ensure the cache folder exists
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        print(f"Created cache folder: {cache_path}")

    # Check if chunk files exist
    file_path = "/home/sam-byron/.cache/huggingface/hub/datasets--Skylion007--openwebtext/snapshots/f3808c30e817981b845ec549c43e82bb467d8144/subsets/*"
    # chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))

    chunk_size = config["chunk_size"]

    tokenize_with_tokenizer = partial(tokenize_sample, tokenizer=tokenizer)

    # Save index of file processed
    print(f"Loading dataset")
    # ds_list = []
    # load as a streaming IterableDataset (no num_proc in streaming)
    stream = load_dataset(
# "Skylion007/openwebtext",
"/home/sam-byron/.cache/huggingface/hub/datasets--Skylion007--openwebtext/snapshots/f3808c30e817981b845ec549c43e82bb467d8144/openwebtext.py",
        cache_dir=file_path,
        # "json   # data_files={"train":[file]},
        trust_remote_code=True,
        streaming=True,
        split="train",
    )

    # wrap the HuggingFace streaming IterableDataset in a PyTorch DataLoader
    # to parallelize I/O with num_workers > 1
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        stream,
        batch_size=chunk_size,
        num_workers=min(12, mp.cpu_count()),
        collate_fn=lambda examples: examples,  # identity: list of raw examples
    )
    
    # chunk_idx = len(chunk_paths)
    chunk_idx = 0
    # chunks = chunked(stream, chunk_size)
    
    pool = Pool(processes=min(mp.cpu_count(), 96))
    # pool = Pool(processes=min(mp.cpu_count(), 96))
    # now iterate batches of size chunk_size in parallel
    for chunk_idx, chunk in enumerate(dataloader):
        # if len(chunk) == 0:
        #     print(f"Empty chunk encountered at chunk index {chunk_idx}, stopping processing.")
        #     break
        print(f"Appending chunk {chunk_idx}, with {len(chunk)} examples")
        chunk_arg = (chunk, chunk_idx, cache_path, tokenize_with_tokenizer)
        # pool.apply(process_and_save_chunk, chunk_arg, tokenizer)
        pool.apply_async(process_and_save_chunk,
                             args=(chunk_arg, tokenizer))
        if len(chunk) == 0:
            print(f"Empty chunk encountered at chunk index {chunk_idx}, stopping processing.")
            break
        del chunk  # free memory
        gc.collect()  # force garbage collection to free memory
    return 1

def check_chunk_file(path):
    """Check if a single chunk file is valid. Returns (path, is_valid, error_msg)"""
    try:
        torch.load(path, map_location="cpu")
        return (path, True, None)
    except Exception as e:  # Catch all exceptions instead of specific ones
        return (path, False, str(e))

def sanitize_chunks_fast(config, max_workers=None):
    """Fast parallel sanitization using all available cores."""
    cache_path = config["cache_path"]
    chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))
    print(f"Found {len(chunk_paths)} chunk files to check...")
    
    if not chunk_paths:
        print("No chunk files found!")
        return 0, 0
    
    # Use all cores if not specified
    if max_workers is None:
        max_workers = min(96, mp.cpu_count())
    
    print(f"Using {max_workers} parallel workers...")
    
    corrupted_files = []
    valid_files = []
    
    # Use ProcessPoolExecutor for true parallelism
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(check_chunk_file, path): path for path in chunk_paths}
        
        # Process results with progress bar
        for future in tqdm(as_completed(future_to_path), total=len(chunk_paths), desc="Checking files"):
            path, is_valid, error_msg = future.result()
            
            if is_valid:
                valid_files.append(path)
            else:
                print(f"\nCorrupted file: {os.path.basename(path)} - {error_msg}")
                corrupted_files.append(path)
    
    print(f"\nSanitization complete:")
    print(f"  Valid files: {len(valid_files)}")
    print(f"  Corrupted files: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\nRemoving {len(corrupted_files)} corrupted files...")
        
        # Sequential deletion is fast enough for small numbers
        for path in corrupted_files:
            try:
                os.remove(path)
                print(f"  Removed: {os.path.basename(path)}")
            except OSError as e:
                print(f"  Failed to remove {os.path.basename(path)}: {e}")
        print("Cleanup complete!")
    else:
        print("No corrupted files found!")
    
    return len(valid_files), len(corrupted_files)

def main():
    parser = argparse.ArgumentParser(description="Tokenize and prepare data for training")    
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    # sanitize flag
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Sanitize chunks in the cache directory"
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)
    if args.sanitize:
        # Sanitize the chunks in the cache directory
        print(f"Sanitizing chunks...")
        valid_count, corrupted_count = sanitize_chunks_fast(config, 96)
        print(f"Valid chunks: {valid_count}, Corrupted chunks removed: {corrupted_count}")
        return
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    prepare_data(
        config, tokenizer, config["cache_path"]
    )

if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    main()