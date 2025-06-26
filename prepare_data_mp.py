import os
import glob
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
    
    pool = Pool(processes=min(mp.cpu_count(), 32))
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