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
        if batch:
            yield batch
        else:
            yield []  # Yield an empty list if no more items are available
        # if not batch:
        #     break
        # yield batch
    

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
    stream = load_dataset(
        "Skylion007/openwebtext",
        # "json",
        # data_files={"train":[file]},
        trust_remote_code=True,
        # num_proc=64,
        streaming=True,
        split="train",
    )
    # chunk_idx = len(chunk_paths)
    chunk_idx = 0
    # chunks = chunked(stream, chunk_size)
    
    chunk_args = []
    for chunk_idx, chunk in enumerate(chunked(stream, chunk_size)):
        if len(chunk) == 0:
            print(f"Empty chunk encountered at chunk index {chunk_idx}, stopping processing.")
            break
        print(f"Appending chunk {chunk_idx}, with {len(chunk)} examples")
        chunk_arg = (chunk, chunk_idx, cache_path, tokenize_with_tokenizer)
        chunk_args.append(chunk_arg)


    # prepare_data_mp_partial = partial(prepare_data_mp, config=config, tokenizer=tokenizer, cache_path=cache_path)
    with Pool(processes=min(mp.cpu_count(), 64)) as pool:
        # Use the pool to process each file in parallel
        pool.starmap(process_and_save_chunk, chunk_args)

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