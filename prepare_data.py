"""
One‐time data preparation: stream, tokenize & save chunk_*.pt files.
Usage:
    python prepare_data.py --config_path config.json
"""
import os
import glob
import json
import torch
import time                                 # ← added
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

# simple streaming batch generator
def batch_generator(streaming_dataset, chunk_size):
    buf = []
    for item in streaming_dataset:
        buf.append(item)
        if len(buf) == chunk_size:
            yield buf
            buf = []
    if buf:
        yield buf

def generate_chunks(config):
    cache_path = config["cache_path"]
    os.makedirs(cache_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    tok_fn = lambda s: tokenizer.encode(
        s + tokenizer.eos_token,
        add_special_tokens=False,
        truncation=True,
    )
    file_path = "/home/sam-byron/.cache/huggingface/hub/datasets--Skylion007--openwebtext/snapshots/f3808c30e817981b845ec549c43e82bb467d8144/subsets/*"
    src_glob = glob.glob(
        file_path,
    )
    files = sorted(src_glob)
    total_files = len(files)
    print(f"[prepare_data] Found {total_files} files to process")

    idx = 0
    for file_idx, fpath in enumerate(files, start=1):
        print(f"\n[prepare_data] ({file_idx}/{total_files}) Loading file: {fpath}")
        t0_file = time.time()
        ds = load_dataset(
            "Skylion007/openwebtext",
            data_files=fpath,
            trust_remote_code=True,
            streaming=True,
        )["train"]

        for batch_idx, batch in enumerate(batch_generator(ds, config["chunk_size"]), start=1):
            texts = [
                item["text"] if isinstance(item, dict) else item
                for item in batch
            ]
            tok_batch = [tok_fn(t) for t in texts]
            chunk_file = os.path.join(cache_path, f"chunk_{idx:05d}.pt")
            torch.save(tok_batch, chunk_file)
            idx += 1

            # per‐chunk logging
            if idx % config.get("log_every", 50) == 0:
                elapsed = time.time() - t0_file
                print(f"[prepare_data]  → file {file_idx}/{total_files}, "
                      f"batch {batch_idx}, chunk #{idx:05d} saved "
                      f"({elapsed:.1f}s elapsed)")

        file_time = time.time() - t0_file
        print(f"[prepare_data] Completed file {file_idx}/{total_files} "
              f"in {file_time:.1f}s, total chunks so far: {idx}")

    print(f"\n[prepare_data] DONE. Wrote {idx} total chunks to {cache_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True,
                        help="Path to JSON config")
    args = parser.parse_args()
    cfg = json.load(open(args.config_path))
    generate_chunks(cfg)