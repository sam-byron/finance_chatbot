from multiprocessing import Pool
# import multiprocessing as mp
# import torch.multiprocessing as tmp
# mp.set_start_method("spawn", force=True)
# tmp.set_sharing_strategy("file_system")

import os
import math
import random
import torch
import json
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator
from dataset import ChatDataset, prepare_data, load_dataset
from utils import save_checkpoint, load_checkpoint, batch_generator_sequential, tokenize_sample, load_chunk, process_and_save_chunk
from evaluation import evaluate_perplexity, create_test_subset
from itertools import chain
import argparse
import glob
from functools import partial
import time
from torch import _dynamo as torchdynamo
from typing import List
import torch.distributed as dist
from collator import Collator
import traceback
from datetime import timedelta
import torch.distributed as dist
# from transformer_utils import parse_midi_files, parse_midi_files_toEvents, reconstruct_midi_from_events

class EmptyDataset(Dataset):
    def __len__(self): 
        return 0
    def __getitem__(self, idx): 
        raise IndexError

# @torch.compile
def build_model(config):
    model_config = GPT2Config(
        vocab_size=config["vocab_size"],
        n_positions=config["n_positions"],
        n_embd=config["n_embed"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        loss_type="cross_entropy", # Original loss used to train GPT-2
    )
    model = GPT2LMHeadModel(model_config)
    # Optimize the forward method using TorchDynamo.
    # model.forward = torchdynamo.optimize(my_compiler)(model.forward)
    return model

# @torch.compile(backend="inductor", fullgraph=False)
def train_loop(accelerator, model, train_loader, test_loader, test_texts, optimizer, scheduler, config, checkpoint_path, tokenizer, scaler, num_cpu, start_epoch, collate_fn):
    num_epochs = config["num_epochs"]
    # scaler = torch.cuda.amp.GradScaler()

    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    block_size = config["block_size"]

    if start_epoch >= num_epochs:
        print("Training already completed. Exiting.")
        return
    
    # Track time for periodic checkpoint saving
    last_checkpoint_time = time.time()
    model.train()
    for epoch in range(start_epoch, num_epochs):
        # model.train()
        total_loss = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            with accelerator.autocast():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                total_loss += loss.item()

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # def collate_fn(batch):
                #     pad_id = tokenizer.pad_token_id
                #     max_len = max(len(seq) for seq in batch)

                #     input_ids = torch.nn.utils.rnn.pad_sequence(
                #         [torch.tensor(seq, dtype=torch.long) for seq in batch],
                #         batch_first=True,
                #         padding_value=pad_id,
                #     )
                #     attention_mask = (input_ids != pad_id).long()
                #     labels = input_ids.clone()

                #     return {
                #         "input_ids": input_ids,
                #         "attention_mask": attention_mask,
                #         "labels": labels,
                #     }

                # accelerator.wait_for_everyone()
                # if accelerator.is_main_process:
                # Detach so we have a tensor rather than a graph
                loss_tensor = loss.detach()
                # Gather across processes: returns a tensor with one element per process
                loss_gathered = accelerator.gather(loss_tensor)
                # Compute average loss across all GPUs
                avg_loss = loss_gathered.mean()
                # accelerator.wait_for_everyone()
                
                current_time = time.time()
                if current_time - last_checkpoint_time >= 3 * 60:  # 3 minutes in seconds
                    if accelerator.is_main_process:
                        save_checkpoint(epoch, model, optimizer, scheduler, scaler, checkpoint_path)
                        last_checkpoint_time = current_time
                        # if accelerator.is_main_process:
                        print(f"Epoch {epoch + 1}, Step {step + 1}/{len(train_loader)}, Loss: {avg_loss:.4f}")
                        # Evaluate perplexity on a subset of the test set
                        test_subset_loader = create_test_subset(
                            test_texts,
                            10000,
                            block_size,
                            batch_size,
                            collate_fn,
                            accelerator.num_processes,
                            accelerator.process_index
                        )
                        test_subset_loss, perplexity = evaluate_perplexity(model, test_subset_loader, accelerator.device)
                        print(f"Epoch {epoch + 1} Subset test Loss: {test_subset_loss:.4f}, Perplexity: {perplexity:.4f}")

        
        # print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # destroy the process group if it was initialized
        # if dist.is_initialized():
        #     dist.destroy_process_group()
        # accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
            save_checkpoint(epoch, model, optimizer, scheduler, scaler, checkpoint_path)


def main():
    print(torch._dynamo.list_backends())
    
    # 2) pin this process to the correct CUDA device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    # device = accelerator.device
    
    
    
    # grab LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT from torchrun/accelerate launch
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")


    # initialize process group with explicit device_id
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=local_rank,
        timeout=timedelta(minutes=10),
        # device_id=local_rank,
        device_id=torch.device(f"cuda:{local_rank}"),   # <-- pass a torch.device
    )
    
    
    
    accelerator = Accelerator()

    parser = argparse.ArgumentParser(description="Chatbot Training Script with Accelerate")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(config["cache_path"], "checkpoint.pt")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    # set_pad_id(tokenizer.pad_token_id)

    # accelerator.wait_for_everyone()

    batch_size = config["batch_size"]
    pad_id = tokenizer.pad_token_id
    collate_fn = Collator(pad_id)
    empty_loader = DataLoader(
        EmptyDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,)
    train_loader = empty_loader
    test_loader  = empty_loader

    # if accelerator.is_main_process:
    # with accelerator.main_process_first():
    test_texts = None
    # if accelerator.is_main_process:
    train_loader, test_loader, test_texts, collate_fn = prepare_data(
        config, tokenizer, config["cache_path"], accelerator
    )
    model = build_model(config)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
   

    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["num_epochs"] * len(train_loader),
    )
    
    # Compile just the forward() so model stays a Module and you can still call model.train()
    # model.forward = torch.compile(
    #     model.forward,
    #     backend="cudagraphs",
    #     fullgraph=False,
    # )
    # load or initialize checkpoint in uncompiled code
    checkpoint = load_checkpoint(checkpoint_path)
    start_epoch = 0
    if checkpoint:
        loaded_state_dict = checkpoint["model_state_dict"]
        # If keys don't start with "module.", add the prefix.
        if not list(loaded_state_dict.keys())[0].startswith("module."):
            new_state_dict = {"module." + k: v for k, v in loaded_state_dict.items()}
        else:
            new_state_dict = loaded_state_dict
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    # compute once, before compile
    num_cpu = os.cpu_count() - 4
     # create the GradScaler exactly once, outside the compiled loop
    scaler = torch.cuda.amp.GradScaler()

    # instead of compiling the model itself, compile the train_loop:
    # compiled_train_loop = torch.compile(
    #     train_loop, backend="inductor", fullgraph=False,
    # )

    #  # NOW wrap the _prepared_ model in torch.compile
    # model = torch.compile(model, backend="inductor", fullgraph=True)


    # Ensure all processes wait until data is loaded
    # accelerator.wait_for_everyone()
    train_loop(accelerator, model, train_loader, test_loader, test_texts, optimizer, scheduler, config, checkpoint_path, tokenizer, scaler, num_cpu, start_epoch, collate_fn)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise