from multiprocessing import Pool
import os
import math
import random
import torch
import json
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
from data_loader import data_loader
from iter_data_loader import iter_data_loader
from utils_mp import save_checkpoint, load_checkpoint
from evaluation import evaluate_perplexity, create_test_subset
from itertools import chain, islice
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
def train_loop(accelerator, model, train_loader, test_loader, val_loader, optimizer, scheduler, config, checkpoint_path, tokenizer, scaler, num_cpu, start_epoch, collate_fn, steps_per_epoch, val_steps_per_epoch):
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
        model.train()
        # total_loss = 0

        for step, batch in enumerate(
            tqdm(train_loader, 
            desc=f"Epoch {epoch + 1}",
            total=steps_per_epoch,
            leave=True,
            )
            
            ):
            with accelerator.autocast():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                # total_loss += loss.item()

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # loss_tensor = loss.detach()
                # # Gather across processes: returns a tensor with one element per process
                # loss_gathered = accelerator.gather(loss_tensor)
                # # Compute average loss across all GPUs
                # avg_loss = loss_gathered.mean()
                # # accelerator.wait_for_everyone()

                #   this does an all_reduce followed by division by world_size
                reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
                
                current_time = time.time()
                if current_time - last_checkpoint_time >= 1 * 60:  # 3 minutes in seconds
                    # if accelerator.is_main_process:
                    # save_checkpoint(epoch, model, optimizer, scheduler, scaler, checkpoint_path)
                    accelerator.wait_for_everyone()
                    # accelerator.save_model(model, checkpoint_path)
                    accelerator.save_state(output_dir=checkpoint_path)
                    last_checkpoint_time = current_time
                    if accelerator.is_main_process:
                        print(f"Epoch {epoch + 1}, Step {step + 1}/{steps_per_epoch}, Reduced Loss: {reduced_loss:.4f}")
                    # Evaluate perplexity on a subset of the test set
                    test_subset_loss, perplexity = evaluate_perplexity(model, val_loader, accelerator, val_steps_per_epoch)
                    if accelerator.is_main_process:
                        print(f"Epoch {epoch + 1} Subset test Loss: {test_subset_loss:.4f}, Perplexity: {perplexity:.4f}")
            # step += 1
        
        # if accelerator.is_main_process:
        print(f"Epoch {epoch + 1} reduced loss: {reduced_loss:.4f}")
        accelerator.wait_for_everyone()
        accelerator.save_state(output_dir=checkpoint_path)


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
    
    # You need to tell Accelerate not to “dispatch” the same batch to every rank when using an IterableDataset (or else split each global batch into per‐rank pieces). Two ways to do this:
    # Option A) Disable dispatching entirely by passing dispatch_batches=False
    dataloader_config = DataLoaderConfiguration(
    dispatch_batches=False,  # Each process fetches its own batch
    split_batches=True       # Split fetched batches across processes
    )
    accelerator = Accelerator(dataloader_config=dataloader_config)

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

    # test_texts = None

    # train_loader, test_loader, test_texts, collate_fn = data_loader(config, tokenizer, config["cache_path"])
    train_loader, val_loader, test_loader, collate_fn, total_train_batches, total_val_batches = iter_data_loader(config, tokenizer, config["cache_path"])
    model = build_model(config)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
   
    model, optimizer, train_loader, test_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader, val_loader
    )

    # model, optimizer, train_loader, test_loader = accelerator.prepare(
    #     model, optimizer, train_loader, test_loader
    # )

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_train_batches,
    )
    
    # checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")  
    start_epoch = 0
    # if checkpoint:
    #     loaded_state_dict = checkpoint["model_state_dict"]
    #     # If keys don't start with "module.", add the prefix.
    #     if not list(loaded_state_dict.keys())[0].startswith("module."):
    #         new_state_dict = {"module." + k: v for k, v in loaded_state_dict.items()}
    #     else:
    #         new_state_dict = loaded_state_dict
    #     model.load_state_dict(new_state_dict)
    #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    #     start_epoch = checkpoint["epoch"] + 1
    # compute once, before compile
    num_cpu = os.cpu_count() - 4
     # create the GradScaler exactly once, outside the compiled loop
    scaler = torch.cuda.amp.GradScaler()

    train_loop(accelerator, model, train_loader, test_loader, val_loader, optimizer, scheduler, config, checkpoint_path, tokenizer, scaler, num_cpu, start_epoch, collate_fn, total_train_batches, total_val_batches)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise