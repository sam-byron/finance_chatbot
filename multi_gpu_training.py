"""
Multi‐GPU training via Accelerate.  
Assumes `prepare_data.py` has already been run.
Usage:
    accelerate launch --num_processes=2 train.py --config_path config.json
"""
import os
import json
import argparse
import time
import torch
from datetime import timedelta
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, get_scheduler
from accelerate import Accelerator
import torch.distributed as dist

from dataset import load_data, ChatDataset        # your load_data helper
from collator import Collator
from evaluation import evaluate_perplexity, create_test_subset

def build_model(config):
    cfg = GPT2Config(
        vocab_size=config["vocab_size"],
        n_positions=config["n_positions"],
        n_embd=config["n_embed"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        loss_type="cross_entropy",
    )
    return GPT2LMHeadModel(cfg)

def train_loop(accelerator, model, train_loader, test_loader, test_texts,
               optimizer, scheduler, config, checkpoint_path, scaler, collate_fn):
    num_epochs   = config["num_epochs"]
    batch_size   = config["batch_size"]
    block_size   = config["block_size"]
    log_interval = config.get("log_interval", 100)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            with accelerator.autocast():
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = out.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()

            if step % log_interval == 0 and accelerator.is_main_process:
                avg = running_loss / step
                print(f"[Epoch {epoch+1}] Step {step}/{len(train_loader)} train_loss={avg:.4f}")

        # end‐of‐epoch gather & checkpoint
        if accelerator.is_main_process:
            avg_epoch_loss = running_loss / len(train_loader)
            print(f"=== Epoch {epoch+1} complete: avg_train_loss={avg_epoch_loss:.4f} ===")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, checkpoint_path)

            # quick subset evaluation
            test_subset = create_test_subset(
                test_texts,
                5000,
                block_size,
                batch_size,
                collate_fn,
                accelerator.num_processes,
                accelerator.process_index,
            )
            val_loss, ppl = evaluate_perplexity(model, test_subset, accelerator.device)
            print(f">>> val_loss={val_loss:.4f}, ppl={ppl:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True,
                        help="Path to JSON config")
    args = parser.parse_args()
    config = json.load(open(args.config_path))

    # make sure each process is pinned before any dist calls
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # init NCCL process group explicitly so device mappings are known
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=local_rank,
        timeout=timedelta(minutes=10),
    )

    accelerator = Accelerator()
    device = accelerator.device

    # tokenizer & collator
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    collate_fn = Collator(pad_id)

    # load the pre‐baked chunks on all ranks
    train_loader, test_loader, test_texts, collate_fn = load_data(
        config, tokenizer, config["cache_path"], accelerator
    )

    model     = build_model(config).to(device)
    optimizer = AdamW(model.parameters(),
                      lr=config["learning_rate"],
                      weight_decay=config["weight_decay"])
    num_train_steps = config["num_epochs"] * len(train_loader)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=num_train_steps,
    )

    # wrap for DDP + AMP
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
    scaler = torch.cuda.amp.GradScaler()

    # train!
    checkpoint_path = os.path.join(config["cache_path"], "checkpoint.pt")
    train_loop(
        accelerator, model, train_loader, test_loader, test_texts,
        optimizer, scheduler, config, checkpoint_path, scaler, collate_fn
    )

if __name__ == "__main__":
    main()