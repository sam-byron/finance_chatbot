# finetune_lora.py
import argparse, math, os, json
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    get_scheduler,
)
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from tqdm import tqdm
from collator import Collator  # Import your custom collator


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True, help="Path to JSON configuration file")
    return p.parse_args()


def load_instruct_dataset(dataset_name, tokenizer, max_len, train_split_size=None):
    """Load and format instruction dataset from HuggingFace or local file"""
    
    # Load dataset - handle both HF datasets and local files
    if os.path.isfile(dataset_name):
        raw = load_dataset("json", data_files=dataset_name, split="train")
    else:
        raw = load_dataset(dataset_name, split="train")
    
    # If train_split_size is specified, take only that portion
    if train_split_size and train_split_size < len(raw):
        raw = raw.select(range(train_split_size))
        print(f"Using {train_split_size} examples from dataset")

    def format_alpaca_example(ex):
        """Format Alpaca-style examples into instruction format"""
        prompt = f"### Instruction:\n{ex['instruction']}\n"
        
        # Handle input field (may be empty string in Alpaca dataset)
        if ex.get("input") and ex["input"].strip():
            prompt += f"\n### Input:\n{ex['input']}\n"
        
        prompt += "\n### Response:\n"
        text = prompt + ex["output"]
        
        # Tokenize with truncation
        ids = tokenizer(text, truncation=True, max_length=max_len)["input_ids"]
        return {"input_ids": ids}

    # Apply formatting and remove original columns
    formatted_ds = raw.map(format_alpaca_example, remove_columns=raw.column_names)
    
    print(f"Loaded {len(formatted_ds)} formatted examples")
    return formatted_ds


def main():
    args = parse_args()
    config = load_config(args.config_path)
    
    acc = Accelerator()
    
    # Expand the tilde and get absolute path
    model_path = os.path.expanduser(config["model_path"])
    model_path = os.path.abspath(model_path)
    print(f"Loading model from {model_path}")
    
    # Load tokenizer from the base model name (since your local model doesn't have tokenizer files)
    base_tokenizer_name = config.get("base_tokenizer", "gpt2")
    print(f"Loading tokenizer from {base_tokenizer_name}")
    tok = AutoTokenizer.from_pretrained(base_tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load model from local path
    print(f"Loading model weights from {model_path}")
    model = GPT2LMHeadModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True  # Ensure we only use local files
    )
    model.config.pad_token_id = tok.pad_token_id

    print(f"Loaded model with {model.num_parameters():,} parameters")
    print(f"Model config: {model.config}")

    # Configure LoRA
    lora_cfg = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=config["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)
    
    if acc.is_main_process:
        model.print_trainable_parameters()

    # Load and format dataset
    ds = load_instruct_dataset(
        config["dataset_name"], 
        tok, 
        config["max_seq_len"],
        config.get("train_split_size")
    )
    
    # Custom wrapper collator
    class CollatorWrapper:
        def __init__(self, collator):
            self.collator = collator
        
        def __call__(self, batch):
            # Extract input_ids from each dict in the batch
            input_ids_list = [item["input_ids"] for item in batch]
            # Pass to your original collator
            return self.collator(input_ids_list)

    collator = CollatorWrapper(Collator(pad_id=tok.pad_token_id))
    loader = DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=config.get("num_workers", 0),
    )

    # Optimizer and scheduler
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.01)
    )
    
    steps_per_epoch = math.ceil(len(loader) / acc.num_processes)
    num_steps = steps_per_epoch * config["num_epochs"]
    warmup_steps = int(config.get("warmup_ratio", 0.05) * num_steps)
    
    sched = get_scheduler(
        config.get("lr_scheduler_type", "cosine"), 
        opt, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=num_steps
    )

    # Prepare for distributed training
    model, loader, opt, sched = acc.prepare(model, loader, opt, sched)

    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(config["num_epochs"]):
        if acc.is_main_process:
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            
        prog = tqdm(loader, disable=not acc.is_main_process, desc=f"Epoch {epoch+1}")
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in prog:
            with acc.accumulate(model):
                # Your custom collator already provides labels, so just pass the batch
                outputs = model(**batch)
                loss = outputs.loss
                acc.backward(loss)
                
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
                
                opt.step()
                sched.step()
                opt.zero_grad()

            global_step += 1
            epoch_loss += loss.item()
            num_batches += 1
            
            if acc.is_main_process and global_step % config.get("logging_steps", 100) == 0:
                prog.set_postfix(
                    loss=f"{loss.item():.3f}",
                    lr=f"{sched.get_last_lr()[0]:.2e}"
                )

        # Save checkpoint each epoch
        if acc.is_main_process:
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.3f}")
            
            os.makedirs(config["output_dir"], exist_ok=True)
            model.save_pretrained(config["output_dir"])
            tok.save_pretrained(config["output_dir"])
            print(f"Model saved to {config['output_dir']}")

    acc.wait_for_everyone()
    if acc.is_main_process:
        print("LoRA fine-tuning completed!")


if __name__ == "__main__":
    main()
