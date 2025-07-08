import random
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from dataset import ChatDataset
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from accelerate import Accelerator
from safetensors.torch import load_file
import argparse
import json
from iter_data_loader import iter_data_loader
import math


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

    return model

def eval_ppl(model, dataloader, device, val_steps_per_epoch):
    import math, torch
    from torch.nn.functional import log_softmax

    model.eval()
    total_logp, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(
                dataloader, 
                desc="Evaluating Val Set Loss and Perplexity",
                total=val_steps_per_epoch,
                leave=True,):
            input_ids = batch["input_ids"].to(device)
            attn       = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask=attn).logits[:, :-1, :]
            labels = input_ids[:, 1:]

            logp = log_softmax(logits, dim=-1)
            token_logp = logp.gather(2, labels.unsqueeze(2)).squeeze(2)

            mask = attn[:, 1:]            # ignore padding
            total_logp   += (token_logp * mask).sum().item()
            total_tokens += mask.sum().item()

    nll = - total_logp / total_tokens
    return math.exp(nll)


# def evaluate_perplexity(model, val_loader, accelerator):
#     """Return (avg_nll, perplexity) on a *clean* validation loader."""
#     model.eval()
#     total_nll = 0.0          # negative log-likelihood (sum over tokens)
#     total_tokens = 0         # number of *active* tokens

#     with torch.no_grad():
#         for batch in tqdm(val_loader, desc="Val perplexity", leave=True):
#             # Move to correct device
#             batch = {k: v.to(accelerator.device) for k, v in batch.items()}
#             input_ids      = batch["input_ids"]
#             attention_mask = batch["attention_mask"]
#             labels         = batch["labels"]

#             # Forward pass : HuggingFace returns mean loss over active tokens
#             outputs = model(input_ids=input_ids,
#                             attention_mask=attention_mask,
#                             labels=labels)
#             mean_loss = outputs.loss          # scalar on this process

#             # How many tokens contributed to that mean?
#             token_count = (labels != -100).sum()

#             # Gather across all processes
#             mean_loss, token_count = accelerator.gather(
#                 (mean_loss, token_count)
#             )

#             # Convert back to python numbers
#             token_count = token_count.sum().item()
#             nll = mean_loss.sum().item() * token_count   # undo the mean

#             total_nll    += nll
#             total_tokens += token_count

#     avg_nll = total_nll / total_tokens
#     perplexity = math.exp(avg_nll)
#     return avg_nll, perplexity


def evaluate_perplexity(model, val_loader, accelerator, val_steps_per_epoch):
    """Return (avg_nll, perplexity) on a *clean* validation loader."""
    model.eval()
    total_mean_loss = 0.0          # negative log-likelihood (sum over tokens)
    total_tokens = 0         # number of *active* tokens

    with torch.no_grad():
        for batch in tqdm(
            val_loader, 
            desc="Evaluating Val Set Loss and Perplexity",
            total=val_steps_per_epoch,
            leave=True,):

            # Move to correct device
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            input_ids      = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels         = batch["labels"]

            # Forward pass : HuggingFace returns mean loss over active tokens
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            mean_loss = outputs.loss          # scalar on this process

            # How many tokens contributed to that mean?
            # token_count = (labels != -100).sum()

            # Gather across all processes
            # loss, token_count = accelerator.gather(
            #     (mean_loss, token_count)
            # )
            loss = accelerator.gather(
                (mean_loss)
            )
            mean_loss = loss.mean()  # average the mean loss across processes
            # Convert back to python numbers
            # token_count = token_count.sum().item()
            # nll = mean_loss.sum().item() * token_count   # undo the mean

            # total_nll    += nll
            total_mean_loss += mean_loss
            # total_tokens += token_count

    avg_nll = total_mean_loss / val_steps_per_epoch  # average loss over all batches
    perplexity = math.exp(avg_nll)
    return avg_nll, perplexity



def create_test_subset(test_texts, num_samples, block_size, batch_size, collate_fn, world_size, rank):
    """Create a subset of the test set and return a DataLoader."""
    # random.seed(42)  # Set seed for reproducibility
    subset_indices = random.sample(range(len(test_texts)), num_samples)  # Randomly sample indices
    test_subset_texts = [test_texts[i] for i in subset_indices]  # Create a subset of test_texts
    test_subset = ChatDataset(test_subset_texts, block_size=block_size)  # Create a ChatDataset for the subset
    test_subset_sampler = DistributedSampler(test_subset, num_replicas=world_size, rank=rank, shuffle=True)
    print(f"Test subset created with {len(test_subset_texts)} samples")

    # Create a DataLoader for the test subset
    test_subset_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        sampler=test_subset_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return test_subset_loader

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration JSON file.")
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer using the model name from the config
    tokenizer = GPT2Tokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize the Accelerator
    accelerator = Accelerator()
    

    model = build_model(config)

    model = accelerator.prepare(model)
    # Load the model weights from the checkpoint
    # Load weights saved by Accelerate
    state_dict = load_file("model_open_web_full/checkpoint.pt/model.safetensors")

    # Step 3: Inspect keys if necessary
    if "lm_head.weight" not in state_dict:
        print("⚠️ WARNING: lm_head.weight missing from checkpoint!")
        print("Keys available:", list(state_dict.keys())[:10])

    model.load_state_dict(state_dict, strict=False)

    train_loader, val_loader, test_loader, collate_fn, total_train_batches, total_val_batches, total_test_batches = iter_data_loader(config, tokenizer, config["cache_path"])

    # perplexity = eval_ppl(model, val_loader, device, total_val_batches)
    avg_nll, perplexity = evaluate_perplexity(model, val_loader, accelerator, total_val_batches)
    print(f"Validation Perplexity: {perplexity:.4f}")