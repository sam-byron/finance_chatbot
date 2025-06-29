import random
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from dataset import ChatDataset
from itertools import islice

# def evaluate_perplexity(model, val_loader, accelerator, val_steps_per_epoch):
#     """Evaluate perplexity on the val set."""
#     model = accelerator.unwrap_model(model)  # Unwrap the model if it's wrapped in DDP or similar
#     model.eval()  # Set the model to evaluation mode
#     total_loss = 0
#     total_tokens = 0
#     device = accelerator.device  # Get the device from the accelerator

#     # if accelerator.is_main_process:
#     with torch.no_grad():  # Disable gradient computation
#         for batch in tqdm(
#             val_loader, 
#             desc="Evaluating Val Set Loss and Perplexity",
#             total=val_steps_per_epoch,
#             leave=True,):

#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)

#             # input_ids = batch["input_ids"]
#             # attention_mask = batch["attention_mask"]
#             # labels = batch["labels"]

#             # Forward pass
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss.mean()

#             # Accumulate loss and token count
#             total_loss += loss.item() * input_ids.size(0)  # Multiply by batch size
#             total_tokens += input_ids.size(0)

#     # Calculate average loss and perplexity
#     avg_loss = total_loss / total_tokens
#     perplexity = torch.exp(torch.tensor(avg_loss))
#     return avg_loss, perplexity


def evaluate_perplexity(model, val_loader, accelerator, val_steps_per_epoch):
    """Evaluate perplexity on the val set."""
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_tokens = 0
    device = accelerator.device  # Get the device from the accelerator

    # if accelerator.is_main_process:
    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm(
            val_loader, 
            desc="Evaluating Val Set Loss and Perplexity",
            total=val_steps_per_epoch,
            leave=True,):

            # batch.to(device)  # Move the batch to the appropriate device
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # predictions = outputs.logits.argmax(dim=-1)
            
            # Gather all predictions and targets
            # predictions, labels = accelerator.gather_for_metrics((predictions, labels))

            
            loss = outputs.loss

            # loss = accelerator.gather(loss.detach())
            loss = accelerator.gather(loss).mean()  # Gather and average the loss across all processes

            # Accumulate loss and token count
            total_loss += loss.item() * input_ids.size(0)  # Multiply by batch size
            total_tokens += input_ids.size(0)

    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return avg_loss, perplexity


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