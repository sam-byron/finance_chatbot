import argparse
import json
import os

from transformers import AutoTokenizer
from iter_data_loader import iter_data_loader
from collator import Collator


parser = argparse.ArgumentParser(description="Chatbot Training Script with Accelerate")
parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
args = parser.parse_args()

with open(args.config_path, "r") as config_file:
    config = json.load(config_file)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = os.path.join(config["cache_path"], "checkpoint.pt")
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
tokenizer.pad_token = tokenizer.eos_token

batch_size = config["batch_size"]
pad_id = tokenizer.pad_token_id
collate_fn = Collator(pad_id)

test_texts = None

# train_loader, test_loader, test_texts, collate_fn = data_loader(config, tokenizer, config["cache_path"])
train_loader, val_loader, test_loader, collate_fn, total_train_blocks, total_val_blocks = iter_data_loader(config, tokenizer, config["cache_path"])

for step, batch in enumerate(train_loader):
    # print length of batch
    print(f"Batch {step}: {len(batch)} samples")
    # print(f"Batch {step}: {batch}")
print(f"Exhausted train_loader after {step + 1} batches")