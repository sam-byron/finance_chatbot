import torch
import json
import argparse
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import os
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors.torch import load_file

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

def load_checkpoint(checkpoint_path="checkpoint.pt"):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint
    else:
        print("No checkpoint found. Starting from scratch.")
        return None
    
def log_probs_from_logits(logits, labels):
     logp = F.log_softmax(logits, dim=-1)
     logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
     return logp_label

def sequence_logprob(model, labels, input_len=0):
        with torch.no_grad():
            output = model(labels)
            log_probs = log_probs_from_logits(
                output.logits[:, :-1, :], labels[:, 1:])
            seq_log_prob = torch.sum(log_probs[:, input_len:])
        return seq_log_prob.cpu().numpy()

def start_chat_session(model_path, config):
    """Start an interactive chat session with the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer using the model name from the config
    tokenizer = GPT2Tokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize the Accelerator
    accelerator = Accelerator()
    
    model = build_model(config)

    model = accelerator.prepare(model)

    # Load the checkpoint using Accelerator's method
    print(f"Loading checkpoint from {model_path}")
    accelerator.load_state(model_path)
    # # Load the model weights from the checkpoint
    # # Load weights saved by Accelerate
    # state_dict = load_file("model_open_web_full/checkpoint.pt/pytorch_model.bin")

    # # Step 3: Inspect keys if necessary
    # if "lm_head.weight" not in state_dict:
    #     print("⚠️ WARNING: lm_head.weight missing from checkpoint!")
    #     print("Keys available:", list(state_dict.keys())[:10])

    # model.load_state_dict(state_dict, strict=False)
    # model.tie_weights() # <- critical

    # Load the checkpoint using Accelerator's method
    print(f"Loading checkpoint from {model_path}")
    accelerator.load_state(model_path)
    
    # checkpoint = torch.load(model_path, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Chat session started (type 'quit' to exit)")


    while True:
        text = input("You: ")
        if text.lower() == "quit":
            break

        usr_input = f"{text}\n"

        # Define max_new_tokens as a variable for consistency
        max_new_tokens = config.get("max_new_tokens", 50)  # Default to 50 if not specified

        # Tokenize the conversation history
        encoded = tokenizer(
            usr_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        gen_kwargs = dict(
            max_new_tokens=120,
            do_sample=True,          # turn on stochastic decoding
            temperature=0.8,         # soften next-token distribution
            top_p=0.92,              # nucleus sampling
            top_k=0,                 # let top_p do the truncation
            repetition_penalty=1.15, # discourages exact repeats
            no_repeat_ngram_size=3,  # stop short loops
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        output = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

        # # Generate the bot's response
        # output_beam = model.module.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     do_sample=False,
        #     # temperature=config["temperature"],
        #     # top_p=config["top_p"],
        #     num_beams=5,
        #     pad_token_id=tokenizer.eos_token_id,  # Silences the warning
        #     no_repeat_ngram_size=5
        # )

        # Decode the response
        logp = sequence_logprob(model, output, input_len=len(input_ids[0]))
        response = tokenizer.decode(output[0])
        print(f"\nlog-prob: {logp:.2f}")

        # Print the bot's response
        print(f"Prompt: {response}")

    

    # # Load the model weights from the checkpoint
    model_path = os.path.join(checkpoint_path, "model.safetensors")
    model = accelerator.load_state(checkpoint_path)
    print(model)
    # checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # model.load_state_dict(checkpoint["model_state_dict"])
    model = accelerator.unwrap_model(model)
    model.to(device)
    model.eval()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Start an interactive chat session with a GPT-2 model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration JSON file.")
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)

    # Start the chat session
    start_chat_session(model_path=args.model_path, config=config)