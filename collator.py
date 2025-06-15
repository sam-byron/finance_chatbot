import torch

class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        # batch is a list of 1D tensors or lists of token‐IDs
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [
                seq.detach().clone().long() if isinstance(seq, torch.Tensor)
                else torch.tensor(seq, dtype=torch.long)
                for seq in batch
            ],
            batch_first=True,
            padding_value=self.pad_id,
        )
        attention_mask = (input_ids != self.pad_id).long()
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         input_ids.clone(),
        }