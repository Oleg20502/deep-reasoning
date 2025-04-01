import torch


def split_cot(text, by=">> <<"):
    if text.startswith('<<'):
        text = text[2:]
    if text.endswith('>>'):
        text = text[:-2]

    return text.split(by)


def make_segment(input_tokens, loss=False):
    input_ids = torch.tensor(input_tokens)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor(input_tokens) if loss else torch.tensor([-100] * len(input_ids))
    # labels = torch.tensor(input_tokens)
    labels_mask = torch.ones_like(input_ids) if loss else torch.zeros_like(input_ids)

    return {'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'labels_mask': labels_mask.bool()
            }
