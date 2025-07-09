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

def make_segment_with_mem(input_tokens, loss=False, mem_token=None, num_mem_tokens=0):
    input_tokens_with_mem = [mem_token]*num_mem_tokens + input_tokens + [mem_token]*num_mem_tokens
    if loss:
        labels = torch.tensor(input_tokens)
    else:
        labels = torch.tensor([-100] * len(input_tokens))
    
    input_ids = torch.tensor(input_tokens_with_mem)
    
    read_mem_mask = torch.zeros(input_ids.shape, dtype=torch.long)
    read_mem_mask[:num_mem_tokens] = 1

    write_mem_mask = torch.zeros(input_ids.shape, dtype=torch.long)
    write_mem_mask[-num_mem_tokens:] = 1

    text_mask = torch.ones(input_ids.shape, dtype=torch.long) - read_mem_mask - write_mem_mask

    attention_mask = torch.ones_like(input_ids)

    if loss:
        labels_mask = torch.ones(len(input_tokens))
    else:
        labels_mask = torch.zeros(len(input_tokens))

    return {'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'labels_mask': labels_mask.bool(),
            'text_mask': text_mask.bool(),
            'read_mem_mask': read_mem_mask.bool(),
            'write_mem_mask': write_mem_mask.bool()
            }