import numpy as np
import warnings
import torch

def formatting_func_flan_it(example, tokenizer):
    messages = [
        {"role": "user", "content": example['inputs']},
        {"role": "assistant", "content": example['targets']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_flan(example):
    input_prompt = (f"{example['inputs']}\n\n"
        f"{example['targets']}")
    return {"text" : input_prompt}


def formatting_func_dolly_it(example, tokenizer):
    if example.get("context", "") != "":
        messages = [
            {"role": "user", "content": example['instruction'] + "\n" + example['context']},
            {"role": "assistant", "content": example['response']}
        ]
    else:
        messages = [
            {"role": "user", "content": example['instruction']},
            {"role": "assistant", "content": example['response']}
        ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_synth_it(example, tokenizer):
    messages = [
        {"role": "user", "content": example['question'] + "\n" + example['context']},
        {"role": "assistant", "content": example['answer']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_synth_it_debug(example, tokenizer):
    messages = [
        {"role": "user", "content": f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\nInstruction:\n{example['question']}\n\nContext:\n{example['context']}"},
        {"role": "assistant", "content": example['answer']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}


def formatting_func_dolly(example):
    if example.get("context", "") != "":
        input_prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Input: \n"
        f"{example['context']}\n\n"
        f"### Response: \n"
        f"{example['response']}")
    else:
        input_prompt = (f"Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Response:\n"
        f"{example['response']}")
    return {"text" : input_prompt}


def formatting_func_unnat_it(example, tokenizer):
    if "None" not in example['instances'][0]["constraints"]:
        messages = [
            {"role": "user", "content": example['instances'][0]['instruction_with_input'] + example['instances'][0]['constraints']},
            {"role": "assistant", "content": example['instances'][0]['output']}
        ]
    else:
        messages = [
            {"role": "user", "content": example['instances'][0]['instruction_with_input']},
            {"role": "assistant", "content": example['instances'][0]['output']}
        ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}

def formatting_func_unnat(example):
    input_prompt = (f"{example['instances'][0]['instruction_with_input']} + {example['instances'][0]['constraints'] if 'None' not in example['instances'][0]['constraints'] else ''}\n\n"
        f"{example['instances'][0]['output']}")
    return {"text" : input_prompt}

def formatting_func_la_it(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}

def formatting_func_babilong_it(example, tokenizer):
    template = "{} {}Answer with a single word."
    context = example['input']
    messages = [
        {"role": "user", "content": template.format(context, example['question'])},
        {"role": "assistant", "content": example['target']}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False
    )
    input_ids_short = tokenizer.apply_chat_template(
        [{"role": "user", "content": template.format(context, example['question'])}],
        tokenize=True,
        add_generation_prompt=False
    )
    labels_mask = torch.zeros(len(input_ids))
    labels_mask[len(input_ids_short) + 1:] = True
    return {"text" : input_ids, "labels_mask": labels_mask}

def formatting_func_dfq_golden_it(example, tokenizer):
    messages = [
        {"role": "user", "content": example['gold_evidence_text'] + "\n" + example['Question']},
        {"role": "assistant", "content": example['Answer']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text" : text}

def mask_non_completion(input, response_template, tokenizer, ignore_index=-100):
    # adapted from huggingface DataCollatorForCompletionOnlyLM https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L87
    if isinstance(response_template, str):
        # The user provides a string, must tokenize
        response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    else:
        # The user already provides the token ids
        response_token_ids = response_template
    response_token_ids_start_idx = None
    output = np.array(input.copy())
    for idx in np.where(output == response_token_ids[0])[0]:
        # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
        if (
            response_token_ids
            == output[idx : idx + len(response_token_ids)].tolist()
        ):
            response_token_ids_start_idx = idx

    if response_token_ids_start_idx is None:
        warnings.warn(
            f"Could not find response key `{response_template}` in the "
            f'following instance: {tokenizer.decode(output)} '
            f"This instance will be ignored in loss calculation. "
            f"Note, if this happens often, consider increasing the `max_seq_length`."
        )
        output[:] = ignore_index
    else:
        response_token_ids_end_idx = response_token_ids_start_idx + len(response_token_ids)

        # Make pytorch loss function ignore all tokens up through the end of the response key
        output[:response_token_ids_end_idx] = ignore_index
    return output.tolist()


def mask_non_completion_multi(input, response_template, end_response_token_ids, tokenizer, ignore_index=-100):
    # adapted from huggingface DataCollatorForCompletionOnlyLM https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L87
    if isinstance(response_template, str):
        # The user provides a string, must tokenize
        response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    else:
        # The user already provides the token ids
        response_token_ids = response_template
    response_token_ids_start_idx = None
    response_token_ids_start_ids = []
    output = np.array(input.copy())
    end_response_token_start_ids = np.where(output == end_response_token_ids)[0]
    for idx in np.where(output == response_token_ids[0])[0]:
        # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
        if (
            response_token_ids
            == output[idx : idx + len(response_token_ids)].tolist()
        ):
            response_token_ids_start_idx = idx
            response_token_ids_start_ids.append(response_token_ids_start_idx)
            #print(idx)

    #end_response_token_start_ids = np.where(output == end_response_token_ids)[0]
    #print(end_response_token_start_ids)
    if response_token_ids_start_idx is None:
        warnings.warn(
            f"Could not find response key `{response_template}` in the "
            f'following instance: {tokenizer.decode(output)} '
            f"This instance will be ignored in loss calculation. "
            f"Note, if this happens often, consider increasing the `max_seq_length`."
        )
        output[:] = ignore_index
    else:
        if len(response_token_ids_start_ids) == 1:
            # case of one reply from assistant
            response_token_ids_end_idx = response_token_ids_start_idx + len(response_token_ids)
            output[:response_token_ids_end_idx] = ignore_index
        else:
            # case of multiple step conversation
            response_token_ids_start_ids = [el + len(response_token_ids) for el in response_token_ids_start_ids]
            response_token_ids_end_ids = []
            for el in response_token_ids_start_ids:
                buf_end_toks = end_response_token_start_ids - el
                closest_end_idx = end_response_token_start_ids[np.where(buf_end_toks > 0)[0][np.argmin((buf_end_toks)[np.where(buf_end_toks > 0)])]]
                #print(buf_end_toks)
                response_token_ids_end_ids.append(closest_end_idx)
            #print(response_token_ids_start_ids)
            #print(response_token_ids_end_ids)
            #response_token_ids_start_ids = [el + len(response_token_ids) for el in response_token_ids_start_ids]
            mask = np.zeros_like(output)
            for response_idx in range(len(response_token_ids_start_ids)):
                start_idx = response_token_ids_start_ids[response_idx]
                try:
                    end_idx = response_token_ids_end_ids[response_idx]
                except:
                    end_idx = -1
                mask[start_idx:end_idx] = 1
            output[np.where(mask == 0)[0]] = ignore_index

        # Make pytorch loss function ignore all tokens up through the end of the response key
        #output[:response_token_ids_end_idx] = ignore_index
    return output.tolist()


def formatting_func_dispatcher(task_name):
    if "flan" in task_name:
        return formatting_func_flan_it
    elif "dolly" in task_name or "alpaca" in task_name:
        return formatting_func_dolly_it
    elif "LongAlign" in task_name or "LAMix" in task_name or "sharegpt" in task_name or "ultrachat" in task_name or "smoltalk" in task_name:
        return formatting_func_la_it
    elif "unnatural" in task_name:
        return formatting_func_unnat_it
    elif "synth_reduced_1k_2k_debug" in task_name:
        return formatting_func_synth_it_debug
    elif "synth" in task_name:
        return formatting_func_synth_it
    elif "docfinqa_golden" in task_name:
        return formatting_func_dfq_golden_it
    elif "babilong" in task_name:
        return formatting_func_babilong_it
    else:
        raise NotImplementedError(f"There is no formatting function for task {task_name}!")

def formatting_func_dispatcher_non_it(task_name):
    if "flan" in task_name:
        return formatting_func_flan
    elif "dolly" in task_name or "alpaca" in task_name:
        return formatting_func_dolly
    elif "LongAlign" in task_name or "LAMix" in task_name or "sharegpt" in task_name or "ultrachat" in task_name:
        return None
    elif "unnatural" in task_name:
        return formatting_func_unnat
    else:
        raise NotImplementedError(f"There is no formatting function for task {task_name}!")