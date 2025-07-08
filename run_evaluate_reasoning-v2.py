import logging
import os
from pathlib import Path
import torch
import numpy as np
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from lm_experiments_tools.utils import get_cls_by_name
from utils.reasoning import make_segment, split_cot

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')

parser = HfArgumentParser([])
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cpt', type=str, default=None, help='pretrained model checkpoint path')
parser.add_argument('--dataset_name', type=str, required=True, help='HuggingFace dataset name')
parser.add_argument('--task_name', type=str, default='gsm8k', help='Task name (default: gsm8k)')

parser.add_argument('--max_cot_steps', type=int, default=None, help='Maximum number of cot steps')
parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of new tokens to generate')

parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
parser.add_argument('--device', type=str, default='cuda', help='Device for evaluation')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)
    device = args.device

    model = AutoModelForCausalLM.from_pretrained(args.from_pretrained)
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    bos = [tokenizer.bos_token_id]
    eos = [tokenizer.eos_token_id]
    think = tokenizer.encode("<issue_start>")
    ans = tokenizer.encode("<issue_closed>")
    
        
    
    if args.model_cpt:
        if "safetensors" in args.model_cpt:
            print(model)
            from safetensors.torch import load_model
            load_model(model, args.model_cpt, device="cuda:0")
        else:
            if ".bin" in args.model_cpt:
                model_cpt = args.model_cpt
            elif "model_best" in os.listdir(args.model_cpt):
                model_cpt = os.path.join(args.model_cpt, "model_best", "pytorch_model.bin")
            else:
                dir_files = os.listdir(args.model_cpt)
                checkpoint_dir = [el for el in dir_files if "checkpoint-" in el][0]
                model_cpt = os.path.join(args.model_cpt, checkpoint_dir, "pytorch_model.bin")
            cpt = torch.load(model_cpt, map_location='cpu')
            model.load_state_dict(cpt, strict=False)
        logger.info(f'Loaded RMT state dict from: {args.model_cpt}')

    def collate_fn(batch):
        input_ids, labels, labels_mask, attention_mask = [], [], [], []
        for sample in batch:
            task, lab, cot = sample['task'], sample['labels'], sample['cot']
            task_tokens = tokenizer.encode(task, add_special_tokens=False)
            labels_tokens = tokenizer.encode(lab, add_special_tokens=False)
            cot_tokens = tokenizer.encode(cot, add_special_tokens=False)


            full_input = task_tokens + think
            inp_ids = torch.tensor(full_input)
            input_ids.append(inp_ids)

            lab = torch.tensor(full_input)
            lab[:len(task_tokens)] = -100
            labels.append(lab)

            lab_mask = torch.ones_like(inp_ids)
            lab_mask[:len(task_tokens)] = 0
            labels_mask.append(lab_mask)
            attention_mask.append(torch.ones_like(inp_ids))

        input_ids = pad_sequence(input_ids, padding_value=pad, batch_first=True)
        attention_mask = pad_sequence(attention_mask, padding_value=0, batch_first=True)
        labels = pad_sequence(labels, padding_value=pad, batch_first=True)
        labels_mask = pad_sequence(labels_mask, padding_value=0, batch_first=True)

        collated = {'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': attention_mask,
                    }
        return collated

    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = datasets.load_dataset(args.dataset_name)
    valid_dataset = dataset['valid'] if 'valid' in dataset else dataset['validation']

    if args.max_cot_steps is not None:
        valid_dataset = valid_dataset.filter(lambda x: x['cot_len'] <= args.max_cot_steps)

    def evaluate(model, dataset, device='cpu', bs=16, max_new_tokens=25):
        all_preds, all_labels = [], []
        all_preds_cot, all_labels_cot = [], []
        all_preds_ans, all_labels_ans = [], []

        for start_ind in tqdm(range(0, len(dataset), bs)):
            batch = dataset.select(range(start_ind, min(len(dataset), start_ind + bs)))
            collated = collate_fn(batch)
            task = collated['segments'][0]
            task = {k:v.to(device) for k,v in task.items()}

            with torch.no_grad():
                preds_full = model.generate([task], max_new_tokens=max_new_tokens, pad_token_id=eos[0])
            
            labels = collated['labels']
            labels_masks = labels > 0
            labels_full = [lab[m] for lab, m in zip(labels, labels_masks)]

            for lab_tokens, pred_tokens in zip(labels_full, preds_full):
                lab_tokens = [t.item() for t in lab_tokens if t != bos[0]]
                ans_start_index_l = max(i for i, x in enumerate(lab_tokens) if x == ans[0])
                if ans[0] in pred_tokens:
                    ans_start_index_p = max(i for i, x in enumerate(pred_tokens) if x == ans[0])
                else:
                    ans_start_index_p = ans_start_index_l

                pred_cot_tokens = pred_tokens[:ans_start_index_p].tolist()
                lab_cot_tokens = lab_tokens[:ans_start_index_l]
                all_preds_cot.append(pred_cot_tokens)
                all_labels_cot.append(lab_cot_tokens)
                all_preds_ans.append(pred_tokens[ans_start_index_p:].tolist())
                all_labels_ans.append(lab_tokens[ans_start_index_l:])
                all_preds.append(pred_tokens.tolist())
                all_labels.append(lab_tokens)
        cot_correct = [p == l for p, l in zip(all_preds_cot, all_labels_cot)]
        ans_correct = [p == l for p, l in zip(all_preds_ans, all_labels_ans)]
        res = {'accuracy_cot': np.mean(cot_correct), 'accuracy_ans': np.mean(ans_correct)}
        return res

    logger.info("Starting evaluation...")

    model.to(device)
    model.eval()

    results = evaluate(model, valid_dataset, device=device, bs=args.batch_size)
    logger.info(f"Evaluation results: {results}")
    print("Evaluation results:", results) 