import datasets
from functools import partial
from sklearn.model_selection import train_test_split
from lm_experiments_tools.instruction_utils import formatting_func_dispatcher
import numpy as np

# class DatasetMix:
#     def __init__(self, datasets, ratios=None, random_seed=42):
#         self.datasets = datasets 
#         self.gen = np.random.default_rng(seed=random_seed)
        
#         if ratios is None:
#             ratios = [1/len(datasets) for _ in datasets]
#         self.ratios = ratios
#         self.column_names = datasets[0].column_names
#         self.seed = random_seed

#     def __getitem__(self, ind):
#         dataset_ind = self.gen.choice(len(self.datasets), p=self.ratios)
#         dataset = self.datasets[dataset_ind]
#         return dataset[ind]

#     def __len__(self):
#         return min([len(d) for d in self.datasets])
    
#     def map(self, *args, **kwargs):
#         results = [d.map(*args, **kwargs) for d in self.datasets]
#         return DatasetMix(results, self.ratios, self.seed)

def combine_datasets(datasets_, ratios, target_size):
    target_sizes = [int(target_size * r) for r in ratios]
    datasets_scaled = []
    for dataset, target_size in zip(datasets_, target_sizes):
        if len(dataset) >= target_size:
            dataset_scaled = dataset.select(range(target_size))
        else:
            repeated_dataset = [dataset] * (int(target_size / len(dataset)) + 1)
            dataset_scaled = datasets.concatenate_datasets(repeated_dataset).select(range(target_size))
        datasets_scaled.append(dataset_scaled)
    return datasets.concatenate_datasets(datasets_scaled)

    
def load_and_preprocess_task(task_name, tokenizer, sample_size,
                             max_n_segments, num_mem_tokens,
                             use_length_filtering, reduce_eval):
    """ Loads and preprocess dataset based on task_name
    """
    if 'wikitext' in task_name:
        raw_datasets = datasets.load_dataset('wikitext', task_name)
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
    elif 'arxiv' in task_name:
        tokenized_datasets = datasets.load_from_disk('/home/bulatov/bulatov/datasets/arxiv_pile/processed/')
    elif 'babilong' in task_name:
        dname, task, split = task_name.split('_')
        dataset_train = datasets.load_dataset("RMT-team/babilong-train-5k-samples", split)[task]
        dataset_test = datasets.load_dataset("RMT-team/babilong", split)[task]

        formatting_func = partial(formatting_func_dispatcher(task_name), tokenizer=tokenizer)
        formatted_dataset_train = dataset_train.map(formatting_func)
        formatted_dataset_test = dataset_test.map(formatting_func)

        indices = np.arange(len(formatted_dataset_train))
        splits = ["train", "test", "test"]
        formatted_dataset = datasets.DatasetDict(dict(train=formatted_dataset_train,
                                                      validation=formatted_dataset_test,
                                                      test=formatted_dataset_test))
        def tokenize_function(examples):
            # return tokenizer(examples["text"])
            return {"input_ids": examples["text"], 'labels_mask': examples['labels_mask']}
        column_names = formatted_dataset["train"].column_names
        # print('\n\n\n\n\nBABILONG')
        # formatted_dataset.save_to_disk('/home/jovyan/rmt/armt-instruct-tune/babilong_tmp')
        tokenized_datasets = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
    elif 'flan' in task_name or "dolly" in task_name or "longdolly" in task_name or "LongAlign" in task_name or "LAMix" in task_name or "sharegpt" in task_name or "ultrachat" in task_name or "synth" in task_name or "smoltalk" in task_name or "docfinqa_golden" in task_name:
        if "longdolly" in task_name:
            dname = f"long_dolly_15k/dolly_15k_{sample_size+2*num_mem_tokens*max_n_segments}"
            dataset = datasets.load_from_disk(f'/home/jovyan/gkuzmin/rmt_it/data/long_dolly_15k/dolly_15k_{sample_size+2*num_mem_tokens*max_n_segments}')
            data_dict = {"train": dataset}
            dataset = datasets.DatasetDict(data_dict)
        elif "LongAlign" in task_name:
            dname = "THUDM/LongAlign-10k"
            dataset = datasets.load_dataset(dname)
            dataset["train"] = dataset["train"].filter(lambda example: example["length"] < sample_size * max_n_segments)
        elif "LAMix" in task_name:
            dname = "LAMix"
            dataset = datasets.load_dataset("../data/LongAlign_sg_16k_mix")
            dataset["train"] = dataset["train"].filter(lambda example: example["length"] < sample_size * max_n_segments)
        elif "flan_mix" in task_name:
            dname = "flan_long_mix"
            dataset = datasets.load_from_disk("../data/flan_long_mix")
        elif "sharegpt" in task_name:
            dname = "sharegpt"
            dataset = datasets.load_from_disk("../data/sharegpt_raw_cleaned/hf_dataset_sharegpt")
        elif "ultrachat" in task_name:
            dname = "ultrachat"
            dataset = datasets.load_from_disk("../data/ultrachat_200k_flattened")
        elif "dolly_mix" in task_name:
            dname = "dolly_mix"
            dataset = datasets.load_dataset("../data/dolly_long_mix")
        elif "dolly_long" in task_name:
            dname = task_name
            dataset = datasets.load_from_disk(f"../data/{task_name}")
        elif "short_synth" in task_name:
            dname = "short_synth"
            dataset = datasets.load_from_disk("../data/long_context_raw/llama31_8b_short_pile_labelled")
            dataset = datasets.DatasetDict({"train": dataset})
        elif "2k_synth" in task_name:
            dname = "2k_synth"
            dataset = datasets.load_dataset("AIRI-NLP/long_context_it_pile_unc_len2k_llama31_8b")
        elif "synth_reduced" in task_name:
            dname = task_name
            dataset = datasets.load_from_disk(f"../data/{task_name}")
            if not isinstance(dataset, datasets.DatasetDict):
                dataset = datasets.DatasetDict({"train": dataset})
        elif "smoltalk_long" in task_name:
            dname = task_name
            dataset = datasets.load_from_disk(f"../data/{task_name}")
        elif "smoltalk" in task_name:
            dname = "smoltalk"
            dataset = datasets.load_dataset("HuggingFaceTB/smoltalk", "all")
        elif "docfinqa_golden" in task_name:
            dname = "docfinqa_golden"
            dataset = datasets.load_dataset("AIRI-NLP/docfinqa_golden")
        else:
            dname = "glkuzi/flan_collection" if 'flan' in task_name else "databricks/databricks-dolly-15k"
            try:
                dataset = datasets.load_dataset(dname)
            except:
                dataset = datasets.load_from_disk(dname)
        formatting_func = partial(formatting_func_dispatcher(task_name), tokenizer=tokenizer)
        formatted_dataset = dataset.map(formatting_func)
        #formatted_dataset = dataset
        indices = np.arange(len(formatted_dataset["train"]))
        # in case of flan -- use only 1% for test
        if "flan" in dname:
            train_ids, test_ids = train_test_split(indices, test_size=0.01, train_size=0.99, random_state=42)
            val_ids, test_ids = train_test_split(test_ids, test_size=0.5, train_size=0.5, random_state=42)
        elif "smoltalk" in dname:
            # for smoltalk we have train and test, so we simply split val from train
            train_ids, val_ids = train_test_split(indices, test_size=0.05, train_size=0.95, random_state=42)
        else:
            train_ids, test_ids = train_test_split(indices, test_size=0.1, train_size=0.9, random_state=42)
            val_ids, test_ids = train_test_split(test_ids, test_size=0.5, train_size=0.5, random_state=42)
        if reduce_eval is not None:
            val_ids = val_ids[:int(reduce_eval*len(val_ids))]
        splits = ["train", "validation", "test"]
        if not(all([key in formatted_dataset.keys() for key in splits])):
            if "smoltalk" in dname:
                splits = ["train", "validation"]
                all_indices = [train_ids, val_ids]
                data_dict = {"test": formatted_dataset["test"]}
            else:
                all_indices = [train_ids, val_ids, test_ids]
                data_dict = {}
            for split, indices in zip(splits, all_indices):
                data_dict[split] = formatted_dataset["train"].select(indices)
            formatted_dataset = datasets.DatasetDict(data_dict)
        #formatted_dataset.save_to_disk(f"../data/{task_name}_splitted")
        def tokenize_function(examples):
            return tokenizer(examples["text"])
        column_names = formatted_dataset["train"].column_names
        # print('\n\n\n\n\nDOLLY MV')
        # formatted_dataset.save_to_disk('/home/jovyan/rmt/armt-instruct-tune/dolly_tmp')
        tokenized_datasets = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
        if use_length_filtering:
            tokenized_datasets = tokenized_datasets.filter(lambda x: len(x["input_ids"]) < sample_size)
    else:
        raise ValueError(f"Unknown dataset {task_name}")
    return tokenized_datasets

# def load_and_preprocess_task(task_name, tokenizer, sample_size,
#                              max_n_segments, num_mem_tokens,
#                              use_length_filtering):
#     """ Loads and preprocess dataset based on task_name
#     """
#     if 'wikitext' in task_name:
#         raw_datasets = datasets.load_dataset('wikitext', task_name)
#         column_names = raw_datasets["train"].column_names
#         text_column_name = "text" if "text" in column_names else column_names[0]

#         def tokenize_function(examples):
#             return tokenizer(examples[text_column_name])

#         tokenized_datasets = raw_datasets.map(
#             tokenize_function,
#             batched=True,
#             remove_columns=column_names,
#             desc="Running tokenizer on dataset",
#         )
#     elif 'arxiv' in task_name:
#         tokenized_datasets = datasets.load_from_disk('/home/bulatov/bulatov/datasets/arxiv_pile/processed/')
#     elif 'babilong' in task_name:
#         dname, task, split = task_name.split('_')
#         dataset_train = datasets.load_dataset("RMT-team/babilong-train-5k-samples", split)[task]
#         dataset_test = datasets.load_dataset("RMT-team/babilong", split)[task]

#         formatting_func = partial(formatting_func_dispatcher(task_name), tokenizer=tokenizer)
#         formatted_dataset_train = dataset_train.map(formatting_func)
#         formatted_dataset_test = dataset_test.map(formatting_func)

#         indices = np.arange(len(formatted_dataset_train))
#         splits = ["train", "test", "test"]
#         formatted_dataset = datasets.DatasetDict(dict(train=formatted_dataset_train,
#                                                       validation=formatted_dataset_test,
#                                                       test=formatted_dataset_test))
#         def tokenize_function(examples):
#             return tokenizer(examples["text"])
#         column_names = formatted_dataset["train"].column_names
#         tokenized_datasets = formatted_dataset.map(
#             tokenize_function,
#             batched=True,
#             remove_columns=column_names,
#             desc="Running tokenizer on dataset",
#         )
#     elif 'flan' in task_name or "dolly" in task_name or "longdolly" in task_name or "LongAlign" in task_name or "LAMix" in task_name or "sharegpt" in task_name or "ultrachat" in task_name or "synth" in task_name:
#         if "longdolly" in task_name:
#             dname = f"long_dolly_15k/dolly_15k_{sample_size+2*num_mem_tokens*max_n_segments}"
#             dataset = datasets.load_from_disk(f'/home/jovyan/gkuzmin/rmt_it/data/long_dolly_15k/dolly_15k_{sample_size+2*num_mem_tokens*max_n_segments}')
#             data_dict = {"train": dataset}
#             dataset = datasets.DatasetDict(data_dict)
#         elif "LongAlign" in task_name:
#             dname = "THUDM/LongAlign-10k"
#             dataset = datasets.load_dataset(dname)
#             dataset["train"] = dataset["train"].filter(lambda example: example["length"] < sample_size * max_n_segments)
#         elif "LAMix" in task_name:
#             dname = "LAMix"
#             dataset = datasets.load_dataset("../data/LongAlign_sg_16k_mix")
#             dataset["train"] = dataset["train"].filter(lambda example: example["length"] < sample_size * max_n_segments)
#         elif "flan_mix" in task_name:
#             dname = "flan_long_mix"
#             dataset = datasets.load_from_disk("../data/flan_long_mix")
#         elif "sharegpt" in task_name:
#             dname = "sharegpt"
#             dataset = datasets.load_from_disk("../data/sharegpt_raw_cleaned/hf_dataset_sharegpt")
#         elif "ultrachat" in task_name:
#             dname = "ultrachat"
#             dataset = datasets.load_from_disk("../data/ultrachat_200k_flattened")
#         elif "dolly_mix" in task_name:
#             dname = "dolly_mix"
#             dataset = datasets.load_dataset("../data/dolly_long_mix")
#         elif "dolly_long" in task_name:
#             dname = task_name
#             dataset = datasets.load_from_disk(f"../data/{task_name}")
#         elif "short_synth" in task_name:
#             dname = "short_synth"
#             dataset = datasets.load_from_disk("../data/long_context_raw/llama31_8b_short_pile_labelled")
#             dataset = datasets.DatasetDict({"train": dataset})
#         else:
#             dname = "glkuzi/flan_collection" if 'flan' in task_name else "databricks/databricks-dolly-15k"
#             try:
#                 dataset = datasets.load_dataset(dname)
#             except:
#                 dataset = datasets.load_from_disk(dname)
#         formatting_func = partial(formatting_func_dispatcher(task_name), tokenizer=tokenizer)
#         formatted_dataset = dataset.map(formatting_func)
#         indices = np.arange(len(formatted_dataset["train"]))
#         # in case of flan -- use only 1% for test
#         if "flan" in dname:
#             train_ids, test_ids = train_test_split(indices, test_size=0.01, train_size=0.99, random_state=42)
#             val_ids, test_ids = train_test_split(test_ids, test_size=0.5, train_size=0.5, random_state=42)
#         else:
#             train_ids, test_ids = train_test_split(indices, test_size=0.1, train_size=0.9, random_state=42)
#             val_ids, test_ids = train_test_split(test_ids, test_size=0.5, train_size=0.5, random_state=42)
#         splits = ["train", "validation", "test"]
#         all_indices = [train_ids, val_ids, test_ids]
#         data_dict = {}
#         for split, indices in zip(splits, all_indices):
#             data_dict[split] = formatted_dataset["train"].select(indices)
#         formatted_dataset = datasets.DatasetDict(data_dict)
#         def tokenize_function(examples):
#             return tokenizer(examples["text"])
#         column_names = formatted_dataset["train"].column_names
#         tokenized_datasets = formatted_dataset.map(
#             tokenize_function,
#             batched=True,
#             remove_columns=column_names,
#             desc="Running tokenizer on dataset",
#         )
#         if use_length_filtering:
#             tokenized_datasets = tokenized_datasets.filter(lambda x: len(x["input_ids"]) < sample_size)
#     else:
#         raise ValueError(f"Unknown dataset {task_name}")
#     return tokenized_datasets
