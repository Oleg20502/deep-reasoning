from datasets import Dataset, DatasetDict
import json

# Load your JSON file
with open("prosqa_example.json", "r") as f:
    data = json.load(f)

# Flatten the nested structure: each row is a test_example dict
examples = [v["test_example"] for v in data.values()]

# Create the HuggingFace Dataset
ds = Dataset.from_list(examples)

# Optionally, wrap in a DatasetDict for a 'train' split
ds_dict = DatasetDict({"train": ds})

# Push to HuggingFace Hub
# Replace with your actual dataset name and token
ds_dict.push_to_hub('alexlegeartis/prosqa', token='token')