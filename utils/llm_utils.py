from datasets import load_dataset
from random import sample
import csv
import sys
import json
csv.field_size_limit(sys.maxsize)

def load_dataset_from_file(data_path, test_size=0.2, seed=None):
    data = load_dataset("csv", data_files=data_path)
    data = data['train']
    data = data.train_test_split(test_size=test_size, seed=seed)
    return data

def process_textbook_dataset(data, tokenizer, block_size=512, overlap=True, overlap_space=1, debug=False,
                             debug_file="dataset_debug.csv", old=False):
    def preprocess_function(examples):
        if old:
            return tokenizer([" ".join(x) for x in examples["text"]])
        return tokenizer(examples["text"])
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        if overlap:
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, overlap_space) if i + block_size < total_length]
                for k, t in concatenated_examples.items()
            }
        else:
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        result["labels"] = result["input_ids"].copy()
        return result
    tokenized_data = data.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=data["train"].column_names,
    )
    if debug:
        num_samples = len(tokenized_data['train'])
        list_keys = list(range(num_samples))
        sample_keys = sample(list_keys, 5)
        for key in sample_keys:
            text = tokenized_data['train'][key]
            print(text)
            output_text = tokenizer.decode(text['input_ids'], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=True)
            print(f"tokenized_data {key}: {output_text}")

    lm_dataset = tokenized_data.map(group_texts, batched=True, num_proc=4)
    if debug:
        num_samples = len(lm_dataset['train'])
        list_keys = list(range(num_samples))
        sample_keys = sample(list_keys, 5)
        for key in sample_keys:
            text = lm_dataset['train'][key]
            print(text)
            output_text = tokenizer.decode(text['input_ids'], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=True)
            print(f"lm_dataset {key}: {output_text}")
        with open(debug_file, 'w', newline='') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['id', 'inputs', 'outputs'])
            for key in list_keys:
                text = lm_dataset['train'][key]
                input = tokenizer.decode(text['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                output = tokenizer.decode(text['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                csvwriter.writerow([key, input, output])
    return lm_dataset


def load_results(json_file):
    with open(json_file, 'r') as f:
        results = json.load(f)
    return results
