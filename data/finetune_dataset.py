from torch.utils.data import Dataset
from PIL import Image
import PIL
import os
import numpy as np
import json


class FinetuneDataset(Dataset):
    def __init__(self, tokenizer, data_path, seq_length=1024, mode='train'):
        self.beg_prompt = ""
        self.tokenizer = tokenizer
        self.data = self.get_question_data(data_path)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.ignore_token_id = -100
        self.mode = mode
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        qid, question_text, answer_text = data['qid'], data['question'], data['answer']

        # process question and answer as separate sequences and combine them into a single sequence
        question_text = self.beg_prompt + question_text + self.tokenizer.eos_token
        answer_text = answer_text + self.tokenizer.eos_token

        tokenized_question = self.tokenizer(question_text)
        tokenized_answer = self.tokenizer(answer_text)
        question_input_ids = tokenized_question['input_ids']
        answer_input_ids = tokenized_answer['input_ids']

        input_ids = question_input_ids + answer_input_ids

        # ignore the initial prompt for generation
        ignore_input_tokens = [self.ignore_token_id] * len(question_input_ids)
        label_ids = ignore_input_tokens + answer_input_ids
        attention_mask = [1] * len(input_ids)

        if len(input_ids) < self.seq_length:
            padding_length = self.seq_length - len(input_ids)
            input_ids = np.pad(input_ids, (0, padding_length), 'constant', constant_values=self.pad_token_id)
            attention_mask = np.pad(attention_mask, (0, padding_length), 'constant', constant_values=0)
            label_ids = np.pad(label_ids, (0, padding_length), 'constant', constant_values=self.ignore_token_id)
        elif len(input_ids) == self.seq_length:
            pass
        else:
            trunc_length = len(input_ids) - self.seq_length
            input_ids = input_ids[:-trunc_length]
            attention_mask = attention_mask[:-trunc_length]
            label_ids = label_ids[:-trunc_length]

        assert len(input_ids) == len(attention_mask), f"{len(input_ids)} != {len(attention_mask)}"
        assert len(input_ids) == len(label_ids), f"{len(input_ids)} != {len(label_ids)}"

        print(f"input_ids shape: {len(input_ids)}, attention_mask shape: {len(attention_mask)}, labels shape: {len(label_ids)}")

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label_ids}


class MimicFinetuneDataset(FinetuneDataset):
    def get_question_data(self, data_path):
        # Load the JSON file
        with open(data_path, 'r') as f:
            data = json.load(f)
        # Initialize an empty list to store the filtered data
        filtered_data = []
        for i, entry in enumerate(data):
            image_path = entry.get('image_path', '')
            answer = entry.get('answer', [])
            # ignore questions with multiple answers
            if len(answer) != 1:
                continue
            answer = answer[0]
            if answer not in ['yes', 'no']:
                continue
            content_type = entry.get('content_type', '')
            # Construct the full image path
            full_image_path = self.img_root + image_path
            if os.path.exists(full_image_path) and answer:
                # Image exists and is valid, include this entry
                question = entry['question']
                idx = entry['idx']
                filtered_data.append({"qid": idx, "img_id": idx, "question": question, "img_name": image_path,
                                      "answer": answer, "q_lang": "en", "content_type": content_type})
            else:
                # Image does not exist or answer is empty, skip this entry
                pass
        return filtered_data

    @property
    def img_root(self):
        # return "/data1/mimic_shared/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
        return "/local2/yijia.xiao/images/2.0.0/files/"


class KGFinetuneDataset(FinetuneDataset):
    def get_question_data(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        question_data = []
        for i, entry in enumerate(data):
            question = entry['question']
            answer = entry['answer']
            qid = entry['qid']
            question_data.append({"qid": qid, "img_id": None, "question": question, "img_name": None,
                                  "answer": answer, "q_lang": "en", "content_type": "KG"})
        return question_data