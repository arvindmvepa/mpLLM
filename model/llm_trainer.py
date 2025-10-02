from transformers import DefaultDataCollator
import os
import csv
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from data.dataset_factory import get_dataset_factory
from utils.llm_utils import load_dataset_from_file, process_textbook_dataset
from model.base_model import BaseLLM


class LLMTrainer(BaseLLM):
    def __init__(self, exp_file=None, use_wandb=False):
        super().__init__(exp_file=exp_file)
        self.use_wandb = use_wandb

    def setup(self):
        super(LLMTrainer, self).setup()
        self.setup_tokenizer()

    def run(self):
        self.train()
        self.evaluate()

    def evaluate(self):
        """Evaluate the model on specified questions and write the answers to a csv file"""
        model = self.load_inf_model()
        model.eval()
        decoding_kwargs = self.params['inf']["decoding_kwargs"]
        content = []
        if isinstance(self.params['inf']['questions'], str) and os.path.exists(self.params['inf']['questions']):
            if ".csv" in self.params['inf']['questions']:
                question_df = pd.read_csv(self.params['inf']['questions'])
                questions = question_df['question'].tolist()
            elif ".json" in self.params['inf']['questions']:
                with open(self.params['inf']['questions'], 'r') as f:
                    data = json.load(f)
                questions = [datum['question'] for datum in data]
            else:
                raise ValueError("questions file must be a csv or json file")
        elif isinstance(self.params['inf']['questions'], list):
            questions = self.params['inf']['questions']
        else:
            raise ValueError("questions must be a list or file path")
        # Loop through questions and generate answers
        for question in questions:
            question = self.prepare_question(question)
            inputs = self.tokenizer(question, return_tensors="pt").to("cuda")
            generated_ids = model.generate(**inputs, num_return_sequences=1,
                                           max_new_tokens=self.params['inf']['max_new_tokens'],
                                           pad_token_id=self.tokenizer.pad_token_id,
                                           **decoding_kwargs)
            answer_raw = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False,
                                                     clean_up_tokenization_spaces=False)[0]
            answer_clean = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True)[0]
            question_clean = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)[0]
            answer_clean = answer_clean.split(question_clean)[-1]
            content.append((question, answer_clean))
            print(f"question: {question}")
            print(f"answer (raw): {answer_raw}")
            print(f"answer (clean): {answer_clean}")

        with open(os.path.join(self.output_dir, self.params['inf']['save_file']), mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["question", "answer"])
            for line in content:
                writer.writerow(line)

    def get_train_data(self):
        if "textbook" in self.params['data']['train_dataset']:
            return self.get_textbook_data()
        if "finetune" in self.params['data']['train_dataset']:
            return self.get_finetune_data()
        else:
            raise ValueError("train_data_type must be either 'textbook' or 'finetune'")

    def get_textbook_data(self):
        """Process the csv entries into overlapping chunks for language modeling"""
        unprocessed_dataset = load_dataset_from_file(self.params['data']["data_path"],
                                                     seed=self.params['data']["data_seed"],
                                                     test_size=self.params['data']["test_size"])
        data = process_textbook_dataset(unprocessed_dataset, self.tokenizer, block_size=self.params['data']["block_size"],
                                        overlap=self.params['data']["overlap"],
                                        overlap_space=self.params['data']["overlap_space"])
        return data

    def get_finetune_data(self):
        dataset_factory = get_dataset_factory(dataset_type=self.params['data']['train_dataset'])
        full_data = dataset_factory.create_dataset(tokenizer=self.tokenizer, data_path=self.params['data']['data_path'],
                                                   seq_length=self.params['data']['seq_length'])
        indices = list(range(len(full_data)))
        train_indices, test_indices = train_test_split(indices, test_size=self.params['data']['test_size'],
                                                       random_state=self.params['data']['data_seed'])
        train_data = Subset(full_data, train_indices)
        test_data = Subset(full_data, test_indices)
        return {"train": train_data, "test": test_data}

    def get_data_collator(self):
        return DefaultDataCollator()

    def load_train_model(self):
        model = self.load_llm_model(model_name=self.params['train']["model_name"],
                                    use_quantization=self.params['train']["use_quantization"],
                                    r=self.params['train']["r"], lora_alpha=self.params['train']["lora_alpha"],
                                    target_modules=self.params['train']["target_modules"],
                                    lora_dropout=self.params['train']["lora_dropout"],
                                    bias=self.params['train']["bias"],
                                    task_type=self.params['train']["task_type"])
        model.config.use_cache = False
        return model

    def load_inf_model(self):
        model = self.load_llm_model(model_name=self.params['inf']["model_name"],
                                    use_quantization=self.params['inf']["use_quantization"],
                                    r=self.params['inf']["r"], lora_alpha=self.params['inf']["lora_alpha"],
                                    target_modules=self.params['inf']["target_modules"],
                                    lora_dropout=self.params['inf']["lora_dropout"],
                                    bias=self.params['inf']["bias"],
                                    task_type=self.params['inf']["task_type"])
        return model

    @property
    def required_params(self):
        required_params = super(LLMTrainer, self).required_params
        required_params["data"] = required_params["data"] + ["train_dataset", "block_size", "overlap", "overlap_space",
                                                             "seq_length"]
        required_params["inf"] = ["model_name", "use_quantization", "r", "lora_alpha", "target_modules", "lora_dropout",
                                  "bias", "task_type", "decoding_kwargs", "questions", "save_file", "max_new_tokens"]
        assert isinstance(required_params, dict), "required_params must be a dictionary"
        return required_params
