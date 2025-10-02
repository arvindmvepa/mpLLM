from abc import abstractmethod
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch
from transformers import DefaultDataCollator
from tqdm import tqdm
import os
import json
import numpy as np
from model.base_model import BaseLLM
from data.dataset_factory import get_dataset_factory
from utils.vision_to_llm_utils import process_textbook_fig_cap_dataset
from utils.llm_eval_utils import collect_mc_answers_from_llm_output


class VisionToLLMTrainer(BaseLLM):
    def __init__(self, exp_file=None, use_wandb=False):
        super().__init__(exp_file=exp_file, use_wandb=use_wandb)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):
        """Evaluate the model on specified questions and write the answers to a json file"""
        inf_data = self.get_inf_data()
        print(f"Number of samples: {len(inf_data)}")
        model, image_processor = self.load_inf_model()
        inf_data.update_transforms_w_processor(image_processor)
        model.eval()
        decoding_kwargs = self.params['inf']["decoding_kwargs"]
        content = []
        print("test saving the results")
        if hasattr(model, "language_projection"):
            if hasattr(model.language_projection, "save_stats"):
                model.language_projection.save_stats(filename=os.path.join(self.output_dir, "moe_stats.json"))
        for i in range(len(inf_data)):
            sample = inf_data[i]
            qid = sample['qid']
            print(f"Question ID: {qid}")
            orig_question_text = sample['question']
            question_text = self.prepare_question(orig_question_text)
            print(f"Question with context: {question_text}")
            inputs = self.apply_tokenizer(question_text).to(self.device)
            pixel_values = sample['pixel_values'].to(self.device, dtype=self.pixel_values_dtype)
            results = {}
            if hasattr(model, "add_multitask") and model.add_multitask:
                if torch.all(pixel_values == 0) or self.params["inf"]["llm_only"]:
                    model_results = self.language_model_generate(model, inputs, decoding_kwargs)
                else:
                    model_results = self.generate(model, inputs, pixel_values, decoding_kwargs)
                generated_ids = model_results['sequences']
                area_logits = model_results["area_logits"][0]
                shape_logits = model_results["shape_logits"][0]
                satellite_logits = model_results["satellite_logits"][0]
                region_logits = model_results["region_logits"][0]
                answer_raw = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False,
                                                         clean_up_tokenization_spaces=False)[0]
                answer_clean = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=True)[0]
                answer_raw_wo_question = answer_raw.split(question_text)[-1]
                results.update({'area_logits': area_logits, 'area_label': sample['area_label'],
                                'shape_logits': shape_logits, 'shape_label': sample['shape_label'],
                                'satellite_logits': satellite_logits, 'satellite_label': sample['satellite_label'],
                                'region_logits': region_logits, 'region_label': sample['region_label']})
                if hasattr(model, "add_multitask_unknown"):
                    unknown_logits = model_results["unknown_logits"][0]
                    results.update({'unknown_logits': unknown_logits,
                                    'unknown_label': sample['unknown_label']})
            else:
                if torch.all(pixel_values == 0) or self.params["inf"]["llm_only"]:
                    generated_ids = self.language_model_generate(model, inputs, decoding_kwargs)
                else:
                    generated_ids = self.generate(model, inputs, pixel_values, decoding_kwargs)
                answer_raw = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False,
                                                         clean_up_tokenization_spaces=False)[0]
                answer_clean = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=True)[0]
                answer_raw_wo_question = answer_raw.split(question_text)[-1]
            # add additional content for the save file (for possibly re-loading to train/evaluate)
            answer = sample['answer']
            q_lang = sample['q_lang']
            img_id = sample.get('img_id', None)
            img_name = sample.get('img_name', None)
            location = sample.get('location', None)
            modality = sample.get('modality', None)
            answer_type = sample.get('answer_type', None)
            base_type = sample.get('base_type', None)
            content_type = sample.get('content_type', None)
            triple = sample.get('triple', None)
            results.update({'img_id': img_id, 'img_name': img_name, 'orig_question': orig_question_text,
                            'question': question_text, 'model_answer': answer_clean,
                            'model_raw_answer_wo_question': answer_raw_wo_question, 'answer': answer, 'q_lang': q_lang,
                            'location': location, 'modality': modality, 'answer_type': answer_type,
                            'base_type': base_type, 'content_type': content_type, 'triple': triple, "qid": qid})
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, torch.Tensor):  # This handles torch.Tensor and monai.MetaTensor
                    # Detach tensor from graph, move to CPU (safer), convert to list
                    serializable_results[key] = value.detach().cpu().tolist()
                elif isinstance(value, np.ndarray):  # Handle numpy arrays too
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value
            content.append(serializable_results)
            print(f"Answer: {answer_raw}")
        if hasattr(model, "language_projection"):
            if hasattr(model.language_projection, "save_stats"):
                model.language_projection.save_stats(filename=os.path.join(self.output_dir, "moe_stats.json"))

        # Save results to json file
        save_file = os.path.join(self.output_dir, self.params['inf']['save_file'])

        with open(save_file, 'w') as f:
            json.dump(content, f, indent=4)

        # clean answers
        if self.params['inf']['clean_mc']:
            collect_mc_answers_from_llm_output(save_file, save_file)

        # Optionally process answers and calculate metrics
        self.get_eval_metrics(save_file, os.path.join(self.output_dir, self.params['inf']['results_file']))

    def generate_dataset_image_embeddings(self):
        """Generate image embeddings from the vision encoder"""
        inf_data = self.get_inf_data()
        print(f"Number of samples: {len(inf_data)}")
        model, image_processor = self.load_inf_model()
        inf_data.update_transforms_w_processor(image_processor)
        embedding_dir = os.path.join(self.output_dir, "image_embeddings")
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)
        print("Generating image embeddings...")
        for i in tqdm(range(len(inf_data))):
            sample = inf_data[i]
            qid = sample['qid']
            location = sample.get('location')
            modality = sample.get('modality')
            label_name = sample.get('label_name')

            # Get pixel values and convert to NumPy float16
            pixel_values = sample['pixel_values'].to(self.device, dtype=self.pixel_values_dtype)
            image_embeddings = self.generate_image_embeddings(model, pixel_values)
            image_embeddings_np = image_embeddings.detach().cpu().numpy().astype(np.float16)

            # Generate a unique file name
            file_name = f"qid-{qid}_loc-{location}_mod-{modality}_label-name-{label_name}.npy"
            file_path = os.path.join(embedding_dir, file_name)

            # Save the embedding to a file
            np.savez_compressed(file_path, image_embeddings=image_embeddings_np)

    @abstractmethod
    def load_train_model(self):
        raise NotImplementedError()

    @abstractmethod
    def load_inf_model(self):
        raise NotImplementedError()

    @abstractmethod
    def load_vision_model(self):
        raise NotImplementedError()

    def get_train_data(self):
        if self.params['data']['train_dataset'][-2:] == "lm":
            return self.get_fig_cap_train_data()
        elif "instruct_vqa" in self.params['data']['train_dataset']:
            return self.get_fig_cap_train_data()
        elif self.params['data']['train_dataset'][-3:] == "vqa":
            return self.get_vqa_train_data()
        else:
            raise ValueError("train_data_type must be either 'fig_cap' or 'vqa'")

    def get_fig_cap_train_data(self):
        """Load the figure/caption dataset for training"""
        if not os.path.exists(os.path.join(self.output_dir, self.params['data']['data_path'])):
            process_textbook_fig_cap_dataset(base_dir=self.params['data']['base_dir'],
                                             save_file=self.params['data']['data_path'])
        means, stds = self.get_vision_model_norm_params()
        dataset_factory = get_dataset_factory(dataset_type=self.params['data']['train_dataset'])
        full_data = dataset_factory.create_dataset(tokenizer=self.tokenizer, data_path=self.params['data']['data_path'],
                                                   prompt_type=self.params['data']['prompt_type'], beg_prompt="",
                                                   mid_prompt="", end_prompt="", img_dir=self.params['data']['img_dir'],
                                                   height=self.params['data']['height'],
                                                   width=self.params['data']['width'],
                                                   num_channels=self.params['data']['num_channels'],
                                                   means=means, stds=stds, img_tokens=self.params['data']['img_tokens'],
                                                   pad_token_str=self.pad_token,
                                                   img_token_str=self.img_token,
                                                   seq_length=self.params['data']['seq_length'], mode='train')
        indices = list(range(len(full_data)))
        train_indices, test_indices = train_test_split(indices, test_size=self.params['data']['test_size'],
                                                       random_state=self.params['data']['data_seed'])
        train_data = full_data
        train_data = Subset(train_data, train_indices)
        test_data = dataset_factory.create_dataset(tokenizer=self.tokenizer, data_path=self.params['data']['data_path'],
                                                   prompt_type=self.params['data']['prompt_type'], beg_prompt="",
                                                   mid_prompt="", end_prompt="", img_dir=self.params['data']['img_dir'],
                                                   height=self.params['data']['height'],
                                                   width=self.params['data']['width'],
                                                   num_channels=self.params['data']['num_channels'],
                                                   means=means, stds=stds, img_tokens=self.params['data']['img_tokens'],
                                                   pad_token_str=self.pad_token,
                                                   img_token_str=self.img_token,
                                                   seq_length=self.params['data']['seq_length'], mode='val')
        test_data = Subset(test_data, test_indices)
        return {"train": train_data, "test": test_data}

    def get_vqa_train_data(self):
        """Load the vqa dataset for training"""
        means, stds = self.get_vision_model_norm_params()
        dataset_factory = get_dataset_factory(dataset_type=self.params['data']['train_dataset'])
        train_data = dataset_factory.create_dataset(tokenizer=self.tokenizer, beg_prompt="", mid_prompt="", end_prompt="",
                                                    prompt_type=self.params['data']['prompt_type'],
                                                    data_path=self.params['data']['data_path'],
                                                    img_dir=self.params['data']['img_dir'],
                                                    height=self.params['data']['height'],
                                                    width=self.params['data']['width'],
                                                    num_channels=self.params['data']['num_channels'],
                                                    means=means, stds=stds,
                                                    img_tokens=self.params['data']['img_tokens'],
                                                    pad_token_str=self.pad_token,
                                                    img_token_str=self.img_token,
                                                    seq_length=self.params['data']['seq_length'], mode='train')
        test_data = dataset_factory.create_dataset(tokenizer=self.tokenizer, beg_prompt="", mid_prompt="", end_prompt="",
                                                   prompt_type=self.params['data']['prompt_type'],
                                                   data_path=self.params['data']['inf_data_path'],
                                                   img_dir=self.params['data']['inf_img_dir'],
                                                   height=self.params['data']['height'],
                                                   width=self.params['data']['width'],
                                                   num_channels=self.params['data']['num_channels'],
                                                   means=means, stds=stds, img_tokens=self.params['data']['img_tokens'],
                                                   pad_token_str=self.pad_token,
                                                   img_token_str=self.img_token,
                                                   seq_length=self.params['data']['seq_length'], mode='val')
        return {"train": train_data, "test": test_data}

    def get_inf_data(self):
        dataset_factory = get_dataset_factory(dataset_type=self.params['data']['inf_dataset'])
        return dataset_factory.create_dataset(tokenizer=self.tokenizer, prompt_type=self.params['data']['prompt_type'],
                                              beg_prompt=self.params['inf']['beg_prompt'],
                                              mid_prompt=self.params['inf']['mid_prompt'],
                                              end_prompt=self.params['inf']['end_prompt'],
                                              replace_prompt=self.params['inf']["replace_prompt"],
                                              data_path=self.params['data']['inf_data_path'],
                                              img_dir=self.params['data']['inf_img_dir'],
                                              height=self.params['data']['height'], width=self.params['data']['width'],
                                              num_channels=self.params['data']['num_channels'],
                                              img_tokens=self.params['data']['img_tokens'],
                                              img_token_str=self.img_token if self.params['inf']['include_img'] else "",
                                              seq_length=self.params['data']['seq_length'], mode='test')

    def get_eval_metrics(self, pred_file, results_file):
        """Calculate the metrics for the generated answers"""
        inf_data = self.get_inf_data()
        inf_data.calculate_metrics(self.params['data']['inf_data_path'], pred_file, results_file)

    def generate(self, model, inputs, pixel_values, decoding_kwargs):
        generated_ids = model.generate(**inputs, pixel_values=pixel_values, num_return_sequences=1,
                                       max_new_tokens=self.params['inf']['max_new_tokens'], **decoding_kwargs)
        return generated_ids

    def generate_image_embeddings(self, model, pixel_values):
        image_embeddings = model.generate_image_embeddings(pixel_values)
        return image_embeddings

    def language_model_generate(self, model, inputs, decoding_kwargs):
        generated_ids = model.language_model.generate(**inputs, num_return_sequences=1,
                                                      max_new_tokens=self.params['inf']['max_new_tokens'],
                                                      **decoding_kwargs)
        return generated_ids

    def apply_tokenizer(self, text):
        return self.tokenizer(text, return_tensors="pt")

    def get_vision_model_norm_params(self):
        raise NotImplementedError()

    def get_data_collator(self):
        return DefaultDataCollator()

    @property
    def required_params(self):
        required_params = super().required_params
        required_params["data"] = required_params["data"] + ["train_dataset", "inf_dataset", "base_dir", "img_dir",
                                                             "height", "width", "num_channels", "img_tokens",
                                                             "seq_length", "inf_img_dir", "inf_data_path",
                                                             "kg_embedder_params", "prompt_type"]
        required_params["train"] = required_params["train"] + ["llm_model_name"]
        required_params["inf"] = required_params["inf"] + ["context_prompt", "replace_prompt", "max_new_tokens", "llm_only",
                                                           "decoding_kwargs", "save_file", "results_file", "top_k",
                                                           "similarity_threshold", "clean_mc", "include_img"]
        assert isinstance(required_params, dict), "required_params must be a dictionary"
        return required_params

    @property
    def img_token(self):
        return "<image>"

    @property
    def pad_token(self):
        return '<|finetune_right_pad_id|>'

    @property
    @abstractmethod
    def pixel_values_dtype(self):
        raise NotImplementedError()
