from torch.utils.data import Dataset
from torchvision import transforms
import monai.transforms as mtf
from data.randaugment import RandomAugment
from utils.huggingface_utils import load_tokenizer_from_huggingface
from PIL import Image
import PIL
import torch
import numpy as np
import os
import re
import pandas as pd
import json
from data.llm_eval import VQAEval
import cv2


class VQADataset(Dataset):
    def __init__(self, tokenizer, prompt_type, beg_prompt, mid_prompt, end_prompt, data_path, replace_prompt=None,
                 img_dir='converted_neph_vqa_all_mc_updated/', height=224, width=224, num_channels=3,
                 img_tokens=197, pad_token_str='<|finetune_right_pad_id|>', img_token_str="<image>", seq_length=1024,
                 mode='train', means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225), filter_key='q_lang',
                 filter_cond='en', calculate_mae=True):
        self.prompt_type = prompt_type
        self.beg_prompt = beg_prompt
        self.mid_prompt = mid_prompt
        self.end_prompt = end_prompt
        self.replace_prompt = replace_prompt
        self.img_root = img_dir
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(pad_token_str)
        self.ignore_token_id = -100
        self.mode = mode
        self.filter_key = filter_key
        self.filter_cond = filter_cond
        self.calculate_mae = calculate_mae
        self.img_padding = [self.ignore_token_id for _ in range(img_tokens)]
        self.img_token_str = img_token_str
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.data = self.get_question_data(data_path)

        normalize = transforms.Normalize(means, stds)
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((self.height, self.width), interpolation=Image.BICUBIC),
                # transforms.RandomResizedCrop((self.height, self.width), scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                # transforms.RandomHorizontalFlip(),
                RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
                transforms.ToTensor(),
                normalize,
            ])
        elif self.mode == 'test' or self.mode == 'val':
            self.transform = transforms.Compose([
                transforms.Resize((self.height, self.width), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            raise ValueError('mode must be in [train, test]')

    def __getitem__(self, idx):
        data = self.data[idx]
        question_text, answer_text = data['question'], data['answer']
        if self.replace_prompt is not None:
            question_text = self.replace_prompt
        transform_img = self.prepare_image(data)
        question_text, answer_text = self.add_language_model_prompt(question_text, answer_text)
        item = {}
        if self.mode in ["train", "val"]:
            tokenized_question_answer = self.apply_tokenizer(question_text + answer_text)
            tokenized_question = self.apply_tokenizer(question_text)
            input_ids = tokenized_question_answer['input_ids']
            question_input_ids = tokenized_question['input_ids']

            #input_ids = input_ids[:-1]
            ignore_input_tokens = [self.ignore_token_id] * len(question_input_ids)
            #ignore_input_tokens = [self.ignore_token_id] * (len(question_input_ids) - 1)
            label_ids = ignore_input_tokens + input_ids[len(question_input_ids):]
            attention_mask = [1] * len(input_ids)

            if len(input_ids) < self.seq_length:
                padding_length = self.seq_length - len(input_ids)
                input_ids = np.pad(input_ids, (0, padding_length), 'constant', constant_values=self.pad_token_id)
                attention_mask = np.pad(attention_mask, (0, padding_length), 'constant', constant_values=0)
                label_ids = np.pad(label_ids, (0, padding_length), 'constant', constant_values=self.ignore_token_id)
            else:
                trunc_length = len(input_ids) - self.seq_length
                input_ids = input_ids[:-trunc_length]
                attention_mask = attention_mask[:-trunc_length]
                label_ids = label_ids[:-trunc_length]

            assert len(input_ids) == len(attention_mask), f"{len(input_ids)} != {len(attention_mask)}"
            assert len(input_ids) == len(label_ids), f"{len(input_ids)} != {len(label_ids)}"

            item['pixel_values'] = transform_img
            item['input_ids'] = input_ids
            item['attention_mask'] = attention_mask
            item['labels'] = label_ids
        elif self.mode == "test":
            item = dict(data)
            item['question'] = question_text
            if isinstance(transform_img, np.ndarray):
                item['pixel_values'] = np.expand_dims(transform_img, axis=0)
            else:
                item['pixel_values'] = transform_img.unsqueeze(0)
        else:
            raise ValueError(f'self.mode must be in [train, test], instead {self.mode}')
        return item

    def prepare_image(self, data):
        image_path = data['img_name']
        full_image_path = self.img_root + image_path
        img = PIL.Image.open(full_image_path).convert('RGB')
        transform_img = self.transform(img)
        return transform_img

    def apply_tokenizer(self, prompt):
        return self.tokenizer(prompt)

    def get_bos_token(self):
        return self.tokenizer.bos_token

    def get_eos_token(self):
        return self.tokenizer.eos_token

    def update_transforms_w_processor(self, image_processor):
        if image_processor is not None:
            self.transform = lambda x: image_processor.preprocess(x, return_tensors='pt')['pixel_values']

    def __len__(self):
        return len(self.data)

    def save_data(self, save_dir, save_file="questions.csv", save_img_dir="png_images"):
        assert self.mode == "test", "can only save data in test mode"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file = os.path.join(save_dir, save_file)
        save_img_dir = os.path.join(save_dir, save_img_dir)
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        data = []
        for idx in range(len(self)):
            item = self[idx]
            question = item['question']
            pixel_values = item['pixel_values']
            img_path = os.path.join(save_img_dir, f"{idx}.png")
            img = transforms.ToPILImage()(pixel_values.squeeze(0))
            img.save(img_path)
            data.append([idx, question, img_path])
        df = pd.DataFrame(data, columns=["idx", "question", "image_path"])
        df.to_csv(save_file, index=False)

    def calculate_metrics(self, gt_file, pred_file, results_file):
        eval = VQAEval(gt_file, pred_file, calculate_mae=self.calculate_mae)
        eval.evaluate()
        eval.write_results(results_file)

    def get_question_data(self, data_path, image_header="In question: "):
        with open(data_path, "r") as f:
            questions = json.load(f)
        questions = [q for q in questions if q['q_lang'] == "en" and q['img_name'] is not None]

        return questions

    def add_language_model_prompt(self, question_text, answer_text):
        if self.prompt_type == "standard":
            question_text = self.get_bos_token() + f'{self.img_token_str}{self.beg_prompt}{question_text}{self.mid_prompt}' + self.get_eos_token()
            answer_text = self.get_bos_token() + answer_text + self.get_eos_token()
        elif self.prompt_type == "standard_viz_after":
            question_text = self.get_bos_token() + f'{self.beg_prompt}{question_text}{self.mid_prompt}{self.img_token_str}' + self.get_eos_token()
            answer_text = self.get_bos_token() + answer_text + self.get_eos_token()
        elif self.prompt_type == "llama3":
            llama_system_prompt = (self.get_bos_token()+'<|start_header_id|>system<|end_header_id|>'
                                   '\n\n'
                                   "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.<|eot_id|>")
            llama_user_prompt = ('<|start_header_id|>\n'
                                 'user<|end_header_id|>'
                                 '\n\n'
                                 f'{self.img_token_str}'
                                 '\n'
                                 f'{self.beg_prompt}{question_text}{self.mid_prompt}<|eot_id|>')
            llama_assistant_prompt = '<|start_header_id|>assistant<|end_header_id|>\n\n'
            llama_assistant_prompt += f'{self.end_prompt}'
            question_text = llama_system_prompt + llama_user_prompt + llama_assistant_prompt
            answer_text = answer_text + self.get_eos_token()
        elif self.prompt_type == "mistral":
            question_text = self.get_bos_token() + "[INST]" + f'{self.img_token_str}' + '\n' + f'{self.beg_prompt}{question_text}{self.mid_prompt}' + "[/INST]"
            answer_text = answer_text + self.get_eos_token()
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_type}")
        return question_text, answer_text


class NephVQADataset(VQADataset):

    def prepare_image(self, data):
        image_path = data['img_name']
        full_img_paths = [self.img_root + img_path_ for img_path_ in image_path]
        imgs = []
        for img_path in full_img_paths:
            img = PIL.Image.open(img_path).convert('RGB')
            imgs.append(img)
        if len(imgs) > 0:
            # Determine the max height
            common_height = max(img.height for img in imgs)
            # Resize all images to the common height while maintaining aspect ratio
            resized_imgs = []
            for img in imgs:
                width, height = img.size
                new_width = int((common_height / height) * width)
                resized_img = img.resize((new_width, common_height))
                resized_imgs.append(resized_img)
            imgs_arr = [np.array(img) for img in resized_imgs]
            combined_arr = np.hstack(imgs_arr)
            combined_img = Image.fromarray(combined_arr)
            transform_img = self.transform(combined_img)
        else:
            transform_img = torch.zeros((3, self.height, self.width))
        return transform_img

    def get_question_data(self, data_path, num_im_cols=24, image_header="In question: "):
        questions_w_images = []
        if data_path.endswith(".csv"):
            self.data = pd.read_csv(data_path, header=0)
            for idx, sample in self.data.iterrows():
                caption = sample['caption']
                img_path = sample['renderURL'].split('/')[-1].strip()
                questions_w_images.append({"qid": idx, "question": "Please provide the figure caption for this image.",
                                           "img_name": [img_path], "answer": caption,
                                           "q_lang": "en"})
        else:
            im_keys = [f"image_{i}" for i in range(1, num_im_cols + 1, 1)]
            with open(data_path, "r") as f:
                questions = json.load(f)
            for question in questions:
                question_text = question["question"]
                clean_question_text = re.sub(r'Question \d+\n', '', question_text)
                image_paths = [question[key][len(image_header):] for key in im_keys if image_header in question[key]]
                answer_text = question["clean answer"]
                qid = question["qid"]
                if len(image_paths) > 0:
                    questions_w_images.append({"qid": qid, "question": clean_question_text, "img_name": image_paths,
                                               "answer": answer_text, "q_lang": "en"})
        return questions_w_images

    def calculate_metrics(self, gt_file, pred_file, results_file):
        eval = VQAEval(gt_file, pred_file, filter_key=self.filter_key, filter_cond=self.filter_cond,
                       calculate_mae=self.calculate_mae)
        eval.evaluate()
        eval.write_results(results_file)


class NephVQAInstructFinetuneDataset(NephVQADataset):

    def get_question_data(self, data_path, num_im_cols=24, image_header="In question: "):
        questions_w_images = []
        self.data = pd.read_csv(data_path, header=0)
        for idx, sample in self.data.iterrows():
            img_name = sample['image_name'].strip()
            question = sample['question'].strip()
            answer = sample['answer'].strip()
            questions_w_images.append({"qid": idx, "question": question,
                                       "img_name": [img_name], "answer": answer,
                                       "q_lang": "en"})
        return questions_w_images


class SlakeVQADataset(VQADataset):
    pass

class SlakeLlavaVQADataset(SlakeVQADataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from llava.mm_utils import tokenizer_image_token as llava_med_tokenizer
        self.llava_med_tokenizer = llava_med_tokenizer

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.mode == 'train' or self.mode == 'val':
            item["images"] = item.pop("pixel_values")
        return item

    def apply_tokenizer(self, prompt, return_tensors=None):
        tokens = dict()
        tokens['input_ids'] = self.llava_med_tokenizer(prompt, self.tokenizer, return_tensors=return_tensors)
        return tokens


class Brats3DVQADataset(VQADataset):

    def __init__(self, tokenizer, prompt_type, beg_prompt, mid_prompt, end_prompt, data_path,
                 included_modalities=("t1c", "t1n", "t2f", "t2w"), replace_prompt=None,
                 img_dir='',  img_tokens=1024,  pad_token_str='<|finetune_right_pad_id|>', img_token_str="<image>",
                 seq_length=1536, mode='train', filter_key='q_lang', filter_cond='en', calculate_mae=True,
                 add_vqa_summary_token=False, vqa_summary_token="<VQA_SUMMARY>", use_multitask_unknown=True,
                 calculate_multitask=True):
        self.prompt_type = prompt_type
        self.beg_prompt = beg_prompt
        self.mid_prompt = mid_prompt
        self.end_prompt = end_prompt
        self.replace_prompt = replace_prompt
        self.included_modalities = included_modalities
        self.img_root = img_dir
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(pad_token_str)
        self.ignore_token_id = -100
        self.mode = mode
        self.filter_key = filter_key
        self.filter_cond = filter_cond
        self.calculate_mae = calculate_mae
        self.calculate_multitask = calculate_multitask
        self.img_padding = [self.ignore_token_id for _ in range(img_tokens)]
        self.img_token_str = img_token_str
        self.seq_length = seq_length
        self.add_vqa_summary_token = add_vqa_summary_token
        self.vqa_summary_token = vqa_summary_token
        self.use_multitask_unknown = use_multitask_unknown
        self.area_index = 0
        self.num_area_labels = 8
        self.region_index = 1
        self.num_region_labels = 11
        self.shape_index = 2
        self.num_shape_labels = 7
        self.satellite_index = 3
        self.num_satellite_labels = 5
        self.num_unknown_labels = 1
        self.unknown_index = 4
        self.data = self.get_question_data(data_path)

        if self.mode == 'train':
            self.transform = mtf.Compose(
                [
                    mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                    mtf.RandFlip(prob=0.10, spatial_axis=0),
                    mtf.RandFlip(prob=0.10, spatial_axis=1),
                    mtf.RandFlip(prob=0.10, spatial_axis=2),
                    mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                    mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                    mtf.ToTensor(dtype=torch.float16),
                ]
            )
        elif self.mode == 'test' or self.mode == 'val':
            self.transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float16),
                ]
            )
        else:
            raise ValueError('mode must be in [train, test]')

    def __getitem__(self, idx):
        data = self.data[idx]
        item = super().__getitem__(idx)
        non_image_labels = data.get('answer_vqa_numeric', None)
        if non_image_labels is not None:
            # convert labels to hote encoding (hot for region and one-hot for others)
            item['area_label'] = self._multi_hot(non_image_labels[self.area_index], num_labels=self.num_area_labels)
            item['region_label'] = self._multi_hot(non_image_labels[self.region_index], num_labels=self.num_region_labels)
            item['shape_label'] = self._multi_hot(non_image_labels[self.shape_index], num_labels=self.num_shape_labels)
            item['satellite_label'] = self._multi_hot(non_image_labels[self.satellite_index], num_labels=self.num_satellite_labels)
            if self.use_multitask_unknown:
                item['unknown_label'] = torch.tensor([non_image_labels[self.unknown_index]], dtype=torch.int)
        return item

    def add_language_model_prompt(self, question_text, answer_text):
        question_text, answer_text = super().add_language_model_prompt(question_text, answer_text)
        if self.add_vqa_summary_token:
            eos_token_index = answer_text.index(self.get_eos_token())
            answer_text = answer_text[:eos_token_index] + self.vqa_summary_token + answer_text[eos_token_index:]
        return question_text, answer_text

    @staticmethod
    def _multi_hot(idx_list, num_labels):
        vec = torch.zeros(num_labels, dtype=torch.int)
        vec[idx_list] = 1
        return vec

    def prepare_image(self, data):
        image = []
        for modality in self.included_modalities:
            image_abs_path = data["volume_non_seg_files"][modality]
            new_image_abs_path = self.convert_file_path_to_npy(image_abs_path)
            image_ = np.load(new_image_abs_path)
            image_ = self.transform(image_)
            image.append(image_)
        image = torch.stack(image, axis=0)
        return image

    def convert_file_path_to_npy(self, image_abs_path):
        volume_abs_dir = os.path.dirname(image_abs_path)
        base_dir = os.path.dirname(volume_abs_dir)
        new_base_dir = base_dir + "_npy"

        volume_dir = os.path.basename(volume_abs_dir)
        image_file = os.path.basename(image_abs_path)
        new_image_abs_path = os.path.join(new_base_dir, volume_dir, image_file + ".npy")
        return new_image_abs_path

    def get_question_data(self, data_path, image_header="In question: "):
        with open(data_path, 'r') as f:
            questions = json.load(f)
        return questions

    def calculate_metrics(self, gt_file, pred_file, results_file):
        eval = VQAEval(gt_file, pred_file, calculate_mae=self.calculate_mae,
                       calculate_multitask=self.calculate_multitask,
                       use_multitask_unknown=self.use_multitask_unknown)
        eval.evaluate()
        eval.write_results(results_file)


class Brats3D2DVQADataset(Brats3DVQADataset):

    def __init__(self, included_modalities=("t1c",), height=224, width=224, num_channels=3, means=(0.485, 0.456, 0.406),
                 stds=(0.229, 0.224, 0.225),*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.included_modalities = included_modalities

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.mode == 'train' or self.mode == 'val':
            item["images"] = item.pop("pixel_values")
        return item

    def prepare_image(self, data):
        image = []
        for modality in self.included_modalities:
            image_abs_path = data["volume_non_seg_files"][modality]
            new_image_abs_path = self.convert_file_path_to_npy(image_abs_path)
            image_ = np.load(new_image_abs_path)
            image.append(image_)
        image = np.stack(image, axis=0)
        # keep only the middle slice
        image = image[..., 15, :, :]
        image = image.squeeze()
        image = np.stack([image, image, image], axis=0)
        image = self.transform(image)
        return image


class MimicVQADataset(VQADataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            content_type = entry.get('content_type', '')
            # Construct the full image path
            full_image_path = self.img_root + image_path
            if os.path.exists(full_image_path) and answer:
                # Attempt to open the image to check for corruption
                try:
                    with PIL.Image.open(full_image_path) as img:
                        if self.mode in ["train", "val"]:
                            img = img.convert('RGB')  # Verify that it is, indeed, an image
                        else:
                            img.verify()
                except (PIL.UnidentifiedImageError, OSError) as e:
                    print(f"Corrupted image found and skipped: {full_image_path}. Error: {e}")
                    continue  # Skip this entry
                # Image exists and is valid, include this entry
                question = entry['question']
                idx = entry['idx']
                filtered_data.append({"qid": idx, "img_id": idx, "question": question, "img_name": image_path,
                                      "answer": answer, "q_lang": "en", "content_type": content_type})
            else:
                # Image does not exist or answer is empty, skip this entry
                pass
        return filtered_data


class ROCCOv2VQADataset(VQADataset):

    def get_question_data(self, data_path):
        df = pd.read_csv(data_path, header=0)
        questions = []
        for idx, row in df.iterrows():
            img_name = row['ID'] + '.jpg'
            question = "Please describe the provided image."
            answer = row['Caption']
            qid = idx
            questions.append({'question': question, 'img_name': img_name, 'answer': answer, 'qid': qid, 'q_lang': 'en'})
        return questions

