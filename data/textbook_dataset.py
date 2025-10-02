from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from data.randaugment import RandomAugment
from PIL import Image
import PIL
import numpy as np
import copy


class TextbookDataset(Dataset):
    def __init__(self, tokenizer, data_path,
                 img_dir='Nephrology/TextBook/pdffigures2/figure/image/merge/', height=224,
                 width=224, num_channels=3, img_tokens=256, seq_length=1024, mode='train', means=(0.485, 0.456, 0.406),
                 stds=(0.229, 0.224, 0.225), start=0):
        self.beg_prompt = "Please describe the image."
        self.img_root = img_dir
        self.data = pd.read_csv(data_path).iloc[start:]
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.ignore_token_id = -100
        self.mode = mode
        self.img_padding = [self.ignore_token_id for _ in range(img_tokens)]
        self.height = height
        self.width = width
        self.num_channels = num_channels

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
        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([
                transforms.Resize((self.height, self.width), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            raise ValueError('mode must be in [train, val, test]')
        self.mode = mode
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        # process image
        image_path = self.img_root + sample['renderURL'].split('/')[-1].strip()
        img = PIL.Image.open(image_path).convert('RGB')
        transform_img = self.transform(img)

        # process prompt and caption as separate sequences and combine them into a single sequence
        caption = sample['caption']
        prompt = self.beg_prompt + self.tokenizer.eos_token
        caption = caption + self.tokenizer.eos_token

        tokenized_prompt = self.tokenizer(prompt)
        tokenized_caption = self.tokenizer(caption)
        prompt_input_ids = tokenized_prompt['input_ids']
        caption_input_ids = tokenized_caption['input_ids']

        input_ids = prompt_input_ids + caption_input_ids

        # ignore the initial prompt for generation
        ignore_input_tokens = [self.ignore_token_id] * len(prompt_input_ids)
        label_ids = self.img_padding + ignore_input_tokens + caption_input_ids
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
        assert len(input_ids) == (len(label_ids) - len(self.img_padding)), f"{len(input_ids)} != {len(label_ids)} - {len(self.img_padding)}"

        return {'pixel_values': transform_img,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label_ids}
