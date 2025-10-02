import csv
import os
from tqdm import tqdm
import openai
import base64
import pandas as pd
from openai import OpenAI
from model.base_model import BaseModel
from data.kg_extract import get_full_prompts_and_calculate_token_lengths


class OpenaiLLM(BaseModel):
    def __init__(self, exp_file=None):
        super().__init__(exp_file=exp_file)
        self.client = OpenAI(api_key=self.params["openai"]["api_key"])

    def run(self):
        self.inference()

    def inference(self):
        """Call the OpenAI API on the list of questions/paragraphs and write the output to a csv file."""
        # append the prompt to each question
        if isinstance(self.params["data"]["aug_prompts"], list):
            paras = [self.params["data"]["prompt"] + "\n" + aug_prompt for aug_prompt in self.params["data"]["aug_prompts"] for _ in range(self.params["data"]["num_aug_samples"])]
        elif isinstance(self.params["data"]["questions"], list):
            paras = [self.params["data"]["prompt"]+"\n"+question for question in self.params["data"]["questions"]]
        elif ".csv" in self.params["data"]["questions"]:
            data = pd.read_csv(self.params["data"]["questions"])
            paras = [self.params["data"]["prompt"] + "\n" + row for row in data["caption"].tolist()]
            if self.params["data"]["image_dir"]:
                image_names = data["image_name"].to_list()
                encoded_ims = [encode_image(self.params["data"]["image_dir"] + image_name) for image_name in image_names]
                paras = [(para, encoded_im) for para, encoded_im in zip(paras, encoded_ims)]
        else:
            paras = get_full_prompts_and_calculate_token_lengths(self.params["data"]["data_dir"],
                                                                 prompt=self.params["data"]["prompt"],
                                                                 min_tokens=self.params["inf"]["min_tokens"],
                                                                 max_tokens=self.params["inf"]["max_tokens"])
            if self.params["data"]["debug_num_samples"] is not None:
                paras = paras[:self.params["data"]["debug_num_samples"]]
        # call the OpenAI API on each question and write to a csv file
        with open(os.path.join(self.params["exp"]["output_dir"], self.params["inf"]["save_file"]), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["question", "answer"])
            for para in tqdm(paras):
                if self.params["data"]["image_dir"]:
                    para, encoded_im = para
                    outputs = self.generate_n_inference(para, encoded_im=encoded_im)
                else:
                    outputs = self.generate_n_inference(para)
                for output in outputs:
                    writer.writerow([para, output])
                    file.flush()

    def generate_n_inference(self, para, encoded_im=None):
        """Call the OpenAI API on a single question/paragraph and return the response."""
        try:
            if self.params["inf"]["model_name"] == "gpt-3.5-turbo-instruct":
                outputs = self.get_completions_n_responses(para)
            else:
                outputs = self.get_chat_n_responses(para, encoded_im=encoded_im)
        except openai.APIError as e:
            # Handle API error here, e.g. retry or log
            outputs = f"OpenAI API returned an API Error: {e}"
            print(outputs + f" for prompt: {para}")
            outputs = [outputs]
        except openai.APIConnectionError as e:
            # Handle connection error here
            outputs = f"Failed to connect to OpenAI API: {e}"
            print(outputs + f" for prompt: {para}")
            outputs = [outputs]
        except openai.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            outputs = "OpenAI API request exceeded rate limit: {e}"
            print(outputs + f" for prompt: {para}")
            outputs = [outputs]
        return outputs

    def get_completions_n_responses(self, para):
        """OpenAI API code for completions response"""
        response = self.client.completions.create(
            model=self.params["inf"]["model_name"],
            prompt=para,
            max_tokens=self.params["inf"]["max_tokens"],
            n=self.params["inf"]["n"]
        )
        return [choice.text for choice in response.choices]

    def get_chat_n_responses(self, para, encoded_im=None):
        """OpenAI API code for chat response"""
        if encoded_im is not None:
            content = [{"type": "text", "text": para},
                       {"type": "image_url", "image_url": {"url":  f"data:image/png;base64,{encoded_im}"}}]
        else:
            content = para
        response = self.client.chat.completions.create(
            model=self.params["inf"]["model_name"],
            messages=[{"role": "user", "content":  content}],
            max_tokens=self.params["inf"]["max_tokens"],
            n=self.params["inf"]["n"],
            temperature=self.params["inf"]["temperature"],
            top_p=self.params["inf"]["top_p"]
        )
        return [choice.message.content for choice in response.choices]

    @property
    def required_params(self):
        required_params = super(OpenaiLLM, self).required_params
        required_params["openai"] = ["api_key"]
        required_params["data"] = ["prompt", "aug_prompts", "num_aug_samples", "num_aug_samples", "data_dir",
                                   "questions", "debug_num_samples", "image_dir"]
        required_params["inf"] = ["temperature", "min_tokens", "max_tokens", "n", "top_p", "save_file", "model_name"]
        assert isinstance(required_params, dict), "required_params must be a dictionary"
        return required_params


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


if __name__ == '__main__':
    openai_llm = OpenaiLLM("openai_kg_extract.yml")
    openai_llm.setup()
    openai_llm.run()