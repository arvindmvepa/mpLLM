from abc import abstractmethod, ABC
import os
import shutil
import torch
import transformers
from transformers import TrainerCallback
from utils.misc_utils import load_yaml, assert_required_params_list
from utils.huggingface_utils import load_tokenizer_from_huggingface, load_llm_from_huggingface


class BaseModel(ABC):
    """Abstract class for LLM"""

    def __init__(self, exp_file=None):
        self.exp_file = exp_file
        self.params = load_yaml(exp_file)
        print("params: ", self.params)
        self.check_params()

    def setup(self):
        self.setup_exp_dir()

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    def setup_exp_dir(self):
        """Setup the experiment directory and save the exp file in the directory"""
        output_dir = self.params['exp']['output_dir']
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if self.exp_file is not None:
            save_exp_file = os.path.join(output_dir, "exp.yml")
            try:
                shutil.copyfile(self.exp_file, save_exp_file)
            except shutil.SameFileError:
                pass

    def check_params(self):
        """For each outermost yaml section, confirm all required params are included and no extra params are included."""
        assert isinstance(self.params, dict), "params must be a dictionary"
        required_sections = list(self.required_params.keys())
        included_sections = list(self.params.keys())
        assert_required_params_list(required_sections, included_sections)

        for section in required_sections:
            required_section_params = self.required_params[section]
            included_section_params = self.params[section]
            assert_required_params_list(required_section_params, included_section_params, header=section)

    @property
    def required_params(self):
        required_params = {"exp": ["output_dir"]}
        assert isinstance(required_params, dict), "required_params must be a dictionary (base class)"
        return required_params


class BaseLLM(BaseModel):
    def __init__(self, exp_file=None, use_wandb=False):
        super().__init__(exp_file=exp_file)
        self.use_wandb = use_wandb

    def setup(self):
        super().setup()
        self.setup_tokenizer()

    def run(self):
        self.train()

    def train(self):
        """Train the Huggingface model using the huggingface Trainer class."""
        data = self.get_train_data()
        print("Number of training samples: ", len(data["train"]))
        print("Number of test samples: ", len(data["test"]))
        data_collator = self.get_data_collator()
        model, image_processor = self.load_train_model()
        data['train'].update_transforms_w_processor(image_processor)
        data['test'].update_transforms_w_processor(image_processor)
        training_args = self.get_training_args()
        trainer_class = self.get_trainer_class()
        trainer = trainer_class(
            model=model,
            tokenizer=self.tokenizer,
            train_dataset=data["train"],
            eval_dataset=data["test"],
            args=training_args,
            data_collator=data_collator
        )
        if self.params['train']["evaluate_start"]:
            eval_at_start_callback = EvalAtStartCallback()
            eval_at_start_callback.trainer = trainer
            trainer.add_callback(eval_at_start_callback)
        if self.params['train']["gen_train_outputs"]:
            stepwise_train_outputs_callback = StepwiseTrainOutputsCallback(tokenizer=self.tokenizer)
            trainer.add_callback(stepwise_train_outputs_callback)
        if self.params['train']["gen_llava_med_train_outputs"]:
            stepwise_train_outputs_callback = StepwiseTrainOutputsLlavaMedCallback(tokenizer=self.tokenizer)
            trainer.add_callback(stepwise_train_outputs_callback)
        trainer.train(resume_from_checkpoint=self.params['train']["resume_from_checkpoint"])
        self.save_model(model)

    def get_training_args(self):
        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=self.params['train']["per_device_train_batch_size"],
            per_device_eval_batch_size=self.params['train']["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.params['train']["gradient_accumulation_steps"],
            num_train_epochs=self.params['train']["num_train_epochs"],
            learning_rate=self.params['train']["learning_rate"],
            fp16=self.params['train']["fp16"],
            save_total_limit=self.params['train']["save_total_limit"],
            logging_steps=self.params['train']["logging_steps"],
            label_names=self.params['train']["label_names"],
            output_dir=self.output_dir,
            save_strategy=self.params['train']["save_strategy"],
            evaluation_strategy=self.params['train']["evaluation_strategy"],
            eval_steps=self.params['train']["eval_steps"],
            save_steps=self.params['train']["save_steps"],
            optim=self.params['train']["optim"],
            lr_scheduler_type=self.params['train']["lr_scheduler_type"],
            warmup_ratio=self.params['train']["warmup_ratio"],
            load_best_model_at_end=True,
            report_to="wandb" if self.use_wandb else "tensorboard",
        )
        return training_args

    def get_trainer_class(self):
        return transformers.Trainer

    def save_model(self, model):
        model.save_pretrained(os.path.join(self.output_dir, self.params['train']["save_model_name"]))

    def setup_tokenizer(self):
        self.tokenizer = load_tokenizer_from_huggingface(self.params['data']["tokenizer_name"])
        self.tokenizer.add_bos_token = False

    def prepare_question(self, question):
        return question

    @abstractmethod
    def get_train_data(self):
        raise NotImplementedError()

    @abstractmethod
    def get_data_collator(self):
        raise NotImplementedError()

    def load_llm_model(self, *args, **kwargs):
        model = load_llm_from_huggingface(*args, **kwargs)
        return model

    def load_train_model(self):
        raise NotImplementedError()

    def load_inf_model(self):
        raise NotImplementedError()

    @property
    def required_params(self):
        required_params = super(BaseLLM, self).required_params
        required_params["data"] = ["tokenizer_name", "data_path", "test_size", "data_seed"]
        required_params["train"] = ["model_name", "save_model_name", "use_quantization", "r", "lora_alpha",
                                    "target_modules", "lora_dropout", "bias", "task_type",
                                    "per_device_train_batch_size", "per_device_eval_batch_size",
                                    "gradient_accumulation_steps", "num_train_epochs", "learning_rate", "fp16",
                                    "save_total_limit", "logging_steps", "save_strategy", "evaluation_strategy",
                                    "eval_steps", "save_steps", "optim", "lr_scheduler_type", "warmup_ratio",
                                    "resume_from_checkpoint", "evaluate_start", "gen_train_outputs",
                                    "gen_llava_med_train_outputs", "label_names"]
        required_params["inf"] = ["model_name", "beg_prompt", "mid_prompt", "end_prompt"]
        assert isinstance(required_params, dict), "required_params must be a dictionary"
        return required_params

class EvalAtStartCallback(TrainerCallback):
    """A callback to run evaluation at the start of training."""
    def on_train_begin(self, args, state, control, **kwargs):
        print("Running evaluation at the start of training")
        self.trainer.evaluate()

class StepwiseTrainOutputsCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_train_begin(self, args, state, control, **kwargs):
        print("Debugging train outputs...")
        model = kwargs.get('model')
        dataloader = kwargs.get('train_dataloader')
        for step, batch in enumerate(dataloader):
            print("batch.keys(): ", batch.keys())
            batch_input_ids = batch['input_ids']
            batch_pixel_values = batch['pixel_values']
            batch_attention_mask = batch['attention_mask']
            batch_labels = batch['labels']
            for input_ids, pixel_values, attention_mask, labels in zip(batch_input_ids, batch_pixel_values,
                                                                       batch_attention_mask, batch_labels):
                # Adjust attention mask to match the length of input_ids
                attention_mask = attention_mask[:len(input_ids)]
                model.eval()
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=input_ids.unsqueeze(0),
                        pixel_values=pixel_values.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                        max_new_tokens=50,
                        num_beams=1,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id
                    )[0]

                input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                prediction_text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
                filtered_labels = self.filter_invalid_token_ids(labels)
                label_text = self.tokenizer.decode(filtered_labels, skip_special_tokens=False)

                print(f"Evaluation Step {step + 1}")
                print("len(input_ids): ", len(input_ids))
                print("len(output_ids): ", len(output_ids))
                print("len(labels): ", len(labels))
                print(f"Input: {input_text}")
                print(f"Prediction: {prediction_text}")
                print(f"Ground Truth: {label_text}")
                print("-" * 30)

    def filter_invalid_token_ids(self, labels):
        return [token_id for token_id in labels if token_id >= 0]


class StepwiseTrainOutputsLlavaMedCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        from llava.constants import IMAGE_TOKEN_INDEX
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX


    def on_train_begin(self, args, state, control, **kwargs):
        print("Debugging train outputs...")
        model = kwargs.get('model')
        dataloader = kwargs.get('train_dataloader')
        for step, batch in enumerate(dataloader):
            print("batch.keys(): ", batch.keys())
            batch_input_ids = batch['input_ids']
            batch_pixel_values = batch['images']
            batch_attention_mask = batch['attention_mask']
            batch_labels = batch['labels']
            for input_ids, pixel_values, attention_mask, labels in zip(batch_input_ids, batch_pixel_values,
                                                                       batch_attention_mask, batch_labels):
                # Adjust attention mask to match the length of input_ids
                attention_mask = attention_mask[:len(input_ids)]
                model.eval()
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids.unsqueeze(0),
                        images=pixel_values.unsqueeze(0),
                        max_new_tokens=50,
                        num_beams=1,
                        num_return_sequences=1
                    )[0]

                print(f"Evaluation Step {step + 1}")
                print("len(input_ids): ", len(input_ids))
                print("len(output_ids): ", len(output_ids))
                print("len(labels): ", len(labels))
                print("input_ids: ", input_ids)
                print("labels: ", labels)

                input_ids[input_ids == self.IMAGE_TOKEN_INDEX] = 0
                input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                prediction_text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
                filtered_labels = self.filter_invalid_token_ids(labels)
                label_text = self.tokenizer.decode(filtered_labels, skip_special_tokens=False)

                print(f"Input: {input_text}")
                print(f"Prediction: {prediction_text}")
                print(f"Ground Truth: {label_text}")
                print("-" * 30)

    def filter_invalid_token_ids(self, labels):
        return [token_id for token_id in labels if token_id >= 0]


class RAGMixin:

    def setup(self):
        super().setup()
        self.setup_kg_embedder()

    def setup_kg_embedder(self):
        from utils.run_utils import get_model
        self.kg_embedder = get_model("kg_embed", self.params["data"]["kg_embedder_params"])
        self.kg_embedder.setup()
        self.kg_embedder.set_index_for_kg_embeddings()

    def prepare_question(self, question):
        retrieved_kg_entries = self.kg_embedder.retrieve_top_kg_entries(question)
        context = ". ".join(retrieved_kg_entries)
        if context:
            question = context + ". " + question
        return super().prepare_question(question)

    @property
    def required_params(self):
        required_params = super().required_params
        required_params["inf"] = required_params["inf"] + ["context_prompt"]
        assert isinstance(required_params, dict), "required_params must be a dictionary"
        return required_params
