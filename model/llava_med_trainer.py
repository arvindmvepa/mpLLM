from model.vision_to_llm_trainer import VisionToLLMTrainer
from llava.model.builder import load_pretrained_model
from llava.train.llava_trainer import LLaVATrainer
from llava.train.train import TrainingArguments, smart_tokenizer_and_embedding_resize, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from llava.mm_utils import tokenizer_image_token as llava_med_tokenizer
import torch
from types import SimpleNamespace
import os


class LlavaMedTrainer(VisionToLLMTrainer):

    def load_train_model(self):
        llm_model = self.load_llm_model(model_name=self.params['train']["model_name"],
                                        use_quantization=self.params['train']["use_quantization"],
                                        r=self.params['train']["r"], lora_alpha=self.params['train']["lora_alpha"],
                                        target_modules=self.params['train']["target_modules"],
                                        lora_dropout=self.params['train']["lora_dropout"],
                                        bias=self.params['train']["bias"], task_type=self.params['train']["task_type"])
        print(f"Adding pad token as {self.pad_token}")
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=self.pad_token),
            tokenizer=self.tokenizer,
            model=llm_model,
        )
        args = SimpleNamespace(vision_tower=self.params["train"]["vision_tower"],
                               mm_vision_select_layer=self.params["train"]["mm_vision_select_layer"],
                               mm_vision_select_feature=self.params["train"]["mm_vision_select_feature"],
                               pretrain_mm_mlp_adapter=self.params["train"]["pretrain_mm_mlp_adapter"],
                               mm_patch_merge_type=self.params["train"]["mm_patch_merge_type"],
                               mm_projector_type=self.params["train"]["mm_projector_type"])
        llm_model.get_model().initialize_vision_modules(
            model_args=args,
            fsdp=None
        )
        vision_tower = llm_model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16, device="cuda")
        image_processor = vision_tower.image_processor

        if self.params['train']["freeze_vision_model"]:
            vision_tower.requires_grad_(False)
        if self.params['train']["freeze_llm_model"]:
            llm_model.model.requires_grad_(False)
        if self.params['train']["freeze_projector"]:
            for p in llm_model.get_model().mm_projector.parameters():
                p.requires_grad = False
        else:
            for p in llm_model.get_model().mm_projector.parameters():
                p.requires_grad = True
        if not self.params['train']["freeze_embedding_layer"]:
            for p in llm_model.get_input_embeddings().parameters():
                p.requires_grad = False
            for p in llm_model.get_output_embeddings().parameters():
                p.requires_grad = False

        return llm_model, image_processor

    def load_inf_model(self):
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=self.params['inf']["model_name"],
            model_base=self.params['inf']["model_name"],
            model_name=self.params['inf']["model_name"],
            projector_name=self.params['inf']["projector_name"]
        )
        return model, image_processor

    def load_vision_model(self):
        pass

    def get_vision_model_norm_params(self):
        # Dummy values - are not used
        return (0, 0, 0), (0, 0, 0)

    def generate(self, model, inputs, pixel_values, decoding_kwargs):
        generated_ids = model.generate(inputs, images=pixel_values, num_return_sequences=1,
                                       max_new_tokens=self.params['inf']['max_new_tokens'], **decoding_kwargs)
        return generated_ids

    def get_trainer_class(self):
        return LLaVATrainer

    def get_training_args(self):
        training_args = TrainingArguments(
            freeze_mm_mlp_adapter=self.params['train']["freeze_projector"],
            per_device_train_batch_size=self.params['train']["per_device_train_batch_size"],
            per_device_eval_batch_size=self.params['train']["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.params['train']["gradient_accumulation_steps"],
            num_train_epochs=self.params['train']["num_train_epochs"],
            learning_rate=self.params['train']["learning_rate"],
            double_quant=False, # ignoring these arguments in the llava model training
            quant_type=None, # ignoring these arguments in the llava model training
            bits=None, # ignoring these arguments in the llava model training
            lora_enable=True if self.params['train']["r"] else False,
            lora_r=self.params['train']["r"],
            lora_alpha=self.params['train']["lora_alpha"],
            lora_dropout=self.params['train']["lora_dropout"],
            lora_weight_path="",
            lora_bias=self.params['train']["bias"],
            group_by_modality_length=self.params['train']["group_by_modality_length"],
            fp16=self.params['train']["fp16"],
            save_total_limit=self.params['train']["save_total_limit"],
            logging_steps=self.params['train']["logging_steps"],
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

    def save_model(self, model):
        assert self.params['train']["r"] is not None, f"save_model is only valid if lora r is set"
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), self.params['train']["bias"]
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        save_model_dir = os.path.join(self.output_dir, self.params['train']["save_model_name"])
        self.tokenizer.save_pretrained(save_model_dir)
        model.config.save_pretrained(save_model_dir)
        model.save_pretrained(save_model_dir, state_dict=state_dict)
        torch.save(non_lora_state_dict, os.path.join(save_model_dir, 'non_lora_trainables.bin'))

    def apply_tokenizer(self, prompt, return_tensors='pt'):
        return llava_med_tokenizer(prompt, self.tokenizer, return_tensors=return_tensors).unsqueeze(0)

    def get_bos_token(self):
        return self.tokenizer.bos_token

    @property
    def required_params(self):
        required_params = super(LlavaMedTrainer, self).required_params
        required_params["train"] = required_params["train"] + ["projector_name", "group_by_modality_length",
                                                               "mm_patch_merge_type",
                                                               "pretrain_mm_mlp_adapter", "vision_tower",
                                                               "vision_tower_use_s2", "mm_projector_type",
                                                               "mm_vision_select_layer", "mm_vision_select_feature",
                                                               "freeze_llm_model",
                                                               "freeze_vision_model", "freeze_projector",
                                                               "freeze_embedding_layer"]
        required_params["inf"] = required_params["inf"] + ["projector_name"]
        assert isinstance(required_params, dict), "required_params must be a dictionary"
        return required_params

    @property
    def pad_token(self):
        return '<pad>'

    @property
    def pixel_values_dtype(self):
        return torch.float16
