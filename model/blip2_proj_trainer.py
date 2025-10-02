import torch
from torch import nn
from transformers import Blip2Config, Blip2Model
from model.vision_to_llm_trainer import VisionToLLMTrainer
from utils.huggingface_utils import load_blip_qformer_model_from_huggingface, load_blip_vision_model_from_huggingface, \
    load_blip_vision_model_norm_params_from_huggingface


class Blip2ProjTrainer(VisionToLLMTrainer):

    def load_train_model(self):
        """Load custom BLIP-2 model with frozen BLIP-2 Vision encoder and Qformer, specified frozen LLM, and trainable projection layer"""
        vision_model = self.load_vision_model()
        # Freeze the Vision encoder
        for param in vision_model.parameters():
            param.requires_grad = False
        qformer_model = self.load_qformer_model()
        # Freeze the Qformer
        for param in qformer_model.parameters():
            param.requires_grad = False
        # Freeze the LLM
        llm_model = self.load_llm_model(model_name=self.params['train']["llm_model_name"])
        for param in llm_model.parameters():
            param.requires_grad = False
        model = Blip2ModelCustom(Blip2Config.from_pretrained(self.params['train']["model_name"]),
                                 vision_model=vision_model, qformer_model=qformer_model, language_model=llm_model)
        return model

    def load_qformer_model(self):
        model = load_blip_qformer_model_from_huggingface(self.params['train']["qformer_model_name"])
        return model

    def load_vision_model(self):
        model = load_blip_vision_model_from_huggingface(self.params['train']["vision_model_name"])
        return model

    def get_vision_model_norm_params(self):
        return load_blip_vision_model_norm_params_from_huggingface(self.params['train']["vision_model_name"])

    @property
    def required_params(self):
        required_params = super(VisionToLLMTrainer, self).required_params
        required_params["train"] = required_params["train"] + ["qformer_model_name"]
        assert isinstance(required_params, dict), "required_params must be a dictionary"
        return required_params


class Blip2ModelCustom(Blip2Model):

    def __init__(self, config: Blip2Config, vision_model, qformer_model, language_model):
        super().__init__(config)

        self.vision_model = vision_model
        self.qformer = qformer_model
        self.language_model = language_model

        config.vision_config = self.vision_model.config
        config.qformer_config = self.qformer.config
        config.text_config = self.language_model.config

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)

        # Update _tied_weights_keys using the base model used.
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        # Initialize weights and apply final processing
        self.post_init()