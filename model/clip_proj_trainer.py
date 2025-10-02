import torch
from model.vision_language_model import VisionLanguageModel
from model.base_model import RAGMixin
from model.vision_to_llm_trainer import VisionToLLMTrainer
from utils.huggingface_utils import load_clip_vision_model, load_clip_vision_model_norm_params, convert_meta_to_tensor


class ClipProjTrainer(VisionToLLMTrainer):

    def setup_tokenizer(self):
        super().setup_tokenizer()
        self.tokenizer.add_tokens([self.img_token])
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(self.img_token)

    def load_train_model(self):
        """Load vision-language model with CLIP Vision encoder, specified frozen LLM, and trainable projection layer"""
        vision_model, _ = self.load_vision_model()
        llm_model = self.load_llm_model(model_name=self.params['train']["llm_model_name"],
                                        use_quantization=self.params['train']["use_quantization"],
                                        r=self.params['train']["r"], lora_alpha=self.params['train']["lora_alpha"],
                                        target_modules=self.params['train']["target_modules"],
                                        lora_dropout=self.params['train']["lora_dropout"],
                                        bias=self.params['train']["bias"], task_type=self.params['train']["task_type"])
        if self.params['train']['pretrained']:
            model = VisionLanguageModel.from_pretrained(self.params['train']["model_name"],
                                                            vision_model=vision_model,
                                                            img_token_id=self.img_token_id,
                                                            img_tokens=self.params['data']['img_tokens'],
                                                            language_model=llm_model,
                                                            load_projection_matrix=self.params['train']["load_projection_matrix"])
        else:
            model = VisionLanguageModel(vision_model=vision_model, language_model=llm_model,
                                        img_token_id=self.img_token_id, img_tokens=self.params['data']['img_tokens'])
        if self.params['train']["freeze_vision_model"]:
            for param in model.vision_model.parameters():
                param.requires_grad = False
        if self.params['train']["freeze_llm_model"]:
            for param in model.language_model.parameters():
                param.requires_grad = False
        return model, None

    def load_inf_model(self):
        vision_model, _ = self.load_vision_model()
        llm_model = self.load_llm_model(model_name=self.params['inf']["llm_model_name"],
                                        use_quantization=self.params['inf']["use_quantization"],
                                        r=self.params['inf']["r"], lora_alpha=self.params['inf']["lora_alpha"],
                                        target_modules=self.params['inf']["target_modules"],
                                        lora_dropout=self.params['inf']["lora_dropout"],
                                        bias=self.params['inf']["bias"], task_type=self.params['inf']["task_type"])
        model = VisionLanguageModel.from_pretrained(self.params['inf']["model_name"], vision_model=vision_model,
                                                    img_token_id=self.img_token_id,
                                                    img_tokens=self.params['data']['img_tokens'],
                                                    language_model=llm_model,
                                                    load_projection_matrix=self.params['inf']["load_projection_matrix"])
        return model, None

    def load_vision_model(self):
        # make sure to only retain the trunk of the CLIP model
        model = load_clip_vision_model(self.params['train']["vision_model_name"]).trunk
        return model, None

    def get_vision_model_norm_params(self):
        return load_clip_vision_model_norm_params(self.params['train']["vision_model_name"])

    @property
    def required_params(self):
        required_params = super(ClipProjTrainer, self).required_params
        required_params["train"] = required_params["train"] + ["vision_model_name", "freeze_llm_model",
                                                               "freeze_vision_model", "pretrained",
                                                               "load_projection_matrix", "num_proj_layers"]
        required_params["inf"] = required_params["inf"] + ["llm_model_name", "use_quantization", "r", "lora_alpha",
                                                           "target_modules", "lora_dropout", "bias", "task_type",
                                                           "load_projection_matrix"]
        assert isinstance(required_params, dict), "required_params must be a dictionary"
        return required_params

    @property
    def pixel_values_dtype(self):
        return torch.float


class ClipProjTrainerWithRAG(RAGMixin, ClipProjTrainer):
    pass



