import torch
from model.vision_to_llm_trainer import VisionToLLMTrainer
from model.vision_3D_language_model import Vision3DLanguageModel
from model.medical_3D_vit import ViT3DTower
from data.dataset_factory import get_dataset_factory


class Medical3DLLMTrainer(VisionToLLMTrainer):

    def setup_tokenizer(self):
        super().setup_tokenizer()
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.vqa_summary_token, self.img_token]})
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(self.img_token)
        self.vqa_summary_token_id = self.tokenizer.convert_tokens_to_ids(self.vqa_summary_token)

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
            model = Vision3DLanguageModel.from_pretrained(self.params['train']["model_name"],
                                                          vision_model=vision_model,
                                                          img_token_id=self.img_token_id,
                                                          vqa_summary_token_id=self.vqa_summary_token_id,
                                                          add_vqa_summary_token=self.params['train']['add_vqa_summary_token'],
                                                          add_multitask=self.params['train']['add_multitask'],
                                                          add_multitask_unknown=self.params['train']['add_multitask_unknown'],
                                                          add_multitask_first_eos=self.params['train'][ 'add_multitask_first_eos'],
                                                          add_viz_w_add_multitask=self.params['train']['add_viz_w_add_multitask'],
                                                          multitask_wt=self.params['train']['multitask_wt'],
                                                          multitask_text_ft_wt=self.params['train']['multitask_text_ft_wt'],
                                                          multitask_viz_ft_wt=self.params['train']['multitask_viz_ft_wt'],
                                                          img_tokens=self.params['data']['img_tokens'],
                                                          num_proj_layers=self.params['train']["num_proj_layers"],
                                                          pooling_size=self.params['train']['pooling_size'],
                                                          image_size=self.params['data']['image_size'],
                                                          patch_size=self.params['train']["patch_size"],
                                                          num_modalities=len(self.params['data']['included_modalities']),
                                                          create_self_attn_block=self.params['train']["create_self_attn_block"],
                                                          num_attn_layers=self.params['train']["num_attn_layers"],
                                                          num_attn_heads=self.params['train']["num_attn_heads"],
                                                          add_attn_mlp=self.params['train']["add_attn_mlp"],
                                                          create_x_attn_block=self.params['train']["create_x_attn_block"],
                                                          num_x_attn_heads=self.params['train']["num_x_attn_heads"],
                                                          add_x_attn_mlp=self.params['train']["add_x_attn_mlp"],
                                                          x_attn_query=self.params['train']["x_attn_query"],
                                                          language_model=llm_model,
                                                          create_moe_block=self.params['train']["create_moe_block"],
                                                          moe_use_router=self.params['train']["moe_use_router"],
                                                          moe_router_hidden_dim=self.params['train']["moe_router_hidden_dim"],
                                                          moe_num_proj=self.params['train']['moe_num_proj'],
                                                          moe_fusion_mode=self.params['train']['moe_fusion_mode'],
                                                          moe_use_shared_expert=self.params['train']['moe_use_shared_expert'],
                                                          moe_sum_weights=self.params['train']['moe_sum_weights'],
                                                          moe_use_lite_router=self.params['train']['moe_use_lite_router'],
                                                          moe_router_reg_coeff=self.params['train']['moe_router_reg_coeff'],
                                                          moe_adapted_router=self.params['train']['moe_adapted_router'],
                                                          moe_token_based_router=self.params['train']['moe_token_based_router'],
                                                          moe_w_text=self.params['train']['moe_w_text'],
                                                          moe_token_and_seq_based_router=self.params['train']['moe_token_and_seq_based_router'],
                                                          moe_token_and_seq_based_router_w_viz=self.params['train']['moe_token_and_seq_based_router_w_viz'],
                                                          moe_higher_level_router_num_blocks=self.params['train']['moe_higher_level_router_num_blocks'],
                                                          moe_higher_level_block_kwargs=self.params['train']['moe_higher_level_block_kwargs'],
                                                          load_projection_matrix=self.params['train']["load_projection_matrix"],
                                                          tokenizer=self.tokenizer)
        else:
            model = Vision3DLanguageModel(vision_model=vision_model, language_model=llm_model,
                                          img_token_id=self.img_token_id,
                                          vqa_summary_token_id=self.vqa_summary_token_id,
                                          add_vqa_summary_token=self.params['train']['add_vqa_summary_token'],
                                          add_multitask=self.params['train']['add_multitask'],
                                          add_multitask_unknown=self.params['train']['add_multitask_unknown'],
                                          add_multitask_first_eos=self.params['train']['add_multitask_first_eos'],
                                          add_viz_w_add_multitask=self.params['train']['add_viz_w_add_multitask'],
                                          multitask_wt=self.params['train']['multitask_wt'],
                                          multitask_text_ft_wt=self.params['train']['multitask_text_ft_wt'],
                                          multitask_viz_ft_wt=self.params['train']['multitask_viz_ft_wt'],
                                          num_proj_layers=self.params['train']["num_proj_layers"],
                                          img_tokens=self.params['data']['img_tokens'],
                                          pooling_size=self.params['train']['pooling_size'],
                                          image_size=self.params['data']['image_size'],
                                          patch_size=self.params['train']["patch_size"],
                                          num_modalities=len(self.params['data']['included_modalities']),
                                          create_self_attn_block=self.params['train']["create_self_attn_block"],
                                          num_attn_layers=self.params['train']["num_attn_layers"],
                                          num_attn_heads=self.params['train']["num_attn_heads"],
                                          add_attn_mlp=self.params['train']["add_attn_mlp"],
                                          create_x_attn_block=self.params['train']["create_x_attn_block"],
                                          num_x_attn_heads=self.params['train']["num_x_attn_heads"],
                                          add_x_attn_mlp=self.params['train']["add_x_attn_mlp"],
                                          x_attn_query=self.params['train']["x_attn_query"],
                                          create_moe_block=self.params['train']["create_moe_block"],
                                          moe_use_router=self.params['train']["moe_use_router"],
                                          moe_router_hidden_dim=self.params['train']["moe_router_hidden_dim"],
                                          moe_num_proj=self.params['train']['moe_num_proj'],
                                          moe_fusion_mode=self.params['train']['moe_fusion_mode'],
                                          moe_use_shared_expert=self.params['train']['moe_use_shared_expert'],
                                          moe_sum_weights=self.params['train']['moe_sum_weights'],
                                          moe_use_lite_router=self.params['train']['moe_use_lite_router'],
                                          moe_router_reg_coeff=self.params['train']['moe_router_reg_coeff'],
                                          moe_adapted_router=self.params['train']['moe_adapted_router'],
                                          moe_token_based_router=self.params['train']['moe_token_based_router'],
                                          moe_w_text=self.params['train']['moe_w_text'],
                                          moe_token_and_seq_based_router=self.params['train']['moe_token_and_seq_based_router'],
                                          moe_token_and_seq_based_router_w_viz=self.params['train']['moe_token_and_seq_based_router_w_viz'],
                                          moe_higher_level_router_num_blocks=self.params['train']['moe_higher_level_router_num_blocks'],
                                          moe_higher_level_block_kwargs=self.params['train']['moe_higher_level_block_kwargs'],
                                          tokenizer=self.tokenizer)
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
        model = Vision3DLanguageModel.from_pretrained(self.params['inf']["model_name"], vision_model=vision_model,
                                                      img_token_id=self.img_token_id,
                                                      vqa_summary_token_id=self.vqa_summary_token_id,
                                                      add_vqa_summary_token=self.params['train']['add_vqa_summary_token'],
                                                      add_multitask=self.params['train']['add_multitask'],
                                                      add_viz_w_add_multitask=self.params['train']['add_viz_w_add_multitask'],
                                                      add_multitask_unknown=self.params['train']['add_multitask_unknown'],
                                                      add_multitask_first_eos=self.params['train']['add_multitask_first_eos'],
                                                      multitask_wt=self.params['train']['multitask_wt'],
                                                      multitask_text_ft_wt=self.params['train']['multitask_text_ft_wt'],
                                                      multitask_viz_ft_wt=self.params['train']['multitask_viz_ft_wt'],
                                                      img_tokens=self.params['data']['img_tokens'],
                                                      num_proj_layers=self.params['train']["num_proj_layers"],
                                                      pooling_size=self.params['train']['pooling_size'],
                                                      image_size=self.params['data']['image_size'],
                                                      patch_size=self.params['train']["patch_size"],
                                                      num_modalities=len(self.params['data']['included_modalities']),
                                                      create_self_attn_block=self.params['train']["create_self_attn_block"],
                                                      language_model=llm_model,
                                                      create_x_attn_block=self.params['train']["create_x_attn_block"],
                                                      num_x_attn_heads=self.params['train']["num_x_attn_heads"],
                                                      add_x_attn_mlp=self.params['train']["add_x_attn_mlp"],
                                                      x_attn_query=self.params['train']["x_attn_query"],
                                                      create_moe_block=self.params['train']["create_moe_block"],
                                                      moe_use_router=self.params['train']["moe_use_router"],
                                                      moe_router_hidden_dim=self.params['train']["moe_router_hidden_dim"],
                                                      moe_num_proj=self.params['train']['moe_num_proj'],
                                                      moe_fusion_mode=self.params['train']['moe_fusion_mode'],
                                                      moe_use_shared_expert=self.params['train']['moe_use_shared_expert'],
                                                      moe_sum_weights=self.params['train']['moe_sum_weights'],
                                                      moe_use_lite_router=self.params['train']['moe_use_lite_router'],
                                                      moe_router_reg_coeff=self.params['train']['moe_router_reg_coeff'],
                                                      moe_adapted_router=self.params['train']['moe_adapted_router'],
                                                      moe_token_based_router=self.params['train']['moe_token_based_router'],
                                                      moe_w_text=self.params['train']['moe_w_text'],
                                                      moe_token_and_seq_based_router=self.params['train']['moe_token_and_seq_based_router'],
                                                      moe_token_and_seq_based_router_w_viz=self.params['train']['moe_token_and_seq_based_router_w_viz'],
                                                      moe_higher_level_router_num_blocks=self.params['train']['moe_higher_level_router_num_blocks'],
                                                      moe_higher_level_block_kwargs=self.params['train']['moe_higher_level_block_kwargs'],
                                                      load_projection_matrix=self.params['train']["load_projection_matrix"],
                                                      tokenizer=self.tokenizer)
        return model, None

    def load_vision_model(self):
        model = ViT3DTower.from_pretrained(self.params['train']["vision_model_name"],
                                           vision_select_layer=self.params['train']["vision_select_layer"],
                                           vision_select_feature=self.params['train']["vision_select_feature"],
                                           image_channel=self.params['train']["image_channel"],
                                           image_size=self.params['data']["image_size"],
                                           patch_size=self.params['train']["patch_size"])
        return model, None

    def get_vqa_train_data(self):
        """Load the vqa dataset for training"""
        dataset_factory = get_dataset_factory(dataset_type=self.params['data']['train_dataset'])
        train_data = dataset_factory.create_dataset(tokenizer=self.tokenizer, beg_prompt="", mid_prompt="", end_prompt="",
                                                    prompt_type=self.params['data']['prompt_type'],
                                                    data_path=self.params['data']['data_path'],
                                                    included_modalities=self.params['data']['included_modalities'],
                                                    img_dir=self.params['data']['img_dir'],
                                                    img_tokens=self.params['data']['img_tokens'],
                                                    pad_token_str=self.pad_token,
                                                    img_token_str=self.img_token,
                                                    seq_length=self.params['data']['seq_length'],
                                                    add_vqa_summary_token=self.params['train']['add_vqa_summary_token'],
                                                    vqa_summary_token=self.vqa_summary_token,
                                                    calculate_mae=self.params['data']['calculate_mae'],
                                                    use_multitask_unknown=self.params['train']['add_multitask_unknown'],
                                                    calculate_multitask=self.params['train']['add_multitask'], mode='train')
        test_data = dataset_factory.create_dataset(tokenizer=self.tokenizer, beg_prompt="", mid_prompt="", end_prompt="",
                                                   prompt_type=self.params['data']['prompt_type'],
                                                   data_path=self.params['data']['inf_data_path'],
                                                   included_modalities=self.params['data']['included_modalities'],
                                                   img_dir=self.params['data']['img_dir'],
                                                   img_tokens=self.params['data']['img_tokens'],
                                                   pad_token_str=self.pad_token,
                                                   img_token_str=self.img_token,
                                                   seq_length=self.params['data']['seq_length'],
                                                   add_vqa_summary_token=self.params['train']['add_vqa_summary_token'],
                                                   vqa_summary_token=self.vqa_summary_token,
                                                   calculate_mae=self.params['data']['calculate_mae'],
                                                   use_multitask_unknown=self.params['train']['add_multitask_unknown'],
                                                   calculate_multitask=self.params['train']['add_multitask'], mode='val')
        return {"train": train_data, "test": test_data}

    def get_inf_data(self):
        dataset_factory = get_dataset_factory(dataset_type=self.params['data']['inf_dataset'])
        return dataset_factory.create_dataset(tokenizer=self.tokenizer, beg_prompt="", mid_prompt="", end_prompt="",
                                              prompt_type=self.params['data']['prompt_type'],
                                              data_path=self.params['data']['inf_data_path'],
                                              included_modalities=self.params['data']['included_modalities'],
                                              img_dir=self.params['data']['img_dir'],
                                              img_tokens=self.params['data']['img_tokens'],
                                              pad_token_str=self.pad_token,
                                              img_token_str=self.img_token,
                                              add_vqa_summary_token=self.params['train']['add_vqa_summary_token'],
                                              vqa_summary_token=self.vqa_summary_token,
                                              seq_length=self.params['data']['seq_length'],
                                              calculate_mae=self.params['data']['calculate_mae'],
                                              use_multitask_unknown=self.params['train']['add_multitask_unknown'],
                                              calculate_multitask=self.params['train']['add_multitask'], mode='test')

    @property
    def required_params(self):
        required_params = super().required_params
        required_params["data"] = required_params["data"] + ["image_size", "included_modalities", "calculate_mae"]
        required_params["data"] = {key for key in required_params["data"] if key not in ['height', 'width', 'num_channels']}
        required_params["train"] = required_params["train"] + ["vision_model_name", "vision_select_layer",
                                                               "vision_select_feature", "pooling_size", "patch_size",
                                                               "image_channel", "freeze_llm_model",
                                                               "freeze_vision_model",
                                                               "pretrained", "load_projection_matrix",
                                                               "vision_select_layer", "vision_select_feature",
                                                               "image_channel", "num_proj_layers",
                                                               "create_self_attn_block",  "num_attn_layers",
                                                               "num_attn_heads", "add_attn_mlp", 'create_x_attn_block',
                                                               "create_moe_block", "moe_use_router",
                                                               "moe_router_hidden_dim", "moe_num_proj",
                                                               "moe_fusion_mode", 'num_x_attn_heads', 'add_x_attn_mlp',
                                                               'x_attn_query', 'moe_use_shared_expert',
                                                               'moe_sum_weights', 'moe_use_lite_router',
                                                               'moe_router_reg_coeff', 'moe_adapted_router',
                                                               'moe_token_based_router',
                                                               'add_multitask', 'add_vqa_summary_token',
                                                               'add_viz_w_add_multitask', 'add_multitask_unknown',
                                                               'add_multitask_first_eos',
                                                               'multitask_wt', 'multitask_text_ft_wt',
                                                               'multitask_viz_ft_wt', 'moe_w_text',
                                                               'moe_token_and_seq_based_router',
                                                               'moe_token_and_seq_based_router_w_viz',
                                                               'moe_higher_level_router_num_blocks',
                                                               'moe_higher_level_block_kwargs']
        required_params["inf"] = required_params["inf"] + ["llm_model_name", "use_quantization", "r", "lora_alpha",
                                                           "target_modules", "lora_dropout", "bias", "task_type",
                                                           "load_projection_matrix"]
        assert isinstance(required_params, dict), "required_params must be a dictionary"
        return required_params

    @property
    def pixel_values_dtype(self):
        return torch.float

    @property
    def vqa_summary_token(self):
        return "<VQA_SUMMARY>"
