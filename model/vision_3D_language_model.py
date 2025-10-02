import torch
from collections.abc import Iterable
from model.vision_language_model import VisionLanguageModel
from model.moe_block import MoE_Block
from model.higher_level_moe_block import Higher_Level_MoE_Block
from safetensors import safe_open
from torch import nn
import os
from utils.huggingface_utils import convert_meta_to_tensor
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass


@dataclass
class VQAMultiTaskOutput(CausalLMOutputWithPast):
    area_logits: torch.FloatTensor = None
    region_logits: torch.FloatTensor = None
    shape_logits: torch.FloatTensor = None
    satellite_logits: torch.FloatTensor = None
    multitask_loss: torch.FloatTensor = None


class Vision3DLanguageModel(VisionLanguageModel):
    def __init__(self, vision_model, language_model, img_token_id, vqa_summary_token_id, pooling_size=2,
                 img_tokens=1024, num_proj_layers=1, image_size=(32, 256, 256), patch_size=(4, 16, 16), num_modalities=4,
                 create_self_attn_block=True, create_x_attn_block=True, num_attn_layers=1, num_attn_heads=12,
                 add_attn_mlp=True, num_x_attn_heads=12, add_x_attn_mlp=True, x_attn_query="text",
                 create_moe_block=False, moe_use_router=False, moe_router_hidden_dim=128, moe_num_proj=1,
                 moe_fusion_mode="sum", moe_use_shared_expert=False, moe_sum_weights=False, moe_use_lite_router=False,
                 moe_router_reg_coeff=0.01, moe_adapted_router=False, moe_token_based_router=False,
                 add_vqa_summary_token=True, add_multitask=True, add_viz_w_add_multitask=False,
                 add_multitask_unknown=True, add_multitask_first_eos=True, num_area=8, num_shape=7, num_sat=5,
                 num_region=11, multitask_wt=1.0,  multitask_text_ft_wt=1.0, multitask_viz_ft_wt=1.0, moe_w_text=False,
                 moe_token_and_seq_based_router=False, moe_token_and_seq_based_router_w_viz=False,
                 moe_higher_level_router_num_blocks=None, moe_higher_level_block_kwargs=None, tokenizer=None):
        image_size = (1, 1) + tuple(image_size)
        super().__init__(vision_model=vision_model, language_model=language_model, img_token_id=img_token_id,
                         image_size=image_size, img_tokens=img_tokens,
                         num_proj_layers=num_proj_layers,
                         create_projection_layer=False, create_self_attn_block=False, create_x_attn_block=False,
                         num_attn_layers=num_attn_layers, num_attn_heads=num_attn_heads, add_attn_mlp=add_attn_mlp,
                         num_x_attn_heads=num_x_attn_heads, add_x_attn_mlp=add_x_attn_mlp, x_attn_query=x_attn_query)
        if num_modalities < 1:
            raise ValueError("num_modalities must be 1 or greater")
        self.patch_size = patch_size
        self.num_modalities = num_modalities
        self.pooling_size = pooling_size
        self.num_patches_pre = [img // pch for img, pch in zip(self.image_size[2:], self.patch_size)]
        if isinstance(self.pooling_size, int):
            self.num_patches_post = [num // self.pooling_size for num in self.num_patches_pre]
        elif isinstance(self.pooling_size, Iterable):
            self.num_patches_post = [num // pooling_size_ for (num, pooling_size_) in zip(self.num_patches_pre, self.pooling_size)]
        sample_input = torch.randn(*image_size).to("cuda")
        vision_output = self.vision_model(sample_input).size()
        self.vision_hidden_dim = vision_output[2]
        if create_self_attn_block:
            self.set_self_attn_block()
        if create_x_attn_block:
            self.set_x_attn_block()
        self.moe_use_router = moe_use_router
        self.moe_use_lite_router = moe_use_lite_router
        self.moe_router_hidden_dim = moe_router_hidden_dim
        self.moe_fusion_mode = moe_fusion_mode
        self.moe_num_proj = moe_num_proj
        self.moe_use_shared_expert = moe_use_shared_expert
        self.moe_sum_weights = moe_sum_weights
        self.create_moe_block = create_moe_block
        self.moe_router_reg_coeff = moe_router_reg_coeff
        self.moe_adapted_router = moe_adapted_router
        self.moe_token_based_router = moe_token_based_router
        self.moe_w_text = moe_w_text
        self.moe_token_and_seq_based_router = moe_token_and_seq_based_router
        self.moe_token_and_seq_based_router_w_viz = moe_token_and_seq_based_router_w_viz
        self.moe_higher_level_router_num_blocks = moe_higher_level_router_num_blocks
        self.moe_higher_level_block_kwargs = moe_higher_level_block_kwargs
        if create_moe_block:
            if self.vision_model.select_feature == "cls_patch":
                assert self.moe_use_router and not self.moe_token_based_router, "cls_patch feature selection only for MoE block with router"
            self.set_moe_block()
        else:
            assert self.vision_model.select_feature == "patch", "No MoE block (or token based routing) requires patch feature selection"
            self.set_language_projection_layers()
        self.vqa_summary_token_id = vqa_summary_token_id
        self.add_vqa_summary_token = add_vqa_summary_token
        self.add_multitask = add_multitask
        self.add_viz_w_add_multitask = add_viz_w_add_multitask
        self.add_multitask_unknown = add_multitask_unknown
        self.add_multitask_first_eos = add_multitask_first_eos
        self.num_area = num_area
        self.num_shape = num_shape
        self.num_sat = num_sat
        self.num_region = num_region
        self.multitask_wt = multitask_wt
        self.multitask_text_ft_wt = multitask_text_ft_wt
        self.multitask_viz_ft_wt = multitask_viz_ft_wt
        self.tokenizer = tokenizer

        if self.add_multitask:
            D_txt = self.language_model.config.hidden_size
            if self.add_viz_w_add_multitask:
                input_dim_heads = (1 + self.img_tokens) * D_txt # Calculate flattened dimension
                print(f"Multitask heads adapting to FLATTENED input dim: {input_dim_heads} = (1 + {self.img_tokens}) * {D_txt}")
            else:
                input_dim_heads = D_txt
                print(f"Multitask heads using TEXT ONLY input dim: {input_dim_heads}")
            # Initialize heads with the determined input dimension
            self.head_area = nn.Linear(input_dim_heads, self.num_area)
            self.head_shape = nn.Linear(input_dim_heads, self.num_shape)
            self.head_satellite = nn.Linear(input_dim_heads, self.num_sat)
            self.head_region = nn.Linear(input_dim_heads, self.num_region)
            if self.add_multitask_unknown:
                self.head_unknown = nn.Linear(input_dim_heads, 1)
            print(f"Initialized Vision3DLanguageModel with {self.num_modalities} modalities, self-attention block: {create_self_attn_block}, x-attention block: {create_x_attn_block}, with # of higher-level Moe Blocks {self.moe_higher_level_router_num_blocks} and MoE kwargs {self.moe_higher_level_block_kwargs}, moe block: {create_moe_block}, moe_w_text {self.moe_w_text}, multitask head: {self.add_multitask} with unknown {self.add_multitask_unknown} with vision {self.add_viz_w_add_multitask} and vqa summary token {self.add_vqa_summary_token} and first eos {self.add_multitask_first_eos} and router {self.moe_use_router} and token-based routing {self.moe_token_based_router} and lite router {self.moe_use_lite_router} and adapted router {self.moe_adapted_router} and shared expert {self.moe_use_shared_expert} and multitask_wt {self.multitask_wt}, multitask_text_ft_wt {self.multitask_text_ft_wt}, and multitask_viz_ft_wt {self.multitask_viz_ft_wt}")

    def _run_heads(self, feat):
        outputs = {
            "area": self.head_area(feat),
            "shape": self.head_shape(feat),
            "satellite": self.head_satellite(feat),
            "region": self.head_region(feat)
            }
        if self.add_multitask_unknown:
            outputs["unknown"] = self.head_unknown(feat)
        return outputs

    def forward(self, pixel_values, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                output_attentions=None, output_hidden_states=None, labels=None, area_label=None, region_label=None,
                shape_label=None, satellite_label=None, unknown_label=None, return_dict=None):
        (combined_embeddings, attention_mask, labels), image_features, router_reg_loss = self.get_image_and_text_embeddings(pixel_values=pixel_values,
                                                                                                                            input_ids=input_ids,
                                                                                                                            attention_mask=attention_mask,
                                                                                                                            labels=labels)
        outputs = self.language_model(inputs_embeds=combined_embeddings, attention_mask=attention_mask, labels=labels,
                                      output_hidden_states=self.add_multitask)

        multitask_loss = torch.tensor(0.0, device=image_features.device)
        if self.add_multitask:
            assert (area_label is not None) and (region_label is not None) and (shape_label is not None) and (satellite_label is not None), "Multitask labels must be provided"
            # get last-layer hidden state of vqa_summary_token or last token prior to adding
            if self.add_vqa_summary_token:
                vqa_summary_pos = (input_ids == self.vqa_summary_token_id).nonzero(as_tuple=False)  # (B,1,2) → (batch, idx)
                batch_idx = vqa_summary_pos[:, 0]
                token_idx = vqa_summary_pos[:, 1]
                token_matrix = outputs.hidden_states[-1]  # (B, L, D)
                feat_txt = token_matrix[batch_idx, token_idx]  # (B, D)
            else:
                if self.add_multitask_first_eos:
                    eos_mask = (input_ids == self.tokenizer.eos_token_id)  # shape [B, L]
                    # calculate first_eos_idx by looking at the original ipput_ids and adding img_tokens-1 (the embedding inserted into the language model)
                    first_eos_idx = (eos_mask.float().argmax(dim=1) + (self.img_tokens-1)).long()
                    batch_idx = torch.arange(attention_mask.size(0), device=attention_mask.device)
                    feat_txt = outputs.hidden_states[-1][batch_idx, first_eos_idx]  # (B, D)
                else:
                    last_idx = attention_mask.sum(dim=1) - 1  # (B,)  e.g. [23,17,…]
                    last_idx = last_idx.long()  # Ensure it's long for indexing
                    batch_idx = torch.arange(attention_mask.size(0), device=attention_mask.device)
                    feat_txt = outputs.hidden_states[-1][batch_idx, last_idx]  # (B, D)
            # add visual features
            if self.add_viz_w_add_multitask:
                feat_txt = torch.cat((self.multitask_text_ft_wt * feat_txt.unsqueeze(1),
                                      self.multitask_viz_ft_wt * image_features),
                                     dim=1)
                feat_txt = feat_txt.flatten(start_dim=1)

            # multitask heads
            logits = self._run_heads(feat_txt)

            multitask_loss += F.cross_entropy(logits["area"], area_label.float())
            multitask_loss += F.cross_entropy(logits["shape"], shape_label.float())
            multitask_loss += F.cross_entropy(logits["satellite"], satellite_label.float())
            multitask_loss += F.binary_cross_entropy_with_logits(logits["region"], region_label.float())
            if self.add_multitask_unknown:
                multitask_loss += F.binary_cross_entropy_with_logits(logits["unknown"], unknown_label.float())

        if outputs.loss is not None:
            total_loss = outputs.loss + router_reg_loss + multitask_loss*self.multitask_wt
        else:
            total_loss = router_reg_loss + multitask_loss*self.multitask_wt

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens, **decoding_kwargs):
        with torch.no_grad():
            (combined_embeddings, attention_mask,
             labels), image_features, router_reg_loss = self.get_image_and_text_embeddings(pixel_values=pixel_values,
                                                                                           input_ids=input_ids,
                                                                                           attention_mask=attention_mask)

            gen_out = self.language_model.generate(inputs_embeds=combined_embeddings, attention_mask=attention_mask,
                                                   max_new_tokens=max_new_tokens, output_hidden_states=True,
                                                   return_dict_in_generate=True, use_cache=False, **decoding_kwargs)

            if not self.add_multitask:
                return gen_out.sequences
            else:
                # Assuming B=1
                sequences = gen_out.sequences  # Shape: (1, L_sequence)

                # gen_out.hidden_states structure is (time_steps, layers) confirmed by debug logs
                # Each state tensor like hidden_states[t][layer] seems to be (1, 1, D)
                hidden_states_all_steps = gen_out.hidden_states

                if self.add_vqa_summary_token:
                    current_sequence = sequences[0]
                    mask = current_sequence == self.vqa_summary_token_id

                    if not mask.any():
                        # Fallback: Use last layer state from the LAST time step
                        print("VQA Summary Token not found. Using hidden state from the last token/step.")
                        # Shape is (1, 1, D)
                        hidden_state_tensor = hidden_states_all_steps[-1][-1]
                    else:
                        # Summary token found
                        pos_idx = mask.float().argmax().item()  # Get integer index

                        # Retrieve hidden state computed AT time step `pos_idx` for the final layer.
                        # Shape is (1, 1, D)
                        hidden_state_tensor = hidden_states_all_steps[pos_idx][-1]
                else:
                    if self.add_multitask_first_eos:
                        lm_out = self.language_model(
                            inputs_embeds=combined_embeddings,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            use_cache=False,
                        )
                        eos_mask = (input_ids == self.tokenizer.eos_token_id)  # shape [B, L]
                        # calculate first_eos_idx by looking at the original ipput_ids and adding img_tokens-1 (the embedding inserted into the language model)
                        first_eos_idx = (eos_mask.float().argmax(dim=1) + (self.img_tokens - 1)).long()
                        batch_idx = torch.arange(attention_mask.size(0), device=attention_mask.device)
                        hidden_state_tensor = lm_out.hidden_states[-1][batch_idx, first_eos_idx]  # (B, D)
                    else:
                        # Get last layer state from the LAST time step
                        # Shape is (1, 1, D)
                        hidden_state_tensor = hidden_states_all_steps[-1][-1]  # Corrected indexing

                if not isinstance(hidden_state_tensor, torch.Tensor):
                    raise TypeError(f"Extracted hidden_state is not a tensor ({type(hidden_state_tensor)})")

                # We expect shape (1, 1, D), squeeze dim 1 to get (1, D)
                if hidden_state_tensor.dim() == 3 and hidden_state_tensor.shape[1] == 1:
                    feat_txt = hidden_state_tensor.squeeze(1)
                elif hidden_state_tensor.dim() == 2 and hidden_state_tensor.shape[0] == 1:  # Maybe already squeezed?
                    feat_txt = hidden_state_tensor
                else:
                    # Handle unexpected shapes if necessary
                    raise ValueError(
                        f"Unexpected shape for hidden_state_tensor: {hidden_state_tensor.shape}. Cannot squeeze to (1, D).")

                if feat_txt.dim() != 2 or feat_txt.shape[0] != 1:
                    raise ValueError(
                        f"Unexpected shape for feat_txt before combining: {feat_txt.shape}. Expected (1, D)")

                if self.add_viz_w_add_multitask:
                    feat_txt = torch.cat((self.multitask_text_ft_wt * feat_txt.unsqueeze(1),
                                          self.multitask_viz_ft_wt * image_features),
                                         dim=1)
                    feat_txt = feat_txt.flatten(start_dim=1)

                logits = self._run_heads(feat_txt)

                output = {
                    "sequences": sequences,
                    "area_logits": logits["area"],
                    "shape_logits": logits["shape"],
                    "satellite_logits": logits["satellite"],
                    "region_logits": logits["region"]
                }
                if self.add_multitask_unknown:
                    output["unknown_logits"] = logits["unknown"]
                return output

    def get_image_and_text_embeddings(self, pixel_values, input_ids, image_features=None, projected_features=None,
                                      attention_mask=None, labels=None):
        first_seq_input_ids = input_ids[0]
        if self.img_token_id in first_seq_input_ids:
            indices = torch.nonzero(first_seq_input_ids == self.img_token_id, as_tuple=False)
            img_token_pos = indices.item()
        else:
            raise ValueError("Image token not found in input_ids")
        image_features = []
        for modality in range(self.num_modalities):
            image_features_ = self.vision_model.forward_features(pixel_values[:, modality])
            if isinstance(self.pooling_size, Iterable) or self.pooling_size > 1:
                image_features_ = self.spatial_pooling(image_features_)
            image_features.append(image_features_)
        image_features = torch.cat(image_features, dim=1)
        if self.moe_w_text:
            input_ids_wo_image_token = torch.cat((input_ids[:, :img_token_pos],
                                                  input_ids[:, (img_token_pos + 1):]), dim=1)
            attention_mask_wo_image_token = torch.cat((attention_mask[:, :img_token_pos],
                                                       attention_mask[:, (img_token_pos + 1):]), dim=1)
            pre_out = self.language_model(
                input_ids=input_ids_wo_image_token,
                attention_mask=attention_mask_wo_image_token,
                output_hidden_states=True,
            )
            eos_mask = (input_ids_wo_image_token == self.tokenizer.eos_token_id)  # shape [B, L]
            # make sure to only use the first eos token index (so we don't train on the labels)
            first_eos_idx = eos_mask.float().argmax(dim=1).long()
            batch_idx = torch.arange(attention_mask.size(0), device=attention_mask.device)
            prompt_features = pre_out.hidden_states[-1][batch_idx, first_eos_idx]  # (B, D)
            outputs = self.language_projection(image_features, prompt_features)
        else:
            outputs = self.language_projection(image_features)
        if len(outputs) == 2:
            image_features, router_reg_loss = outputs
        else:
            image_features, router_reg_loss = outputs, 0.0
        return (super().get_image_and_text_embeddings(pixel_values=pixel_values, input_ids=input_ids,
                                                      projected_features=image_features, attention_mask=attention_mask,
                                                      labels=labels), image_features, router_reg_loss)


    def set_moe_block(self):
        if self.moe_higher_level_router_num_blocks is not None:
            assert isinstance(self.moe_higher_level_router_num_blocks, int) and self.moe_higher_level_router_num_blocks > 0, "Higher level router num blocks must be greater than 0"
            self.language_projection = Higher_Level_MoE_Block(num_blocks=self.moe_higher_level_router_num_blocks,
                                                              block_kwargs=self.moe_higher_level_block_kwargs,
                                                              num_modalities=self.num_modalities,
                                                              vision_hidden_dim=self.vision_hidden_dim,
                                                              language_emb_dim=self.language_model.config.hidden_size,
                                                              use_router=self.moe_use_router,
                                                              router_hidden_dim=self.moe_router_hidden_dim,
                                                              num_proj=self.moe_num_proj,
                                                              fusion_mode=self.moe_fusion_mode,
                                                              use_shared_expert=self.moe_use_shared_expert,
                                                              sum_weights=self.moe_sum_weights,
                                                              token_based_router=self.moe_token_based_router,
                                                              use_lite_router=self.moe_use_lite_router,
                                                              router_reg_coeff=self.moe_router_reg_coeff,
                                                              adapted_router=self.moe_adapted_router,
                                                              w_text_router=self.moe_w_text,
                                                              token_and_seq_based_router=self.moe_token_and_seq_based_router,
                                                              token_and_seq_based_router_w_viz=self.moe_token_and_seq_based_router_w_viz)
        else:
            self.language_projection = MoE_Block(num_modalities=self.num_modalities,
                                                 vision_hidden_dim=self.vision_hidden_dim,
                                                 language_emb_dim=self.language_model.config.hidden_size,
                                                 use_router=self.moe_use_router,
                                                 router_hidden_dim=self.moe_router_hidden_dim,
                                                 num_proj=self.moe_num_proj,
                                                 fusion_mode=self.moe_fusion_mode,
                                                 use_shared_expert=self.moe_use_shared_expert,
                                                 sum_weights=self.moe_sum_weights,
                                                 token_based_router=self.moe_token_based_router,
                                                 use_lite_router=self.moe_use_lite_router,
                                                 router_reg_coeff=self.moe_router_reg_coeff,
                                                 adapted_router=self.moe_adapted_router,
                                                 w_text_router=self.moe_w_text,
                                                 token_and_seq_based_router=self.moe_token_and_seq_based_router,
                                                 token_and_seq_based_router_w_viz=self.moe_token_and_seq_based_router_w_viz)

    def spatial_pooling(self, x):
        B = x.shape[0]
        # only spatially pool on the non-cls tokens
        if self.vision_model.select_feature == "cls_patch":
            cls_token = x[:, :1, :]
            patch_tokens = x[:, 1:, :]
            to_3d = Rearrange("b (p1 p2 p3) d -> b d p1 p2 p3", b=B, d=self.vision_hidden_dim,
                              p1=self.num_patches_pre[0],
                              p2=self.num_patches_pre[1], p3=self.num_patches_pre[2])
            patch_tokens = to_3d(patch_tokens)
            patch_tokens = F.avg_pool3d(patch_tokens, kernel_size=self.pooling_size, stride=self.pooling_size)
            to_seq = Rearrange("b d p1 p2 p3 -> b (p1 p2 p3) d", b=B, d=self.vision_hidden_dim,
                               p1=self.num_patches_post[0],
                               p2=self.num_patches_post[1], p3=self.num_patches_post[2])
            patch_tokens = to_seq(patch_tokens)
            x = torch.cat([cls_token, patch_tokens], dim=1)
        else:
            to_3d = Rearrange("b (p1 p2 p3) d -> b d p1 p2 p3", b=B, d=self.vision_hidden_dim, p1=self.num_patches_pre[0],
                              p2=self.num_patches_pre[1], p3=self.num_patches_pre[2])
            x = to_3d(x)
            x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)
            to_seq = Rearrange("b d p1 p2 p3 -> b (p1 p2 p3) d", b=B, d=self.vision_hidden_dim, p1=self.num_patches_post[0],
                               p2=self.num_patches_post[1], p3=self.num_patches_post[2])
            x = to_seq(x)
        return x

    @classmethod
    def from_pretrained(cls, save_directory, vision_model, language_model, img_token_id, vqa_summary_token_id,
                        pooling_size=2, img_tokens=1024,
                        num_proj_layers=1, image_size=(32, 256, 256), patch_size=(4, 16, 16),
                        num_modalities=4, create_self_attn_block=True, create_x_attn_block=True, num_attn_layers=1, num_attn_heads=12,
                        add_attn_mlp=True, num_x_attn_heads=12, add_x_attn_mlp=True, x_attn_query="text", add_vqa_summary_token=True,
                        add_multitask=True, add_viz_w_add_multitask=False, add_multitask_unknown=True,
                        add_multitask_first_eos=True, multitask_wt=1.0, multitask_text_ft_wt=1.0, multitask_viz_ft_wt=1.0,
                        create_moe_block=False, moe_use_router=False, moe_router_hidden_dim=128, moe_num_proj=1,
                        moe_fusion_mode="sum", moe_use_shared_expert=False, moe_sum_weights=False,
                        moe_use_lite_router=False, moe_router_reg_coeff=0.01, moe_adapted_router=False,
                        moe_token_based_router=False, moe_w_text=False, moe_token_and_seq_based_router=False,
                        moe_token_and_seq_based_router_w_viz=False, moe_higher_level_router_num_blocks=None,
                        moe_higher_level_block_kwargs=None, load_projection_matrix=False, tokenizer=None):
        # Initialize the model
        model = cls(vision_model=vision_model, language_model=language_model, img_token_id=img_token_id,
                    vqa_summary_token_id=vqa_summary_token_id,
                    pooling_size=pooling_size, img_tokens=img_tokens, num_proj_layers=num_proj_layers,
                    image_size=image_size, patch_size=patch_size,
                    num_modalities=num_modalities, create_self_attn_block=create_self_attn_block,
                    create_x_attn_block=create_x_attn_block, num_attn_layers=num_attn_layers,
                    num_attn_heads=num_attn_heads, add_attn_mlp=add_attn_mlp, num_x_attn_heads=num_x_attn_heads,
                    add_x_attn_mlp=add_x_attn_mlp, x_attn_query=x_attn_query, add_vqa_summary_token=add_vqa_summary_token,
                    add_multitask=add_multitask, add_viz_w_add_multitask=add_viz_w_add_multitask,
                    add_multitask_unknown=add_multitask_unknown, add_multitask_first_eos=add_multitask_first_eos,
                    multitask_wt=multitask_wt,
                    multitask_text_ft_wt=multitask_text_ft_wt, multitask_viz_ft_wt=multitask_viz_ft_wt,
                    create_moe_block=create_moe_block,
                    moe_use_router=moe_use_router, moe_router_hidden_dim=moe_router_hidden_dim,
                    moe_num_proj=moe_num_proj, moe_fusion_mode=moe_fusion_mode,
                    moe_use_shared_expert=moe_use_shared_expert, moe_sum_weights=moe_sum_weights,
                    moe_use_lite_router=moe_use_lite_router, moe_router_reg_coeff=moe_router_reg_coeff,
                    moe_adapted_router=moe_adapted_router, moe_token_based_router=moe_token_based_router,
                    moe_w_text=moe_w_text, moe_token_and_seq_based_router=moe_token_and_seq_based_router,
                    moe_token_and_seq_based_router_w_viz=moe_token_and_seq_based_router_w_viz,
                    moe_higher_level_router_num_blocks=moe_higher_level_router_num_blocks,
                    moe_higher_level_block_kwargs=moe_higher_level_block_kwargs, tokenizer=tokenizer)
        model.to(device="cuda" if torch.cuda.is_available() else "cpu")

        state_dict = None
        if save_directory is not None:
            if "checkpoint" in os.path.basename(save_directory):
                model_path = os.path.join(save_directory, "model.safetensors")
                # Load the state dictionary from the safetensors file
                state_dict = {}
                with safe_open(model_path, framework="pt") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                state_dict = convert_meta_to_tensor(state_dict, device="cuda" if torch.cuda.is_available() else "cpu")
            else:
                # Load the state dictionary of the trainable parameters
                state_dict = torch.load(os.path.join(save_directory, "pytorch_model.bin"), map_location='cpu')
                state_dict = convert_meta_to_tensor(state_dict, device="cuda" if torch.cuda.is_available() else "cpu")
        # Only load the projection matrix weights if specified
        if load_projection_matrix:
            projection_state_dict = {
                'weight': state_dict.get('language_projection.weight'),
                'bias': state_dict.get('language_projection.bias')
            }
            # Check if the weights exist in the state_dict
            if projection_state_dict['weight'] is not None and projection_state_dict['bias'] is not None:
                # Load the projection weights into the model
                model.language_projection.load_state_dict(projection_state_dict)
                print("Projection matrix weights loaded successfully.")
            else:
                print("Projection matrix weights not found in the state dictionary.")
        else:
            model.load_state_dict(state_dict)
        return model
