import os
from safetensors import safe_open
import torch
from torch import nn
from utils.huggingface_utils import convert_meta_to_tensor
from transformers.modeling_outputs import CausalLMOutputWithPast


class VisionLanguageModel(nn.Module):
    def __init__(self, vision_model, language_model, img_token_id, image_size=(1, 3, 224, 224), img_tokens=197,
                 num_proj_layers=1,  create_projection_layer=True, create_self_attn_block=True,
                 create_x_attn_block=True, num_attn_layers=1, num_attn_heads=12, add_attn_mlp=True,
                 num_x_attn_heads=12, add_x_attn_mlp=True, x_attn_query="text"):
        super().__init__()
        self.vision_model = vision_model
        self.language_model = language_model
        self.img_token_id = img_token_id
        self.ignore_token_id = -100
        self.image_size = image_size
        self.img_tokens = img_tokens
        self.num_proj_layers = num_proj_layers
        self.language_projection = None
        if create_projection_layer:
            # calculate projection layer dimensions
            sample_input = torch.randn(1, 3, 224, 224).to("cuda")
            self.vision_hidden_dim = self.vision_model(sample_input).size(1)
            self.set_language_projection_layers()
        self.self_attn_block = None
        self.num_attn_layers = num_attn_layers
        self.num_attn_heads = num_attn_heads
        self.add_attn_mlp = add_attn_mlp
        if create_self_attn_block:
            self.set_self_attn_block()
        self.x_attn_block = None
        self.num_x_attn_heads = num_x_attn_heads
        self.add_x_attn_mlp = add_x_attn_mlp
        self.x_attn_query = x_attn_query
        if create_x_attn_block:
            self.set_x_attn_block()

    def forward(self, pixel_values, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                output_attentions=None, output_hidden_states=None, labels=None, return_dict=None):
        combined_embeddings, attention_mask, labels = self.get_image_and_text_embeddings(pixel_values=pixel_values,
                                                                                         input_ids=input_ids,
                                                                                         attention_mask=attention_mask,
                                                                                         labels=labels)
        # Pass the combined embeddings through the LLM
        outputs = self.language_model(inputs_embeds=combined_embeddings, attention_mask=attention_mask, labels=labels)

        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
    def set_language_projection_layers(self):
        # create a projection layer for the vision embeddings
        if self.num_proj_layers == 1:
            self.language_projection = nn.Linear(self.vision_hidden_dim, self.language_model.config.hidden_size)
            print("Projection layer created with 1 layer.")
        elif self.num_proj_layers == 2:
            self.language_projection = nn.Sequential(
                nn.Linear(self.vision_hidden_dim, self.language_model.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.language_model.config.hidden_size, self.language_model.config.hidden_size)
            )
            print("Projection layer created with 2 layers.")
        else:
            raise ValueError("Invalid number of projection layers specified.")

    def set_self_attn_block(self):
        if self.num_attn_layers == 1:
            self.self_attn_block = VisionSelfAttnBlock(embed_dim=self.language_model.config.hidden_size,
                                                       num_heads=self.num_attn_heads,
                                                       add_mlp=self.add_attn_mlp)
        elif self.num_attn_layers > 1:
            self.self_attn_block = nn.Sequential(*[VisionSelfAttnBlock(embed_dim=self.language_model.config.hidden_size,
                                                                       num_heads=self.num_attn_heads,
                                                                       add_mlp=self.add_attn_mlp) for _ in range(self.num_attn_layers)])
        print(f"Created self-attention block with embed_dim: {self.language_model.config.hidden_size}, "
              f"num_attn_layers: {self.num_attn_layers}, num_heads: {self.num_attn_heads}, add_mlp: {self.add_attn_mlp}")

    def set_x_attn_block(self):
        self.x_attn_block = CrossAttention(embed_dim=self.language_model.config.hidden_size,
                                           num_heads=self.num_x_attn_heads, add_mlp=self.add_x_attn_mlp)
        print(f"Created cross-attention block with embed_dim: {self.language_model.config.hidden_size}, "
              f"num_heads: {self.num_x_attn_heads}, add_mlp: {self.add_x_attn_mlp}, and x_attn_query: {self.x_attn_query}")

    def get_image_and_text_embeddings(self, pixel_values, input_ids, image_features=None, projected_features=None,
                                      attention_mask=None, labels=None):
        if projected_features is None:
            if image_features is None:
                # make sure to only retain the unpooled features
                image_features = self.vision_model.forward_features(pixel_values)
            # project image features to LLM embedding space
            image_features = self.language_projection(image_features)
        else:
            image_features = projected_features
        if self.self_attn_block is not None:
            image_features = self.self_attn_block(image_features)
        # obtain location of image token in batch_id=0
        first_seq_input_ids = input_ids[0]
        if self.img_token_id in first_seq_input_ids:
            indices = torch.nonzero(first_seq_input_ids == self.img_token_id, as_tuple=False)
            img_token_pos = indices.item()
        else:
            raise ValueError("Image token not found in input_ids")
        # Get text embeddings from LLM
        pre_img_text_embeddings = self.language_model.get_input_embeddings()(input_ids[:, :img_token_pos])
        post_img_text_embeddings = self.language_model.get_input_embeddings()(input_ids[:, (img_token_pos+1):])
        if self.x_attn_block is not None:
            text_embeddings = torch.cat((pre_img_text_embeddings, post_img_text_embeddings), dim=1)
            if self.x_attn_query == "text":
                text_embeddings = self.x_attn_block(query_embeds=text_embeddings, key_value_embeds=image_features)
                """
                if attention_mask is not None:
                    pre_img_text_attention_mask = attention_mask[:, :img_token_pos]
                    post_img_text_attention_mask = attention_mask[:, (img_token_pos + 1):]
                    text_query_mask = torch.cat([pre_img_text_attention_mask, post_img_text_attention_mask], dim=1)
                    text_embeddings = self.x_attn_block(query_embeds=text_embeddings, query_mask=text_query_mask,
                                                        key_value_embeds=image_features)
                else:
                    text_embeddings = self.x_attn_block(query_embeds=text_embeddings, key_value_embeds=image_features)
                """
                pre_img_text_embeddings, post_img_text_embeddings = text_embeddings[:, :img_token_pos], text_embeddings[:, img_token_pos:]
            elif self.x_attn_query == "vision":
                image_features = self.x_attn_block(query_embeds=image_features, key_value_embeds=text_embeddings)
            else:
                raise ValueError("Invalid x_attn_query specified.")
        # Concatenate vision and text embeddings
        combined_embeddings = torch.cat((pre_img_text_embeddings, image_features, post_img_text_embeddings), dim=1)
        # Adjust attention mask
        if attention_mask is not None:
            pre_img_attention_mask = attention_mask[:, :img_token_pos]
            img_attention_mask = torch.ones(image_features.size()[:2], device=image_features.device)
            post_img_attention_mask = attention_mask[:, (img_token_pos + 1):]
            attention_mask = torch.cat([pre_img_attention_mask, img_attention_mask, post_img_attention_mask], dim=1)
        else:
            attention_mask = torch.ones(combined_embeddings.size()[:2], device=combined_embeddings.device)
        # Adjust labels
        if labels is not None:
            pre_img_labels = labels[:, :img_token_pos]
            img_labels = torch.full((labels.size()[0], self.img_tokens), fill_value=self.ignore_token_id,
                                    device=labels.device)
            post_img_labels = labels[:, (img_token_pos + 1):]
            labels = torch.cat([pre_img_labels, img_labels, post_img_labels], dim=1)
        return combined_embeddings, attention_mask, labels

    def generate_image_embeddings(self, pixel_values):
        image_features = self.vision_model.forward_features(pixel_values)
        return image_features

    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens, **decoding_kwargs):
        with torch.no_grad():
            # debug stuff
            combined_embeddings, attention_mask, labels = self.get_image_and_text_embeddings(pixel_values=pixel_values,
                                                                                             input_ids=input_ids,
                                                                                             attention_mask=attention_mask)
            # https://github.com/huggingface/transformers/issues/23042
            return self.language_model.generate(inputs_embeds=combined_embeddings, attention_mask=attention_mask,
                                                max_new_tokens=max_new_tokens, use_cache=False, **decoding_kwargs)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        # Save the state dictionary of the trainable parameters
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, save_directory, vision_model, language_model, img_token_id, img_tokens, num_proj_layers=1,
                        create_self_attn_block=True, create_x_attn_block=True, num_attn_layers=1, num_attn_heads=12,
                        add_attn_mlp=True, num_x_attn_heads=12, add_x_attn_mlp=True, x_attn_query="text",
                        load_projection_matrix=False):
        # Initialize the model
        model = cls(vision_model=vision_model, language_model=language_model, img_token_id=img_token_id,
                    img_tokens=img_tokens, num_proj_layers=num_proj_layers,
                    create_self_attn_block=create_self_attn_block,
                    num_attn_layers=num_attn_layers, num_attn_heads=num_attn_heads, add_attn_mlp=add_attn_mlp,
                    create_x_attn_block=create_x_attn_block, num_x_attn_heads=num_x_attn_heads,
                    add_x_attn_mlp=add_x_attn_mlp, x_attn_query=x_attn_query)

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


class VisionSelfAttnBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, add_mlp=True):
        super().__init__()
        self.add_mlp = add_mlp
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)

        if self.add_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
            )
            self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-Attention
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = self.ln1(x)

        # Feed-Forward
        if self.add_mlp:
            mlp_out = self.mlp(x)
            x = x + mlp_out
            x = self.ln2(x)

        return x


class CrossAttention(nn.Module):
    """
    CrossAttention layer where text embeddings attend to vision embeddings.
    - query_embeds (queries) will be updated
    - key_value_embeds (keys/values) remain unchanged here
    """
    def __init__(self, embed_dim, num_heads, add_mlp=True):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(embed_dim)
        # Optionally add an MLP for further processing:
        self.mlp = None
        self.layernorm_mlp = None
        if add_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
            )
            self.layernorm_mlp = nn.LayerNorm(embed_dim)

    def forward(self, query_embeds, key_value_embeds, query_mask=None, key_value_mask=None):
        """
        query_embeds:  [B, T, D]  -- Queries
        key_value_embeds: [B, V, D] -- Keys/Values
        query_mask:   [B, T] (optional) boolean mask for text tokens (True=pad).
        key_value_mask: [B, V] (optional) boolean mask for vision tokens.

        Returns: updated_text of shape [B, T, D]
        """

        # PyTorch MultiheadAttention can take a "key_padding_mask" for the K/V side
        # to indicate which tokens to ignore if there's padding (vision_mask).
        # Typically, shape is [B, V], with True indicating tokens to ignore.
        # If text_mask is needed, it can be used in an attn_mask or we can
        # transform it similarly. For simplicity, we'll just pass vision_mask here.

        updated_query_embeds, _ = self.cross_attn(
            query=query_embeds,            # Q
            key=key_value_embeds,            # K
            value=key_value_embeds,          # V
            key_padding_mask=key_value_mask, # (B, V)
            attn_mask=None,               # optional [T, V] if needed
            need_weights=False            # don't return attention weights
        )

        # Residual connection + LayerNorm
        query_embeds = query_embeds + updated_query_embeds
        query_embeds = self.layernorm(query_embeds)

        # Optionally, a feed-forward (MLP) layer with another residual
        if self.mlp is not None:
            mlp_out = self.mlp(query_embeds)
            query_embeds = query_embeds + mlp_out
            query_embeds = self.layernorm_mlp(query_embeds)

        return query_embeds