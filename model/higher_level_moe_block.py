from torch import nn
import torch
import torch.nn.functional as F
from model.moe_block import MoE_Block


class Higher_Level_MoE_Block(nn.Module):

    def __init__(
            self,
            num_blocks=2,
            block_kwargs=None,
            **kwargs
    ):
        super().__init__()
        assert num_blocks > 1, "Number of blocks must be greater than 1"
        self.num_blocks = num_blocks
        self.block_kwargs = block_kwargs
        if block_kwargs is not None:
            for key, vals in block_kwargs.items():
                assert key in kwargs, f"Key {key} not found in kwargs"
                assert len(vals) == self.num_blocks, f"Length of {key} must be equal to num_blocks"
            self.moe_blocks = nn.ModuleList()
            for i in range(self.num_blocks):
                current_block_kwargs = dict(kwargs)
                current_block_kwargs.update({key: vals[i] for key, vals in block_kwargs.items()})
                self.moe_blocks.append(MoE_Block(**current_block_kwargs))
        else:
            self.moe_blocks = nn.ModuleList([MoE_Block(**kwargs) for _ in range(num_blocks)])
        self.higher_router_input_dims = kwargs['language_emb_dim']
        self.higher_router_hidden_dim = kwargs['router_hidden_dim']
        self.vision_hidden_dim = kwargs['vision_hidden_dim']
        self.num_modalities = kwargs['num_modalities']
        self.higher_router_output_dims = num_blocks
        self.higher_router_mlp = self._create_higher_router(self.higher_router_input_dims,
                                                            self.higher_router_output_dims)
        print(f"Higher-level MoEBlock with num_blocks={self.num_blocks} and block_kwargs={self.block_kwargs}")

    def forward(self, image_features, prompt_features=None):
        block_weights = self._get_higher_level_router_weights(prompt_features, image_features.shape[0])
        fused = []
        for i in range(self.num_blocks):
            if self.moe_blocks[i].token_based_router:
                # make sure to drop CLS token for token_based_router
                B, S, D = image_features.shape
                assert S % self.num_modalities == 0, "Sequence length must be multiple of num_modalities"
                L = S // self.num_modalities  # tokens per modality incl. CLS
                image_features_ = (
                    image_features
                    .contiguous()
                    .view(B, L, self.num_modalities * D)[:, 1:]  # drop CLS
                    .reshape(B, (L - 1) * self.num_modalities, D)  # back to flat
                )
                fused_i, _ = self.moe_blocks[i].forward(image_features=image_features_,
                                                        prompt_features=prompt_features)
            else:
                fused_i, _ = self.moe_blocks[i].forward(image_features=image_features, prompt_features=prompt_features)
            fused.append(block_weights[:, i] * fused_i)
        # sum the features from all blocks
        fused = torch.sum(torch.stack(fused), dim=0)
        return fused, torch.tensor(0.0, device=image_features.device)

    def _create_higher_router(self, input_dims, output_dims=1):
        """
        Creates a router MLP with the specified input and output dimensions.
        """
        return nn.Sequential(
            nn.Linear(input_dims, self.higher_router_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.higher_router_hidden_dim, output_dims)
        )

    def _get_higher_level_router_weights(self, prompt_features, B):

        assert prompt_features.shape[-1] == self.higher_router_input_dims, \
            f"Expected {self.higher_router_input_dims}, got {prompt_features.shape[-1]}"
        # shape [B, self.num_blocks, 1] for indexing and then broadcasting
        block_weights = F.softmax(self.higher_router_mlp(prompt_features), dim=-1).unsqueeze(-1)
        return block_weights
