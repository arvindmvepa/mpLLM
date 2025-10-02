from torch import nn
import torch
from collections import defaultdict
import json
import torch.nn.functional as F
import os


class MoE_Block(nn.Module):
    """
    A Mixture-of-Experts block that can:
      - Learn per-modality gates (via an MLP router or a simple parameter),
      - Optionally include a 'shared expert' projection that blends with each modality's main projection,
      - Combine (fuse) the resulting embeddings via sum or concat.

    Arguments:
      num_modalities (int): Number of modalities M.
      vision_hidden_dim (int): Dimensionality of each modality's feature embeddings.
      language_emb_dim (int): Dimensionality of the target LLM embedding space.
      use_router (bool): Whether gating depends on the input (MLP) or is a learned constant.
      use_shared_expert (bool): If True, add one extra 'shared' linear projection for each modality to mix with.
      router_hidden_dim (int): Hidden size of the router MLP, if use_router=True.
      num_proj (int): Number of projection layers. Must be 1 or M (the same as num_modalities).
                      If use_shared_expert=True, we actually create M+1 projections internally.
      fusion_mode (str): 'sum' or 'concat' to fuse the final embeddings.

    Shape/Usage:
      - We expect input shaped [B, M*N, vision_hidden_dim],
        where M is # modalities, N is # tokens per modality.
      - We'll reshape to [B, M, N, vision_hidden_dim].
      - Gating can be shape [B, M] (router) or [M] (learned param).
      - We'll apply per-modality or single projection. If shared expert is on, we do main_proj + shared_proj.
      - Output shape depends on fusion_mode:
          * 'sum' => [B, N, language_emb_dim]
          * 'concat' => [B, M*N, language_emb_dim]
    """

    def __init__(
            self,
            num_modalities: int,
            vision_hidden_dim: int,
            language_emb_dim: int,
            use_router: bool = False,
            use_shared_expert: bool = False,
            router_hidden_dim: int = 128,
            num_proj: int = 1,
            fusion_mode: str = "sum",
            sum_weights=False,
            use_lite_router=False,
            router_reg_coeff: float = 0.01,
            adapted_router=False,
            token_based_router=False,
            w_text_router=False,
            token_and_seq_based_router=False,
            token_and_seq_based_router_w_viz=False
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.vision_hidden_dim = vision_hidden_dim
        self.language_emb_dim = language_emb_dim

        self.use_router = use_router
        self.use_lite_router = use_lite_router
        if self.use_lite_router:
            assert self.use_router, "use_lite_router can only be used with use_router=True"
        self.use_shared_expert = use_shared_expert
        self.router_hidden_dim = router_hidden_dim
        self.fusion_mode = fusion_mode
        self.sum_weights = sum_weights
        self.router_reg_coeff = router_reg_coeff
        self.adapted_router = adapted_router
        self.token_based_router = token_based_router
        self.w_text_router = w_text_router
        self.token_and_seq_based_router = token_and_seq_based_router
        self.token_and_seq_based_router_w_viz = token_and_seq_based_router_w_viz
        assert not (self.token_based_router) or not (self.token_and_seq_based_router), "token_based_router and token_and_seq_based_router cannot be used together"
        if sum_weights:
            assert self.fusion_mode == "sum", "sum_weights can only be used with fusion_mode='sum'"

        # Validate num_proj must be 1 or num_modalities
        # We'll internally create (num_modalities + 1) if use_shared_expert=True
        # but from the user's perspective, they set num_proj=1 or =M
        if not (num_proj == 1 or num_proj == num_modalities):
            raise ValueError("num_proj must be 1 or equal to num_modalities.")
        if num_proj == 1 and self.use_shared_expert:
            raise ValueError("If num_proj=1, use_shared_expert must be False.")

        # number_of_experts is how many linear layers we actually build:
        # - If we want a shared expert, add +1
        # - If num_proj=1, we only do a single layer (plus possibly the shared)
        # - If num_proj=M, we do M layers (plus possibly the shared)
        if self.use_shared_expert:
            self.num_experts = 1 if num_proj == 1 else (num_modalities + 1)
            # Explanation:
            # If num_proj=1 but we have M modalities, we do M + 1 experts
            # If num_proj=M, we do M + 1 experts
        else:
            self.num_experts = num_proj if num_proj != 1 else 1

        # Build gating mechanism
        # We always produce M gating values (one per modality),
        # The "shared" portion is implicitly computed as (1 - gate_i).
        if self.use_router:
            if self.token_and_seq_based_router:
                self.router_mlp = nn.ModuleList([self._create_router(vision_hidden_dim=vision_hidden_dim,
                                                                     language_emb_dim=language_emb_dim),
                                                 self._create_router(vision_hidden_dim=vision_hidden_dim,
                                                                     language_emb_dim=language_emb_dim)])
                if self.token_and_seq_based_router_w_viz:
                    self.higher_router_input_dims = language_emb_dim + vision_hidden_dim * self.num_modalities
                    self.higher_router_mlp = self._create_higher_router(input_dims=self.higher_router_input_dims)
                else:
                    self.higher_router_input_dims = language_emb_dim
                    self.higher_router_mlp = self._create_higher_router(input_dims=self.higher_router_input_dims)
            else:
                # The router MLP: input = [B, M*vision_hidden_dim], output = [B, M]
                self.router_mlp = self._create_router(vision_hidden_dim=vision_hidden_dim,
                                                      language_emb_dim=language_emb_dim)
            self.gates_param = None  # not used
        else:
            # A simple learnable parameter of shape [M]
            # This will broadcast to [B, M] at runtime
            if self.sum_weights and self.use_shared_expert:
                self.gates_param = nn.Parameter(torch.ones(2*num_modalities))
            elif self.use_shared_expert:
                self.gates_param = nn.Parameter(torch.ones(num_modalities))
            else:
                self.gates_param = nn.Parameter(torch.ones(num_modalities), requires_grad=True)

            self.router_mlp = None

        # Build projection layers
        # If self.num_experts == 1, just a single nn.Linear
        # else we have a ModuleList for each expert
        if self.num_experts == 1:
            self.language_projection = nn.Linear(vision_hidden_dim, language_emb_dim)
        else:
            # e.g. M or M+1
            self.language_projection = nn.ModuleList([
                nn.Linear(vision_hidden_dim, language_emb_dim)
                for _ in range(self.num_experts)
            ])
        self.stats = defaultdict(list)
        print(
            f"[MoEBlock] #modalities={self.num_modalities}, #experts={self.num_experts}, "
            f"use_router={self.use_router}, lite_router={self.use_lite_router}, use_shared_expert={self.use_shared_expert}, "
            f"token based router={self.token_based_router}, text_router={self.w_text_router}, "
            f"higher_weights={self.token_and_seq_based_router} "
            f"with viz={self.token_and_seq_based_router_w_viz} fusion_mode={self.fusion_mode} with "
            f"sum_weights={self.sum_weights} and reg={self.router_reg_coeff} and adapted_router={self.adapted_router}"
        )

    def _create_router(self, vision_hidden_dim, language_emb_dim, num_modalities=4):
        """
        Creates a router MLP with the specified input and output dimensions.
        """
        if self.adapted_router:
            self.router_input_dims = language_emb_dim + vision_hidden_dim if self.w_text_router else vision_hidden_dim
            if self.sum_weights:
                self.router_output_dims = 2
            else:
                self.router_output_dims = 1
        else:
            self.router_input_dims = language_emb_dim + (
                        num_modalities * vision_hidden_dim) if self.w_text_router else (
                        num_modalities * vision_hidden_dim)
            if self.sum_weights:
                self.router_output_dims = 2 * num_modalities
            else:
                self.router_output_dims = num_modalities
        if self.use_lite_router:
            router_mlp = nn.Linear(self.router_input_dims, self.router_output_dims)
        else:
            router_mlp = nn.Sequential(
                nn.Linear(self.router_input_dims, self.router_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.router_hidden_dim, self.router_output_dims)
            )
        return router_mlp

    def _create_higher_router(self, input_dims, output_dims=1):
        """
        Creates a router MLP with the specified input and output dimensions.
        """
        return nn.Sequential(
                nn.Linear(input_dims, self.router_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.router_hidden_dim, output_dims)
            )

    def forward(self, image_features, prompt_features=None):
        """
        image_features: [B, M*N, vision_hidden_dim]
          Where M = #modalities, N = #tokens per modality.

        Returns fused embeddings:
          if fusion_mode='sum': shape [B, N, language_emb_dim]
          if fusion_mode='concat': shape [B, M*N, language_emb_dim]
        """
        B = image_features.size(0)
        # Reshape to [B, M, N, vision_hidden_dim]
        # So each of the M modalities has N tokens
        image_features = image_features.view(
            B, self.num_modalities, -1, self.vision_hidden_dim
        )
        # shape = [B, M, N, vision_hidden_dim]

        # Typically we do gating for each modality.
        # If use_router, we take the CLS from each modality
        # Use the 1st token from each modality as "CLS"
        #   flatten that => shape [B, M * vision_hidden_dim]
        #   pass through router => shape [B, M]
        # If not router, shape = [M] constant gating per modality.
        # --------------------------
        if self.use_router:
            if self.token_and_seq_based_router:
                gates, image_features = self._get_token_and_seq_level_router_weights(image_features=image_features,
                                                                                     prompt_features=prompt_features,
                                                                                     router_mlp=self.router_mlp,
                                                                                     B=B)
            elif self.token_based_router:
                gates = self._get_token_level_router_weights(image_features=image_features,
                                                             prompt_features=prompt_features,
                                                             router_mlp=self.router_mlp,
                                                             B=B)
            else:
                gates, image_features = self._get_seq_level_router_weights(image_features=image_features,
                                                                           prompt_features=prompt_features,
                                                                           router_mlp=self.router_mlp,
                                                                           B=B)
        else:
            # shape [M], broadcast to [B, M] later
            gates = self.gates_param
            gates = self._expand_gates_dims(gates, B)

        # each sequence has a shared and sequence-specific expert with softmax weights
        if self.sum_weights and self.use_shared_expert:
            gates = F.softmax(gates, dim=1)
            for i in range(2*self.num_modalities):
                self.stats[f"gate_{i}"].append(gates[:, i].mean().item())
            if self.training and self.sum_weights and self.router_reg_coeff > 0:
                lb_loss = self.load_balance_loss(gates, num_gates=2*self.num_modalities)
        # # each sequence has a shared and sequence-specific expert with sigmoid weights
        elif self.use_shared_expert:
            # elementwise sigmoid => shape [B, M] or [M]
            gates = torch.sigmoid(gates)
            for i in range(self.num_modalities):
                self.stats[f"gate_{i}"].append(gates[:, i].mean().item())
            if self.training and self.router_reg_coeff > 0:
                lb_loss = self.load_balance_loss(gates, num_gates=2)
        # each sequence has a sequence-specific expert with softmax weights
        elif self.sum_weights:
            # softmax => sum=1 across M
            gates = F.softmax(gates, dim=1)
            for i in range(self.num_modalities):
                self.stats[f"gate_{i}"].append(gates[:, i].mean().item())
            if self.training and self.router_reg_coeff > 0:
                lb_loss = self.load_balance_loss(gates, num_gates=self.num_modalities)
        # 1 sequence for all experts with even weights
        else:
            # softmax => sum=1 across M
            gates = F.softmax(gates, dim=1)

        # We'll iterate over M modalities
        # For each modality i:
        #   - pick the appropriate linear projection
        #   - scale by gates[:, i]
        #   - if shared_expert, also compute shared = (1 - gate_i)*shared_proj
        # Then we either sum or we keep them separate for concat
        N = image_features.shape[2]
        per_modality_list = []

        for i in range(self.num_modalities):
            # shape [B, N, vision_hidden_dim]
            emb_i = image_features[:, i, :, :]

            # pick a projection
            if self.num_experts == 1:
                # single layer for everything
                projected = self.language_projection(emb_i)
            else:
                # either M or M+1 layers
                # if M+1, the last index is the shared expert
                # for the main expert, use index i
                projected = self.language_projection[i](emb_i)

            self.stats[f"modality_specific_output_{i}"].append(projected.norm(dim=-1).mean().item())

            # gating factor for modality i => shape [B]
            gate_i = gates[:, i]  # [B]
            # broadcast multiply => [B, N, language_emb_dim]
            projected = projected * gate_i

            # if we have a shared expert, do that as well
            if self.use_shared_expert:
                # shape = [B, N, language_emb_dim]
                shared_proj_layer = self.language_projection[self.num_experts - 1]
                # "leftover" gate = 1 - gate_i
                if self.sum_weights:
                    shared_gate = gates[:, i + self.num_modalities]
                else:
                    shared_gate = (1.0 - gate_i)
                shared_proj = shared_proj_layer(emb_i)
                self.stats[f"shared_output_{i}"].append(shared_proj.norm(dim=-1).mean().item())
                shared_proj = shared_proj * shared_gate
                # combine them
                projected = projected + shared_proj

            per_modality_list.append(projected)

        if self.fusion_mode == "sum":
            # Sum across M => [B, N, D]
            # but we have a list of length M => each [B, N, D]
            stacked = torch.stack(per_modality_list, dim=1)  # [B, M, N, D]
            fused = stacked.sum(dim=1)  # => [B, N, D]
        elif self.fusion_mode == "concat":
            # Concat along token dimension => [B, M*N, D]
            # But each item is [B, N, D], so we stack on dim=1 => [B, M, N, D]
            # then reshape to [B, M*N, D]
            stacked = torch.stack(per_modality_list, dim=1)  # [B, M, N, D]
            fused = stacked.reshape(B, self.num_modalities * N, self.language_emb_dim)
        else:
            raise ValueError("fusion_mode must be 'sum' or 'concat'.")

        if self.training and self.router_reg_coeff > 0:
            router_reg_loss = self.router_reg_coeff * lb_loss
        else:
            router_reg_loss = torch.tensor(0.0, device=image_features.device)

        return fused, router_reg_loss


    def _get_token_level_router_weights(self, image_features, router_mlp, B, prompt_features=None):
        # image_features: [B, M, N, D]
        B, M, N, D = image_features.shape

        # → [B, N, M*D]  (no contiguous copy; reshape handles it)
        flat_img = image_features.permute(0, 2, 1, 3).reshape(B, N, M * D)

        if self.w_text_router:
            # prompt_broadcast: [B, N, D_p]  (view-only, no data copy)
            prompt_broadcast = prompt_features[:, None, :].expand(-1, N, -1)
            flat_img = torch.cat((flat_img, prompt_broadcast), dim=-1)

        assert flat_img.shape[-1] == self.router_input_dims, \
            f"Expected {self.router_input_dims}, got {flat_img.shape[-1]}"

        gates = router_mlp(flat_img).transpose(1, 2)  # [B, M, N]
        return self._expand_gates_dims(gates, B)

    def _get_seq_level_router_weights(self, image_features, router_mlp, B, prompt_features=None):
        cls_tokens = image_features[:, :, 0, :]  # [B, M, D]
        cls_flat = cls_tokens.reshape(B, self.num_modalities * self.vision_hidden_dim)

        if self.w_text_router:
            cls_flat = torch.cat((cls_flat, prompt_features), dim=-1)

        assert cls_flat.shape[-1] == self.router_input_dims, \
            f"Expected {self.router_input_dims}, got {cls_flat.shape[-1]}"

        gates = router_mlp(cls_flat)  # [B, M]
        image_features = image_features[:, :, 1:, :]  # drop CLS
        gates = self._expand_gates_dims(gates, B)  # → [B, M, 1, 1]
        return gates, image_features

    def _get_token_and_seq_level_router_weights(self, image_features, router_mlp, B, prompt_features=None):
        cls_tokens = image_features[:, :, 0, :]
        cls_flat = cls_tokens.reshape(B, self.num_modalities * self.vision_hidden_dim)

        higher_inp = (torch.cat((cls_flat, prompt_features), dim=-1)
                      if self.token_and_seq_based_router_w_viz
                      else prompt_features)

        assert higher_inp.shape[-1] == self.higher_router_input_dims, \
            f"Expected {self.higher_router_input_dims}, got {higher_inp.shape[-1]}"

        # mixture weight α  ∈ (0,1)  → shape [B, 1, 1, 1] for broadcasting
        alpha = torch.sigmoid(self.higher_router_mlp(higher_inp)).view(B, 1, 1, 1)

        # get sequence- & token-level gates
        seq_gates, img_wo_cls = self._get_seq_level_router_weights(
            image_features=image_features, router_mlp=router_mlp[1],
            B=B, prompt_features=prompt_features)

        tok_gates = self._get_token_level_router_weights(
            image_features=img_wo_cls, router_mlp=router_mlp[0],
            B=B, prompt_features=prompt_features)

        # convex combination
        gates = alpha * seq_gates + (1.0 - alpha) * tok_gates  # [B, M, N, 1]
        return gates, img_wo_cls

    def _expand_gates_dims(self, gates, B):
        if gates.dim() == 1:          # [M] → [1, M]
            gates = gates.unsqueeze(0)
        while gates.dim() < 4:
            gates = gates.unsqueeze(-1)
        if gates.size(0) == 1:
            gates = gates.expand(B, *gates.shape[1:])
        assert gates.dim() == 4, f"Unexpected gates shape: {gates.shape}. Expected [B, M, N, 1]."
        return gates


    def load_balance_loss(self, gates: torch.Tensor, num_gates: int) -> torch.Tensor:
        """
        gates: [B, num_gates] gating distribution across the batch
        num_gates: number of gates (M or 2*M if sum_weights=True)

        Returns a scalar that is small if usage is near-uniform
        and large if one gate dominates.
        """
        # average usage per gate across the batch => shape [num_gates]
        mean_usage = gates.mean(dim=0)
        # we want each gate to be ~ 1/num_gates
        target = 1.0 / num_gates
        # sum of squared differences
        loss = ((mean_usage - target) ** 2).sum()
        return loss

    def save_stats(self, filename, reset_after_save=True):
        """
        Writes out all collected statistics to file in JSON format.
        Optionally resets the stats so each epoch is fresh.
        """
        # 1. Convert defaultdict(list) into a normal dict for JSON
        stats_dict = dict(self.stats)

        # 2. Write to JSON
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(stats_dict, f, indent=2)

        # 3. (Optional) Reset stats
        if reset_after_save:
            self.stats = defaultdict(list)