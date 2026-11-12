"""
DINOv3 Aligner Module

This module provides alignment layers to match DINOv3 features with Qwen3VL features.
The aligner uses a two-stage linear projection strategy with layer normalization:
1. Linear1: Align head_dim (dinov3_head_dim -> qwen_head_dim)
2. Linear2: Align num_heads (dinov3_num_heads -> qwen_num_heads)
3. LayerNorm: Stabilize output distribution
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class VisualTokenAligner(nn.Module):
    """
    Aligns ViT block-output hidden states (tokens) to Qwen LLM hidden dimension.

    For token_concat ablation: takes ViT layer outputs and projects them to the LLM
    hidden size so they can be prepended to LLM hidden states at specified layers.

    Works for both SigLIP (Qwen ViT, hidden_size=1024) and DINOv3 (hidden_size=1280).

    Input:  {vit_layer: Tensor[B, N, encoder_hidden_dim]}
    Output: {qwen_layer: Tensor[B, N, qwen_hidden_dim]}

    One Linear + LayerNorm per Qwen target layer.
    """

    def __init__(
        self,
        encoder_hidden_dim: int,       # ViT hidden_size (e.g. 1024 for SigLIP, 1280 for DINOv3)
        qwen_hidden_dim: int,           # Qwen LLM hidden_size  (e.g. 2560)
        layer_mapping: Dict[int, int],  # {qwen_layer: vit_layer}
    ):
        super().__init__()
        self.encoder_hidden_dim = encoder_hidden_dim
        self.qwen_hidden_dim = qwen_hidden_dim
        self.layer_mapping = layer_mapping

        self.proj = nn.ModuleDict()
        self.norm = nn.ModuleDict()

        for qwen_layer in layer_mapping.keys():
            key = str(qwen_layer)
            self.proj[key] = nn.Linear(encoder_hidden_dim, qwen_hidden_dim, bias=False)
            self.norm[key] = nn.LayerNorm(qwen_hidden_dim)

    def forward(
        self,
        token_states: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Args:
            token_states: {vit_layer: Tensor[B, N, encoder_hidden_dim]}
        Returns:
            {qwen_layer: Tensor[B, N, qwen_hidden_dim]}
        """
        # Build vit_layer → [qwen_layers] mapping.
        # Must support many-to-one (e.g. LAST1X16 where a single dino layer
        # is repeated 16 times to feed 16 different qwen layers).
        # The naive `{v: k for k, v in ...}` dict comprehension silently drops
        # all but the last qwen_layer when multiple qwen layers share the same
        # vit_layer, so we use a list-valued defaultdict instead.
        from collections import defaultdict
        vit_to_qwen: Dict[int, list] = defaultdict(list)
        for qwen_layer, vit_layer in self.layer_mapping.items():
            vit_to_qwen[vit_layer].append(qwen_layer)

        output: Dict[int, torch.Tensor] = {}

        for vit_layer, tokens in token_states.items():
            if vit_layer not in vit_to_qwen:
                continue
            # tokens: [B, N, encoder_hidden_dim]
            for qwen_layer in vit_to_qwen[vit_layer]:
                key = str(qwen_layer)
                # output: [B, N, qwen_hidden_dim]
                output[qwen_layer] = self.norm[key](self.proj[key](tokens))

        return output


# Backward-compatible alias
SigLIPTokenAligner = VisualTokenAligner


class DINOv3QKVAligner(nn.Module):
    """
    Aligner for DINOv3 QKV states to match Qwen3VL dimensions.

    Supports Grouped Query Attention (GQA) where K/V have fewer heads than Q.

    Uses a two-stage alignment:
    1. Align head dimension: dinov3_head_dim -> qwen_head_dim
    2. Align number of heads: dinov3_num_heads -> qwen_q_num_heads (for Q) or qwen_kv_num_heads (for K/V)
    3. Apply LayerNorm for stability

    Two weight-sharing modes (controlled by ``shared``):
    - ``shared=False`` (default): each Qwen layer gets its own independent set of Q/K/V
      linear layers.  Keys are ``layer_{qwen_layer}_{q|k|v}``.
    - ``shared=True``: a single shared set of Q/K/V linear layers is used for **all**
      Qwen target layers.  Keys are ``shared_{q|k|v}``.  This halves the number of
      aligner parameters (relative to per-layer) when many layers are mapped.
    """

    def __init__(
        self,
        dinov3_num_heads: int,
        dinov3_head_dim: int,
        qwen_q_num_heads: int,
        qwen_kv_num_heads: int,
        qwen_head_dim: int,
        layer_mapping: Dict[int, int],  # {qwen_layer: dinov3_layer}
        shared: bool = False,
        mlp: bool = False,
        expert_inject_gate: bool = False,
    ):
        """
        Args:
            dinov3_num_heads: Number of attention heads in DINOv3
            dinov3_head_dim: Head dimension in DINOv3
            qwen_q_num_heads: Number of query heads in Qwen (e.g., 32)
            qwen_kv_num_heads: Number of key/value heads in Qwen (e.g., 8)
            qwen_head_dim: Head dimension in Qwen
            layer_mapping: Mapping from Qwen layer indices to DINOv3 layer indices
            shared: If True, use a single shared Q/K/V aligner for all layers instead
                    of independent per-layer aligners.
            mlp: If True, replace each single Linear stage with a 2-layer MLP
                    (Linear(in, hidden, bias=False) -> GELU -> Linear(hidden, out, bias=False),
                    hidden = 2*max(in_dim, out_dim)). Default False keeps the released
                    single-Linear behavior bit-identical.
            expert_inject_gate: If True, add one learnable scalar gate per Qwen
                    target layer (init 1.0) that multiplies the aligned expert
                    **value** output. Because the attention softmax weights depend
                    only on Q/K, scaling V scales that layer's expert contribution
                    to the VL output rows exactly linearly; gate=1.0 is bit-identical
                    to the released behavior, gate=0.0 removes the expert value
                    contribution (expert tokens still occupy softmax mass via K).
                    Default False creates no parameter and changes nothing.
        """
        super().__init__()
        self.dinov3_num_heads = dinov3_num_heads
        self.dinov3_head_dim = dinov3_head_dim
        self.qwen_q_num_heads = qwen_q_num_heads
        self.qwen_kv_num_heads = qwen_kv_num_heads
        self.qwen_head_dim = qwen_head_dim
        self.layer_mapping = layer_mapping
        self.shared = shared
        self.mlp = mlp
        self.expert_inject_gate_enabled = bool(expert_inject_gate)
        # Gate index follows the sorted Qwen target layers so the parameter
        # ordering is deterministic regardless of mapping insertion order.
        self.expert_inject_gate_layers = sorted(layer_mapping)
        self._expert_inject_gate_index = {
            qwen_layer: index
            for index, qwen_layer in enumerate(self.expert_inject_gate_layers)
        }
        if self.expert_inject_gate_enabled:
            self.expert_inject_gate = nn.Parameter(
                torch.ones(len(self.expert_inject_gate_layers))
            )

        # Create linear layers and layer norms for each (layer, qkv_type) combination.
        # 📌 重要：Q 和 K/V 使用不同数量的头（GQA）
        #
        # shared=False: keys are  "layer_{qwen_layer}_{q|k|v}"  (per-layer weights)
        # shared=True:  keys are  "shared_{q|k|v}"              (one shared set for all layers)
        self.linear1_dict = nn.ModuleDict()  # Align head_dim
        self.linear2_dict = nn.ModuleDict()  # Align num_heads
        self.layernorm_dict = nn.ModuleDict()  # Normalize output

        def _make_stage(in_dim: int, out_dim: int) -> nn.Module:
            if not mlp:
                return nn.Linear(in_dim, out_dim, bias=False)
            hidden_dim = 2 * max(in_dim, out_dim)
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim, bias=False),
            )

        if shared:
            # Create a single shared set of Q/K/V aligners
            for qkv_type in ['q', 'k', 'v']:
                key = f'shared_{qkv_type}'

                self.linear1_dict[key] = _make_stage(dinov3_head_dim, qwen_head_dim)

                target_num_heads = qwen_q_num_heads if qkv_type == 'q' else qwen_kv_num_heads
                self.linear2_dict[key] = _make_stage(dinov3_num_heads, target_num_heads)

                self.layernorm_dict[key] = nn.LayerNorm(qwen_head_dim)
        else:
            for qwen_layer in layer_mapping.keys():
                for qkv_type in ['q', 'k', 'v']:  # query, key, value
                    key = f'layer_{qwen_layer}_{qkv_type}'

                    # Linear1: [dinov3_head_dim] -> [qwen_head_dim]
                    self.linear1_dict[key] = _make_stage(dinov3_head_dim, qwen_head_dim)

                    # Linear2: [dinov3_num_heads] -> [qwen_q_num_heads or qwen_kv_num_heads]
                    # Query 使用 qwen_q_num_heads，Key/Value 使用 qwen_kv_num_heads
                    target_num_heads = qwen_q_num_heads if qkv_type == 'q' else qwen_kv_num_heads
                    self.linear2_dict[key] = _make_stage(dinov3_num_heads, target_num_heads)

                    # LayerNorm: normalize over qwen_head_dim
                    self.layernorm_dict[key] = nn.LayerNorm(qwen_head_dim)

    def align_qkv(
        self,
        qkv_tensor: torch.Tensor,
        qwen_layer: int,
        qkv_type: str  # 'query', 'key', or 'value'
    ) -> torch.Tensor:
        """
        Align a single QKV tensor from DINOv3 to Qwen dimensions.

        Supports Grouped Query Attention (GQA):
        - Query: aligned to qwen_q_num_heads heads
        - Key/Value: aligned to qwen_kv_num_heads heads

        Args:
            qkv_tensor: Shape [batch, dinov3_num_heads, seq_len, dinov3_head_dim]
            qwen_layer: Target Qwen layer index
            qkv_type: One of 'query', 'key', 'value'

        Returns:
            Aligned tensor of shape:
            - Query: [batch, qwen_q_num_heads, seq_len, qwen_head_dim]
            - Key/Value: [batch, qwen_kv_num_heads, seq_len, qwen_head_dim]
        """
        # Get the corresponding modules.
        # shared=True  → one shared key per Q/K/V type, reused for every layer.
        # shared=False → per-layer key, independent weights for every Qwen layer.
        qkv_initial = qkv_type[0]  # 'q', 'k', or 'v'
        key = f'shared_{qkv_initial}' if self.shared else f'layer_{qwen_layer}_{qkv_initial}'

        if key not in self.linear1_dict:
            # No aligner for this layer, return original
            return qkv_tensor

        linear1 = self.linear1_dict[key]
        linear2 = self.linear2_dict[key]
        layernorm = self.layernorm_dict[key]

        # Input shape: [batch, dinov3_num_heads, seq_len, dinov3_head_dim]
        batch_size, dinov3_num_heads, seq_len, dinov3_head_dim = qkv_tensor.shape

        # Step 1: Align head_dim with Linear1
        # [batch, dinov3_num_heads, seq_len, dinov3_head_dim] -> [batch, dinov3_num_heads, seq_len, qwen_head_dim]
        qkv_step1 = linear1(qkv_tensor)

        # Step 2: Permute to apply Linear2 on num_heads dimension
        # [batch, dinov3_num_heads, seq_len, qwen_head_dim] -> [batch, seq_len, qwen_head_dim, dinov3_num_heads]
        qkv_step2 = qkv_step1.permute(0, 2, 3, 1)

        # Step 3: Align num_heads with Linear2
        # [batch, seq_len, qwen_head_dim, dinov3_num_heads] -> [batch, seq_len, qwen_head_dim, qwen_num_heads]
        qkv_step3 = linear2(qkv_step2)

        # Step 4: Permute back to correct format
        # [batch, seq_len, qwen_head_dim, qwen_num_heads] -> [batch, seq_len, qwen_num_heads, qwen_head_dim]
        qkv_step4 = qkv_step3.permute(0, 1, 3, 2)

        # Step 5: Apply LayerNorm on the last dimension
        qkv_step5 = layernorm(qkv_step4)

        # Step 6: Final permute to [batch, qwen_num_heads, seq_len, qwen_head_dim]
        qkv_final = qkv_step5.permute(0, 2, 1, 3)

        # Optional per-layer expert injection gate: scales the aligned expert
        # value so that this layer's expert contribution to the VL output rows
        # is weighted by a learnable scalar (init 1.0 = released behavior).
        if self.expert_inject_gate_enabled and qkv_type == 'value':
            gate_index = self._expert_inject_gate_index.get(qwen_layer)
            if gate_index is not None:
                qkv_final = qkv_final * self.expert_inject_gate[gate_index]

        return qkv_final

    def forward(
        self,
        dinov3_qkv_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Align all QKV states from DINOv3 to Qwen dimensions.

        Args:
            dinov3_qkv_states: Dict mapping DINOv3 layer indices to QKV dicts
                               Each QKV dict has keys 'query', 'key', 'value'
                               with tensors of shape [batch, dinov3_num_heads, seq_len, dinov3_head_dim]

        Returns:
            Aligned QKV states with Qwen layer indices as keys
            Output shapes (GQA):
            - query: [batch, qwen_q_num_heads, seq_len, qwen_head_dim]
            - key: [batch, qwen_kv_num_heads, seq_len, qwen_head_dim]
            - value: [batch, qwen_kv_num_heads, seq_len, qwen_head_dim]
        """
        aligned_qkv_states = {}

        for qwen_layer, dinov3_layer in self.layer_mapping.items():
            if dinov3_layer not in dinov3_qkv_states:
                continue

            qkv_dict = dinov3_qkv_states[dinov3_layer]
            aligned_qkv = {}

            for qkv_type in ['query', 'key', 'value']:
                if qkv_type in qkv_dict:
                    aligned_qkv[qkv_type] = self.align_qkv(
                        qkv_dict[qkv_type],
                        qwen_layer,
                        qkv_type
                    )

            aligned_qkv_states[qwen_layer] = aligned_qkv

        return aligned_qkv_states
