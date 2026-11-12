"""
dinov3_concat_projector.py

FusedMLP projector for concatenating Qwen3-VL ViT features and DINOv3 patch
features before feeding into the LLM, following the OpenVLA DinoSigLIP concat style.

Pipeline (per image):
  Qwen ViT (post-merger): [N_tokens, qwen_out_dim]
  DINOv3 (last layer, patches only): [N_dino_patch, dino_dim]
    -> bilinear interpolate to match Qwen merged grid -> [N_tokens, dino_dim]
  Concat: [N_tokens, qwen_out_dim + dino_dim]
  FusedMLP: [N_tokens, qwen_out_dim + dino_dim] -> [N_tokens, llm_dim]
"""

import torch
import torch.nn as nn


class DINOv3ConcatProjector(nn.Module):
    """
    3-layer FusedMLP projector (OpenVLA style) that maps concatenated
    [Qwen-ViT || DINOv3-patch] features to LLM hidden dimension.

    Args:
        qwen_out_dim: Output dim of Qwen3-VL's PatchMerger (typically 2560).
        dino_dim: DINOv3 hidden size (e.g. 1280 for ViT-H).
        llm_dim: LLM hidden size to project to (typically 2560).
        pre_norm: When True, apply an entry LayerNorm over the fused
            (concatenated) feature before the MLP. This removes the ~25x
            per-token scale mismatch between the DINOv3 half and the Qwen
            half (OpenVLA/Prismatic-style fused-feature LN; same precedent
            as the DINOv3QKVAligner trailing LN). Default False keeps the
            released behavior (and released state-dict keys) bit-identical.
    """

    def __init__(
        self,
        qwen_out_dim: int,
        dino_dim: int,
        llm_dim: int,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        fused_dim = qwen_out_dim + dino_dim
        self.fused_dim = fused_dim
        self.llm_dim = llm_dim

        self.pre_norm = nn.LayerNorm(fused_dim) if pre_norm else None

        # 3-layer MLP: fused_dim -> fused_dim -> llm_dim -> llm_dim
        self.projector = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_features: [N_tokens, qwen_out_dim + dino_dim]
        Returns:
            projected: [N_tokens, llm_dim]
        """
        if self.pre_norm is not None:
            fused_features = self.pre_norm(fused_features)
        return self.projector(fused_features)
