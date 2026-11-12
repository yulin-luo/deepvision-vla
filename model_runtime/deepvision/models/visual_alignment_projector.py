"""
visual_alignment_projector.py

Lightweight projector for Visual Representation Alignment, following BlindVLA
(arXiv 2510.25616):

  "Don't Blind Your VLA: Aligning Visual Representations for OOD Generalization"

During VLA training, the model's mid-layer LLM visual token representations are
projected toward a frozen teacher (DINOv3) patch features via cosine similarity,
anchoring perception while action learning proceeds.

Architecture (per layer):
  LayerNorm → Linear(llm_dim → proj_dim) → SiLU → Dropout
  → Linear(proj_dim → proj_dim) → SiLU → Dropout
  → Linear(proj_dim → teacher_dim)
"""

import torch
import torch.nn as nn


class VisualAlignmentProjector(nn.Module):
    """
    Projects VLA LLM hidden states (at visual token positions) to the
    frozen teacher's feature space for cosine similarity alignment.

    A single shared projector is applied to each specified align layer.

    Args:
        llm_dim: VLA LLM hidden dimension (e.g. 2560 for Qwen3-VL-4B).
        proj_dim: Inner projection dimension (default 1024).
        teacher_dim: Frozen teacher (DINOv3) hidden dimension (e.g. 1280 for ViT-H).
        dropout: Dropout rate (default 0.1, following BlindVLA).
    """

    def __init__(
        self,
        llm_dim: int,
        proj_dim: int,
        teacher_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, proj_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, teacher_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N_vis_tokens, llm_dim]
        Returns:
            projected: [B, N_vis_tokens, teacher_dim]
        """
        return self.net(x)
