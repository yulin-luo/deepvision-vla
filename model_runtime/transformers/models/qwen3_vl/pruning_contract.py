"""Shared contracts for multi-layer attention voting and patch selection."""

from __future__ import annotations

import math
from typing import Sequence

import torch


PRUNING_VOTE_CONTRACTS = frozenset(
    {
        "historical_floor_global_fallback",
        "strict_ceil_fail_closed",
    }
)


def resolve_pruning_vote_threshold(
    vote_ratio: float,
    num_layers: int,
    contract: str,
) -> int:
    vote_ratio = float(vote_ratio)
    num_layers = int(num_layers)
    if contract not in PRUNING_VOTE_CONTRACTS:
        raise ValueError(
            f"pruning_vote_contract must be one of {sorted(PRUNING_VOTE_CONTRACTS)}; "
            f"got {contract!r}."
        )
    if not 0.0 < vote_ratio <= 1.0:
        raise ValueError(f"pruning vote ratio must be in (0, 1]; got {vote_ratio}.")
    if num_layers <= 0:
        raise ValueError(f"pruning vote requires at least one layer; got {num_layers}.")

    raw_threshold = vote_ratio * num_layers
    if contract == "historical_floor_global_fallback":
        return max(1, int(raw_threshold))
    return max(1, math.ceil(raw_threshold))


def aggregate_sparse_topk_attention(
    attention_maps: Sequence[torch.Tensor],
    *,
    topk: int,
    mode: str,
    vote_ratio: float,
    contract: str,
) -> torch.Tensor:
    """Aggregate per-layer top-k membership into sparse token scores."""
    if mode not in {"vote", "intersection"}:
        raise ValueError(f"Sparse top-k aggregation mode must be vote or intersection; got {mode!r}.")
    if not attention_maps:
        raise ValueError("Sparse top-k aggregation requires at least one attention map.")

    first = attention_maps[0]
    if first.ndim != 2:
        raise ValueError(f"Attention maps must have shape [batch, tokens]; got {tuple(first.shape)}.")
    batch_size, num_tokens = first.shape
    topk = max(1, min(int(topk), int(num_tokens)))
    counts = torch.zeros(
        batch_size,
        num_tokens,
        device=first.device,
        dtype=torch.float32,
    )
    for layer_index, layer_attention in enumerate(attention_maps):
        if tuple(layer_attention.shape) != (batch_size, num_tokens):
            raise ValueError(
                "All pruning attention maps must share [batch, tokens] shape; "
                f"layer 0 has {(batch_size, num_tokens)} and layer {layer_index} has "
                f"{tuple(layer_attention.shape)}."
            )
        top_indices = torch.topk(layer_attention, k=topk, dim=1).indices
        counts.scatter_add_(1, top_indices, torch.ones_like(top_indices, dtype=torch.float32))

    if mode == "intersection":
        threshold = len(attention_maps)
    else:
        threshold = resolve_pruning_vote_threshold(vote_ratio, len(attention_maps), contract)
    sparse_scores = counts * (counts >= threshold).to(counts.dtype)

    if contract == "historical_floor_global_fallback":
        # Preserve the exact historical batch-global fallback. If another sample
        # has a non-empty vote, an empty sample remains all-zero and top-k ties
        # are resolved by torch.topk as before.
        if sparse_scores.sum().item() == 0:
            return counts
        return sparse_scores

    empty_samples = torch.nonzero(sparse_scores.sum(dim=1) == 0, as_tuple=False).flatten()
    if empty_samples.numel() > 0:
        raise ValueError(
            f"Strict {mode} aggregation selected no Qwen visual tokens for batch indices "
            f"{empty_samples.tolist()}; lower the vote ratio or use the explicit historical fallback contract."
        )
    return sparse_scores


def select_pruning_patch_indices(
    attention_map: torch.Tensor,
    *,
    num_tokens_to_keep: int,
    order: str,
    eligibility_mask: torch.Tensor | None = None,
    fail_on_insufficient_candidates: bool = False,
) -> torch.Tensor:
    """Select final patch indices from the exact score map used by the model."""
    if attention_map.ndim != 2:
        raise ValueError(
            f"Pruning attention map must have shape [batch, patch_tokens]; got {tuple(attention_map.shape)}."
        )
    if not torch.isfinite(attention_map).all():
        raise ValueError("Pruning attention map contains non-finite values.")
    num_tokens = int(attention_map.shape[1])
    num_tokens_to_keep = max(1, min(int(num_tokens_to_keep), num_tokens))
    selection_scores = attention_map
    if eligibility_mask is not None:
        if tuple(eligibility_mask.shape) != tuple(attention_map.shape):
            raise ValueError(
                "Pruning eligibility mask must match the attention map; "
                f"got mask {tuple(eligibility_mask.shape)} and map {tuple(attention_map.shape)}."
            )
        eligibility_mask = eligibility_mask.to(device=attention_map.device, dtype=torch.bool)
        candidate_counts = eligibility_mask.sum(dim=1)
        if fail_on_insufficient_candidates:
            insufficient = torch.nonzero(
                candidate_counts < num_tokens_to_keep,
                as_tuple=False,
            ).flatten()
            if insufficient.numel() > 0:
                details = {
                    int(index): int(candidate_counts[index].item())
                    for index in insufficient.tolist()
                }
                raise ValueError(
                    "Strict pruning aggregation has fewer eligible expert patches than the requested "
                    f"top-k={num_tokens_to_keep}: {details}."
                )
        selection_scores = attention_map.masked_fill(
            ~eligibility_mask,
            torch.finfo(attention_map.dtype).min,
        )
    indices = torch.topk(
        selection_scores,
        k=num_tokens_to_keep,
        dim=1,
        sorted=True,
    ).indices
    if order == "importance":
        return indices
    if order == "spatial":
        return torch.sort(indices, dim=1).values
    raise ValueError(f"expert_token_order must be importance or spatial; got {order!r}.")
