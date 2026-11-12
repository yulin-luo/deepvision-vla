import atexit
import os, sys, pathlib
import argparse
import tqdm
import shutil
import hashlib
import platform
import importlib.metadata
import re
import torch
import torch.nn as nn

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_VENDORED_LIFT3D = _REPO_ROOT / "third_party" / "lift3d"
sys.path.insert(0, str(_REPO_ROOT))
if _VENDORED_LIFT3D.is_dir():
    sys.path.insert(0, str(_VENDORED_LIFT3D))

from transformers import Qwen3VLForConditionalGeneration, AutoConfig, AutoProcessor, AutoModel, AutoImageProcessor
from transformers.models.qwen3_vl.pruning_contract import PRUNING_VOTE_CONTRACTS

from deepvision.models import (
    ACTION_QUERY_CONTRACTS,
    CHUNK_EXECUTION_CONTRACTS,
    ActionTokenizer,
    DINOv3ConcatProjector,
    DINOv3QKVAligner,
    PerStepSharedActionHead,
    VisualTokenAligner,
    action_head_contract,
    action_query_count,
    build_action_head,
    select_action_for_execution,
)
from utils.rlbench_protocol import (
    checkpoint_provenance,
    profile_metadata,
    resolve_dataset_image_path,
    resolve_profile,
    resolve_siglip_token_contract,
    resolve_task_runtime,
)
from utils.rlbench_image_preprocessing import (
    EVAL_IMAGE_PROTOCOLS,
    IMAGE_AUG_SCOPES,
    prepare_eval_image_views,
    resolve_eval_dino_source,
    resolve_eval_image_protocol,
)
from utils.rlbench_action_representation import (
    decode_action_for_execution,
    validate_saved_action_representation,
)
from utils.expert_attention_contract import (
    EXPERT_ROPE_CONTRACTS,
    QWEN_POSITION_CONTRACTS,
    resolve_expert_rope_runtime,
    sync_expert_qwen_rope_runtime,
)
import numpy as np
import os
import PIL.Image
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torchvision
import json
import argparse
import copy
import random
from collections import Counter
from typing import List, Dict

from termcolor import cprint, colored

try:
    from lift3d.helpers.common import Logger
except ModuleNotFoundError:
    class Logger:
        """Dataset-only fallback; simulator rollout still imports LIFT3D in main()."""

        @staticmethod
        def log_info(message):
            print(f"[INFO] {message}")

        @staticmethod
        def log_warning(message):
            print(f"[WARN] {message}")

        @staticmethod
        def log_error(message):
            print(f"[ERROR] {message}")

        @staticmethod
        def print_seperator():
            print("=" * 80)
import logging
import time
import threading
from datetime import datetime

import numpy as np
import pickle

import torch
from dataclasses import dataclass
from PIL import Image

from scipy.spatial.transform import Rotation as R


def _maybe_ablate_dino(pixel_values):
    """诊断开关（默认关，不影响任何既有行为）。

    EVAL_DINO_ABLATE=zero|flip 时破坏 DINO 塔的输入，用来检验 DINO 是否真的影响输出。
    接线通 != 有影响：若破坏输入后逐 episode 结果一位不差，说明 DINO 分支是装饰。
    flip 用左右翻转（确定性），保持输入在分布内，比 zero 更公平。
    """
    mode = os.environ.get("EVAL_DINO_ABLATE", "")
    if not mode:
        return pixel_values
    if mode == "zero":
        return torch.zeros_like(pixel_values)
    if mode == "flip":
        return torch.flip(pixel_values, dims=[-1])
    raise ValueError(f"EVAL_DINO_ABLATE 只接受 zero|flip，收到 {mode!r}")


# Debugpy support for remote debugging
try:
    import debugpy
    DEBUGPY_AVAILABLE = True
except ImportError:
    DEBUGPY_AVAILABLE = False
    debugpy = None


def _allow_random_init(args) -> bool:
    return bool(int(getattr(args, "allow_random_init", 0)))


def _missing_component(args, message: str):
    if _allow_random_init(args):
        Logger.log_warning(f"{message}; continuing because --allow-random-init=1")
        return
    raise FileNotFoundError(message)


def resolve_eval_action_contract(training_config, *, action_chunk, action_dim):
    """Resolve one fail-closed action-query contract from checkpoint metadata."""
    query_contract = training_config.get(
        "action_query_contract",
        "scalar_per_dim_v1",
    )
    if query_contract not in ACTION_QUERY_CONTRACTS:
        raise ValueError(
            "training_config action_query_contract must be one of "
            f"{sorted(ACTION_QUERY_CONTRACTS)}, got {query_contract!r}"
        )
    expected_query_count = action_query_count(
        action_chunk,
        action_dim,
        query_contract,
    )
    configured_query_count = int(
        training_config.get("action_query_count", expected_query_count)
    )
    if configured_query_count != expected_query_count:
        raise ValueError(
            "training_config action_query_count does not match the action query contract: "
            f"got {configured_query_count}, expected {expected_query_count}"
        )
    expected_head_contract = action_head_contract(query_contract)
    configured_head_contract = training_config.get(
        "action_head_contract",
        expected_head_contract,
    )
    if configured_head_contract != expected_head_contract:
        raise ValueError(
            "training_config action_head_contract does not match action_query_contract: "
            f"got {configured_head_contract!r}, expected {expected_head_contract!r}"
        )
    execution_contract = training_config.get("chunk_execution_contract", "reject")
    if execution_contract not in CHUNK_EXECUTION_CONTRACTS:
        raise ValueError(
            "training_config chunk_execution_contract must be one of "
            f"{sorted(CHUNK_EXECUTION_CONTRACTS)}, got {execution_contract!r}"
        )
    # Missing metadata keeps historical loading compatible. New per-dim runs may
    # explicitly use safe pad/eos IDs so GT values never reach mRoPE parsing.
    legacy_placeholder_contract = (
        "ground_truth_scalar_token_ids_then_zero_embedding_v1"
        if query_contract in ("scalar_per_dim_v1", "decoupled_per_dim_v1")
        else "active_pad_or_eos_token_ids_then_zero_embedding_v1"
    )
    configured_placeholder_contract = training_config.get(
        "action_query_placeholder_contract",
        legacy_placeholder_contract,
    )
    allowed_placeholder_contracts = {
        legacy_placeholder_contract, "active_pad_or_eos_token_ids_then_zero_embedding_v1",
    }
    if configured_placeholder_contract not in allowed_placeholder_contracts:
        raise ValueError(
            "training_config action_query_placeholder_contract mismatch: "
            f"got {configured_placeholder_contract!r}, "
            f"expected one of {sorted(allowed_placeholder_contracts)!r}"
        )
    return {
        "action_query_contract": query_contract,
        "action_query_count": expected_query_count,
        "action_query_placeholder_contract": configured_placeholder_contract,
        "action_head_contract": expected_head_contract,
        "chunk_execution_contract": execution_contract,
    }


def build_eval_action_head(
    *, hidden_dim, action_chunk, action_dim, action_query_contract
):
    return build_action_head(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        action_chunk=action_chunk,
        action_query_contract=action_query_contract,
    )


def append_action_query_placeholders(
    *,
    base_input_ids,
    base_attention_mask,
    embedding_layer,
    action_query_count,
    placeholder_token_id,
):
    action_query_count = int(action_query_count)
    if action_query_count <= 0:
        raise ValueError(
            f"action_query_count must be positive, got {action_query_count}"
        )
    if base_input_ids.ndim != 2 or base_attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must both have shape [B,L]")
    if base_input_ids.shape != base_attention_mask.shape:
        raise ValueError(
            "input_ids and attention_mask shape mismatch: "
            f"{tuple(base_input_ids.shape)} != {tuple(base_attention_mask.shape)}"
        )
    batch_size = base_input_ids.shape[0]
    query_ids = torch.full(
        (batch_size, action_query_count),
        int(placeholder_token_id),
        dtype=base_input_ids.dtype,
        device=base_input_ids.device,
    )
    full_input_ids = torch.cat((base_input_ids, query_ids), dim=-1)
    inputs_embeds = embedding_layer(full_input_ids)
    inputs_embeds[:, -action_query_count:, :] = 0
    query_attention = torch.ones(
        (batch_size, action_query_count),
        dtype=base_attention_mask.dtype,
        device=base_attention_mask.device,
    )
    full_attention_mask = torch.cat(
        (base_attention_mask, query_attention),
        dim=-1,
    )
    return full_input_ids, inputs_embeds, full_attention_mask


def denormalize_action_chunks(
    normalized_actions,
    statistic,
    *,
    action_chunk,
    action_dim,
):
    """Decode one normalized [H,D] prediction with checkpoint statistics."""
    action_chunk = int(action_chunk)
    action_dim = int(action_dim)
    normalized_actions = np.asarray(normalized_actions).copy()
    expected_shape = (action_chunk, action_dim)
    if normalized_actions.shape != expected_shape:
        raise ValueError(
            "Predicted action chunk shape mismatch: "
            f"got {normalized_actions.shape}, expected {expected_shape}"
        )

    normalized_actions[:, -1] = np.where(
        normalized_actions[:, -1] < 0.5,
        0,
        1,
    )
    action_mask = np.asarray(statistic['action_mask']).reshape(expected_shape)
    action_min = np.asarray(statistic['action_min']).reshape(expected_shape)
    action_max = np.asarray(statistic['action_max']).reshape(expected_shape)
    actions = np.where(
        action_mask,
        0.5 * (normalized_actions + 1) * (action_max - action_min) + action_min,
        normalized_actions,
    )
    return actions[0] if action_chunk == 1 else actions


def _sync_dinov3_rope_config(vl_model, use_qwen_rope):
    use_qwen_rope = bool(int(use_qwen_rope))
    updated = sync_expert_qwen_rope_runtime(vl_model, use_qwen_rope)
    if updated == 0:
        Logger.log_warning("Cannot locate Qwen text layers to sync dinov3_use_qwen_rope")
    Logger.log_info(f"Synced dinov3_use_qwen_rope={use_qwen_rope} to {updated} Qwen attention layer(s)")


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_optional_flag(name: str):
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def require_square_token_grid(num_tokens: int, label: str):
    side = int(int(num_tokens) ** 0.5)
    if int(num_tokens) <= 0 or side * side != int(num_tokens):
        raise ValueError(
            f"{label} token count must be a positive square for current DINO/Qwen alignment; "
            f"got {int(num_tokens)}. Use square image resolutions or add explicit H/W grid plumbing."
        )
    return side, side


def _token_id(tokenizer, token: str):
    vocab = getattr(tokenizer, "vocab", None)
    if isinstance(vocab, dict) and token in vocab:
        return vocab[token]
    token_id = tokenizer.convert_tokens_to_ids(token)
    return None if token_id == getattr(tokenizer, "unk_token_id", None) else token_id


def _infer_qwen_pruning_ranges(processor, base_input_ids: torch.Tensor, device):
    tokenizer = processor.tokenizer
    vision_start_id = _token_id(tokenizer, "<|vision_start|>")
    vision_end_id = _token_id(tokenizer, "<|vision_end|>")
    im_end_id = _token_id(tokenizer, "<|im_end|>")

    prefix_lens = []
    instruction_ranges = []
    for row in base_input_ids.detach().cpu().tolist():
        if vision_start_id is not None and vision_start_id in row:
            prefix_len = row.index(vision_start_id) + 1
        else:
            prefix_len = 5

        if vision_end_id is not None and vision_end_id in row:
            vision_end_pos = row.index(vision_end_id)
            instruction_start = vision_end_pos + 1
            if im_end_id is not None:
                try:
                    instruction_end = row.index(im_end_id, vision_end_pos + 1)
                except ValueError:
                    instruction_end = len(row)
            else:
                instruction_end = len(row)
        else:
            instruction_start = 0
            instruction_end = 0

        prefix_lens.append(prefix_len)
        instruction_ranges.append((instruction_start, instruction_end))

    return (
        torch.tensor(prefix_lens, dtype=torch.long, device=device),
        torch.tensor(instruction_ranges, dtype=torch.long, device=device),
    )


def _prune_dinov3_qkv_by_cls_similarity(
    all_qkv_states,
    last_hidden_state,
    pruning_ratio,
    needed_dinov3_layers,
    num_register_tokens,
):
    """Mirror train_no_gen_encoder_rlbench.py dinov3_cls pruning."""
    num_prefix_tokens = 1 + int(num_register_tokens)
    cls_token = last_hidden_state[:, 0:1, :]
    patch_tokens = last_hidden_state[:, num_prefix_tokens:, :]
    num_patch_tokens = patch_tokens.shape[1]
    if num_patch_tokens <= 0:
        raise RuntimeError("DINOv3 output has no patch tokens to prune")

    cls_norm = cls_token / (cls_token.norm(dim=-1, keepdim=True) + 1e-6)
    patch_norm = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + 1e-6)
    scores = (patch_norm * cls_norm).sum(dim=-1)

    pruning_ratio = float(pruning_ratio)
    if pruning_ratio > 1:
        num_tokens_to_keep = min(int(pruning_ratio), num_patch_tokens)
    else:
        num_tokens_to_keep = int(num_patch_tokens * (1.0 - pruning_ratio))
    num_tokens_to_keep = max(1, num_tokens_to_keep)

    _, token_keep_indices = torch.topk(scores, k=num_tokens_to_keep, dim=1, sorted=True)

    pruned_all_qkv_states = []
    for layer_idx, qkv_dict in enumerate(all_qkv_states):
        if layer_idx not in needed_dinov3_layers:
            pruned_all_qkv_states.append(qkv_dict)
            continue

        batch_size = qkv_dict["query"].shape[0]
        num_heads = qkv_dict["query"].shape[1]
        head_dim = qkv_dict["query"].shape[3]

        cls_indices = torch.zeros(batch_size, 1, dtype=torch.long, device=token_keep_indices.device)
        register_indices = torch.arange(1, num_prefix_tokens, dtype=torch.long, device=token_keep_indices.device)
        register_indices = register_indices.unsqueeze(0).expand(batch_size, -1)
        patch_indices = token_keep_indices + num_prefix_tokens
        global_indices = torch.cat([cls_indices, register_indices, patch_indices], dim=1)
        expanded_indices = global_indices.unsqueeze(1).unsqueeze(-1).expand(
            batch_size, num_heads, -1, head_dim
        )

        pruned_qkv = {}
        for qkv_type in ["query", "key", "value"]:
            if qkv_type in qkv_dict:
                pruned_qkv[qkv_type] = torch.gather(qkv_dict[qkv_type], dim=2, index=expanded_indices)
        pruned_all_qkv_states.append(pruned_qkv)

    Logger.log_info(
        f"Token pruning (dinov3_cls): keeping {num_tokens_to_keep}/{num_patch_tokens} tokens "
        f"(pruning ratio: {pruning_ratio})"
    )
    return pruned_all_qkv_states


def load_and_merge_training_config(args):
    """
    从 training_config.json 严格加载训练配置
    设计理念：最小化测试脚本的超参数输入，所有模型配置从训练配置加载
    """
    config_path = os.path.join(os.path.dirname(args.model_path), 'training_config.json')
    
    if not os.path.exists(config_path):
        print(f"[ERROR] training_config.json 不存在于路径: {config_path}")
        if not _allow_random_init(args):
            raise FileNotFoundError(
                f"training_config.json is required for reproducible eval: {config_path}. "
                "Pass --allow-random-init=1 only for local debugging."
            )
        print("[WARN] --allow-random-init=1，使用默认配置继续；该结果不可用于复现或报告。")
        # 设置默认值以继续运行（与训练保存的 key 一致，便于测试脚本不依赖 .sh 传参）
        setattr(args, 'use_dinov3', 0)
        setattr(args, 'dinov3_image_size', 224)
        setattr(args, 'bins', 256)
        setattr(args, 'pruning_mode', 'none')
        setattr(args, 'pruning_attn_aggregation', 'none')
        setattr(args, 'pruning_attn_aggregation_layer_start', 3)
        setattr(args, 'pruning_attn_aggregation_layer_end', 10)
        setattr(args, 'pruning_attn_aggregation_vote_ratio', 0.6)
        setattr(args, 'pruning_vote_contract', 'historical_floor_global_fallback')
        setattr(args, 'dinov3_efficient', 0)
        setattr(args, 'dinov3_concat_mode', 0)
        setattr(args, 'align_mode', 0)
        setattr(args, 'use_siglip_mot', 0)
        setattr(args, 'siglip_qwen_mapping', {})
        setattr(args, 'use_visual_token_concat', 0)
        setattr(args, 'image_aug', 0)
        setattr(args, 'image_aug_scope', 'qwen_only')
        setattr(args, 'image_aug_crop_scale', 0.9)
        setattr(args, 'qwen_position_contract', 'historical_inputs_embeds_1d')
        setattr(args, 'action_dim', getattr(args, 'action_dim', 7))
        setattr(args, 'action_chunk', getattr(args, 'action_chunk', 1))
        action_contract = resolve_eval_action_contract(
            {},
            action_chunk=args.action_chunk,
            action_dim=args.action_dim,
        )
        for key, value in action_contract.items():
            setattr(args, key, value)
        setattr(args, 'need_to_sub', getattr(args, 'need_to_sub', 3))
        args._training_config = {}
        args.action_representation = "xyz_delta_euler_absolute_v1"
        args._action_representation_metadata = {"action_representation": args.action_representation}
        return args
    
    with open(config_path, 'r') as f:
        training_config = json.load(f)
    args._training_config = training_config
    args.action_normalization_contract = training_config.get('action_normalization_contract', 'quantile_linear_clipped_v1')
    if args.action_normalization_contract not in ('quantile_linear_clipped_v1', 'quantile_linear_unclipped_v1'):
        raise ValueError('Unknown saved action normalization contract')
    args._action_representation_metadata = validate_saved_action_representation(
        training_config, os.path.join(os.path.dirname(args.model_path), "stats_data.json"),
    )
    args.action_representation = args._action_representation_metadata["action_representation"]
    expected_representation = getattr(args, "expected_action_representation", "")
    if expected_representation and expected_representation != args.action_representation:
        raise ValueError("Requested action representation differs from checkpoint metadata")
    
    print(f"[配置加载] 从 {config_path} 加载训练配置")
    
    # ========== 严格从训练配置加载，不使用命令行覆盖 ==========
    # 这些配置必须与训练时完全一致
    
    # DINOv3 配置（关键）
    args.use_dinov3 = training_config.get('use_dinov3', 0)
    args.dinov3_image_size = training_config.get('dinov3_image_size', 224)
    args.dinov3_qwen_mapping = training_config.get('dinov3_qwen_mapping', '[]')
    args.dinov3_qkv_after_rope = training_config.get('dinov3_qkv_after_rope', 1)
    args.dinov3_use_qwen_rope = training_config.get('dinov3_use_qwen_rope', 0)
    args.expert_rope_contract = training_config.get(
        'expert_rope_contract',
        'explicit_flags',
    )
    if args.expert_rope_contract_override:
        args.expert_rope_contract = args.expert_rope_contract_override
    args.expert_rope_runtime = resolve_expert_rope_runtime(
        args.expert_rope_contract,
        dinov3_qkv_after_rope=training_config.get(
            'dinov3_qkv_after_rope',
            args.dinov3_qkv_after_rope,
        ),
        dinov3_use_qwen_rope=training_config.get(
            'dinov3_use_qwen_rope',
            args.dinov3_use_qwen_rope,
        ),
        phase='eval',
    )
    args.dinov3_qkv_after_rope = int(args.expert_rope_runtime.dinov3_qkv_after_rope)
    args.dinov3_use_qwen_rope = int(args.expert_rope_runtime.dinov3_use_qwen_rope)
    args.expert_token_order = training_config.get('expert_token_order', 'importance')
    args.qwen_position_contract = training_config.get(
        'qwen_position_contract',
        'historical_inputs_embeds_1d',
    )
    if args.qwen_position_contract not in QWEN_POSITION_CONTRACTS:
        raise ValueError(
            "training_config qwen_position_contract must be one of "
            f"{sorted(QWEN_POSITION_CONTRACTS)}, got {args.qwen_position_contract!r}"
        )
    args.vl_position_contract = training_config.get(
        'vl_position_contract',
        'historical_expert_offset_all_layers',
    )
    args.pruning_attention_index_contract = training_config.get(
        'pruning_attention_index_contract',
        'historical_base_indices',
    )
    args.pruning_layer_index_base = training_config.get('pruning_layer_index_base', 'zero')
    args.action_task_attention_contract = training_config.get(
        'action_task_attention_contract',
        'historical_joint_mean',
    )
    args.attention_collection_mask_contract = training_config.get(
        'attention_collection_mask_contract',
        # historical_sdpa_eager_mismatch 是 bug-repro 契约（语义静默错），缺 key 的旧
        # ckpt 一律落到修正后的因果 mask。campaign 产出的 ckpt 都在 training_config 里
        # 显式带该 key，不会走此 fallback（2026-08-19 复审修正）。
        'corrected_causal',
    )
    if args.attention_collection_mask_contract_override:
        args.attention_collection_mask_contract = args.attention_collection_mask_contract_override
    args.vla_attention_mask_contract = training_config.get(
        'vla_attention_mask_contract',
        'historical_causal',
    )
    args.expert_private_to_action = int(training_config.get('expert_private_to_action', 0))
    args.dinov3_aligner_mlp = int(training_config.get('dinov3_aligner_mlp', 0))
    args.expert_inject_gate = int(training_config.get('expert_inject_gate', 0))
    args.train_dinov3_full_finetune = training_config.get('train_dinov3_full_finetune', 0)
    
    # dinov3_path 特殊处理：优先使用命令行（允许重定向到不同路径）
    if not hasattr(args, 'dinov3_path') or args.dinov3_path is None:
        args.dinov3_path = training_config.get('dinov3_path', '')
    
    # Token Pruning 配置（关键）
    args.pruning_mode = training_config.get('pruning_mode', 'none')
    args.pruning_ref_layer = training_config.get('pruning_ref_layer', -1)
    args.pruning_ratio = training_config.get('pruning_ratio', 0.0)
    args.pruning_attn_aggregation = training_config.get('pruning_attn_aggregation', 'none')
    args.pruning_attn_aggregation_layer_start = training_config.get('pruning_attn_aggregation_layer_start', 3)
    args.pruning_attn_aggregation_layer_end = training_config.get('pruning_attn_aggregation_layer_end', 10)
    args.pruning_attn_aggregation_vote_ratio = training_config.get('pruning_attn_aggregation_vote_ratio', 0.6)
    args.pruning_vote_contract = training_config.get(
        'pruning_vote_contract',
        'historical_floor_global_fallback',
    )
    if args.pruning_vote_contract not in PRUNING_VOTE_CONTRACTS:
        raise ValueError(
            "training_config pruning_vote_contract must be one of "
            f"{sorted(PRUNING_VOTE_CONTRACTS)}, got {args.pruning_vote_contract!r}"
        )
    args.action_attn_mode = training_config.get('action_attn_mode', 'none')
    args.action_img_attn_start_layer = training_config.get('action_img_attn_start_layer', -1)
    args.action_attn_roi_layer_start = training_config.get('action_attn_roi_layer_start', 3)
    args.action_attn_roi_layer_end = training_config.get('action_attn_roi_layer_end', 10)
    args.action_attn_roi_ratio = training_config.get('action_attn_roi_ratio', 0.5)

    # Efficient DINOv3 模式
    args.dinov3_efficient = training_config.get('dinov3_efficient', 0)
    if int(args.dinov3_efficient):
        raise NotImplementedError(
            "dinov3_efficient checkpoints are disabled for release eval. "
            "This branch needs a dedicated implementation/probe before its results can be trusted."
        )

    # Concat 模式（OpenVLA 风格）
    args.dinov3_concat_mode = training_config.get('dinov3_concat_mode', 0)
    # Early-fusion concat fix 开关（旧 checkpoint 无此键 → 0 = released 行为）
    args.dinov3_concat_pre_norm = int(training_config.get('dinov3_concat_pre_norm', 0))
    # DINOv3 2D-RoPE train-mode 坐标 rescale 上界（旧 checkpoint 无此键 → 2.0 =
    # released 行为；仅 train() 模式生效，eval 无行为差异，此处仅为记录/校验）
    args.dinov3_pos_embed_rescale = float(training_config.get('dinov3_pos_embed_rescale', 2.0))
    # DINOv3 分支独立学习率开关（旧 checkpoint 无此键 → 0.0 = released 行为；
    # 仅优化器分组生效，eval 无行为差异，此处仅为记录/校验）
    args.dinov3_branch_lr = float(training_config.get('dinov3_branch_lr', 0.0))

    # BlindVLA Alignment 模式（推理时不需要 align loss，但需知道是否加载了 projector）
    args.align_mode = training_config.get('align_mode', 0)

    # SigLIP MOT 模式（Qwen ViT QKV 注入 LLM attention）
    args.use_siglip_mot = training_config.get('use_siglip_mot', 0)
    _sm = training_config.get('siglip_qwen_mapping', {})
    args.siglip_qwen_mapping = {int(k): int(v) for k, v in _sm.items()} if isinstance(_sm, dict) else {}
    args.siglip_token_contract = training_config.get(
        'siglip_token_contract',
        'published_legacy_225' if int(args.use_siglip_mot) else 'corrected_256',
    )
    if int(args.use_siglip_mot):
        resolve_siglip_token_contract(args.siglip_token_contract)
    # Visual Token Concat 模式（ablation vs MOT）
    args.use_visual_token_concat = training_config.get('use_visual_token_concat', training_config.get('use_siglip_token_concat', 0))
    
    # Action 相关配置（关键，测试 .sh 可不传，全部从 config 读）
    args.bins = training_config.get('bins', 256)
    args.input_image_num = training_config.get('input_image_num', 1)
    args.action_dim = training_config.get('action_dim', getattr(args, 'action_dim', 7))
    args.action_chunk = training_config.get('action_chunk', getattr(args, 'action_chunk', 1))
    action_contract = resolve_eval_action_contract(
        training_config,
        action_chunk=args.action_chunk,
        action_dim=args.action_dim,
    )
    for key, value in action_contract.items():
        setattr(args, key, value)
    args.need_to_sub = training_config.get('need_to_sub', getattr(args, 'need_to_sub', 3))
    args.image_aug = int(training_config.get('image_aug', 0))
    args.image_aug_scope = training_config.get('image_aug_scope', 'qwen_only')
    args.image_aug_crop_scale = float(training_config.get('image_aug_crop_scale', 0.9))
    if args.image_aug_scope not in IMAGE_AUG_SCOPES:
        raise ValueError(
            f"training_config image_aug_scope must be one of {sorted(IMAGE_AUG_SCOPES)}, "
            f"got {args.image_aug_scope!r}"
        )
    if not 0.0 < args.image_aug_crop_scale <= 1.0:
        raise ValueError(
            "training_config image_aug_crop_scale must be in (0, 1], "
            f"got {args.image_aug_crop_scale}"
        )
    
    # Qwen 图像分辨率（与训练一致，影响 processor 与输入尺寸）
    args.qwen_image_resolution = training_config.get('qwen_image_resolution', 'native')
    if isinstance(args.qwen_image_resolution, int):
        args.qwen_image_resolution = str(args.qwen_image_resolution)
    
    # 其他配置（允许命令行覆盖）
    if not hasattr(args, 'use_robot_state') or args.use_robot_state is None:
        args.use_robot_state = training_config.get('robot_state', 0)
    if not hasattr(args, 'image_generation') or args.image_generation is None:
        args.image_generation = training_config.get('image_generation', 0)
    
    # 打印加载的关键配置
    print(f"[配置] use_dinov3: {args.use_dinov3}")
    if args.use_dinov3:
        print(f"[配置] dinov3_path: {args.dinov3_path}")
        print(f"[配置] dinov3_image_size: {args.dinov3_image_size}")
        print(f"[配置] dinov3_qwen_mapping: {args.dinov3_qwen_mapping}")
        print(f"[配置] dinov3_qkv_after_rope: {args.dinov3_qkv_after_rope}")
        print(f"[配置] dinov3_use_qwen_rope: {args.dinov3_use_qwen_rope}")
        print(f"[配置] expert_rope_contract: {args.expert_rope_contract}")
        print(f"[配置] expert_rope_effective_contract: {args.expert_rope_runtime.effective_contract}")
    
    print(f"[配置] pruning_mode: {args.pruning_mode}")
    if args.pruning_mode != 'none':
        print(f"[配置] pruning_ref_layer: {args.pruning_ref_layer}")
        print(f"[配置] pruning_ratio: {args.pruning_ratio}")
    
    print(f"[配置] bins: {args.bins}")
    print(f"[配置] action_dim: {args.action_dim} (from training_config)")
    print(f"[配置] action_chunk: {args.action_chunk} (from training_config)")
    print(f"[配置] action_query_contract: {args.action_query_contract}")
    print(f"[配置] action_query_count: {args.action_query_count}")
    print(f"[配置] action_head_contract: {args.action_head_contract}")
    print(f"[配置] chunk_execution_contract: {args.chunk_execution_contract}")
    print(f"[配置] need_to_sub: {args.need_to_sub} (from training_config)")
    print(f"[配置] image_aug: {args.image_aug} (from training_config)")
    print(f"[配置] image_aug_scope: {args.image_aug_scope} (from training_config)")
    print(f"[配置] image_aug_crop_scale: {args.image_aug_crop_scale} (from training_config)")
    print(f"[配置] qwen_image_resolution: {args.qwen_image_resolution}")
    print(f"[配置] qwen_position_contract: {args.qwen_position_contract}")
    
    return args


def _resolve_rlbench_profile_runtime(args):
    checkpoint_root = os.path.dirname(args.model_path)
    stats_path = os.path.join(checkpoint_root, "stats_data.json")
    provenance = checkpoint_provenance(
        getattr(args, "_training_config", {}),
        stats_path,
    )
    profile_resolution = resolve_profile(
        args.rlbench_env_profile,
        provenance,
        profile_path=args.rlbench_profile_path or None,
        allow_unknown=bool(args.allow_unknown_rlbench_profile),
        allow_mismatch=bool(args.allow_rlbench_profile_mismatch),
    )

    profile = profile_resolution.get("profile")
    if profile is None:
        profile_camera_resolution = args.eval_camera_resolution or None
        if profile_camera_resolution is None:
            raise ValueError(
                "An unversioned RLBench profile requires --eval-camera-resolution"
            )
    else:
        profile_camera_resolution = int(profile["simulator"]["camera_resolution"])

    qwen_resolution = getattr(args, "qwen_image_resolution", "native")
    qwen_image_size = None if qwen_resolution == "native" else int(qwen_resolution)
    image_contract = resolve_eval_image_protocol(
        protocol=args.eval_image_protocol,
        image_aug=getattr(args, "image_aug", 0),
        image_aug_scope=getattr(args, "image_aug_scope", "qwen_only"),
        crop_scale=getattr(args, "image_aug_crop_scale", 0.9),
        qwen_image_size=qwen_image_size,
        dinov3_image_size=getattr(args, "dinov3_image_size", None),
        use_dinov3=getattr(args, "use_dinov3", 0),
        profile_camera_resolution=profile_camera_resolution,
    )

    camera_override = args.eval_camera_resolution or None
    env_camera = os.environ.get("EVAL_CAM_RES", "")
    if camera_override is None and env_camera:
        if not env_camera.isdigit() or int(env_camera) <= 0:
            raise ValueError(f"EVAL_CAM_RES must be a positive integer, got {env_camera!r}")
        camera_override = int(env_camera)
    if camera_override is None:
        camera_override = int(image_contract["camera_resolution"])

    env_crop = os.environ.get("EVAL_CENTER_CROP", "")
    if env_crop:
        image_contract["center_crop_scale"] = _env_float(
            "EVAL_CENTER_CROP", float(image_contract["center_crop_scale"])
        )
        image_contract["environment_override"] = True
    env_dino_raw = _env_optional_flag("EVAL_DINO_RAW")
    if env_dino_raw is not None:
        image_contract["dino_source"] = "raw" if env_dino_raw else "qwen_resized"
        image_contract["environment_override"] = True
    image_contract["camera_resolution"] = int(camera_override)

    runtime = resolve_task_runtime(
        profile_resolution,
        args.task_name,
        explicit_yaw_offset=args.task_base_yaw_offset,
        camera_resolution_override=camera_override,
    )
    args._rlbench_profile_resolution = profile_resolution
    args._rlbench_runtime = runtime
    args._eval_image_contract = image_contract
    args.task_base_yaw_offset = runtime["resolved_task_base_yaw_offset"]
    args.eval_camera_resolution = runtime["resolved_camera_resolution"]
    return args


def validate_eval_action_contract(args, stats_data, dataset_name):
    expected_action_len = int(args.action_dim) * int(args.action_chunk)
    if dataset_name not in stats_data:
        raise KeyError(f"dataset_name={dataset_name!r} not found in stats_data.json")
    action_stats = stats_data[dataset_name].get("action", {})
    for key in ("mask", "q01", "q99"):
        value = action_stats.get(key)
        if len(value) != expected_action_len:
            raise ValueError(
                f"Eval action statistics length mismatch for {key}: got {len(value)}, "
                f"expected action_dim*action_chunk={expected_action_len}. "
                "The checkpoint training_config.json and stats_data.json are not the same action protocol."
            )


def _sha256_file(path: str):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _validation_runtime_metadata(training_config):
    import transformers

    def distribution_version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    snapshot_root = os.path.realpath(os.environ.get("PROJECT_PATH", str(_REPO_ROOT)))
    source_manifest_sha = os.environ.get("VALIDATION_SOURCE_MANIFEST_SHA256", "")
    if source_manifest_sha and snapshot_root != str(_REPO_ROOT.resolve()):
        raise ValueError("Validation PROJECT_PATH does not match the imported evaluator snapshot")
    identities = {
        "validation_arm_sha256": ("VALIDATION_ARM_SHA256", "repro_arm_sha256"),
        "validation_recipe_sha256": ("VALIDATION_RECIPE_SHA256", "repro_recipe_sha256"),
    }
    values = {}
    for field, (env_name, config_name) in identities.items():
        value = os.environ.get(env_name, "")
        if value and value != training_config.get(config_name):
            raise ValueError(f"{env_name} differs from checkpoint {config_name}")
        values[field] = value
    return {
        "validation_snapshot_root": snapshot_root,
        "validation_transformers_import_path": str(pathlib.Path(transformers.__file__).resolve()),
        "validation_torchvision_import_version": str(torchvision.__version__),
        "validation_source_manifest_sha256": source_manifest_sha,
        **values,
        "validation_runtime_observed": {
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "torchvision_version": distribution_version("torchvision"),
            "transformers_distribution_version": distribution_version("transformers"),
            "transformers_import_version": str(transformers.__version__),
            "tokenizers_version": distribution_version("tokenizers"),
            "accelerate_version": distribution_version("accelerate"),
            "deepspeed_version": distribution_version("deepspeed"),
            "numpy_version": str(np.__version__),
            "pillow_version": str(PIL.__version__),
        },
    }


def _write_eval_metadata(args, output_dir, model_image_size, dataset_summary=None):
    image_contract = getattr(args, "_eval_image_contract", None)
    if image_contract is None:
        default_crop = (
            float(getattr(args, "image_aug_crop_scale", 0.9))
            if int(getattr(args, "image_aug", 0))
            else 0.0
        )
        eval_crop_scale = _env_float("EVAL_CENTER_CROP", default_crop)
        eval_dino_source = resolve_eval_dino_source(
            image_aug=getattr(args, "image_aug", 0),
            image_aug_scope=getattr(args, "image_aug_scope", "qwen_only"),
            dino_raw_override=_env_optional_flag("EVAL_DINO_RAW"),
        )
        image_contract = {
            "eval_image_protocol": "legacy_fallback",
            "camera_resolution": model_image_size or 0,
            "center_crop_scale": eval_crop_scale,
            "dino_source": eval_dino_source,
            "environment_override": True,
        }
    eval_crop_scale = float(image_contract["center_crop_scale"])
    eval_dino_source = str(image_contract["dino_source"])
    training_config = getattr(args, "_training_config", {})
    training_config_path = os.path.join(os.path.dirname(args.model_path), "training_config.json")
    eval_source_archive = os.environ.get(
        "HF10_EVAL_SOURCE_ARCHIVE",
        os.environ.get("RLBENCH4_EVAL_SOURCE_ARCHIVE", ""),
    )
    eval_source_archive_sha256 = os.environ.get(
        "HF10_EVAL_SOURCE_ARCHIVE_SHA256",
        os.environ.get("RLBENCH4_EVAL_SOURCE_ARCHIVE_SHA256", ""),
    )
    eval_source_commit = os.environ.get(
        "HF10_EVAL_SOURCE_COMMIT",
        os.environ.get("RLBENCH4_EVAL_SOURCE_COMMIT", ""),
    )
    metadata = {
        **_validation_runtime_metadata(training_config),
        **getattr(args, "_action_representation_metadata", {}),
        "action_normalization_contract": args.action_normalization_contract,
        "task_name": args.task_name,
        "exp_name": args.exp_name,
        "model_path": args.model_path,
        "model_path_resolved": os.path.realpath(args.model_path),
        "training_config_sha256": _sha256_file(training_config_path),
        "training_repro_campaign_id": training_config.get("repro_campaign_id", ""),
        "training_repro_arm_id": training_config.get("repro_arm_id", ""),
        "training_repro_arm_sha256": training_config.get("repro_arm_sha256", ""),
        "training_repro_recipe_sha256": training_config.get("repro_recipe_sha256", ""),
        "training_data_origin": training_config.get("repro_data_origin", {}),
        "training_source_archive": training_config.get(
            "repro_source_archive",
            training_config.get("rlbench4_source_archive", ""),
        ),
        "training_source_archive_sha256": training_config.get(
            "repro_source_archive_sha256",
            training_config.get("rlbench4_source_archive_sha256", ""),
        ),
        "training_source_commit": training_config.get(
            "repro_source_commit",
            training_config.get("rlbench4_reference_source_commit", ""),
        ),
        "eval_source_archive": eval_source_archive,
        "eval_source_archive_sha256": eval_source_archive_sha256,
        "eval_source_commit": eval_source_commit,
        "evaluator_script": os.path.realpath(__file__),
        "evaluator_script_sha256": _sha256_file(__file__),
        "eval_campaign_id": os.environ.get("HF10_EVAL_CAMPAIGN_ID", ""),
        "eval_arm": os.environ.get("HF10_EVAL_ARM", ""),
        "dataset_name": args.dataset_name,
        "replay_or_predict": args.replay_or_predict,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "eval_seed": args.eval_seed,
        "task_base_yaw_offset": getattr(args, "task_base_yaw_offset", 0.0),
        "profile_task_base_yaw_offset": getattr(args, "_rlbench_runtime", {}).get(
            "profile_task_base_yaw_offset"
        ),
        "explicit_task_base_yaw_offset": getattr(args, "_rlbench_runtime", {}).get(
            "explicit_task_base_yaw_offset"
        ),
        "action_dim": args.action_dim,
        "action_chunk": args.action_chunk,
        "action_query_contract": args.action_query_contract,
        "action_query_count": args.action_query_count,
        "action_query_placeholder_contract": args.action_query_placeholder_contract,
        "action_head_contract": args.action_head_contract,
        "chunk_execution_contract": args.chunk_execution_contract,
        "allow_random_init": int(getattr(args, "allow_random_init", 0)),
        "use_robot_state": args.use_robot_state,
        "qwen_image_resolution": getattr(args, "qwen_image_resolution", "native"),
        "image_aug": getattr(args, "image_aug", 0),
        "image_aug_scope": getattr(args, "image_aug_scope", "qwen_only"),
        "image_aug_crop_scale": getattr(args, "image_aug_crop_scale", 0.9),
        "use_dinov3": getattr(args, "use_dinov3", 0),
        "dinov3_image_size": getattr(args, "dinov3_image_size", None),
        "pruning_mode": getattr(args, "pruning_mode", "none"),
        "pruning_ratio": getattr(args, "pruning_ratio", 0.0),
        "pruning_vote_contract": getattr(
            args,
            "pruning_vote_contract",
            "historical_floor_global_fallback",
        ),
        "expert_rope_contract": getattr(args, "expert_rope_contract", "explicit_flags"),
        "expert_rope_contract_override": getattr(args, "expert_rope_contract_override", ""),
        "expert_rope_effective_contract": getattr(
            getattr(args, "expert_rope_runtime", None),
            "effective_contract",
            None,
        ),
        "dinov3_qkv_after_rope": getattr(args, "dinov3_qkv_after_rope", None),
        "dinov3_use_qwen_rope": getattr(args, "dinov3_use_qwen_rope", None),
        "expert_token_order": getattr(args, "expert_token_order", "importance"),
        "qwen_position_contract": getattr(
            args,
            "qwen_position_contract",
            "historical_inputs_embeds_1d",
        ),
        "vl_position_contract": getattr(
            args,
            "vl_position_contract",
            "historical_expert_offset_all_layers",
        ),
        "pruning_attention_index_contract": getattr(
            args,
            "pruning_attention_index_contract",
            "historical_base_indices",
        ),
        "pruning_layer_index_base": getattr(args, "pruning_layer_index_base", "zero"),
        "action_task_attention_contract": getattr(
            args,
            "action_task_attention_contract",
            "historical_joint_mean",
        ),
        "attention_collection_mask_contract": getattr(
            args,
            "attention_collection_mask_contract",
            "historical_sdpa_eager_mismatch",
        ),
        "attention_collection_mask_contract_override": getattr(
            args,
            "attention_collection_mask_contract_override",
            "",
        ),
        "vla_attention_mask_contract": getattr(
            args,
            "vla_attention_mask_contract",
            "historical_causal",
        ),
        "expert_private_to_action": int(getattr(args, "expert_private_to_action", 0)),
        "dinov3_aligner_mlp": int(getattr(args, "dinov3_aligner_mlp", 0)),
        "dinov3_concat_pre_norm": int(getattr(args, "dinov3_concat_pre_norm", 0)),
        "dinov3_pos_embed_rescale": float(getattr(args, "dinov3_pos_embed_rescale", 2.0)),
        "action_attn_mode": getattr(args, "action_attn_mode", "none"),
        "action_img_attn_start_layer": getattr(args, "action_img_attn_start_layer", -1),
        "action_attn_roi_layer_start": getattr(args, "action_attn_roi_layer_start", 3),
        "action_attn_roi_layer_end": getattr(args, "action_attn_roi_layer_end", 10),
        "action_attn_roi_ratio": getattr(args, "action_attn_roi_ratio", 0.5),
        "eval_camera_resolution": model_image_size,
        "eval_image_protocol": image_contract["eval_image_protocol"],
        "eval_image_contract": image_contract,
        "eval_center_crop_scale": eval_crop_scale,
        "eval_dino_source": eval_dino_source,
        "env_EVAL_CAM_RES": os.environ.get("EVAL_CAM_RES", ""),
        "env_EVAL_CENTER_CROP": os.environ.get("EVAL_CENTER_CROP", ""),
        "env_EVAL_DINO_RAW": os.environ.get("EVAL_DINO_RAW", ""),
        "env_EVAL_DINO_ABLATE": os.environ.get("EVAL_DINO_ABLATE", ""),
        "eval_data_path": getattr(args, "eval_data_path", ""),
        "eval_data_root": getattr(args, "eval_data_root", ""),
        "eval_demo_root": getattr(args, "eval_demo_root", ""),
        "eval_demo_map": getattr(args, "eval_demo_map", ""),
        "eval_variation": getattr(args, "eval_variation", 0),
        "eval_episode_offset": getattr(args, "eval_episode_offset", 0),
        "eval_episodes": getattr(args, "eval_episodes", ""),
        "dataset_max_samples": getattr(args, "dataset_max_samples", 0),
        "dataset_pos_l2_threshold": getattr(args, "dataset_pos_l2_threshold", 0.05),
        "dataset_rot_l2_threshold": getattr(args, "dataset_rot_l2_threshold", 0.25),
    }
    if model_image_size:
        crop_size = (
            int(int(model_image_size) * eval_crop_scale)
            if 0.0 < eval_crop_scale < 1.0
            else int(model_image_size)
        )
        crop_size = max(1, crop_size)
        metadata["eval_center_crop_pixels"] = crop_size
        metadata["eval_effective_linear_scale"] = int(model_image_size) / crop_size
    profile_resolution = getattr(args, "_rlbench_profile_resolution", None)
    if profile_resolution is not None:
        metadata.update(profile_metadata(profile_resolution))
    runtime = getattr(args, "_rlbench_runtime", None)
    if runtime is not None:
        metadata["rlbench_runtime"] = runtime
    if getattr(args, "eval_data_path", ""):
        metadata["eval_data_path_resolved"] = os.path.realpath(args.eval_data_path)
        metadata["eval_data_sha256"] = _sha256_file(args.eval_data_path)
    if getattr(args, "eval_demo_map", ""):
        metadata["eval_demo_map_resolved"] = os.path.realpath(args.eval_demo_map)
        metadata["eval_demo_map_sha256"] = _sha256_file(args.eval_demo_map)
    if dataset_summary is not None:
        metadata["dataset_summary"] = dataset_summary
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _resolve_dataset_path(raw_path, data_path, data_root):
    if isinstance(raw_path, list):
        raw_path = raw_path[0] if raw_path else ""
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Invalid image path in dataset row: {raw_path!r}")
    resolved, _ = resolve_dataset_image_path(raw_path, data_path, data_root or "")
    return resolved


def _episode_step_from_path(path):
    match = re.search(r"episode(\d+)/image(\d+)\.(?:png|jpg|jpeg)$", str(path))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _sample_task_name(sample):
    values = []

    def _walk(x):
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
        elif isinstance(x, list):
            for v in x:
                _walk(v)
        elif isinstance(x, (str, int, float)):
            values.append(str(x))

    _walk(sample)
    text = " ".join(values)
    for task in (
        "close_box",
        "close_laptop_lid",
        "toilet_seat_down",
        "sweep_to_dustpan",
        "close_fridge",
        "phone_on_base",
        "take_umbrella_out_of_umbrella_stand",
        "take_frame_off_hanger",
        "place_wine_at_rack_location",
        "water_plants",
    ):
        if task in text:
            return task
    return None


def _load_dataset_rows_for_task(args):
    if not args.eval_data_path:
        raise ValueError("--eval-data-path is required for dataset_predict mode")
    with open(args.eval_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset at {args.eval_data_path}, got {type(data).__name__}")
    rows = [row for row in data if _sample_task_name(row) == args.task_name]
    if not rows:
        raise ValueError(f"No rows for task={args.task_name!r} in {args.eval_data_path}")
    if args.dataset_max_samples and args.dataset_max_samples > 0:
        rows = rows[: args.dataset_max_samples]
    return rows


def _dataset_rows_summary(args, rows):
    task_counter = Counter()
    declared_res_counter = Counter()
    actual_res_counter = Counter()
    missing_paths = []
    episode_ids = []

    for row in rows:
        task_counter[_sample_task_name(row) or "UNKNOWN"] += 1
        declared_res = row.get("input_image_resolution")
        if isinstance(declared_res, list):
            declared_res_counter["x".join(str(x) for x in declared_res)] += 1

        image_path = _resolve_dataset_path(row.get("input_image"), args.eval_data_path, args.eval_data_root)
        episode_id, _ = _episode_step_from_path(image_path)
        if episode_id is not None:
            episode_ids.append(episode_id)

        if not os.path.exists(image_path):
            missing_paths.append(image_path)
            continue
        with Image.open(image_path) as im:
            actual_res_counter[f"{im.height}x{im.width}"] += 1

    if missing_paths:
        preview = "\n".join(missing_paths[:5])
        raise FileNotFoundError(
            f"dataset_predict found {len(missing_paths)} missing input image(s); first paths:\n{preview}"
        )

    return {
        "selected_rows": len(rows),
        "task_counts": dict(sorted(task_counter.items())),
        "declared_input_resolution_counts": dict(sorted(declared_res_counter.items())),
        "actual_input_resolution_counts": dict(sorted(actual_res_counter.items())),
        "episode_min": min(episode_ids) if episode_ids else None,
        "episode_max": max(episode_ids) if episode_ids else None,
        "episode_count": len(set(episode_ids)) if episode_ids else 0,
    }


def _normalize_action_for_metric(action, statistic, contract='quantile_linear_clipped_v1'):
    from utils.rlbench_action_normalization import normalize_action
    action = np.array(action, dtype=np.float32).reshape(-1)
    return normalize_action(action, statistic['action_min'], statistic['action_max'], statistic['action_mask'], contract)


def _angle_abs_diff(pred_rot, target_rot):
    diff = np.array(pred_rot, dtype=np.float32) - np.array(target_rot, dtype=np.float32)
    return np.abs(np.arctan2(np.sin(diff), np.cos(diff)))


def _rotation_geodesic_angle(pred_rot, target_rot):
    pred_rot = np.array(pred_rot, dtype=np.float32).reshape(-1, 3)
    target_rot = np.array(target_rot, dtype=np.float32).reshape(-1, 3)
    pred_r = R.from_euler("xyz", pred_rot, degrees=False)
    target_r = R.from_euler("xyz", target_rot, degrees=False)
    return (pred_r.inv() * target_r).magnitude()


def _reshape_action_chunks(action, args, label):
    action = np.array(action, dtype=np.float32).reshape(-1)
    expected = int(args.action_dim) * int(args.action_chunk)
    if action.shape[0] != expected:
        raise ValueError(
            f"{label} action length mismatch: got {action.shape[0]}, "
            f"expected action_dim*action_chunk={expected}"
        )
    return action.reshape(int(args.action_chunk), int(args.action_dim))


def _action_metric_breakdown(pred, target, args, current_state=None):
    pred_chunks = _reshape_action_chunks(pred, args, "Predicted")
    target_chunks = _reshape_action_chunks(target, args, "Dataset target")
    if int(args.action_dim) < 7:
        raise ValueError("dataset_predict metrics require action_dim >= 7 for xyz/euler/gripper actions")
    if getattr(args, "action_representation", "xyz_delta_euler_absolute_v1") == "xyz_delta_euler_relative_wrapped_v1":
        if (int(args.action_chunk), int(args.action_dim)) != (1, 7) or current_state is None:
            raise ValueError("Relative Euler dataset metrics require H1/7D and the row's current state")
        pred_chunks = decode_action_for_execution(pred_chunks[0], current_state, args.action_representation)[None, :]
        target_chunks = decode_action_for_execution(target_chunks[0], current_state, args.action_representation)[None, :]

    pos_l2_by_chunk = np.linalg.norm(pred_chunks[:, :3] - target_chunks[:, :3], axis=1)
    rot_l2_by_chunk = np.linalg.norm(_angle_abs_diff(pred_chunks[:, 3:6], target_chunks[:, 3:6]), axis=1)
    rot_geodesic_by_chunk = _rotation_geodesic_angle(pred_chunks[:, 3:6], target_chunks[:, 3:6])
    gripper_ok_by_chunk = np.round(pred_chunks[:, 6]).astype(int) == np.round(target_chunks[:, 6]).astype(int)

    within_by_chunk = (
        (pos_l2_by_chunk <= float(args.dataset_pos_l2_threshold))
        & (rot_geodesic_by_chunk <= float(args.dataset_rot_l2_threshold))
        & gripper_ok_by_chunk
    )
    return {
        "pos_l2_by_chunk": pos_l2_by_chunk.tolist(),
        "rot_l2_periodic_by_chunk": rot_l2_by_chunk.tolist(),
        "rot_geodesic_by_chunk": rot_geodesic_by_chunk.tolist(),
        "gripper_ok_by_chunk": [bool(x) for x in gripper_ok_by_chunk.tolist()],
        "within_threshold_by_chunk": [bool(x) for x in within_by_chunk.tolist()],
        "pos_l2": float(np.mean(pos_l2_by_chunk)),
        "pos_l2_max": float(np.max(pos_l2_by_chunk)),
        "rot_l2_periodic": float(np.mean(rot_l2_by_chunk)),
        "rot_geodesic": float(np.mean(rot_geodesic_by_chunk)),
        "rot_geodesic_max": float(np.max(rot_geodesic_by_chunk)),
        "gripper_ok": bool(np.all(gripper_ok_by_chunk)),
        "within_threshold": bool(np.all(within_by_chunk)),
    }


def _episode_ids(args):
    if getattr(args, "eval_episodes", ""):
        return [int(x.strip()) for x in str(args.eval_episodes).split(",") if x.strip()]
    return [int(args.eval_episode_offset) + i for i in range(args.num_episodes)]


def _load_eval_demo_map(args):
    map_path = getattr(args, "eval_demo_map", "")
    if not map_path:
        return {}
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_maps = data.get("tasks", data)
    task_entries = task_maps.get(args.task_name, {})
    if not isinstance(task_entries, dict) or not task_entries:
        raise KeyError(f"No demo-map entries for task={args.task_name!r} in {map_path}")

    episode_map = {}
    for json_episode, item in task_entries.items():
        if isinstance(item, dict):
            raw_episode = item.get("best_raw_episode")
            if raw_episode is None:
                raw_episode = item.get("raw_episode")
        else:
            raw_episode = item
        if raw_episode is None:
            continue
        episode_map[int(json_episode)] = int(raw_episode)

    if not episode_map:
        raise ValueError(f"Demo-map has no usable raw episode ids for task={args.task_name!r}: {map_path}")
    return episode_map


def _mapped_demo_episode(args, episode_id):
    episode_map = getattr(args, "_eval_demo_episode_map", None)
    if not episode_map:
        return int(episode_id)
    if int(episode_id) not in episode_map:
        raise KeyError(
            f"JSON episode {episode_id} for task={args.task_name!r} is missing from "
            f"--eval-demo-map={args.eval_demo_map}"
        )
    return int(episode_map[int(episode_id)])


def _load_demo(args, episode_id):
    raw_episode_id = _mapped_demo_episode(args, episode_id)
    candidates = [
        os.path.join(
            args.eval_demo_root,
            args.task_name,
            f"variation{args.eval_variation}",
            "episodes",
            f"episode{raw_episode_id}",
            "low_dim_obs.pkl",
        ),
        os.path.join(
            args.eval_demo_root,
            args.task_name,
            "episodes",
            f"episode{raw_episode_id}",
            "low_dim_obs.pkl",
        ),
        os.path.join(
            args.eval_demo_root,
            args.task_name,
            f"episode{raw_episode_id}",
            "low_dim_obs.pkl",
        ),
        os.path.join(
            args.eval_demo_root,
            f"variation{args.eval_variation}",
            "episodes",
            args.task_name,
            f"episode{raw_episode_id}",
            "low_dim_obs.pkl",
        ),
    ]
    demo_path = next((p for p in candidates if os.path.exists(p)), None)
    if demo_path is None:
        raise FileNotFoundError(
            "demo low_dim_obs.pkl missing; tried:\n" + "\n".join(candidates)
        )
    with open(demo_path, "rb") as f:
        return pickle.load(f), demo_path, raw_episode_id


def _env_chain(env):
    seen = set()
    cur = env
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = getattr(cur, "env", None)


def _reset_video_frames(wrapper, source_env):
    if not hasattr(wrapper, "frames"):
        return
    wrapper.frames = []
    if hasattr(source_env, "get_rgb"):
        wrapper.frames.append(source_env.get_rgb())
    wrapper.step_count = 1


def _set_env_variation(env, variation):
    for obj in _env_chain(env):
        method = getattr(obj, "set_variation", None)
        if callable(method):
            method(int(variation))
            return

    for obj in _env_chain(env):
        rlbench_env = getattr(obj, "env", None)
        task_env = getattr(rlbench_env, "rlbench_task_env", None)
        method = getattr(task_env, "set_variation", None)
        if callable(method):
            method(int(variation))
            return

    raise AttributeError("Cannot set RLBench task variation on this environment wrapper stack")


def _reset_env_to_demo(env, demo):
    for obj in _env_chain(env):
        method = getattr(obj, "reset_to_demo", None)
        if callable(method):
            obs_dict = method(demo)
            _reset_video_frames(env, obj)
            return obs_dict

    for obj in _env_chain(env):
        rlbench_env = getattr(obj, "env", None)
        task_env = getattr(rlbench_env, "rlbench_task_env", None)
        method = getattr(task_env, "reset_to_demo", None)
        if not callable(method):
            continue

        if hasattr(obj, "cur_step"):
            obj.cur_step = 0
        method(demo)
        if not hasattr(obj, "get_obs"):
            raise AttributeError("Underlying RLBench env reset_to_demo exists, but wrapper has no get_obs()")
        obs_dict = obj.get_obs()
        if getattr(obj, "cinematic_flag", False):
            obj.tr.take_snap(obs=None)
        _reset_video_frames(env, obj)
        return obs_dict

    raise AttributeError("Cannot reset RLBench environment to demo on this wrapper stack")


def _task_description_from_env(task_name, env_text):
    task_description = env_text
    if task_name == "close_box":
        task_description = "close box"
    elif task_name == "close_laptop_lid":
        task_description = "close laptop lid"
    elif task_name == "phone_on_base":
        task_description = "put the phone on the base"
    elif task_name == "sweep_to_dustpan":
        task_description = "sweep the dirt up"
    return task_description


def run_dataset_predict(args):
    args.result_dir = os.path.join(args.result_dir, "dataset_predict_results")
    if args.exp_name is None:
        args.exp_name = args.task_name

    out_dir = os.path.join(args.result_dir, args.task_name, args.exp_name)
    recreate_directory(out_dir)
    pred_dir = os.path.join(out_dir, "pre_image")
    os.makedirs(pred_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    recreate_directory(log_dir)
    logger = setup_logger(log_dir)

    rows = _load_dataset_rows_for_task(args)
    dataset_summary = _dataset_rows_summary(args, rows)
    _write_eval_metadata(args, out_dir, model_image_size=None, dataset_summary=dataset_summary)
    logger.info("Dataset rows selected: %s", json.dumps(dataset_summary, sort_keys=True))

    model_components = model_load(args)
    vl_gpt = model_components["vl_gpt"]
    processor = model_components["processor"]
    action_tokenizer = model_components["action_tokenizer"]
    action_head = model_components["action_head"]
    statistic = model_components["statistic"]
    dinov3_model = model_components["dinov3_model"]
    dinov3_processor = model_components["dinov3_processor"]
    dinov3_qwen_mapping = model_components["dinov3_qwen_mapping"]

    metrics = []
    predictions_path = os.path.join(out_dir, "dataset_predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as pred_f:
        for idx, row in enumerate(rows):
            image_path = _resolve_dataset_path(row.get("input_image"), args.eval_data_path, args.eval_data_root)
            image = [Image.open(image_path).convert("RGB")]
            target = np.array(row["action"], dtype=np.float32).reshape(-1)
            _reshape_action_chunks(target, args, "Dataset target")

            task_description = row.get("input_prompt") or row.get("language_instruction") or args.task_name.replace("_", " ")
            state = np.array(row.get("state", []), dtype=np.float32)
            cur_robot_state = state if args.use_robot_state and state.size else None
            sample_dir = os.path.join(pred_dir, f"sample{idx:05d}")
            os.makedirs(sample_dir, exist_ok=True)

            pred = model_predict(
                args,
                vl_gpt,
                processor,
                action_tokenizer,
                action_head,
                statistic,
                task_description,
                image,
                cur_robot_state,
                pointcloud=None,
                pre_image_dir=sample_dir,
                step=idx,
                dinov3_model=dinov3_model,
                dinov3_processor=dinov3_processor,
                dinov3_qwen_mapping=dinov3_qwen_mapping,
            )
            pred = np.array(pred, dtype=np.float32).reshape(-1)
            _reshape_action_chunks(pred, args, "Predicted")
            pred_norm = _normalize_action_for_metric(pred, statistic, args.action_normalization_contract)
            target_norm = _normalize_action_for_metric(target, statistic, args.action_normalization_contract)
            norm_diff = pred_norm - target_norm
            metric = _action_metric_breakdown(pred, target, args, current_state=state)
            episode_id, step_id = _episode_step_from_path(image_path)
            item = {
                "idx": idx,
                "episode_id": episode_id,
                "step_id": step_id,
                "image_path": image_path,
                "task_prompt": task_description,
                "target_action": target.tolist(),
                "pred_action": pred.tolist(),
                "action_representation": args.action_representation,
        "action_normalization_contract": args.action_normalization_contract,
                "rotation_metric_representation": "absolute_xyz_euler_radians",
                "norm_l1": float(np.mean(np.abs(norm_diff))),
                "norm_mse": float(np.mean(norm_diff ** 2)),
                **metric,
            }
            pred_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            metrics.append(item)
            if idx % 25 == 0:
                logger.info(
                    "dataset_sample %d/%d norm_mse=%.6f pos_l2=%.4f rot_geo=%.4f within=%s",
                    idx + 1,
                    len(rows),
                    item["norm_mse"],
                    item["pos_l2"],
                    item["rot_geodesic"],
                    item["within_threshold"],
                )

    n = len(metrics)
    mean_norm_l1 = float(np.mean([m["norm_l1"] for m in metrics])) if metrics else 0.0
    mean_norm_mse = float(np.mean([m["norm_mse"] for m in metrics])) if metrics else 0.0
    mean_pos_l2 = float(np.mean([m["pos_l2"] for m in metrics])) if metrics else 0.0
    mean_rot_l2 = float(np.mean([m["rot_l2_periodic"] for m in metrics])) if metrics else 0.0
    mean_rot_geodesic = float(np.mean([m["rot_geodesic"] for m in metrics])) if metrics else 0.0
    gripper_acc = float(np.mean([m["gripper_ok"] for m in metrics]) * 100.0) if metrics else 0.0
    within_rate = float(np.mean([m["within_threshold"] for m in metrics]) * 100.0) if metrics else 0.0
    summary = {
        "task_name": args.task_name,
        "action_representation": args.action_representation,
        "action_normalization_contract": args.action_normalization_contract,
        "rotation_metric_representation": "absolute_xyz_euler_radians",
        "samples": n,
        "mean_norm_l1": mean_norm_l1,
        "mean_norm_mse": mean_norm_mse,
        "mean_pos_l2": mean_pos_l2,
        "mean_rot_l2_periodic": mean_rot_l2,
        "mean_rot_geodesic": mean_rot_geodesic,
        "gripper_accuracy_pct": gripper_acc,
        "within_threshold_pct": within_rate,
        "pos_l2_threshold": args.dataset_pos_l2_threshold,
        "rot_l2_threshold": args.dataset_rot_l2_threshold,
        "predictions_path": predictions_path,
    }
    with open(os.path.join(out_dir, "dataset_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    logger.info(
        "DatasetEval. %s * %d. norm_l1 %.6f norm_mse %.6f pos_l2 %.6f rot_geo %.6f gripper %.2f%% within %.2f%%",
        args.task_name,
        n,
        mean_norm_l1,
        mean_norm_mse,
        mean_pos_l2,
        mean_rot_geodesic,
        gripper_acc,
        within_rate,
    )
    # Keep a Finished line so cluster aggregation can collect dataset-threshold
    # rates, but the DatasetEval line above is the authoritative metric label.
    logger.info(f"Finished. {args.task_name} * {n}. Success rate {within_rate}%")
    with open(os.path.join(args.result_dir, args.task_name, f"{args.exp_name}_success_rate.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"DatasetEval. {args.task_name} * {n}. norm_l1 {mean_norm_l1} norm_mse {mean_norm_mse} "
            f"pos_l2 {mean_pos_l2} rot_l2_periodic {mean_rot_l2} rot_geodesic {mean_rot_geodesic} "
            f"gripper {gripper_acc}% within {within_rate}%\n"
        )
        f.write(f"Finished. {args.task_name} * {n}. Success rate {within_rate}%\n")


def setup_logger(log_dir):
    log_filename = os.path.join(log_dir, "output.log")
    
    logger = logging.getLogger("RLBenchLogger")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def recreate_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)


def _write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temporary, path)


def _rollout_step_error(info, step, stage="env.step"):
    """Classify typed environment errors; never infer infrastructure safety from text."""
    if not isinstance(info, dict) or not any(info.get(k) for k in ("error", "error_type", "error_category")):
        return None
    category = info.get("error_category", "infrastructure")
    if category not in ("policy_action", "infrastructure"):
        category = "infrastructure"
    return {
        "step": int(step), "stage": stage,
        "error_type": str(info.get("error_type") or "UnclassifiedEnvironmentError"),
        "error_category": category,
        "message": str(info.get("error") or info.get("message") or ""),
    }


def _rollout_exception_error(exc, step, stage):
    return {
        "step": int(step), "stage": str(stage),
        "error_type": type(exc).__name__, "error_category": "infrastructure",
        "message": str(exc),
    }


def _rollout_error_counts(records):
    errors = [error for row in records for error in row.get("step_errors", []) + row.get("reset_errors", [])]
    return {
        "infrastructure_error_count": sum(error["error_category"] == "infrastructure" for error in errors),
        "policy_action_error_count": sum(error["error_category"] == "policy_action" for error in errors),
    }


def model_load(args):
    # 与训练一致：Qwen3VLForConditionalGeneration.from_pretrained，推理只用基座 + action_head
    full_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    vl_base = full_model.model
    del full_model
    # 包装成与原先接口一致：下游使用 vl_gpt.model(...) 与 vl_gpt.get_input_embeddings()
    class _VLBaseWrapper:
        def __init__(self, model):
            self.model = model
            self.get_input_embeddings = model.get_input_embeddings
    vl_gpt = _VLBaseWrapper(vl_base)
    _sync_dinov3_rope_config(vl_gpt.model, getattr(args, "dinov3_use_qwen_rope", 0))

    # 与训练一致：local_files_only=True，action_bins 从 training_config 已合并到 args.bins
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        action_bins=getattr(args, "bins", 256),
    )
    tokenizer = processor.tokenizer
    # 与训练一致：固定 Qwen 输入分辨率时关闭 processor 的 resize，由上游提供已 resize 的图
    qwen_res = getattr(args, "qwen_image_resolution", "native")
    if qwen_res != "native":
        try:
            if hasattr(processor, "image_processor") and processor.image_processor is not None and hasattr(processor.image_processor, "do_resize"):
                processor.image_processor.do_resize = False
            Logger.log_info(f"Using fixed Qwen image size={qwen_res} (square)")
        except Exception:
            pass
    Logger.log_info("Loaded Qwen3VLModel (base only, continuous action); lm_head not loaded")
    
    # ========== 加载 DINOv3（如果训练时启用）==========
    dinov3_model = None
    dinov3_processor = None
    dinov3_qwen_mapping = None
    
    if getattr(args, 'use_dinov3', 0) == 1:
        if not getattr(args, 'dinov3_path', None) or not str(args.dinov3_path).strip():
            if not _allow_random_init(args):
                raise ValueError("use_dinov3=1 but dinov3_path is missing or empty in training_config/CLI.")
            Logger.log_warning("use_dinov3=1 but dinov3_path is missing or empty; skipping DINOv3 load because --allow-random-init=1.")
            setattr(args, 'use_dinov3', 0)
            dinov3_model = None
            dinov3_processor = None
            dinov3_qwen_mapping = None
            vl_gpt.dinov3_aligner = None
        else:
            Logger.log_info("DINOv3 is enabled, loading DINOv3 model...")
            checkpoint_dir = os.path.dirname(args.model_path)
            dinov3_checkpoint_path = os.path.join(checkpoint_dir, 'dinov3_model')

            # 与训练一致：DINOv3 分辨率等均从 training_config 加载（已合并到 args）
            dinov3_config = AutoConfig.from_pretrained(args.dinov3_path, trust_remote_code=True)
            original_image_size = getattr(dinov3_config, 'image_size', 224)
            dinov3_image_size = args.dinov3_image_size  # 来自 training_config（load_and_merge_training_config）
            if dinov3_image_size != original_image_size:
                Logger.log_info(f"Adjusting DINOv3 image_size from {original_image_size} to {dinov3_image_size} (from training_config)")
                dinov3_config.image_size = dinov3_image_size

            expects_finetuned_dino = bool(int(getattr(args, 'train_dinov3_full_finetune', 0)))
            if os.path.exists(dinov3_checkpoint_path):
                Logger.log_info(f"Loading fine-tuned DINOv3 from {dinov3_checkpoint_path}")
                dinov3_model = AutoModel.from_pretrained(
                    args.dinov3_path,
                    config=dinov3_config,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )
                dinov3_weights_path = os.path.join(dinov3_checkpoint_path, 'pytorch_model.bin')
                if os.path.exists(dinov3_weights_path):
                    state_dict = torch.load(dinov3_weights_path, map_location='cpu')
                    dinov3_model.load_state_dict(state_dict)
                    Logger.log_info("Loaded fine-tuned DINOv3 weights")
                else:
                    _missing_component(args, f"DINOv3 checkpoint directory exists but weights are missing: {dinov3_weights_path}")
            else:
                if expects_finetuned_dino:
                    _missing_component(args, f"train_dinov3_full_finetune=1 but finetuned DINOv3 directory is missing: {dinov3_checkpoint_path}")
                Logger.log_info(f"Loading pre-trained DINOv3 from {args.dinov3_path}")
                dinov3_model = AutoModel.from_pretrained(
                    args.dinov3_path,
                    config=dinov3_config,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )

            dinov3_processor = AutoImageProcessor.from_pretrained(
                args.dinov3_path,
                trust_remote_code=True,
            )
            if hasattr(dinov3_processor, "size") and dinov3_image_size:
                dinov3_processor.size = {
                    "height": dinov3_image_size,
                    "width": dinov3_image_size,
                }
            dinov3_model = dinov3_model.to('cuda')
            dinov3_model.eval()

            # 注册到主模型
            vl_gpt.dinov3_model = dinov3_model

            # 解析 dinov3_qwen_mapping（与训练一致：qwen_layer -> dinov3_layer）
            if hasattr(args, 'dinov3_qwen_mapping') and args.dinov3_qwen_mapping:
                if isinstance(args.dinov3_qwen_mapping, dict):
                    dinov3_qwen_mapping = args.dinov3_qwen_mapping
                elif isinstance(args.dinov3_qwen_mapping, list):
                    dinov3_qwen_mapping = dict(args.dinov3_qwen_mapping)  # [[qwen_layer, dinov3_layer], ...]
                else:
                    import ast
                    dinov3_qwen_mapping = dict(ast.literal_eval(str(args.dinov3_qwen_mapping)))
                Logger.log_info(f"DINOv3-Qwen mapping: {dinov3_qwen_mapping}")
            else:
                Logger.log_warning("No DINOv3-Qwen mapping found, DINOv3 may not work correctly!")
                dinov3_qwen_mapping = {}

            # 加载 DINOv3 Aligner（如果存在）：与训练脚本一致，从 config 获取维度并实例化后加载权重
            aligner_path = os.path.join(checkpoint_dir, 'dinov3_aligner.pt')
            if os.path.exists(aligner_path) and dinov3_qwen_mapping:
                # 与 train_no_gen_encoder.py 一致：从 dinov3_config 和 Qwen config 获取维度
                dinov3_num_heads = getattr(dinov3_config, 'num_attention_heads', 12)
                dinov3_head_dim = getattr(dinov3_config, 'hidden_size', 768) // dinov3_num_heads
                _cfg = vl_gpt.model
                if hasattr(_cfg, 'config'):
                    qwen_config = _cfg.config
                else:
                    qwen_config = getattr(vl_gpt, 'config', None)
                if qwen_config is None:
                    raise RuntimeError("Cannot get Qwen config for DINOv3 aligner (vl_gpt.model.config / vl_gpt.config).")
                if hasattr(qwen_config, 'text_config'):
                    qwen_q_num_heads = qwen_config.text_config.num_attention_heads
                    qwen_kv_num_heads = getattr(qwen_config.text_config, 'num_key_value_heads', qwen_q_num_heads)
                    qwen_head_dim = getattr(qwen_config.text_config, 'head_dim',
                                            qwen_config.text_config.hidden_size // qwen_q_num_heads)
                else:
                    hidden_dim = getattr(qwen_config, 'hidden_size', 2560)
                    qwen_q_num_heads = getattr(qwen_config, 'num_attention_heads', 32)
                    qwen_kv_num_heads = getattr(qwen_config, 'num_key_value_heads', qwen_q_num_heads)
                    qwen_head_dim = getattr(qwen_config, 'head_dim', hidden_dim // qwen_q_num_heads)
                layer_mapping = {int(k): int(v) for k, v in dinov3_qwen_mapping.items()}
                dinov3_aligner = DINOv3QKVAligner(
                    dinov3_num_heads=dinov3_num_heads,
                    dinov3_head_dim=dinov3_head_dim,
                    qwen_q_num_heads=qwen_q_num_heads,
                    qwen_kv_num_heads=qwen_kv_num_heads,
                    qwen_head_dim=qwen_head_dim,
                    layer_mapping=layer_mapping,
                    mlp=bool(getattr(args, 'dinov3_aligner_mlp', 0)),
                    expert_inject_gate=bool(getattr(args, 'expert_inject_gate', 0)),
                ).to(dtype=torch.bfloat16)
                aligner_state = torch.load(aligner_path, map_location='cpu')
                dinov3_aligner.load_state_dict(aligner_state, strict=True)
                dinov3_aligner.eval()
                vl_gpt.dinov3_aligner = dinov3_aligner
                Logger.log_info(f"DINOv3 aligner loaded from {aligner_path} (DINOv3 {dinov3_num_heads}h×{dinov3_head_dim}d -> Qwen Q{qwen_q_num_heads}h K/V{qwen_kv_num_heads}h×{qwen_head_dim}d)")
            else:
                vl_gpt.dinov3_aligner = None
                if os.path.exists(aligner_path) and not dinov3_qwen_mapping:
                    Logger.log_warning("dinov3_aligner.pt exists but dinov3_qwen_mapping is empty; aligner not loaded.")
                elif dinov3_qwen_mapping and not getattr(args, 'dinov3_concat_mode', 0) and not getattr(args, 'use_visual_token_concat', 0):
                    _missing_component(args, f"DINOv3-Qwen mapping is enabled but dinov3_aligner.pt is missing: {aligner_path}")

            Logger.log_info(f"DINOv3 model loaded successfully (image_size={dinov3_image_size})")

            # ===== DINOv3 Concat Projector（OpenVLA 风格，训练时 dinov3_concat_mode=1）=====
            if getattr(args, 'dinov3_concat_mode', 0):
                dinov3_config_for_proj = AutoConfig.from_pretrained(args.dinov3_path, trust_remote_code=True)
                dino_dim = getattr(dinov3_config_for_proj, 'hidden_size', 1280)
                _vl_cfg = vl_gpt.model
                if hasattr(_vl_cfg, 'config') and hasattr(_vl_cfg.config, 'text_config'):
                    qwen_out_dim = _vl_cfg.config.text_config.hidden_size
                else:
                    qwen_out_dim = 2560
                llm_dim = qwen_out_dim
                concat_proj = DINOv3ConcatProjector(
                    qwen_out_dim=qwen_out_dim,
                    dino_dim=dino_dim,
                    llm_dim=llm_dim,
                    pre_norm=bool(getattr(args, 'dinov3_concat_pre_norm', 0)),
                ).to(dtype=torch.bfloat16)
                concat_proj_path = os.path.join(checkpoint_dir, 'dinov3_concat_projector.pt')
                if os.path.exists(concat_proj_path):
                    concat_proj_state = torch.load(concat_proj_path, map_location='cpu')
                    incompatible = concat_proj.load_state_dict(concat_proj_state, strict=False)
                    if incompatible.missing_keys or incompatible.unexpected_keys:
                        _missing_component(
                            args,
                            f"dinov3_concat_projector state-dict mismatch with "
                            f"dinov3_concat_pre_norm={int(getattr(args, 'dinov3_concat_pre_norm', 0))}: "
                            f"missing={list(incompatible.missing_keys)} "
                            f"unexpected={list(incompatible.unexpected_keys)}",
                        )
                    Logger.log_info(
                        f"DINOv3 concat projector loaded from {concat_proj_path} "
                        f"(pre_norm={int(getattr(args, 'dinov3_concat_pre_norm', 0))})"
                    )
                else:
                    _missing_component(args, f"dinov3_concat_projector.pt not found at {concat_proj_path}")
                concat_proj = concat_proj.to('cuda').eval()
                vl_gpt.dinov3_concat_projector = concat_proj
            else:
                vl_gpt.dinov3_concat_projector = None
    else:
        Logger.log_info("DINOv3 is disabled")
        vl_gpt.dinov3_aligner = None
        vl_gpt.dinov3_concat_projector = None

    # ===== SigLIP MOT Aligner（Qwen ViT QKV 注入 LLM attention）=====
    if getattr(args, 'use_siglip_mot', 0) and getattr(args, 'siglip_qwen_mapping', {}):
        checkpoint_dir = os.path.dirname(args.model_path)
        siglip_map = args.siglip_qwen_mapping  # {qwen_layer: vit_layer}
        vit_cfg = vl_gpt.model.visual.config
        _s_num_heads = vit_cfg.num_heads
        _s_head_dim = vit_cfg.hidden_size // _s_num_heads
        _llm_cfg = vl_gpt.model.config.text_config
        _q_num = _llm_cfg.num_attention_heads
        _kv_num = getattr(_llm_cfg, 'num_key_value_heads', _q_num)
        _hd = getattr(_llm_cfg, 'head_dim', _llm_cfg.hidden_size // _q_num)
        _siglip_aligner = DINOv3QKVAligner(
            dinov3_num_heads=_s_num_heads,
            dinov3_head_dim=_s_head_dim,
            qwen_q_num_heads=_q_num,
            qwen_kv_num_heads=_kv_num,
            qwen_head_dim=_hd,
            layer_mapping=siglip_map,
        ).to(dtype=torch.bfloat16)
        siglip_aligner_path = os.path.join(checkpoint_dir, 'siglip_aligner.pt')
        if os.path.exists(siglip_aligner_path):
            _siglip_aligner.load_state_dict(torch.load(siglip_aligner_path, map_location='cpu'))
            Logger.log_info(f"SigLIP aligner loaded from {siglip_aligner_path}")
        else:
            _missing_component(args, f"siglip_aligner.pt not found at {siglip_aligner_path}")
        _siglip_aligner = _siglip_aligner.eval()
        vl_gpt.siglip_aligner = _siglip_aligner
    else:
        vl_gpt.siglip_aligner = None

    # ===== Visual Token Concat Aligner (ablation vs MOT) =====
    if getattr(args, 'use_visual_token_concat', 0):
        checkpoint_dir = os.path.dirname(args.model_path)
        _tc_mapping = getattr(args, 'siglip_qwen_mapping', {}) or getattr(args, 'dinov3_qwen_mapping', {})
        _tc_source  = 'siglip' if getattr(args, 'siglip_qwen_mapping', {}) else 'dinov3'

        if _tc_mapping:
            vit_cfg = vl_gpt.model.visual.config
            _tc_hidden_dim = (
                vit_cfg.hidden_size if _tc_source == 'siglip'
                else getattr(getattr(dinov3_model, "config", None), "hidden_size", 1280)
            )
            _llm_cfg = vl_gpt.model.config.text_config
            _tc_qwen_dim = _llm_cfg.hidden_size

            _visual_token_aligner = VisualTokenAligner(
                encoder_hidden_dim=_tc_hidden_dim,
                qwen_hidden_dim=_tc_qwen_dim,
                layer_mapping=_tc_mapping,
            ).to(dtype=torch.bfloat16)
            _tc_ckpt_path = os.path.join(checkpoint_dir, 'visual_token_aligner.pt')
            if os.path.exists(_tc_ckpt_path):
                _visual_token_aligner.load_state_dict(torch.load(_tc_ckpt_path, map_location='cpu'))
                Logger.log_info(f"Visual token concat aligner loaded from {_tc_ckpt_path}")
            else:
                _missing_component(args, f"visual_token_aligner.pt not found at {_tc_ckpt_path}")
            _visual_token_aligner = _visual_token_aligner.eval()
            vl_gpt.visual_token_aligner = _visual_token_aligner
        else:
            Logger.log_warning("use_visual_token_concat=1 but no mapping found; disabling")
            args.use_visual_token_concat = 0
            vl_gpt.visual_token_aligner = None
    else:
        vl_gpt.visual_token_aligner = None
    
    action_tokenizer = ActionTokenizer(tokenizer, need_to_sub=args.need_to_sub)

    # Load action head for continuous action prediction
    # Get hidden_size from base model (Qwen3VLModel; should be 2560 for Qwen3VL-4B)
    _cfg = vl_gpt.model
    if hasattr(_cfg, 'language_model') and hasattr(_cfg.language_model, 'config'):
        hidden_dim = _cfg.language_model.config.hidden_size
    elif hasattr(_cfg, 'config') and getattr(_cfg.config, 'text_config', None) is not None:
        hidden_dim = _cfg.config.text_config.hidden_size
    else:
        hidden_dim = getattr(_cfg.config, 'hidden_size', 2560)
    
    Logger.log_info(f"Language model hidden dimension: {hidden_dim}")
    
    action_head = build_eval_action_head(
        hidden_dim=hidden_dim,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        action_query_contract=args.action_query_contract,
    )
    
    # Load action head weights
    action_head_path = os.path.join(os.path.dirname(args.model_path), "action_head.pt")
    if os.path.exists(action_head_path):
        action_head_state_dict = torch.load(action_head_path, map_location='cpu')
        action_head.load_state_dict(action_head_state_dict)
        Logger.log_info(f"Action head loaded from {action_head_path}")
    else:
        _missing_component(args, f"Action head file not found at {action_head_path}")
    
    action_head = action_head.to(torch.bfloat16)

    statistics_path = os.path.join(os.path.dirname(args.model_path), "stats_data.json")
    if not os.path.exists(statistics_path):
        raise FileNotFoundError(f"stats_data.json is required for eval normalization: {statistics_path}")
    validate_saved_action_representation(getattr(args, "_training_config", {}), statistics_path)
    with open(statistics_path, 'r') as f:
        stats_data = json.load(f)
    dataset_name=args.dataset_name
    validate_eval_action_contract(args, stats_data, dataset_name)

    statistic= {}
    statistic['action_mask'] = np.array(stats_data[dataset_name]['action']['mask'])
    statistic['action_min'] = np.array(stats_data[dataset_name]['action']['q01'])
    statistic['action_max'] = np.array(stats_data[dataset_name]['action']['q99'])
    statistic['state_mask'] = np.array(stats_data[dataset_name]['state']['mask'])
    statistic['state_min'] = np.array(stats_data[dataset_name]['state']['q01'])
    statistic['state_max'] = np.array(stats_data[dataset_name]['state']['q99'])

    # 返回所有加载的组件
    model_components = {
        'vl_gpt': vl_gpt,
        'processor': processor,
        'action_tokenizer': action_tokenizer,
        'action_head': action_head,
        'statistic': statistic,
        'dinov3_model': dinov3_model,
        'dinov3_processor': dinov3_processor,
        'dinov3_qwen_mapping': dinov3_qwen_mapping,
    }
    
    return model_components


def model_predict(args, vl_gpt, processor, action_tokenizer, action_head, statistic, task_description, image, state, pointcloud, pre_image_dir, step,
                  dinov3_model=None, dinov3_processor=None, dinov3_qwen_mapping=None):
    device = f'cuda:{args.cuda}'
    vl_gpt.model = vl_gpt.model.to(device).eval()
    action_head = action_head.to(device).eval()
    if dinov3_model is not None:
        dinov3_model = dinov3_model.to(device).eval()
    if getattr(vl_gpt, 'dinov3_concat_projector', None) is not None:
        vl_gpt.dinov3_concat_projector = vl_gpt.dinov3_concat_projector.to(device).eval()
    parallel_size=1
    img_len = 1
    action_query_count = args.action_query_count

    state_tokens = ""
    if args.use_robot_state:
        state = np.array(state, dtype=np.float32)
        normalized_state = np.where(
            statistic['state_mask'],
            np.clip(2 * (state - statistic['state_min']) / (statistic['state_max'] - statistic['state_min'] + 1e-8) - 1, -1, 1),
            state
        )
        state_tokens += action_tokenizer(normalized_state)

    with torch.inference_mode():
        # ===== Qwen3VL 推理路径：参考训练脚本的 apply_chat_template 逻辑 =====
        # image: List[PIL.Image]，当前只支持单视角（用第一张）。
        # Fail-closed（2026-08-19 复审新增）：多视角训练的 ckpt 走这里会静默丢视角，直接拒绝。
        if int(getattr(args, 'input_image_num', 1) or 1) != 1:
            raise ValueError(
                "model_predict currently supports single-view eval only (uses image[0]); "
                f"training_config input_image_num={getattr(args, 'input_image_num', None)} "
                "would silently drop the other views."
            )
        pil_image = image[0]
        # image_aug=1 的随机 crop 在评测时用同尺度 center crop 确定性对齐。
        # qwen_only 保留历史分叉；shared 让 Qwen/DINO 使用同一 crop 后视场。
        qwen_res = getattr(args, "qwen_image_resolution", "native")
        qwen_image_size = None if qwen_res == "native" else int(qwen_res)
        image_contract = getattr(args, "_eval_image_contract", None)
        if image_contract is None:
            _default_crop = (
                float(getattr(args, 'image_aug_crop_scale', 0.9))
                if int(getattr(args, 'image_aug', 0))
                else 0.0
            )
            _crop = _env_float("EVAL_CENTER_CROP", _default_crop)
            dino_source = resolve_eval_dino_source(
                image_aug=getattr(args, 'image_aug', 0),
                image_aug_scope=getattr(args, 'image_aug_scope', 'qwen_only'),
                dino_raw_override=_env_optional_flag("EVAL_DINO_RAW"),
            )
        else:
            _crop = float(image_contract["center_crop_scale"])
            dino_source = str(image_contract["dino_source"])
        pil_image, dino_pil, _ = prepare_eval_image_views(
            pil_image,
            crop_scale=_crop,
            qwen_image_size=qwen_image_size,
            dino_source=dino_source,
        )

        prompt_content = [{"type": "image", "image": pil_image}]
        prompt_content.append({"type": "text", "text": task_description + state_tokens})

        messages = [
            {
                "role": "user",
                "content": prompt_content,
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        base_input_ids = inputs.input_ids.to(device)  # (1, L)
        bsz, base_len = base_input_ids.shape

        # 为动作 query 预留位置；数量由 checkpoint 的 action-query contract 决定。
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id or 0
        base_attn = inputs.attention_mask.to(device)
        full_input_ids, inputs_embeds, full_attention_mask = (
            append_action_query_placeholders(
                base_input_ids=base_input_ids,
                base_attention_mask=base_attn,
                embedding_layer=vl_gpt.get_input_embeddings(),
                action_query_count=action_query_count,
                placeholder_token_id=pad_token_id,
            )
        )

        pixel_values = inputs.pixel_values.to(torch.bfloat16).to(device)
        image_grid_thw = inputs.image_grid_thw.to(device) if hasattr(inputs, "image_grid_thw") else None

        model_kwargs = {
            "inputs_embeds": inputs_embeds,
            "pixel_values": pixel_values,
            "attention_mask": full_attention_mask,
            "return_dict": True,
            "action_length": action_query_count,
            "expert_token_order": getattr(args, 'expert_token_order', 'importance'),
            "vl_position_contract": getattr(
                args, 'vl_position_contract', 'historical_expert_offset_all_layers'
            ),
            "pruning_attention_index_contract": getattr(
                args, 'pruning_attention_index_contract', 'historical_base_indices'
            ),
            "pruning_layer_index_base": getattr(args, 'pruning_layer_index_base', 'zero'),
            "action_task_attention_contract": getattr(
                args, 'action_task_attention_contract', 'historical_joint_mean'
            ),
            "attention_collection_mask_contract": getattr(
                args,
                'attention_collection_mask_contract',
                'historical_sdpa_eager_mismatch',
            ),
            "vla_attention_mask_contract": getattr(
                args, 'vla_attention_mask_contract', 'historical_causal'
            ),
            "expert_private_to_action": int(getattr(args, 'expert_private_to_action', 0)),
            "pruning_vote_contract": getattr(
                args,
                'pruning_vote_contract',
                'historical_floor_global_fallback',
            ),
            "action_attn_mode": getattr(args, 'action_attn_mode', 'none'),
            "action_img_attn_start_layer": getattr(args, 'action_img_attn_start_layer', -1),
            "action_attn_roi_layer_start": getattr(args, 'action_attn_roi_layer_start', 3),
            "action_attn_roi_layer_end": getattr(args, 'action_attn_roi_layer_end', 10),
            "action_attn_roi_ratio": getattr(args, 'action_attn_roi_ratio', 0.5),
        }
        model_kwargs["qwen_position_contract"] = getattr(
            args,
            'qwen_position_contract',
            'historical_inputs_embeds_1d',
        )
        if model_kwargs["qwen_position_contract"] == 'token_ids_mrope':
            model_kwargs["position_input_ids"] = full_input_ids
        if image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = image_grid_thw

        prefix_tokens_lens, instruction_ranges = _infer_qwen_pruning_ranges(
            processor, base_input_ids, device
        )
        model_kwargs["prefix_tokens_lens"] = prefix_tokens_lens
        model_kwargs["instruction_ranges"] = instruction_ranges
        model_kwargs["visual_pos_masks"] = (
            full_input_ids == vl_gpt.model.config.image_token_id
        )

        # ========== Concat Mode（OpenVLA DinoSigLIP 风格）==========
        if getattr(args, 'dinov3_concat_mode', 0) and dinov3_model is not None and dinov3_processor is not None:
            import torch.nn.functional as F

            dinov3_inputs = dinov3_processor(images=[dino_pil], return_tensors="pt")
            dinov3_pixel_values = dinov3_inputs["pixel_values"].to(device).to(torch.bfloat16)
            dinov3_pixel_values = _maybe_ablate_dino(dinov3_pixel_values)

            # 1. Qwen ViT post-merger features
            with torch.no_grad():
                image_embeds_list, _ = vl_gpt.model.get_image_features(
                    pixel_values,
                    image_grid_thw,
                )
            # 2. DINOv3 last-layer patch features (frozen)
            with torch.no_grad():
                dino_outputs = dinov3_model(pixel_values=dinov3_pixel_values, return_dict=True)
            dino_all = dino_outputs.last_hidden_state
            num_prefix = 1 + getattr(dinov3_model.config, 'num_register_tokens', 4)
            dino_patch = dino_all[:, num_prefix:, :]  # [1, N_patch, dino_dim]
            dino_dim = dino_patch.shape[-1]

            # 3. 插值对齐 + concat + project
            fused_list = []
            _sms = vl_gpt.model.visual.spatial_merge_size  # =2
            for i, qwen_feats in enumerate(image_embeds_list):
                thw = image_grid_thw[i]
                H_post, W_post = int(thw[1].item()) // _sms, int(thw[2].item()) // _sms
                N_i = qwen_feats.shape[0]

                dino_i = dino_patch[i]  # [N_patch, dino_dim]
                N_dino = dino_i.shape[0]
                dino_h, dino_w = require_square_token_grid(
                    N_dino,
                    "DINOv3 eval concat patch",
                )

                dino_2d = dino_i.reshape(dino_h, dino_w, dino_dim).permute(2, 0, 1).unsqueeze(0)
                if H_post != dino_h or W_post != dino_w:
                    dino_aligned = F.interpolate(
                        dino_2d.float(), size=(H_post, W_post), mode='bilinear', align_corners=False
                    ).to(dino_i.dtype)
                else:
                    dino_aligned = dino_2d.to(dino_i.dtype)
                dino_flat = dino_aligned.squeeze(0).permute(1, 2, 0).reshape(N_i, dino_dim)

                fused = torch.cat([qwen_feats.to(device), dino_flat.to(device)], dim=-1)
                fused_projected = vl_gpt.dinov3_concat_projector(fused)  # [N_i, llm_dim]
                fused_list.append(fused_projected)

            fused_embeds = torch.cat(fused_list, dim=0)  # [N_total, llm_dim]

            # 4. scatter fused features 进 inputs_embeds
            img_mask, _ = vl_gpt.model.get_placeholder_mask(
                full_input_ids,
                inputs_embeds=inputs_embeds,
                image_features=fused_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(img_mask, fused_embeds)

            # 5. Forward（不传 pixel_values，features 已 scatter）
            model_kwargs["inputs_embeds"] = inputs_embeds
            model_kwargs["pixel_values"] = None
            model_kwargs["dinov3_qkv_states"] = None
            model_kwargs["pruning_mode"] = "none"
            model_kwargs["pruning_ref_layer"] = -1
            model_kwargs["pruning_ratio"] = 0.0

        # ========== SigLIP MOT 分支：注入 Qwen ViT QKV ==========
        elif getattr(args, 'use_siglip_mot', 0) and getattr(vl_gpt, 'siglip_aligner', None) is not None:
            siglip_map = args.siglip_qwen_mapping  # {qwen_layer: vit_layer}
            _needed_vit_layers = set(siglip_map.values())
            _siglip_qkv_collectors = {layer: {} for layer in _needed_vit_layers}

            # 1. 调用 Qwen ViT with collectors（no_grad: ViT 推理时固定）
            _pv_typed = pixel_values.type(vl_gpt.model.visual.dtype)
            with torch.no_grad():
                _vit_features, _ = vl_gpt.model.visual(
                    _pv_typed,
                    grid_thw=image_grid_thw,
                    siglip_qkv_collectors=_siglip_qkv_collectors,
                )

            # 2. scatter ViT features 进 inputs_embeds
            _sms = vl_gpt.model.visual.spatial_merge_size
            _split_sizes = (image_grid_thw.prod(-1) // _sms ** 2).tolist()
            _img_embeds_cat = torch.cat(
                list(torch.split(_vit_features, [int(s) for s in _split_sizes])), dim=0)
            _img_mask, _ = vl_gpt.model.get_placeholder_mask(
                full_input_ids,
                inputs_embeds=inputs_embeds,
                image_features=_img_embeds_cat,
            )
            inputs_embeds = inputs_embeds.masked_scatter(_img_mask, _img_embeds_cat)

            # 3. reshape QKV: [H, total_patches, D] → [B, H, patches_per_img, D]
            _siglip_layer_qkv = {}
            for _vit_layer, _collector in _siglip_qkv_collectors.items():
                if 'query' not in _collector:
                    continue
                _cu = _collector['cu_seqlens']
                _n_imgs = len(_cu) - 1
                _patches = int((_cu[1] - _cu[0]).item())
                _q = _collector['query']
                _k = _collector['key']
                _v = _collector['value']
                _, H, D = _q.shape
                _siglip_layer_qkv[_vit_layer] = {
                    'query': _q.view(_n_imgs, _patches, H, D).permute(0, 2, 1, 3),
                    'key':   _k.view(_n_imgs, _patches, H, D).permute(0, 2, 1, 3),
                    'value': _v.view(_n_imgs, _patches, H, D).permute(0, 2, 1, 3),
                }

            # 4. apply siglip_aligner
            with torch.no_grad():
                _siglip_qkv_states = vl_gpt.siglip_aligner.to(device)(_siglip_layer_qkv)

            _siglip_contract = resolve_siglip_token_contract(args.siglip_token_contract)
            model_kwargs["inputs_embeds"] = inputs_embeds
            model_kwargs["pixel_values"] = None
            model_kwargs["dinov3_qkv_states"] = _siglip_qkv_states
            model_kwargs["pruning_mode"] = getattr(args, 'pruning_mode', 'none')
            model_kwargs["pruning_ref_layer"] = getattr(args, 'pruning_ref_layer', -1)
            model_kwargs["pruning_ratio"] = getattr(args, 'pruning_ratio', 0.0)
            model_kwargs["dinov3_num_register_tokens"] = 0
            model_kwargs["expert_num_prefix_tokens"] = _siglip_contract["expert_num_prefix_tokens"]
            model_kwargs["expert_patch_grid_contract"] = _siglip_contract["expert_patch_grid_contract"]
            # 传入实际 visual token 位置，避免 pruning 路径使用硬编码 fallback=576
            model_kwargs["visual_pos_masks"] = (
                full_input_ids == vl_gpt.model.config.image_token_id
            )
            model_kwargs["pruning_attn_aggregation"] = getattr(args, 'pruning_attn_aggregation', 'none')
            model_kwargs["pruning_attn_aggregation_layer_start"] = getattr(args, 'pruning_attn_aggregation_layer_start', 3)
            model_kwargs["pruning_attn_aggregation_layer_end"] = getattr(args, 'pruning_attn_aggregation_layer_end', 10)
            model_kwargs["pruning_attn_aggregation_vote_ratio"] = getattr(args, 'pruning_attn_aggregation_vote_ratio', 0.6)
            model_kwargs["pruning_vote_contract"] = getattr(
                args,
                'pruning_vote_contract',
                'historical_floor_global_fallback',
            )

        # ========== Visual Token Concat 分支：token-level ablation ==========
        elif getattr(args, 'use_visual_token_concat', 0) and getattr(vl_gpt, 'visual_token_aligner', None) is not None:
            _tc_mapping    = getattr(args, 'siglip_qwen_mapping', {}) or getattr(args, 'dinov3_qwen_mapping', {})
            _use_siglip_src = bool(getattr(args, 'siglip_qwen_mapping', {}))

            if _use_siglip_src:
                _needed_vit_layers = set(_tc_mapping.values())
                _visual_tok_collectors = {layer: {} for layer in _needed_vit_layers}

                _pv_typed = pixel_values.type(vl_gpt.model.visual.dtype)
                with torch.no_grad():
                    _vit_features, _ = vl_gpt.model.visual(
                        _pv_typed,
                        grid_thw=image_grid_thw,
                        visual_token_collectors=_visual_tok_collectors,
                    )

                # Scatter ViT features into inputs_embeds
                _sms        = vl_gpt.model.visual.spatial_merge_size
                _split_sizes = (image_grid_thw.prod(-1) // _sms ** 2).tolist()
                _img_embeds  = torch.cat(list(torch.split(_vit_features, [int(s) for s in _split_sizes])), dim=0)
                _img_mask, _ = vl_gpt.model.get_placeholder_mask(
                    full_input_ids, inputs_embeds=inputs_embeds, image_features=_img_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(_img_mask, _img_embeds)

                # Reshape [total_patches, hidden_dim] → [B, patches_per_img, hidden_dim]
                _visual_raw_tokens = {}
                for _vit_layer, _col in _visual_tok_collectors.items():
                    if 'hidden' not in _col:
                        continue
                    _cu      = _col['cu_seqlens']
                    _n_imgs  = len(_cu) - 1
                    _patches = int((_cu[1] - _cu[0]).item())
                    _h       = _col['hidden']
                    _visual_raw_tokens[_vit_layer] = _h.view(_n_imgs, _patches, _h.shape[-1])

                _visual_token_states = vl_gpt.visual_token_aligner.to(device)(_visual_raw_tokens)

                model_kwargs["inputs_embeds"]       = inputs_embeds
                model_kwargs["pixel_values"]        = None
                model_kwargs["visual_token_states"] = _visual_token_states
                model_kwargs["pruning_mode"]  = getattr(args, 'pruning_mode', 'none')
                model_kwargs["pruning_ref_layer"] = getattr(args, 'pruning_ref_layer', -1)
                model_kwargs["pruning_ratio"] = getattr(args, 'pruning_ratio', 0.0)
            else:
                # ── DINOv3 block outputs ──────────────────────────────────────
                # Forward Qwen ViT normally to get LLM visual tokens
                _pv_typed = pixel_values.type(vl_gpt.model.visual.dtype)
                with torch.no_grad():
                    _vit_features, _ = vl_gpt.model.visual(
                        _pv_typed,
                        grid_thw=image_grid_thw,
                    )
                _sms        = vl_gpt.model.visual.spatial_merge_size
                _split_sizes = (image_grid_thw.prod(-1) // _sms ** 2).tolist()
                _img_embeds  = torch.cat(list(torch.split(_vit_features, [int(s) for s in _split_sizes])), dim=0)
                _img_mask, _ = vl_gpt.model.get_placeholder_mask(
                    full_input_ids, inputs_embeds=inputs_embeds, image_features=_img_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(_img_mask, _img_embeds)

                # Forward DINOv3 with output_hidden_states=True
                _needed_dino_layers = set(_tc_mapping.values())
                dinov3_inputs = dinov3_processor(images=[dino_pil], return_tensors="pt")
                dinov3_pixel_values = dinov3_inputs["pixel_values"].to(device).to(torch.bfloat16)
                dinov3_pixel_values = _maybe_ablate_dino(dinov3_pixel_values)
                with torch.no_grad():
                    _dino_outputs = dinov3_model(
                        pixel_values=dinov3_pixel_values,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                _dino_hidden_states = _dino_outputs.hidden_states
                _dino_config = dinov3_model.config
                _num_prefix = 1 + getattr(_dino_config, 'num_register_tokens', 4)

                _visual_raw_tokens = {}
                for _dino_layer in _needed_dino_layers:
                    _layer_hidden = _dino_hidden_states[_dino_layer + 1]
                    _patch_hidden = _layer_hidden[:, _num_prefix:, :]
                    _visual_raw_tokens[_dino_layer] = _patch_hidden

                _visual_token_states = vl_gpt.visual_token_aligner.to(device)(_visual_raw_tokens)

                model_kwargs["inputs_embeds"]       = inputs_embeds
                model_kwargs["pixel_values"]        = None
                model_kwargs["visual_token_states"] = _visual_token_states
                model_kwargs["pruning_mode"]        = getattr(args, 'pruning_mode', 'none')
                model_kwargs["pruning_ref_layer"]   = getattr(args, 'pruning_ref_layer', -1)
                model_kwargs["pruning_ratio"]       = getattr(args, 'pruning_ratio', 0.0)

        # ========== DINOv3 MOT/Efficient 分支：与训练一致 ==========
        elif dinov3_model is not None and dinov3_processor is not None and dinov3_qwen_mapping:
            dinov3_inputs = dinov3_processor(images=[dino_pil], return_tensors="pt")
            dinov3_pixel_values = dinov3_inputs["pixel_values"].to(device).to(torch.bfloat16)
            dinov3_pixel_values = _maybe_ablate_dino(dinov3_pixel_values)
            qkv_after_rope = bool(getattr(args, "dinov3_qkv_after_rope", 1))

            dinov3_efficient = bool(getattr(args, 'dinov3_efficient', 0))
            pruning_mode = getattr(args, 'pruning_mode', 'none')
            pruning_ratio = getattr(args, 'pruning_ratio', 0.0)

            if dinov3_efficient and pruning_mode in ['qwen_action', 'qwen_task'] and pruning_ratio > 0:
                # Efficient 模式：不预先计算 QKV，将组件传入模型内部处理
                model_kwargs["dinov3_qkv_states"] = None
                model_kwargs["dinov3_qwen_mapping"] = {
                    int(k) if isinstance(k, str) else k: v for k, v in dinov3_qwen_mapping.items()
                }
                model_kwargs["dinov3_model"] = dinov3_model
                model_kwargs["dinov3_aligner"] = getattr(vl_gpt, 'dinov3_aligner', None)
                model_kwargs["dinov3_pixel_values"] = dinov3_pixel_values
                model_kwargs["dinov3_efficient"] = True
                model_kwargs["dinov3_qkv_after_rope"] = qkv_after_rope
                model_kwargs["train_dinov3_full_finetune"] = False
                num_register_tokens = 4
                if hasattr(dinov3_model, "config") and getattr(dinov3_model.config, "num_register_tokens", None) is not None:
                    num_register_tokens = dinov3_model.config.num_register_tokens
                model_kwargs["dinov3_num_register_tokens"] = num_register_tokens
                model_kwargs["expert_num_prefix_tokens"] = 1 + num_register_tokens
                model_kwargs["pruning_mode"] = pruning_mode
                model_kwargs["pruning_ref_layer"] = getattr(args, 'pruning_ref_layer', -1)
                model_kwargs["pruning_ratio"] = pruning_ratio
                model_kwargs["pruning_attn_aggregation"] = getattr(args, 'pruning_attn_aggregation', 'none')
                model_kwargs["pruning_attn_aggregation_layer_start"] = getattr(args, 'pruning_attn_aggregation_layer_start', 3)
                model_kwargs["pruning_attn_aggregation_layer_end"] = getattr(args, 'pruning_attn_aggregation_layer_end', 10)
                model_kwargs["pruning_attn_aggregation_vote_ratio"] = getattr(args, 'pruning_attn_aggregation_vote_ratio', 0.6)
                model_kwargs["pruning_vote_contract"] = getattr(
                    args,
                    'pruning_vote_contract',
                    'historical_floor_global_fallback',
                )
            else:
                # Normal 模式：预先计算 QKV
                with torch.no_grad():
                    dinov3_outputs = dinov3_model(
                        pixel_values=dinov3_pixel_values,
                        output_qkv=True,
                        qkv_after_rope=qkv_after_rope,
                        return_dict=True,
                    )
                all_qkv_states = dinov3_outputs.qkv_states
                needed_dinov3_layers = set(dinov3_qwen_mapping.values())
                num_register_tokens = 4
                if hasattr(dinov3_model, "config") and getattr(dinov3_model.config, "num_register_tokens", None) is not None:
                    num_register_tokens = dinov3_model.config.num_register_tokens
                if pruning_mode == 'dinov3_cls' and pruning_ratio > 0:
                    all_qkv_states = _prune_dinov3_qkv_by_cls_similarity(
                        all_qkv_states,
                        dinov3_outputs.last_hidden_state,
                        pruning_ratio,
                        needed_dinov3_layers,
                        num_register_tokens,
                    )
                dinov3_layer_qkv = {i: qkv for i, qkv in enumerate(all_qkv_states) if i in needed_dinov3_layers}
                if getattr(vl_gpt, 'dinov3_aligner', None) is not None:
                    vl_gpt.dinov3_aligner = vl_gpt.dinov3_aligner.to(device)
                    dinov3_qkv_states = vl_gpt.dinov3_aligner(dinov3_layer_qkv)
                else:
                    dinov3_qkv_states = {}
                    for qwen_layer, dinov3_layer in dinov3_qwen_mapping.items():
                        qwen_layer_int = int(qwen_layer) if isinstance(qwen_layer, str) else qwen_layer
                        if dinov3_layer in dinov3_layer_qkv:
                            dinov3_qkv_states[qwen_layer_int] = dinov3_layer_qkv[dinov3_layer]
                model_kwargs["dinov3_qkv_states"] = dinov3_qkv_states
                model_kwargs["dinov3_qwen_mapping"] = {
                    int(k) if isinstance(k, str) else k: v for k, v in dinov3_qwen_mapping.items()
                }
                model_kwargs["dinov3_num_register_tokens"] = num_register_tokens
                model_kwargs["expert_num_prefix_tokens"] = 1 + num_register_tokens

                # Normal 模式同样要传剪枝参数：训练侧 Normal 分支
                # (train_no_gen_encoder_rlbench.py 的 model.model(... pruning_mode=...) )
                # 传了这 7 个参数，而模型 forward 默认 pruning_mode='none'/aggregation='none'。
                # 此处不传 => 剪枝训练出的 ckpt 在评测时被喂全量 DINOv3 token，且 aggregation 退回
                # 单层 ref_layer —— 静默的 train/eval 错配，成绩被系统性低估。
                if pruning_mode in ['qwen_action', 'qwen_task'] and pruning_ratio > 0:
                    model_kwargs["pruning_mode"] = pruning_mode
                    model_kwargs["pruning_ref_layer"] = getattr(args, 'pruning_ref_layer', -1)
                    model_kwargs["pruning_ratio"] = pruning_ratio
                    model_kwargs["pruning_attn_aggregation"] = getattr(args, 'pruning_attn_aggregation', 'none')
                    model_kwargs["pruning_attn_aggregation_layer_start"] = getattr(args, 'pruning_attn_aggregation_layer_start', 3)
                    model_kwargs["pruning_attn_aggregation_layer_end"] = getattr(args, 'pruning_attn_aggregation_layer_end', 10)
                    model_kwargs["pruning_attn_aggregation_vote_ratio"] = getattr(args, 'pruning_attn_aggregation_vote_ratio', 0.6)
                    model_kwargs["pruning_vote_contract"] = getattr(
                        args,
                        'pruning_vote_contract',
                        'historical_floor_global_fallback',
                    )
                elif pruning_mode == 'dinov3_cls' and pruning_ratio > 0:
                    model_kwargs["pruning_mode"] = "none"
                    model_kwargs["pruning_ratio"] = 0.0

        outputs = vl_gpt.model(**model_kwargs)
        hidden_states = outputs.last_hidden_state

        actions_hidden_states = hidden_states[:, -action_query_count:, :]
        
        # Predict continuous actions using action head
        predicted_actions = action_head.predict_action(actions_hidden_states)  # (batch_size, action_chunk, action_dim)
        # bfloat16 无法直接转 numpy，这里先转成 float32 再转 numpy
        normalized_actions = predicted_actions.to(torch.float32).cpu().numpy()
        
        if normalized_actions.shape[0] != 1:
            raise ValueError(
                "RLBench evaluator expects batch size 1, got "
                f"{normalized_actions.shape[0]}"
            )
        normalized_actions = normalized_actions[0]
        if not np.isfinite(normalized_actions).all():
            raise ValueError("Predicted normalized actions contain non-finite values")

        actions = denormalize_action_chunks(
            normalized_actions,
            statistic,
            action_chunk=args.action_chunk,
            action_dim=args.action_dim,
        )
        if not np.isfinite(actions).all():
            raise ValueError("Predicted denormalized actions contain non-finite values")
        return actions


def main(args):
    if args.replay_or_predict == 'replay' and getattr(args, 'action_representation', 'xyz_delta_euler_absolute_v1') != 'xyz_delta_euler_absolute_v1':
        raise ValueError("Relative Euler checkpoints support predict/dataset_predict; legacy replay uses an absolute-action dataset")
    args = _resolve_rlbench_profile_runtime(args)
    # Report the arguments
    Logger.log_info(f'Running {colored(__file__, "red")} with arguments:')
    Logger.log_info(f'task name: {args.task_name}')
    Logger.log_info(f'number of episodes: {args.num_episodes}')
    Logger.log_info(f'eval seed: {args.eval_seed}'
                    + (' (未固定：每次 reset 重新随机，跨臂比较非配对)' if args.eval_seed < 0 else ''))
    Logger.log_info(
        f"rlbench profile: requested={args._rlbench_profile_resolution['requested_profile']} "
        f"resolved={args._rlbench_profile_resolution['resolved_profile']} "
        f"source={args._rlbench_profile_resolution['resolution_source']}"
    )
    Logger.log_info(
        f"task base yaw offset: {args.task_base_yaw_offset} rad "
        f"(profile={args._rlbench_runtime['profile_task_base_yaw_offset']}, "
        f"explicit={args._rlbench_runtime['explicit_task_base_yaw_offset']})"
    )
    Logger.log_info(f'result directory: {args.result_dir}')
    Logger.log_info(f'exp name: {args.exp_name}')
    Logger.log_info(f'actions chunk: {args.action_chunk}')
    Logger.log_info(f'replay or predict: {args.replay_or_predict}')
    Logger.log_info(f'max steps: {args.max_steps}')
    Logger.log_info(f'cuda used: {args.cuda}')
    # cprint('-' * os.get_terminal_size().columns, 'cyan')

    if args.replay_or_predict == 'dataset_predict':
        run_dataset_predict(args)
        return

    from lift3d.envs.rlbench_env import RLBenchEnv, RLBenchActionMode, RLBenchObservationConfig
    from lift3d.helpers.gymnasium import VideoWrapper
    from lift3d.helpers.graphics import EEpose
    from rlbench.backend.exceptions import TaskEnvironmentError

    model_image_size = args.eval_camera_resolution
    Logger.log_info(
        f"[环境配置] RLBench 相机分辨率: {model_image_size}x{model_image_size} "
        f"(profile={args._rlbench_runtime['profile_camera_resolution']}, "
        f"override={args._rlbench_runtime['camera_resolution_override']})"
    )

    action_mode = RLBenchActionMode.eepose_then_gripper_action_mode(absolute=True)
    obs_config = RLBenchObservationConfig.single_view_config(camera_name='front', image_size=(model_image_size, model_image_size))
    # obs_config.front_camera.rgb = True
    # obs_config.front_camera.depth = False
    # obs_config.front_camera.point_cloud = False
    # obs_config.front_camera.mask = False
    env = RLBenchEnv(
        task_name=args.task_name,
        action_mode=action_mode,
        obs_config=obs_config,
        point_cloud_camera_names=['front'],
        # cinematic_record_enabled=True,
        cinematic_record_enabled=False,
        num_points=1024,
        use_point_crop=True,
        task_base_yaw_offset=args.task_base_yaw_offset,
    )
    env = VideoWrapper(env)
    env_closed = False

    def close_rlbench_env():
        nonlocal env_closed
        if env_closed:
            return
        env.close()
        env_closed = True

    atexit.register(close_rlbench_env)
    if args.eval_demo_root:
        _set_env_variation(env, args.eval_variation)
        args._eval_demo_episode_map = _load_eval_demo_map(args)
        Logger.log_info(
            f"[环境配置] 使用 demo-backed reset: root={args.eval_demo_root}, "
            f"variation={args.eval_variation}, offset={args.eval_episode_offset}, "
            f"episodes={args.eval_episodes or 'range(num_episodes)'}, "
            f"map={args.eval_demo_map or 'identity'}"
        )
        if args._eval_demo_episode_map:
            Logger.log_info(
                f"[环境配置] demo map loaded: task={args.task_name}, "
                f"entries={len(args._eval_demo_episode_map)}"
            )

    if args.replay_or_predict == 'predict':
        args.result_dir = os.path.join(args.result_dir, 'predict_results')
    elif args.replay_or_predict == 'replay':
        args.result_dir = os.path.join(args.result_dir, 'replay_results')
    
    if args.exp_name is None:
        args.exp_name = args.task_name

    video_dir = os.path.join(
        args.result_dir, args.task_name, args.exp_name, "videos"
    )
    recreate_directory(video_dir)
    
    log_dir = os.path.join(
        os.path.join(
            args.result_dir, args.task_name, args.exp_name
        ),
        f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    recreate_directory(log_dir)
    logger = setup_logger(log_dir)
    _write_eval_metadata(
        args,
        os.path.join(args.result_dir, args.task_name, args.exp_name),
        model_image_size,
    )

    success_num = 0
    # #----------- for model predict
    if args.replay_or_predict == 'predict':
        model_components = model_load(args)
        vl_gpt = model_components['vl_gpt']
        processor = model_components['processor']
        action_tokenizer = model_components['action_tokenizer']
        action_head = model_components['action_head']
        statistic = model_components['statistic']
        dinov3_model = model_components['dinov3_model']
        dinov3_processor = model_components['dinov3_processor']
        dinov3_qwen_mapping = model_components['dinov3_qwen_mapping']
        
        Logger.log_info("Model loaded. Using Qwen3VL API")
        if dinov3_model is not None:
            Logger.log_info(f"DINOv3 enabled with {len(dinov3_qwen_mapping)} layer mappings")
        
        episode_length = args.max_steps

    selected_episode_ids = _episode_ids(args)
    if args.eval_episodes:
        args.num_episodes = len(selected_episode_ids)

    skipped_episodes = 0
    rollout_records = []
    infrastructure_abort = False
    for i, episode_id in enumerate(selected_episode_ids):

        pre_image_dir = os.path.join(
            args.result_dir, args.task_name, args.exp_name, "pre_image", f"episode{episode_id}"
        )
        recreate_directory(pre_image_dir)

        #----------- for key frames replay
        if args.replay_or_predict == 'replay':
            dat = np.load(os.path.join(args.replay_data_dir, args.task_name, f'episode{episode_id}.npy'), allow_pickle=True)
            task_description = dat[0]['language_instruction']
            episode_length = len(dat)

        logger.info(f'episode: {episode_id}, steps: {episode_length}')
        # 逐 episode 独立播种：场景 = f(eval_seed, i)，与"前面跑了哪些 episode"无关。
        # 只在开头播一次是脆的——各臂的 reset 重试次数不同会让后续 episode 整体错位。
        _ep_seed = None if args.eval_seed < 0 else args.eval_seed + i * 1009
        # RLBench 随机放置物体时可能触发 BoundaryError（如 close_fridge），重试 reset 提高稳定性
        MAX_RESET_RETRIES = 5
        obs_dict = None
        reset_attempts = 0
        reset_seeds_attempted = []
        successful_reset_seed = None
        actual_reset_seed = None
        reset_seed_trace = []
        reset_errors = []
        if args.eval_demo_root:
            reset_attempts = 1
            try:
                demo, demo_path, raw_episode_id = _load_demo(args, episode_id)
                obs_dict = _reset_env_to_demo(env, demo)
                logger.info(
                    "reset_to_demo json_episode%d raw_episode%d from %s",
                    episode_id,
                    raw_episode_id,
                    demo_path,
                )
            except Exception as e:
                logger.warning("reset_to_demo 失败 episode%d: %s", episode_id, str(e))
                logger.error("跳过本 episode (episode %d)", episode_id)
                reset_errors.append(_rollout_exception_error(e, -1, "reset_to_demo"))
                infrastructure_abort = True
        else:
            for _reset_attempt in range(MAX_RESET_RETRIES):
                reset_attempts = _reset_attempt + 1
                reset_seed = (
                    None
                    if _ep_seed is None
                    else _ep_seed + _reset_attempt * 7919
                )
                reset_seeds_attempted.append(reset_seed)
                try:
                    # 重试错开 seed，理由同 lift3d/envs/rlbench_env.py:reset
                    obs_dict = env.reset(seed=reset_seed)
                    successful_reset_seed = reset_seed
                    break
                except TaskEnvironmentError as e:
                    logger.warning(
                        "env.reset() 放置任务失败 (attempt %d/%d): %s，重试...",
                        _reset_attempt + 1, MAX_RESET_RETRIES, str(e),
                    )
                    if _reset_attempt == MAX_RESET_RETRIES - 1:
                        logger.error("env.reset() 达到最大重试次数，跳过本 episode (episode %d)", episode_id)
                except Exception as e:
                    reset_errors.append(_rollout_exception_error(e, -1, "env.reset"))
                    infrastructure_abort = True
                    break
        if obs_dict is None:
            skipped_episodes += 1
            rollout_records.append(
                {
                    "episode_index": i,
                    "episode_id": int(episode_id),
                    "episode_seed": _ep_seed,
                    "reset_status": "failed",
                    "reset_attempts": reset_attempts,
                    "reset_seeds_attempted": reset_seeds_attempted,
                    "successful_reset_seed": successful_reset_seed,
                    "actual_reset_seed": actual_reset_seed,
                    "reset_seed_trace": reset_seed_trace,
                    "reset_errors": reset_errors,
                    "step_errors": [],
                    "termination_reason": "infrastructure_error" if infrastructure_abort else "reset_failed",
                    "success": False,
                    "steps_executed": 0,
                }
            )
            if infrastructure_abort:
                break
            continue
        actual_reset_seed = obs_dict.get("actual_reset_seed", successful_reset_seed)
        reset_seed_trace = list(obs_dict.get("reset_seed_trace", []))
        terminated = False
        success = False
        gripper_open = None
        steps_executed = 0
        step_errors = []
        termination_reason = "max_steps"

        for j in range(episode_length):
            
            # #--------- for key frames replay
            if args.replay_or_predict == 'replay':
                action = dat[j]['action']
                robo_state = dat[j]['state']

                
                sum_first_3_rows = np.sum(action.reshape(8, 7)[:, :3], axis=0)
                last_row_last_4 = action[-4:]
                action = np.concatenate([sum_first_3_rows, last_row_last_4])

                # print(action[3:6],robo_state[3:6])
                # action[3:6] = unique_euler_xyz_rad(action[3:6])
                # robo_state[3:6] = unique_euler_xyz_rad(robo_state[3:6])
                # print(action[3:6],robo_state[3:6])


                action[:3] += robo_state[:3]
                gripper_open = action[-1]
                action = EEpose.pose_6DoF_to_7DoF(action[:-1])
                action = np.append(action, gripper_open)
                print(j, "  :", action)
                obs_dict, reward, terminated, truncated, info = env.step(action)
                steps_executed = j + 1
                success = success or bool(reward)

                
            # # #----------- for model predict
            if args.replay_or_predict == 'predict':
                stage = "observation_preparation"
                try:
                    image = [Image.fromarray(obs_dict['image'])]
                    task_description = _task_description_from_env(args.task_name, env.text)
                    robot_state = EEpose.pose_7DoF_to_6DoF(obs_dict['robot_state'][7:14])
                    robot_state = np.concatenate([robot_state, np.array([gripper_open if gripper_open is not None else 1])])
                    cur_robot_state = robot_state if args.use_robot_state else None
                    point_cloud = obs_dict['point_cloud'] if args.load_pointcloud else None
                    stage = "model_predict"
                    actions = model_predict(
                        args, vl_gpt, processor, action_tokenizer, action_head,
                        statistic, task_description, image, cur_robot_state,
                        point_cloud, pre_image_dir, step=j,
                        dinov3_model=dinov3_model,
                        dinov3_processor=dinov3_processor,
                        dinov3_qwen_mapping=dinov3_qwen_mapping,
                    )
                    stage = "action_decode"
                    predicted_chunks = _reshape_action_chunks(actions, args, "Predicted rollout")
                    action = np.asarray(select_action_for_execution(
                        predicted_chunks, args.chunk_execution_contract,
                    ), dtype=np.float32)
                    action = decode_action_for_execution(action, robot_state, args.action_representation)
                    if not np.isfinite(action).all():
                        raise ValueError("Predicted action contains non-finite values")
                    # Both representations retain xyz deltas; relative Euler has
                    # already been restored using the current measured pose.
                    action[:3] += obs_dict['robot_state'][7:10]
                    gripper_open = action[-1]
                    action = EEpose.pose_6DoF_to_7DoF(action[:-1])
                    action = np.append(action, gripper_open)
                    logger.info(f"task description: {task_description}")
                    logger.info("%d  : %s", j, action)
                    stage = "env.step"
                    obs_dict, reward, terminated, truncated, info = env.step(action)
                except Exception as e:
                    step_errors.append(_rollout_exception_error(e, j, stage))
                    termination_reason = "infrastructure_error"
                    infrastructure_abort = True
                    success = False
                    logger.exception("Infrastructure failure at episode %s step %s stage %s", episode_id, j, stage)
                    break
                steps_executed = j + 1
                step_error = _rollout_step_error(info, j)
                if step_error is not None:
                    step_errors.append(step_error)
                    termination_reason = step_error["error_category"] + "_error"
                    infrastructure_abort = step_error["error_category"] == "infrastructure"
                    success = False
                    break
                success = success or bool(reward)
                if terminated or truncated or success:
                    termination_reason = "success" if success else "terminated" if terminated else "truncated"
                    break
                
        if success:
            success_num += 1

        image_dir = os.path.join(
            args.result_dir, args.task_name, args.exp_name, "images", f"episode{episode_id}"
        )
        recreate_directory(image_dir)

        try:
            env.save_video(os.path.join(video_dir, f'episode{episode_id}_video_steps.mp4'))
            env.save_images(image_dir, quiet=True)
        except Exception as e:
            step_errors.append(_rollout_exception_error(e, steps_executed, "save_episode_artifacts"))
            termination_reason = "infrastructure_error"
            infrastructure_abort = True
        logger.info(f'episode{episode_id}_{success}')
        rollout_records.append(
            {
                "episode_index": i,
                "episode_id": int(episode_id),
                "episode_seed": _ep_seed,
                "reset_status": "completed",
                "reset_attempts": reset_attempts,
                "reset_seeds_attempted": reset_seeds_attempted,
                "successful_reset_seed": successful_reset_seed,
                "actual_reset_seed": actual_reset_seed,
                "reset_seed_trace": reset_seed_trace,
                "reset_errors": reset_errors,
                "step_errors": step_errors,
                "termination_reason": termination_reason,
                "success": bool(success),
                "steps_executed": int(steps_executed),
            }
        )
        Logger.print_seperator()
        if infrastructure_abort:
            break
    
    completed_episodes = sum(row["reset_status"] == "completed" for row in rollout_records)
    # Claim-grade RLBench scores must use the requested episode count as the
    # denominator. Reset-placement skips are environment failures for the run,
    # not successful exclusions; otherwise two arms can get different effective
    # denominators and incomparable success rates.
    success_rate_pct = success_num / args.num_episodes * 100 if args.num_episodes > 0 else 0.0
    if skipped_episodes > 0:
        logger.info(f'Finished. {args.task_name} * {args.num_episodes}. Completed {completed_episodes}, skipped {skipped_episodes} (placement failure; counted as failures). Success rate {success_rate_pct}%')
    else:
        logger.info(f'Finished. {args.task_name} * {args.num_episodes}. Success rate {success_rate_pct}%')
    with open(os.path.join(args.result_dir, args.task_name, f'{args.exp_name}_success_rate.txt'), "w", encoding="utf-8") as file:
        file.write(f'Finished. {args.task_name} * {args.num_episodes}. Completed {completed_episodes}, skipped {skipped_episodes}. Success rate {success_rate_pct}%\n')
    rollout_summary = {
        "schema_version": 2,
        "task_name": args.task_name,
        "requested_episodes": args.num_episodes,
        "completed_episodes": completed_episodes,
        "skipped_episodes": skipped_episodes,
        "unattempted_episodes": args.num_episodes - len(rollout_records),
        "status": "FAIL" if infrastructure_abort else "COMPLETE",
        **_rollout_error_counts(rollout_records),
        "action_representation": args.action_representation,
        "action_normalization_contract": args.action_normalization_contract,
        "successes": success_num,
        "success_rate_pct": success_rate_pct,
        "eval_seed": args.eval_seed,
        "max_steps": args.max_steps,
        "episode_records": rollout_records,
    }
    _write_json_atomic(
        os.path.join(
            args.result_dir,
            args.task_name,
            args.exp_name,
            "rollout_summary.json",
        ),
        rollout_summary,
    )
    close_rlbench_env()
    atexit.unregister(close_rlbench_env)
    if infrastructure_abort:
        raise RuntimeError("RLBench evaluation stopped after an infrastructure error; see rollout_summary.json")


if __name__ == '__main__':
    """
    测试脚本参数说明：
    - 最小化超参数输入，模型配置严格从 training_config.json 加载
    - 只保留测试相关的必要参数（任务、路径、设备等）
    - 参考设计：deepvision-vla/scripts_archive/scripts/test_rlbench.py
    """
    parser = argparse.ArgumentParser(description='RLBench 测试脚本（最小化参数设计）')
    
    # ========== 测试基础参数（必需）==========
    parser.add_argument('--task-name', type=str, default='close_box',
                        help='RLBench 任务名称')
    parser.add_argument('--dataset-name', type=str, default='rlbench',
                        help='数据集名称')
    parser.add_argument('--model-path', type=str, default='',
                        help='模型路径（必须包含 training_config.json）')
    
    # ========== 测试行为参数 ==========
    parser.add_argument('--replay-or-predict', type=str, default='predict',
                        choices=['validate', 'predict', 'replay', 'dataset_predict'],
                        help='测试模式：predict (仿真推理), dataset_predict (JSON/PNG 行级动作复现), replay (重放)')
    parser.add_argument('--num-episodes', type=int, default=20,
                        help='每个任务的测试轮数')
    parser.add_argument('--max-steps', type=int, default=10,
                        help='每轮最大步数')
    parser.add_argument('--eval-seed', type=int, default=42,
                        help='评测场景 seed。episode i 的物体摆放 = f(eval_seed, i)，'
                             '与 rollout 历史无关，故各 ckpt 看到同一批场景 -> 跨臂比较是配对的，'
                             '场景难度差不再进入两臂之差。传 -1 恢复旧的每次 reset 重新随机。')
    parser.add_argument('--rlbench-env-profile', type=str,
                        default=os.environ.get('RLBENCH_ENV_PROFILE', 'auto'),
                        help='RLBench data/simulator profile id, or auto to resolve from checkpoint provenance.')
    parser.add_argument('--rlbench-profile-path', type=str,
                        default=os.environ.get('RLBENCH_PROFILE_PATH', ''),
                        help='Optional path to a versioned RLBench profile registry JSON.')
    parser.add_argument('--allow-unknown-rlbench-profile', type=int, default=0,
                        help='Debug only: allow a checkpoint whose RLBench protocol cannot be identified.')
    parser.add_argument('--allow-rlbench-profile-mismatch', type=int, default=0,
                        help='Debug only: allow an explicit profile that conflicts with checkpoint fingerprints.')
    parser.add_argument('--eval-camera-resolution', type=int, default=0,
                        help='Final simulator camera override. 0 uses the resolved profile; EVAL_CAM_RES remains supported.')
    parser.add_argument(
        '--eval-image-protocol',
        type=str,
        choices=sorted(EVAL_IMAGE_PROTOCOLS),
        default='matched',
        help=(
            'matched follows the checkpoint augmentation scope; paper_bug reproduces '
            'the historical no-crop/Qwen-resized-DINO evaluator; split_legacy applies '
            'the deterministic crop only to Qwen and keeps DINO raw; matched_no_crop '
            'isolates the effect of omitting the deterministic test crop.'
        ),
    )
    parser.add_argument('--task-base-yaw-offset', type=float, default=None,
                        help='在 RLBench task base_rotation_bounds 的 z 轴上下界上增加的弧度。'
                             '默认由 profile 决定；显式值是最终覆盖。')

    # ========== 输出和日志 ==========
    parser.add_argument('--result-dir', type=str, default='./result',
                        help='测试结果保存目录')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='实验名称（用于结果组织）')
    
    # ========== 设备和环境 ==========
    parser.add_argument('--cuda', type=str, default='7',
                        help='CUDA 设备 ID')
    
    # ========== Action 相关（必需）==========
    parser.add_argument('--expected-action-representation',
                        choices=('', 'xyz_delta_euler_absolute_v1', 'xyz_delta_euler_relative_wrapped_v1'),
                        default='', help='Optional assertion against saved representation; never overrides it.')
    parser.add_argument('--action-dim', type=int, default=7,
                        help='Action 维度（默认 7：xyz + rotation + gripper）')
    parser.add_argument('--action-chunk', type=int, default=1,
                        help='Action chunk 大小')
    
    # ========== 可选功能开关（默认值可从训练配置覆盖）==========
    parser.add_argument('--use_robot_state', type=int, default=None,
                        help='是否使用机器人状态（从 training_config 加载）')
    parser.add_argument('--load-pointcloud', type=int, default=0,
                        help='是否加载点云（通常为 0）')
    
    # ========== DINOv3 路径（允许重定向到不同位置）==========
    parser.add_argument('--dinov3-path', type=str, default=None,
                        help='DINOv3 模型路径（可选：允许重定向到不同位置，其他配置从 training_config 加载）')
    parser.add_argument('--dinov3_aligner_mlp', type=int, default=0,
                        help='DINOv3 aligner stage architecture（默认 0；正式评测一律被 training_config 合并覆盖）')
    parser.add_argument('--dinov3_concat_pre_norm', type=int, default=0,
                        help='Early-fusion concat projector entry LayerNorm 开关（默认 0；正式评测一律被 '
                             'training_config 合并覆盖）')
    parser.add_argument('--dinov3_pos_embed_rescale', type=float, default=2.0,
                        help='DINOv3 2D-RoPE train-mode 坐标 rescale 上界（默认 2.0；仅 train() 模式生效，'
                             '正式评测一律被 training_config 合并覆盖，此处仅为记录）')
    
    # ========== 其他数据路径（可选）==========
    parser.add_argument('--need-to-sub', type=int, default=3,
                        help='需要减去的值（历史遗留参数）')
    parser.add_argument('--replay_data_dir', type=str, 
                        default='/path/to/data/rlbench/keyframe_fast_slow_chunk8_addlast_0806/for_rlds',
                        help='Replay 数据目录（仅用于 replay 模式）')
    parser.add_argument('--eval-data-path', type=str, default='',
                        help='dataset_predict 模式读取的 *_train.json；HF JSON/PNG 快照用这个评估动作复现。')
    parser.add_argument('--eval-data-root', type=str, default='',
                        help='dataset_predict 模式解析相对图片路径的根目录；例如 HF snapshot root。')
    parser.add_argument('--dataset-max-samples', type=int, default=0,
                        help='dataset_predict 每个任务最多评估多少行；0 表示该任务全部样本。')
    parser.add_argument('--dataset-pos-l2-threshold', type=float, default=0.05,
                        help='dataset_predict 的位置 L2 阈值，仅用于 within-threshold 汇总。')
    parser.add_argument('--dataset-rot-l2-threshold', type=float, default=0.25,
                        help='dataset_predict 的周期角度 L2 阈值，仅用于 within-threshold 汇总。')
    parser.add_argument('--eval-demo-root', type=str, default='',
                        help='可选 raw rollout 根目录；若提供，predict/replay 用 reset_to_demo 恢复 episode 初始状态。')
    parser.add_argument('--eval-demo-map', type=str, default='',
                        help='可选 JSON episode -> raw demo episode 映射；用于 HF JSON/PNG 与 raw demo 非同编号时的 demo reset。')
    parser.add_argument('--eval-variation', type=int, default=0,
                        help='raw rollout variation id，配合 --eval-demo-root 使用。')
    parser.add_argument('--eval-episode-offset', type=int, default=0,
                        help='未显式传 --eval-episodes 时，从该 episode id 开始评测 num_episodes 个。')
    parser.add_argument('--eval-episodes', type=str, default='',
                        help='逗号分隔的 episode id 列表；配合 --eval-demo-root 或 replay 精确选 episode。')
    
    # ========== Debugpy 远程调试 ==========
    parser.add_argument('--debug', type=int, default=0,
                        help='Enable debugpy remote debugging (1=enabled, 0=disabled)')
    parser.add_argument('--debug_port', type=int, default=5678,
                        help='Debugpy port for remote debugging')
    parser.add_argument('--allow-random-init', type=int, default=0,
                        help='Debug only: allow missing checkpoint components and random initialization. Reproducible eval must keep this 0.')
    parser.add_argument(
        '--expert-rope-contract-override',
        choices=[''] + sorted(EXPERT_ROPE_CONTRACTS),
        default='',
        help='Audit-only override for evaluating an existing checkpoint under a specific expert RoPE contract.',
    )
    parser.add_argument(
        '--attention-collection-mask-contract-override',
        choices=['', 'historical_sdpa_eager_mismatch', 'corrected_causal'],
        default='',
        help='Audit-only override for attention collection mask semantics.',
    )
    
    args = parser.parse_args()
    
    # ========== 可选：启动 debugpy 等待调试器连接 ==========
    if args.debug and DEBUGPY_AVAILABLE:
        if not debugpy.is_client_connected():
            try:
                debugpy.listen(("0.0.0.0", args.debug_port))
                print(f"[debugpy] 等待调试器连接 port {args.debug_port}...")
                debugpy.wait_for_client()
                print("[debugpy] 调试器已连接")
            except Exception as e:
                print(f"[debugpy] 启动失败: {e}")
        else:
            print("[debugpy] 调试器已连接")
    elif args.debug and not DEBUGPY_AVAILABLE:
        print("[WARN] 已指定 --debug 但未安装 debugpy，请执行: pip install debugpy")
    
    # ========== 严格从训练配置加载所有模型参数 ==========
    print("=" * 80)
    print("[测试脚本] 开始加载训练配置...")
    print("=" * 80)
    args = load_and_merge_training_config(args)
    print("=" * 80)
    print("[测试脚本] 配置加载完成，开始测试")
    print("=" * 80)
    
    main(args)
