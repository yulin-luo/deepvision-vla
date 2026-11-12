"""Runtime contracts for expert QKV positional encoding."""

from __future__ import annotations

from dataclasses import dataclass


EXPERT_ROPE_CONTRACTS = frozenset(
    {
        "historical_train_eval_mismatch",
        "explicit_flags",
        "dino_only",
        "qwen_only",
        "double",
        "none",
    }
)

EXPERT_ROPE_CONTRACT_ALIASES = {
    "legacy_flags": "explicit_flags",
    "historical_double": "double",
}

QWEN_POSITION_CONTRACTS = frozenset(
    {
        "historical_inputs_embeds_1d",
        "token_ids_mrope",
    }
)


@dataclass(frozen=True)
class ExpertRopeRuntime:
    contract: str
    phase: str
    dinov3_qkv_after_rope: bool
    dinov3_use_qwen_rope: bool

    @property
    def effective_contract(self) -> str:
        if self.dinov3_qkv_after_rope and self.dinov3_use_qwen_rope:
            return "double"
        if self.dinov3_qkv_after_rope:
            return "dino_only"
        if self.dinov3_use_qwen_rope:
            return "qwen_only"
        return "none"


def resolve_expert_rope_runtime(
    contract: str,
    *,
    dinov3_qkv_after_rope: int | bool,
    dinov3_use_qwen_rope: int | bool,
    phase: str = "eval",
) -> ExpertRopeRuntime:
    contract = EXPERT_ROPE_CONTRACT_ALIASES.get(str(contract), str(contract))
    phase = str(phase)
    if contract not in EXPERT_ROPE_CONTRACTS:
        raise ValueError(
            f"expert_rope_contract must be one of {sorted(EXPERT_ROPE_CONTRACTS)}; "
            f"got {contract!r}."
        )
    if phase not in {"train", "eval"}:
        raise ValueError(f"phase must be 'train' or 'eval'; got {phase!r}.")

    after_rope = bool(int(dinov3_qkv_after_rope))
    use_qwen_rope = bool(int(dinov3_use_qwen_rope))
    if contract == "historical_train_eval_mismatch":
        # The paper-era trainer changed config after constructing the attention
        # modules, so foundation-model training kept the layer-local default False.
        # The evaluator later synchronized the serialized True value into each layer.
        use_qwen_rope = use_qwen_rope if phase == "eval" else False
    elif contract == "dino_only":
        after_rope, use_qwen_rope = True, False
    elif contract == "qwen_only":
        after_rope, use_qwen_rope = False, True
    elif contract == "double":
        after_rope, use_qwen_rope = True, True
    elif contract == "none":
        after_rope, use_qwen_rope = False, False

    return ExpertRopeRuntime(
        contract=contract,
        phase=phase,
        dinov3_qkv_after_rope=after_rope,
        dinov3_use_qwen_rope=use_qwen_rope,
    )


def sync_expert_qwen_rope_runtime(vl_model, enabled: int | bool) -> int:
    """Synchronize config and already-constructed Qwen attention modules."""
    enabled = bool(int(enabled))
    config = getattr(vl_model, "config", None)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        text_config.dinov3_use_qwen_rope = enabled

    language_model = getattr(vl_model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        layers = getattr(vl_model, "layers", None)
    if layers is None:
        return 0

    updated = 0
    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None and hasattr(self_attn, "dinov3_use_qwen_rope"):
            self_attn.dinov3_use_qwen_rope = enabled
            updated += 1
    return updated
