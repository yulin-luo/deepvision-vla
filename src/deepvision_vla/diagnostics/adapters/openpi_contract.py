"""Optional OpenPI integration contract; no OpenPI/JAX/Torch import at module load."""

from dataclasses import dataclass
import re
import sys


@dataclass(frozen=True)
class OpenPIContract:
    variant: str
    upstream_commit: str
    checkpoint_identity: str
    config_name: str
    action_horizon: int
    native_action_dim: int
    output_action_dim: int
    input_state_definition: str
    required_state_definition: str
    normalizer_state_definition: str
    normalizer_source: str
    normalizer_asset_id: str
    output_action_definition: str

    def __post_init__(self):
        if self.variant not in {"pi0", "pi05"}:
            raise ValueError("pi0 and pi05 are distinct supported variants")
        if not re.fullmatch(r"[0-9a-f]{40}", self.upstream_commit):
            raise ValueError("An exact upstream source commit is required")
        for value in (self.checkpoint_identity, self.config_name, self.input_state_definition,
                      self.required_state_definition, self.normalizer_state_definition,
                      self.normalizer_source, self.normalizer_asset_id, self.output_action_definition):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Checkpoint, state, normalizer and output definitions must be explicit")
        for value in (self.action_horizon, self.native_action_dim, self.output_action_dim):
            if type(value) is not int or value <= 0:
                raise ValueError("Horizon and action dimensions must be positive integers")
        if self.output_action_dim > self.native_action_dim:
            raise ValueError("Output definition cannot invent extra native channels")
        if self.input_state_definition != self.required_state_definition:
            raise ValueError("Input state is not the state representation required by this policy")
        if self.normalizer_state_definition != self.required_state_definition:
            raise ValueError("Normalizer belongs to a different physical state representation")

    @property
    def noise_shape(self):
        return (1, self.action_horizon, self.native_action_dim)

    def validate_noise_shape(self, shape):
        if tuple(shape) != self.noise_shape:
            raise ValueError("Fixed diffusion noise must retain the complete native horizon and dimension")


def prefix_positions(image_name, image_names, observed_token_counts):
    """Image prefix has no LLM BOS; masked camera slots still occupy tokens."""
    names = tuple(image_names)
    counts = tuple(observed_token_counts)
    if len(set(names)) != len(names) or len(names) != len(counts) or image_name not in names:
        raise ValueError("An actual ordered image layout is required")
    if any(type(x) is not int or x <= 0 for x in counts):
        raise ValueError("Image token counts must come from the actual vision outputs")
    index = names.index(image_name)
    offset = sum(counts[:index])
    return tuple(range(offset, offset + counts[index]))


class OpenPINativeAdapter:
    """Wrap an official, fully configured PyTorch Policy without creating inputs.

    The caller must supply real policy-native observations after establishing
    their physical state mapping and checkpoint normalizer. This class does
    not pad Bridge EEF state into a joint vector, choose DROID statistics,
    truncate outputs to seven channels, or create an imitation model backend.
    """

    def __init__(self, policy, contract: OpenPIContract):
        if sys.version_info < (3, 11):
            raise RuntimeError("Use the independent official OpenPI Python3.11+ environment")
        import transformers
        if transformers.__version__ != "4.53.2":
            raise RuntimeError("Official OpenPI PyTorch support requires Transformers4.53.2 with its upstream replacements")
        from openpi.policies.policy import Policy
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

        if not isinstance(policy, Policy) or not isinstance(policy._model, PI0Pytorch):
            raise TypeError("An official OpenPI PyTorch Policy is required")
        if not policy._is_pytorch_model:
            raise TypeError("This adapter does not instrument the JAX backend")
        model = policy._model
        if bool(model.config.pi05) != (contract.variant == "pi05"):
            raise ValueError("Checkpoint model variant and diagnostic variant differ")
        if model.config.pytorch_compile_mode is not None:
            raise ValueError("Set official Pi0Config.pytorch_compile_mode=None before loading for observable hooks")
        if model.config.action_horizon != contract.action_horizon or model.config.action_dim != contract.native_action_dim:
            raise ValueError("Native action specification differs from the supplied contract")
        self.policy, self.contract, self.model = policy, contract, model

    def predict(self, observation, *, fixed_noise, layer=None, dropped_positions=()):
        import numpy as np
        from ..interventions import zero_visual_tokens

        self.contract.validate_noise_shape(fixed_noise.shape)
        noise = np.asarray(fixed_noise, dtype=np.float32)
        if not np.isfinite(noise).all():
            raise ValueError("Diffusion noise must be finite")
        # infer accepts a complete [B,H,D] noise tensor. Each call gets a copy
        # so even an upstream in-place change cannot alter another ratio.
        if layer is None:
            if dropped_positions:
                raise ValueError("Dropped visual positions require an explicit prefix layer")
            result = self.policy.infer(observation, noise=noise.copy())
            hook = None
        else:
            layers = self.model.paligemma_with_expert.paligemma.language_model.layers
            if type(layer) is not int or not 0 <= layer < len(layers):
                raise ValueError("Prefix layer index is outside the actual model")
            with zero_visual_tokens(layers[layer], dropped_positions) as hook:
                result = self.policy.infer(observation, noise=noise.copy())
            if hook["prefill_calls"] != 1 or hook["cached_decode_calls"] != 0:
                raise RuntimeError("Expected exactly one prefix pass and no action-expert masking")
        actions = np.asarray(result["actions"])
        expected = (self.contract.action_horizon, self.contract.output_action_dim)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError("Official output does not match its declared native action definition")
        return {"actions": actions, "hook": hook, "action_definition": self.contract.output_action_definition}
