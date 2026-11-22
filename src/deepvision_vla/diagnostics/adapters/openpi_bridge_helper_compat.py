"""Explicit constructor-only repair for the pinned LeRobot Pi0 helper.

The upstream constructor's default path works, but its explicit branches read
unassigned self attributes and use two wrong dictionary references. The real
neural classes, methods, tensors and weights are never replaced here.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path


ORIGINAL_HELPER_SHA256 = "1aba460337aacf005fd1da9396a320febb22491b383e3ad76f3636f1f33ce8ae"
COMPATIBILITY_REVISION = "pi0-provided-subconfig-constructor-v1"


def _provided_subconfig(module, value, expected_type):
    if isinstance(value, dict):
        payload = deepcopy(value)
        model_type = payload.setdefault("model_type", expected_type)
        if model_type != expected_type:
            raise ValueError("Expected " + expected_type + " subconfiguration")
        return module.CONFIG_MAPPING[model_type](**payload)
    if isinstance(value, module.PretrainedConfig):
        if value.model_type != expected_type:
            raise ValueError("Expected " + expected_type + " configuration object")
        return value
    raise TypeError("Expected a dict or genuine Transformers PretrainedConfig")


@contextmanager
def provided_subconfig_constructor_compat(module):
    """Temporarily repair only __init__, retaining the real helper class.

    Uses the unchanged upstream default branch and parent initialization, then
    assigns each explicitly provided subconfig through its correct genuine
    Transformers config class. This avoids the four broken explicit branches.
    The exact original constructor is restored even if strict loading fails.
    """
    path = Path(module.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_HELPER_SHA256:
        raise RuntimeError("Constructor compatibility requires the exact pinned helper")
    cls = module.PaliGemmaWithExpertConfig
    original = cls.__init__
    if getattr(original, "_deepvision_constructor_compat", False):
        raise RuntimeError("Nested constructor compatibility is unsupported")
    neural_forward = module.PaliGemmaWithExpertModel.forward

    def corrected(self, paligemma_config=None, gemma_expert_config=None,
                  freeze_vision_encoder=True, train_expert_only=True,
                  attention_implementation="eager", **kwargs):
        # This original path constructs configuration objects only; it never
        # constructs PaliGemma/Gemma networks or accesses model weights.
        original(self, paligemma_config=None, gemma_expert_config=None,
                 freeze_vision_encoder=freeze_vision_encoder,
                 train_expert_only=train_expert_only,
                 attention_implementation=attention_implementation, **kwargs)
        if paligemma_config is not None:
            self.paligemma_config = _provided_subconfig(module, paligemma_config, "paligemma")
        if gemma_expert_config is not None:
            self.gemma_expert_config = _provided_subconfig(module, gemma_expert_config, "gemma")

    corrected._deepvision_constructor_compat = True
    cls.__init__ = corrected
    audit = {"revision": COMPATIBILITY_REVISION,
             "original_helper_sha256": ORIGINAL_HELPER_SHA256,
             "modified_methods": ["PaliGemmaWithExpertConfig.__init__"],
             "neural_forward_unchanged": True}
    try:
        yield audit
    finally:
        cls.__init__ = original
        if module.PaliGemmaWithExpertModel.forward is not neural_forward:
            raise RuntimeError("Neural forward changed during constructor compatibility")
