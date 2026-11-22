"""Pinned structural compatibility for the absent, unused expert vocab head.

Pi0's manual action path uses the expert decoder and action_out_proj, not its
generic language vocabulary head. The historical converter explicitly makes
that head a zero placeholder; this artifact omits it. No missing active weight
is filled, aliased, ignored or synthesized by this compatibility operation.
"""
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import struct


HEADER_SHA256 = "8bca30b7fbf565fde73b3f970f9385b6e1c4dfc246021d991d47439933b4b973"
HELPER_SHA256 = "1aba460337aacf005fd1da9396a320febb22491b383e3ad76f3636f1f33ce8ae"
CONVERTER_SHA256 = "c08f9e966cd7bda39be681a34262517dc49a4198b619a44702b0e0456d9fea3c"
FORWARD_SOURCE_SHA256 = "4b898e58ca499d28ea4d6d501fc3b51b2de24781f040afdf2b12eace1e71d58c"
ABSENT_KEY = "model.paligemma_with_expert.gemma_expert.lm_head.weight"


def read_exact_header(checkpoint):
    with (Path(checkpoint) / "model.safetensors").open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        if length != 131840:
            raise RuntimeError("Unexpected pinned safetensors header length")
        raw = handle.read(length)
    if hashlib.sha256(raw).hexdigest() != HEADER_SHA256:
        raise RuntimeError("Safetensors header differs from the pinned artifact")
    header = json.loads(raw)
    if ABSENT_KEY in header or any(ABSENT_KEY in (k, v) for k, v in header.get("__metadata__", {}).items()):
        raise RuntimeError("Expert head is present or declared as an alias")
    return header


def verify_unused_head_source(helper):
    path = Path(helper.__file__)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HELPER_SHA256:
        raise RuntimeError("Unexpected neural helper source")
    converter = path.parent / "conversion_scripts/convert_pi0_to_hf_lerobot.py"
    if hashlib.sha256(converter.read_bytes()).hexdigest() != CONVERTER_SHA256:
        raise RuntimeError("Unexpected original converter evidence")
    tree = ast.parse(raw)
    model = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "PaliGemmaWithExpertModel")
    forward = next(x for x in model.body if isinstance(x, ast.FunctionDef) and x.name == "forward")
    # Source hashing is stable across Python AST schema versions (3.10/3.13).
    if hashlib.sha256(ast.get_source_segment(raw.decode(), forward).encode()).hexdigest() != FORWARD_SOURCE_SHA256:
        raise RuntimeError("Manual neural forward changed")
    if any(isinstance(x, ast.Attribute) and x.attr == "lm_head" for x in ast.walk(model)):
        raise RuntimeError("Neural helper accesses the vocabulary head")
    return {"helper_sha256": HELPER_SHA256, "converter_sha256": CONVERTER_SHA256,
            "manual_forward_source_sha256": FORWARD_SOURCE_SHA256, "vocab_head_references": 0}


def _prune_detached_vocab_head(expert, expected_shape):
    """Remove only a real Linear head with no surviving decoder alias."""
    import torch
    if expert.model.embed_tokens is not None:
        raise RuntimeError("Expert token embedding must already be removed upstream")
    head = expert.lm_head
    if type(head) is not torch.nn.Linear or head.bias is not None or tuple(head.weight.shape) != expected_shape:
        raise RuntimeError("Unexpected expert vocabulary projection")
    pointer = head.weight.untyped_storage().data_ptr()
    for name, value in expert.model.named_parameters(remove_duplicate=False):
        if value.untyped_storage().data_ptr() == pointer:
            raise RuntimeError("Vocabulary head aliases an active decoder parameter: " + name)
    audit = {"removed_class": type(head).__name__, "removed_shape": list(head.weight.shape),
             "removed_parameter_count": sum(p.numel() for p in head.parameters()),
             "removed_dtype": str(head.weight.dtype), "removed_key": ABSENT_KEY,
             "weights_created_or_copied": 0, "active_decoder_unchanged": True}
    expert.lm_head = None
    return audit


@contextmanager
def absent_unused_expert_head_compat(helper, checkpoint):
    """Retain the real NN constructor, then prune the proven dead projection."""
    header = read_exact_header(checkpoint)
    evidence = verify_unused_head_source(helper)
    cls = helper.PaliGemmaWithExpertModel
    original = cls.__init__
    forward = cls.forward
    audit = {"revision": "pi0-absent-unused-expert-vocab-head-v1", "source": evidence,
             "header_sha256": HEADER_SHA256, "header_tensor_count": len(header) - 1,
             "constructor_calls": 0, "removed": []}

    def corrected(self, config):
        original(self, config)
        if config.gemma_expert_config.hidden_size != 1024 or config.gemma_expert_config.vocab_size != 257152:
            raise RuntimeError("Unexpected actual expert architecture")
        audit["removed"].append(_prune_detached_vocab_head(self.gemma_expert, (257152, 1024)))
        audit["constructor_calls"] += 1

    cls.__init__ = corrected
    try:
        yield audit
    finally:
        cls.__init__ = original
        if cls.forward is not forward:
            raise RuntimeError("Active neural forward changed during structural compatibility")


def audit_active_state_header(policy, checkpoint):
    """Require exact external weight keys/shapes, allowing proven tied aliases."""
    header = read_exact_header(checkpoint)
    state = policy.state_dict()
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    missing = set(state) - set(tensors)
    aliases = {}
    for key in sorted(missing):
        target = header.get("__metadata__", {}).get(key)
        if target not in tensors or target not in state:
            raise RuntimeError("Active model weight absent from external checkpoint: " + key)
        left, right = state[key], state[target]
        if (left.untyped_storage().data_ptr(), left.storage_offset(), left.shape, left.dtype) != (
                right.untyped_storage().data_ptr(), right.storage_offset(), right.shape, right.dtype):
            raise RuntimeError("Declared checkpoint alias is not a real model alias")
        aliases[key] = target
    if set(tensors) - set(state):
        raise RuntimeError("External checkpoint has unused tensor keys")
    for key, info in tensors.items():
        if list(state[key].shape) != info["shape"]:
            raise RuntimeError("External checkpoint shape mismatch: " + key)
        dtype_names = {"BF16": "torch.bfloat16", "F32": "torch.float32", "F16": "torch.float16",
                       "F64": "torch.float64", "I64": "torch.int64", "I32": "torch.int32", "BOOL": "torch.bool"}
        if dtype_names.get(info["dtype"]) != str(state[key].dtype):
            raise RuntimeError("External checkpoint dtype mismatch: " + key)
    return {"header_tensor_count": len(tensors), "active_state_key_count": len(state),
            "real_tied_aliases": aliases, "missing_active_weights": [], "unexpected_external_weights": []}
