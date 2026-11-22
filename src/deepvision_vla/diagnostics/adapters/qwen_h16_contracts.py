"""Reusable diagnostics extracted from validated native implementations."""

import hashlib

REPOSITORY_NAME_SHA256 = "8e94914254f190018924a49f108f870db030c2de7129e610d15c4dc25cbcd920"

REVISION = 'b131353d58b77daf2b6308e5ea4ea0aa4a417a79'
RUNTIME_EXPORT_ID = "deepvision-vla/qwenvla-oft-h16-runtime"
RUNTIME_EXPORT_FORMAT = "qwen_h16_runtime_export_v1"

def matches_controlled_repository(name):
    """Match the pinned receipt identity without embedding its download location."""
    return (isinstance(name, str)
            and hashlib.sha256(name.encode("utf-8")).hexdigest() == REPOSITORY_NAME_SHA256)


def validate_cpu_receipt(receipt, adapter_sha256):
    loads = receipt.get('strict_loads', [])
    if receipt.get('status') != 'complete' or receipt.get('cuda_initialized') is not False or receipt.get('model_forward_calls') != 0 or (receipt.get('action_chunk') != 16) or (receipt.get('action_dim') != 7) or (receipt.get('lm_head_tied') is not True) or (receipt.get('adapter_sha256') != adapter_sha256) or (len(loads) != 2):
        raise ValueError('A matching real CPU-only H16 load receipt is required')
    for load in loads:
        if load.get('strict') is not True or load.get('missing') or load.get('unexpected') or (load.get('provided_meta_tensors') != 0) or (load.get('loaded_meta_tensors') != 0) or (load.get('provided_keys', 0) <= 0) or (load['provided_keys'] != load.get('loaded_keys')):
            raise ValueError('CPU component loading was not strict and materialized')
