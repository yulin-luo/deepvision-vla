"""Reusable diagnostics extracted from validated native implementations."""
import ast
import hashlib
from pathlib import Path
from contextlib import contextmanager
from deepvision_vla.diagnostics.gradient_capture import capture_residual_input
from deepvision_vla.diagnostics.gradient_capture import token_gradcam
import functools
from deepvision_vla.diagnostics.adapters.openpi_bridge import bridge_eef_state8

LAYERS = (0, 6, 12, 16)

TARGET_NAME = 'single_flow_step_spatial_velocity_squared_mean'

TARGET_FORMULA = 'mean(v_t[..., :6] ** 2), v_t shape[1,4,32], t=.5, x_t=fixed_seed42_full_noise'

HELPER_SHA = '1aba460337aacf005fd1da9396a320febb22491b383e3ad76f3636f1f33ce8ae'

FORWARD_SHA = '4b898e58ca499d28ea4d6d501fc3b51b2de24781f040afdf2b12eace1e71d58c'

ACTUAL_SEAM = 'pi_prefix_layernorm_residual_input'

LOGICAL_SEAM = 'pi_post_block_output_at_next_prefix_layernorm_input'

def validate_layers(layers, count=18):
    layers = tuple(layers)
    if layers != LAYERS or any((type(x) is not int for x in layers)) or count != 18:
        raise ValueError('Pinned Pi0 OUTPUT layers must be0/6/12/16 in its18-layer prefix')
    return layers

def validate_cpu_receipt(receipt):
    loads = receipt.get('strict_loads', [])
    if receipt.get('status') != 'complete' or receipt.get('cuda_initialized') is not False or receipt.get('meta_tensors') != 0 or (receipt.get('model_forward_calls') != 0) or (receipt.get('gpu_calls') != 0) or (len(loads) != 1):
        raise ValueError('Actual complete CPU-only strict load receipt is required')
    load = loads[0]
    if load.get('outer_strict') is not True or load.get('missing') or load.get('unexpected') or (load.get('loaded_keys') != 783):
        raise ValueError('Pi0 checkpoint was not strictly/materially loaded')
    if receipt.get('active_state_header_audit', {}).get('missing_active_weights') != [] or receipt.get('unused_expert_head_audit', {}).get('source', {}).get('manual_forward_source_sha256') != FORWARD_SHA:
        raise ValueError('CPU receipt does not bind the pinned genuine manual forward')

def verify_output_seam_source(proof, helper_path):
    """Hash the real helper+AST source, then derive four aliases in its loop."""
    raw = Path(helper_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != HELPER_SHA:
        raise ValueError('Wrong actual PaliGemma helper source')
    tree = ast.parse(raw.decode('utf8'))
    model = next((x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'PaliGemmaWithExpertModel'))
    forward = next((x for x in model.body if isinstance(x, ast.FunctionDef) and x.name == 'forward'))
    actual = hashlib.sha256(ast.get_source_segment(raw.decode('utf8'), forward).encode()).hexdigest()
    if actual != FORWARD_SHA:
        raise ValueError('The native manual forward changed')
    if proof.get('status') != 'source_identity_derivation_passed' or proof.get('helper_sha256') != HELPER_SHA or proof.get('forward_source_sha256') != FORWARD_SHA or (proof.get('prefix_layer_count') != 18) or (proof.get('no_forward_rewrite') is not True) or (proof.get('protocol_label_required') != LOGICAL_SEAM + '; not a generic INPUT-map relabel'):
        raise ValueError('The provided OUTPUT identity proof differs')
    provided = {r['post_block_output']: r['next_prefix_layernorm_input'] for r in proof.get('seams', [])}
    if provided.get(0) != 1 or provided.get(16) != 17 or any((v != k + 1 for k, v in provided.items())):
        raise ValueError('Proof seam identities changed')
    return {'helper_sha256': HELPER_SHA, 'forward_source_sha256': actual, 'identity_basis': 'exact_pinned_manual_loop_dataflow_not_input_map_relabel', 'runtime_previous_block_tensor_directly_observed': False, 'seams': [{'logical_output_layer': layer, 'actual_prefix_norm_input_layer': layer + 1, 'actual_module_path': f'policy.model.paligemma_with_expert.paligemma.language_model.model.layers[{layer + 1}].input_layernorm', 'logical_seam': LOGICAL_SEAM, 'actual_hook_location': ACTUAL_SEAM} for layer in LAYERS]}

@contextmanager
def capture_post_block_output_alias(norm, *, logical_layer, expected_shape):
    """Observe the exact native next-LN Tensor, preserving all aliases/values.

    Production must first pass verify_output_seam_source on the real helper.
    The shared capture enables a genuine floating leaf's gradient in place;
    it never returns a clone/replacement and includes the original residual.
    """
    if type(logical_layer) is not int or logical_layer not in LAYERS:
        raise ValueError('Unsupported logical OUTPUT alias')
    with capture_residual_input(norm, expected_shape=expected_shape, target_name=TARGET_NAME, hook_location=ACTUAL_SEAM, expected_prefix_calls=1) as capture:
        yield capture

def alias_contribution(capture, positions, logical_layer):
    result = token_gradcam(capture, positions, target_name=TARGET_NAME)
    result['shared_capture_activation_semantics'] = result['activation_semantics']
    result.update(logical_output_layer=logical_layer, actual_prefix_norm_input_layer=logical_layer + 1, logical_seam=LOGICAL_SEAM, actual_hook_location=ACTUAL_SEAM, activation_semantics='source_proven_post_block_output_alias_at_actual_next_layernorm_input', alias_identity_basis='verified_pinned_source_dataflow', runtime_previous_block_tensor_directly_observed=False)
    return result

@contextmanager
def freeze_parameters(parameters):
    unique = {id(p): p for p in parameters}
    original = [(p, p.requires_grad) for p in unique.values()]
    if any((p.grad is not None for p, _ in original)):
        raise RuntimeError('Fresh model parameters must have no gradients')
    for p, _ in original:
        p.requires_grad_(False)
    audit = {'parameter_tensors': len(original), 'optimizer_steps': 0, 'all_real_parameters_frozen': True}
    try:
        yield audit
    finally:
        bad = sum((p.grad is not None for p, _ in original))
        for p, flag in original:
            p.requires_grad_(flag)
        audit.update(parameter_flags_restored=all((p.requires_grad == flag for p, flag in original)), parameter_grads_present=bad)
        if bad:
            raise RuntimeError('Frozen model accumulated parameter gradients')

@contextmanager
def count_native_methods(flow, state):
    """Delegate to unchanged native methods; restore original instance layout."""
    old = {}
    for name in ('embed_prefix', 'denoise_step'):
        original = getattr(flow, name)
        old[name] = (name in flow.__dict__, flow.__dict__.get(name))

        def counted(*args, _name=name, _original=original, **kwargs):
            state['native_calls'][_name] += 1
            return _original(*args, **kwargs)
        setattr(flow, name, functools.wraps(original)(counted))
    try:
        yield
    finally:
        for name, (had, value) in old.items():
            if had:
                setattr(flow, name, value)
            else:
                delattr(flow, name)

def prepare_native_inputs(policy, image, state7, instruction):
    """Run the unchanged real policy's preprocessing with measured raw grip."""
    import numpy as np
    import torch
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError('Use the original uint8 RGB image')
    device = next(policy.parameters()).device
    state8 = bridge_eef_state8(state7)
    batch = {'observation.state': torch.tensor([state8], dtype=torch.float32, device=device), 'observation.images.image_0': torch.from_numpy(array.copy()).permute(2, 0, 1).float()[None].to(device) / 255, 'task': [instruction]}
    batch = policy.normalize_inputs(batch)
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens, lang_masks = policy.prepare_language(batch)
    if tuple(state.shape) != (1, 32) or tuple(lang_tokens.shape) != (1, 48) or len(images) != 1:
        raise RuntimeError('Native prepared state/language/camera contract changed')
    return ((images, img_masks, lang_tokens, lang_masks, state), {'native_state8_before_normalization': state8, 'gripper_input': 'raw_measured_passthrough', 'gripper_clipped_or_binarized': False, 'zero_state_used': False, 'native_preparation': ['normalize_inputs', 'prepare_images', 'prepare_state', 'prepare_language'], 'prepared_state_shape': list(state.shape), 'prepared_state_values': state.detach().cpu().tolist(), 'image_shapes': [list(x.shape) for x in images], 'language_shape': list(lang_tokens.shape), 'language_mask_true_count': int(lang_masks.sum().detach().cpu())})

def native_velocity_step(policy, prepared, noise, make_masks):
    """One real prefix cache and one native denoise_step, without sampling."""
    import torch
    images, img_masks, lang_tokens, lang_masks, state = prepared
    flow = policy.model
    if not policy.config.use_cache:
        raise RuntimeError('Pinned native KV cache must be enabled')
    prefix, pad, attention = flow.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    if tuple(prefix.shape) != (1, 304, 2048):
        raise RuntimeError('Actual native prefix differs from[1,304,2048]')
    masks = make_masks(pad, attention)
    positions = torch.cumsum(pad, dim=1) - 1
    _, cache = flow.paligemma_with_expert.forward(attention_mask=masks, position_ids=positions, past_key_values=None, inputs_embeds=[prefix, None], use_cache=policy.config.use_cache, fill_kv_cache=True)
    timestep = torch.full((1,), 0.5, dtype=torch.float32, device=state.device)
    velocity = flow.denoise_step(state, pad, cache, noise.detach().clone().to(state.device), timestep)
    if tuple(velocity.shape) != (1, 4, 32) or velocity.dtype != torch.float32 or (not bool(torch.isfinite(velocity).all())):
        raise RuntimeError('Expected finite native float32[1,4,32] flow velocity')
    return velocity

def compare_velocity(velocity, baseline):
    import torch
    if tuple(velocity.shape) != (1, 4, 32) or tuple(baseline.shape) != (1, 4, 32) or velocity.dtype != baseline.dtype:
        raise ValueError('Full native velocity shape/dtype is required')
    current = velocity.detach().cpu().contiguous()
    before = baseline.detach().cpu().contiguous()
    exact = torch.equal(current.view(torch.uint8), before.view(torch.uint8))
    return {'bit_exact': bool(exact), 'shape': [1, 4, 32], 'dtype': str(current.dtype), 'max_abs_difference': float((current.double() - before.double()).abs().max()), 'compared_all128_native_values': True, 'includes_padding_and_gripper_channels': True}

def spatial_velocity_target(velocity):
    import torch
    if tuple(velocity.shape) != (1, 4, 32) or not bool(torch.isfinite(velocity).all()):
        raise ValueError('A finite full H4x32 native flow velocity is required')
    return velocity[..., :6].square().mean()

def validate_counts(state):
    if state.get('native_calls') != {'embed_prefix': 5, 'denoise_step': 5} or state.get('backward_calls') != 4 or state.get('probes_completed') != 4:
        raise RuntimeError('Require exactly5 native prefixes/denoise steps and4 backwards')
    if tuple((r.get('logical_output_layer') for r in state.get('probes', []))) != LAYERS or any((r.get('status') != 'complete' or r.get('baseline_comparison', {}).get('bit_exact') is not True for r in state['probes'])):
        raise RuntimeError('All four ordered logical OUTPUT maps must preserve native velocity bitwise')

