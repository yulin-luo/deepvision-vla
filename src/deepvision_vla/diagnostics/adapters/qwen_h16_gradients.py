"""Reusable diagnostics extracted from validated native implementations."""
import math
import struct
from deepvision_vla.diagnostics.adapters.qwen_h16_contracts import REVISION, matches_controlled_repository
from contextlib import contextmanager

TARGET_NAME = 'normalized_h16_action_head_l2_mean'

TARGET_FORMULA = 'mean(predictions.float() ** 2), complete [1,16,7] normalized action chunk'

LAYERS = (0, 34)

def prediction_float32_bytes(actions):
    """Preserve Float32 bit identity, including signed zero, across JSON IO."""
    if len(actions) != 16 or any((len(row) != 7 for row in actions)):
        raise ValueError('Expected a complete [16,7] normalized prediction')
    values = [value for row in actions for value in row]
    if any((type(value) not in (int, float) or not math.isfinite(value) for value in values)):
        raise ValueError('Prediction values must be finite real numbers')
    try:
        packed = struct.pack('<112f', *values)
    except (OverflowError, struct.error) as error:
        raise ValueError('Prediction values exceed Float32') from error
    if any((a != b for a, b in zip(values, struct.unpack('<112f', packed)))):
        raise ValueError('Prediction JSON must preserve exact native Float32 values')
    return packed

def compare_full_prediction(actual, baseline):
    left, right = (prediction_float32_bytes(actual), prediction_float32_bytes(baseline))
    return {'bit_exact': left == right, 'mismatched_float32_channels': sum((left[i:i + 4] != right[i:i + 4] for i in range(0, 448, 4))), 'max_absolute_difference': max((abs(float(a) - float(b)) for x, y in zip(actual, baseline) for a, b in zip(x, y)))}

def validate_baseline(value, sample):
    if value.get('status') != 'complete' or value.get('lease_released') is not True or not matches_controlled_repository(value.get('repo')) or (value.get('revision') != REVISION) or (value.get('sample_id') != sample.get('sample_id')) or (value.get('instruction') != sample.get('instruction')) or (value.get('prediction_calls_completed') != 7) or (value.get('model_forward_calls') != 7) or (value.get('head_forward_calls') != 7):
        raise ValueError('A completed same-model/same-book Qwen H16 canary baseline is required')
    prediction_float32_bytes(value['baseline_normalized_h16'])
    if not isinstance(value.get('layout', {}).get('adapter'), dict):
        raise ValueError('Baseline must retain its actual input/grid layout')

def full_h16_action_l2_mean(prediction):
    """The single declared target, retaining all112 normalized action values."""
    import torch
    if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != (1, 16, 7) or (not prediction.is_floating_point()) or (not bool(torch.isfinite(prediction).all())):
        raise ValueError('The target requires finite floating normalized [1,16,7] actions')
    scalar = prediction.float().square().mean()
    if not bool(torch.isfinite(scalar)):
        raise ValueError('The full-H16 target overflowed')
    return scalar

@contextmanager
def frozen_parameters(parameters):
    """Temporarily freeze real parameters without erasing existing gradients."""
    unique = {id(parameter): parameter for parameter in parameters}
    originals = [(parameter, parameter.requires_grad) for parameter in unique.values()]
    if any((parameter.grad is not None for parameter, _ in originals)):
        raise RuntimeError('Expected a freshly loaded model with no parameter gradients')
    for parameter, _ in originals:
        parameter.requires_grad_(False)
    try:
        yield {'frozen_parameter_tensors': len(originals), 'optimizer_steps': 0}
    finally:
        for parameter, enabled in originals:
            parameter.requires_grad_(enabled)

def append_action_queries(base, attention, embedding_layer, placeholder):
    """The unchanged adapter's sixteen active pad/eos IDs and zero embeddings."""
    import torch
    if base.ndim != 2 or base.shape[0] != 1 or base.shape[1] <= 0 or (tuple(attention.shape) != tuple(base.shape)) or (type(placeholder) is not int) or (placeholder < 0):
        raise ValueError('Expected one actual tokenized prompt and placeholder ID')
    queries = torch.full((1, 16), placeholder, dtype=base.dtype, device=base.device)
    full_ids = torch.cat((base, queries), dim=1)
    embeddings = embedding_layer(full_ids)
    embeddings[:, -16:, :] = 0
    attention = torch.cat((attention, torch.ones((1, 16), dtype=attention.dtype, device=base.device)), dim=1)
    return (full_ids, attention, embeddings)

def prepare_gradient_forward(adapter, image, instruction):
    """Exact H16 input recipe, outside the inference-only prediction method."""
    from PIL import Image
    original_size = image.size
    image = image.convert('RGB').resize((224, 224), resample=Image.Resampling.BICUBIC)
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': instruction}]}]
    inputs = adapter.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
    device = next(adapter.model.parameters()).device
    base = inputs.input_ids.to(device)
    attention = inputs.attention_mask.to(device)
    token = adapter.processor.tokenizer.pad_token_id
    if token is None:
        token = adapter.processor.tokenizer.eos_token_id
    if token is None:
        raise ValueError('The trained placeholder requires an actual pad/eos token')
    full_ids, attention, embeddings = append_action_queries(base, attention, adapter.model.get_input_embeddings(), token)
    positions = (full_ids[0] == adapter.model.config.image_token_id).nonzero().flatten()
    grid = inputs.image_grid_thw.to(device)
    observed = {'image_grid_thw': grid.detach().cpu().tolist(), 'visual_positions': positions.detach().cpu().tolist(), 'action_query_count': 16, 'original_image_size': list(original_size), 'prepared_image_size': [224, 224], 'image_transform': 'full_image_bicubic_resize_224', 'prompt_length': base.shape[1], 'patch_size': adapter.model.config.vision_config.patch_size, 'spatial_merge_size': adapter.model.config.vision_config.spatial_merge_size}
    kwargs = {'inputs_embeds': embeddings, 'pixel_values': inputs.pixel_values.to(device=device, dtype=adapter.torch.bfloat16), 'attention_mask': attention, 'image_grid_thw': grid, 'return_dict': True, 'action_length': 16, 'position_input_ids': full_ids, 'visual_pos_masks': full_ids == adapter.model.config.image_token_id, 'qwen_position_contract': 'token_ids_mrope', 'action_task_attention_contract': 'action_only', 'attention_collection_mask_contract': 'corrected_causal', 'vla_attention_mask_contract': 'prompt_causal_action_bidirectional', 'use_cache': False}
    return (kwargs, observed)

