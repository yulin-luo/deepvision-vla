"""Reusable diagnostics extracted from validated native implementations."""
import math
from contextlib import contextmanager
from deepvision_vla.diagnostics.adapters.openvla_runtime import prepare_prompt_tokens
from deepvision_vla.diagnostics.adapters.openvla_runtime import checked_action

LAYERS = (0, 10, 20, 30)

TARGET_NAME = 'teacher_forced_self_generated_action_token_ce'

TARGET_FORMULA = 'native causal_lm_cross_entropy over exactly seven greedy self-generated action token IDs'

CE_FLOAT32_EPSILON = 2.0 ** (-23)

CE_RELATIVE_TOLERANCE = 8 * CE_FLOAT32_EPSILON

CE_ABSOLUTE_TOLERANCE = 0.0

def compare_native_ce(native, explicit):
    """Predeclared float32 reduction bound; never change the native target.

    Native CE reduces the entire shifted sequence with ignored positions;
    the verification CE reduces only the seven actual target positions.
    Different reduction shapes may round differently at machine precision.
    """
    if any((type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in (native, explicit))):
        raise ValueError('Both native and explicit CE must be finite nonnegative scalars')
    difference = abs(native - explicit)
    bound = CE_RELATIVE_TOLERANCE * abs(explicit) + CE_ABSOLUTE_TOLERANCE
    return {'native_ce': native, 'explicit_seven_token_ce': explicit, 'absolute_difference': difference, 'relative_tolerance': CE_RELATIVE_TOLERANCE, 'absolute_tolerance': CE_ABSOLUTE_TOLERANCE, 'float32_epsilon': CE_FLOAT32_EPSILON, 'allowed_absolute_difference': bound, 'comparison_formula': 'abs(native_ce-explicit_ce) <= (8 * 2**-23) * abs(explicit_ce) + 0', 'within_bound': difference <= bound, 'gradient_target': 'native outputs.loss unchanged'}

def validate_layers(layers, count=32):
    layers = tuple(layers)
    if layers != LAYERS or any((type(x) is not int for x in layers)) or type(count) is not int or (count != 32):
        raise ValueError('The paper OpenVLA probe requires its 32-layer decoder and layers0/10/20/30')
    return layers

def teacher_sequence(prompt, generated):
    """Build original text-space labels; native Prismatic adds patch ignores."""
    if not prompt or len(generated) != 7 or any((type(x) is not int or x < 0 for x in list(prompt) + list(generated))):
        raise ValueError('Declare one nonempty native prompt and seven real generated IDs')
    return (list(prompt) + list(generated), [-100] * len(prompt) + list(generated))

def action_logit_positions(prompt_length, patch_count):
    if any((type(x) is not int or x <= 0 for x in (prompt_length, patch_count))):
        raise ValueError('Actual prompt and projected patch counts are required')
    return tuple(range(prompt_length + patch_count - 1, prompt_length + patch_count + 6))

def validate_generated_sequence(full_ids, prompt, forward_argmax):
    if list(full_ids[:len(prompt)]) != list(prompt) or len(full_ids) != len(prompt) + 7:
        raise RuntimeError('Native generation did not preserve the prompt plus exactly seven tokens')
    generated = list(full_ids[len(prompt):])
    if len(forward_argmax) != 7 or generated != list(forward_argmax):
        raise RuntimeError('Native greedy IDs differ from its seven actual forward argmax values')
    return generated

def validate_counts(state):
    expected = {'generation_requests': 1, 'generation_model_forward_calls': 7, 'generation_projector_calls': 1, 'teacher_forward_calls': 4, 'teacher_projector_calls': 4, 'backward_calls': 4, 'probes_completed': 4}
    if any((type(state.get(k)) is not int or state[k] != v for k, v in expected.items())):
        raise RuntimeError('Expected one native seven-step generation and four teacher forwards/backwards')
    rows = state.get('probes', [])
    if len(rows) != 4 or tuple((x.get('layer') for x in rows)) != LAYERS or any((x.get('status') != 'complete' or x.get('argmax_matches_generated') is not True for x in rows)):
        raise RuntimeError('All four ordered output maps must pass generated-token alignment')

@contextmanager
def freeze_parameters(parameters):
    """Keep original flags and reject stale/accumulated parameter gradients."""
    originals = [(p, p.requires_grad) for p in {id(p): p for p in parameters}.values()]
    if any((p.grad is not None for p, _ in originals)):
        raise RuntimeError('Expected a freshly loaded model without parameter gradients')
    for p, _ in originals:
        p.requires_grad_(False)
    try:
        yield {'parameter_tensors': len(originals), 'optimizer_steps': 0}
    finally:
        for p, flag in originals:
            p.requires_grad_(flag)

def prepare_inputs(runtime, image, instruction):
    """Same prompt, empty-token insertion and BF16 pixels as native runtime."""
    torch = runtime.torch
    prompt = 'In: What action should the robot take to ' + instruction.strip().lower() + '?\nOut:'
    batch = runtime.processor(prompt, image.convert('RGB'), return_tensors='pt')
    ids, mask, appended = prepare_prompt_tokens(batch.input_ids[0].tolist(), batch.attention_mask[0].tolist())
    batch['input_ids'] = torch.tensor([ids], dtype=batch.input_ids.dtype)
    batch['attention_mask'] = torch.tensor([mask], dtype=batch.attention_mask.dtype)
    device = next(runtime.model.parameters()).device
    inputs = {k: v.to(device, dtype=torch.bfloat16) if k == 'pixel_values' else v.to(device) for k, v in batch.items()}
    if set(inputs) != {'input_ids', 'attention_mask', 'pixel_values'}:
        raise ValueError('Unexpected native processor inputs need an explicit reviewed contract')
    return (inputs, ids, appended)

def native_generation(runtime, inputs, prompt, state):
    """Observe actual predict_action without replacing generation or decoding."""
    model, torch = (runtime.model, runtime.torch)
    original = model.generate
    had_instance = 'generate' in model.__dict__
    old_instance = model.__dict__.get('generate')
    sequence = []
    greedy = []
    shapes = []
    handles = []

    def delegate(*args, **kwargs):
        state['generation_requests'] += 1
        if state['generation_requests'] != 1 or kwargs.get('max_new_tokens') != 7 or kwargs.get('do_sample') is not False:
            raise RuntimeError('One actual native greedy seven-token generate call is required')
        result = original(*args, **kwargs)
        if not isinstance(result, torch.Tensor) or tuple(result.shape) != (1, len(prompt) + 7):
            raise RuntimeError('Native generation ended early or returned an unexpected shape')
        sequence.extend(result[0].detach().cpu().tolist())
        return result

    def count(_module, _args, output):
        state['generation_model_forward_calls'] += 1
        if output.logits.ndim != 3 or output.logits.shape[0] != 1:
            raise RuntimeError('Native generation logits are not one batch')
        greedy.append(int(output.logits[0, -1].argmax().detach().cpu()))

    def projector(_module, _args, output):
        state['generation_projector_calls'] += 1
        shapes.append(list(output.shape))
    model.generate = delegate
    try:
        handles.append(model.register_forward_hook(count))
        handles.append(model.projector.register_forward_hook(projector))
        with torch.inference_mode():
            action = model.predict_action(**inputs, unnorm_key='bridge_orig', do_sample=False)
        generated = validate_generated_sequence(sequence, prompt, greedy)
        if len(shapes) != 1:
            raise RuntimeError('Native generation must project the image exactly once')
        state['generation'] = {'generated_ids': generated, 'forward_argmax_ids': greedy, 'native_actions': checked_action(action), 'projected_shapes': shapes}
        return generated
    finally:
        for handle in handles:
            handle.remove()
        if had_instance:
            model.generate = old_instance
        else:
            delattr(model, 'generate')

