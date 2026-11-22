"""Native paper OUTPUT probe body; caller owns sealed inputs, CPU load and lease."""
import gc
import json
import os
import time
from deepvision_vla.diagnostics.adapters import qwen_h16
from deepvision_vla.diagnostics.resources import require_available
from deepvision_vla.diagnostics.model_geometry import actual_visual_layout
from deepvision_vla.diagnostics.resources import require_own_memory
from deepvision_vla.diagnostics.adapters.qwen_h16_gradients import TARGET_NAME, TARGET_FORMULA, compare_full_prediction, full_h16_action_l2_mean, frozen_parameters, prepare_gradient_forward
from deepvision_vla.diagnostics.gradient_output_capture import capture_residual_output, token_gradcam
from deepvision_vla.diagnostics.geometry import project_roi_boxes
from deepvision_vla.diagnostics.resources import verify_cuda_pci

LAYERS = (0, 12, 24, 34)

PROBE_COUNT = 4

def validate_figure_layer_selection(value, available=None):
    selected = tuple(value)
    if selected != LAYERS or any((type(layer) is not int for layer in selected)):
        raise ValueError('The requested Qwen figure requires exactly layers0/12/24/34 in order')
    if available is not None and (type(available) is not int or available <= 34):
        raise ValueError('The actual decoder does not contain requested figure layer34')
    return selected

def validate_paper_probe_counts(state):
    fields = ('model_forward_calls', 'head_forward_calls', 'backward_calls', 'probes_completed')
    if any((type(state.get(name)) is not int or state[name] != PROBE_COUNT for name in fields)):
        raise RuntimeError('The four-layer figure requires exactly4 model/head forwards/backwards')
    probes = state.get('probes', [])
    if len(probes) != PROBE_COUNT or tuple((row.get('layer') for row in probes)) != LAYERS or any((row.get('status') != 'complete' for row in probes)) or any((row.get('baseline_comparison', {}).get('bit_exact') is not True for row in probes)):
        raise RuntimeError('Four complete ordered bit-exact layer results are required')

def run_native(args, sample, image, filtered, cpu, baseline, selected, state, save):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    state['cuda_binding'] = verify_cuda_pci(selected['pci'])
    save()
    import torch
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one selected CUDA device is required')
    torch.cuda.set_per_process_memory_fraction(22000000000 / torch.cuda.get_device_properties(0).total_memory, 0)

    def budget(minimum=8000000000, require_own=True):
        info = require_available(args.gpu, selected['uuid'], sharing=args.allow_gpu_sharing, minimum_free_bytes=minimum, maximum_own_bytes=24000000000)
        if require_own:
            state['last_observed_own_gpu_bytes'] = require_own_memory(info, os.getpid())
    budget(32000000000, False)
    state['status'] = 'loading'
    save()
    adapter = args.loaded_adapter
    budget(32000000000, False)
    adapter.to('cuda:0')
    budget()
    layers = adapter.model.model.language_model.layers
    selected_layers = validate_figure_layer_selection(LAYERS, len(layers))
    parameters = list(adapter.model.parameters()) + list(adapter.head.parameters())
    original_flags = [parameter.requires_grad for parameter in parameters]
    handles = []

    def count_model(_module, _inputs):
        state['model_forward_calls'] += 1

    def count_head(_module, inputs, output):
        state['head_forward_calls'] += 1
        if tuple(inputs[0].shape) != (1, 16, adapter.head.input_dim) or tuple(output.shape) != (1, 16, 7):
            raise RuntimeError('Native action head no longer preserves the full H16 chunk')
    handles.append(adapter.model.model.register_forward_pre_hook(count_model))
    handles.append(adapter.head.model.register_forward_hook(count_head))
    try:
        with frozen_parameters(parameters) as freeze_audit:
            state['parameter_freeze_audit'] = freeze_audit
            for layer in selected_layers:
                if time.time() > state['deadline_epoch']:
                    raise TimeoutError('Original gradient budget expired')
                budget()
                torch.manual_seed(42)
                inputs, observed = prepare_gradient_forward(adapter, image, sample['instruction'])
                if observed != baseline['layout']['adapter']:
                    raise RuntimeError('Gradient input/visual layout differs from the completed canary')
                layout = actual_visual_layout(sample, observed)
                roi = project_roi_boxes([row['xyxy_px'] for row in filtered['candidate_boxes']], image_sha256=sample['image_sha256'], layout=layout)
                state['layout'] = {'adapter': observed, 'grid_shape': list(layout.grid_shape), 'roi_tokens': list(roi)}
                row = {'layer': layer, 'target_name': TARGET_NAME, 'target_formula': TARGET_FORMULA, 'hook_location': 'decoder_residual_output', 'status': 'forward_started'}
                state['probes'].append(row)
                state['status'] = 'probing'
                save()
                with capture_residual_output(layers[layer], expected_shape=tuple(inputs['inputs_embeds'].shape), target_name=TARGET_NAME, hook_location='decoder_residual_output') as captured:
                    outputs = adapter.model.model(**inputs)
                    prediction = adapter.head.predict_action(outputs.last_hidden_state[:, -16:, :])
                    if tuple(prediction.shape) != (1, 16, 7) or not bool(torch.isfinite(prediction).all()):
                        raise RuntimeError('Expected finite normalized [1,16,7] prediction')
                    values = prediction.detach().float().cpu().tolist()[0]
                    row['normalized_actions_h16'] = values
                    row['baseline_comparison'] = compare_full_prediction(values, baseline['baseline_normalized_h16'])
                    save()
                    if not row['baseline_comparison']['bit_exact']:
                        row['status'] = 'prediction_mismatch'
                        save()
                        raise RuntimeError('Gradient-enabled full H16 prediction differs bitwise from the completed canary')
                    scalar = full_h16_action_l2_mean(prediction)
                    if scalar.ndim != 0 or not bool(torch.isfinite(scalar)):
                        raise RuntimeError('The declared full-H16 scalar is non-finite')
                    row['scalar_value'] = float(scalar.detach().cpu())
                    row['status'] = 'backward_started'
                    save()
                    scalar.backward()
                    state['backward_calls'] += 1
                    contribution = token_gradcam(captured, layout.token_positions, target_name=TARGET_NAME)
                    row['contribution'] = {name: value.tolist() if isinstance(value, torch.Tensor) else value for name, value in contribution.items()}
                    row['prefix_calls'] = captured.forward_calls
                    row['captured_backward_calls'] = captured.backward_calls
                if any((parameter.grad is not None for parameter in parameters)):
                    raise RuntimeError('Frozen model accumulated parameter gradients')
                row['status'] = 'complete'
                state['probes_completed'] += 1
                save()
                del scalar, prediction, outputs, captured, inputs
                gc.collect()
                torch.cuda.empty_cache()
                budget()
        state['parameter_flags_restored'] = [p.requires_grad for p in parameters] == original_flags
        state['parameter_grads_present'] = sum((p.grad is not None for p in parameters))
        validate_paper_probe_counts(state)
        if not state['parameter_flags_restored'] or state['parameter_grads_present']:
            raise RuntimeError('Four-layer probe did not preserve parameter freeze/no-gradient contract')
        state.update(status='computation_complete', peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(), peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(), finished_epoch=time.time())
        save()
    finally:
        for handle in handles:
            handle.remove()
