"""Native paper OUTPUT probe body; caller owns sealed inputs, CPU load and lease."""
from dataclasses import asdict
import gc
import hashlib
import os
import sys
import time
from deepvision_vla.diagnostics.adapters.pi0_gradients import LAYERS, TARGET_NAME, ACTUAL_SEAM, LOGICAL_SEAM, validate_layers, verify_output_seam_source, capture_post_block_output_alias, alias_contribution, freeze_parameters, count_native_methods, prepare_native_inputs, native_velocity_step, compare_velocity, spatial_velocity_target, validate_counts
from deepvision_vla.diagnostics.adapters.openpi_bridge import load_bridge_pi0
from deepvision_vla.diagnostics.resources import require_available, require_own_memory, verify_cuda_pci
from deepvision_vla.diagnostics.input_payloads import screened_roi
from deepvision_vla.diagnostics.geometry import project_roi_boxes

def run_native(args, sample, image, state7, input_audit, cpu, proof, selected, state, save):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    state['cuda_binding'] = verify_cuda_pci(selected['pci'])
    save()
    import torch
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one numeric selected-GPU binding is required')
    torch.cuda.set_per_process_memory_fraction(22000000000 / torch.cuda.get_device_properties(0).total_memory, 0)

    def budget(minimum=8000000000, own=True):
        if time.time() > state['deadline_epoch']:
            raise TimeoutError('Original1200s deadline expired')
        observed = require_available(args.gpu, selected['uuid'], sharing=args.allow_gpu_sharing, minimum_free_bytes=minimum, maximum_own_bytes=24000000000)
        if own:
            state['last_observed_own_gpu_bytes'] = require_own_memory(observed, os.getpid())
    budget(32000000000, False)
    adapter = args.loaded_adapter
    if cpu['active_state_header_audit'] != adapter.active_state_header_audit or cpu['unused_expert_head_audit'] != adapter.unused_expert_head_audit:
        raise RuntimeError('Loaded active model differs from actual strict CPU receipt')
    budget(32000000000, False)
    policy = adapter.policy.to('cuda:0').eval()
    budget()
    layers = policy.model.paligemma_with_expert.paligemma.language_model.model.layers
    validate_layers(LAYERS, len(layers))
    helper = sys.modules[type(policy.model.paligemma_with_expert).__module__]
    state['output_alias_source_proof'] = verify_output_seam_source(proof, helper.__file__)
    author = sys.modules[type(policy).__module__]
    make_masks = author.make_att_2d_masks
    layout = adapter.visual_layout(sample['image_sha256'], image.size)
    state['layout'] = asdict(layout)
    filtered = screened_roi(sample)
    roi = project_roi_boxes([r['xyxy_px'] for r in filtered['candidate_boxes']], image_sha256=sample['image_sha256'], layout=layout, membership='patch_center')
    if not roi:
        raise ValueError('Screened ROI has no actual visual tokens')
    state['roi_tokens'] = list(roi)
    state['sample_input_audit'] = input_audit
    params = list(policy.parameters())
    old_flags = [p.requires_grad for p in params]
    with freeze_parameters(params) as audit, count_native_methods(policy.model, state):
        state['parameter_freeze_audit'] = audit
        prepared, native_audit = prepare_native_inputs(policy, image, state7, sample['instruction'])
        state['native_input_audit'] = native_audit
        noise = torch.randn((1, 4, 32), generator=torch.Generator(device='cpu').manual_seed(42), dtype=torch.float32)
        state.update(noise_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest(), flow_time=0.5, x_t_definition='seed42_full_native_noise_not_GT_interpolation', status='baseline')
        save()
        budget()
        with torch.no_grad():
            velocity = native_velocity_step(policy, prepared, noise, make_masks)
        baseline = velocity.detach().cpu().contiguous().clone()
        del velocity
        state['baseline_velocity_full_h4x32'] = baseline.tolist()
        save()
        budget()
        for logical in LAYERS:
            budget()
            row = {'logical_output_layer': logical, 'actual_prefix_norm_input_layer': logical + 1, 'logical_seam': LOGICAL_SEAM, 'actual_hook_location': ACTUAL_SEAM, 'target_name': TARGET_NAME, 'status': 'forward_started'}
            state['probes'].append(row)
            state['status'] = 'probing'
            save()
            with torch.enable_grad(), capture_post_block_output_alias(layers[logical + 1].input_layernorm, logical_layer=logical, expected_shape=(1, 304, 2048)) as captured:
                velocity = native_velocity_step(policy, prepared, noise, make_masks)
                row['baseline_comparison'] = compare_velocity(velocity, baseline)
                save()
                if not row['baseline_comparison']['bit_exact']:
                    raise RuntimeError('Gradient forward changed full native velocity bytes')
                scalar = spatial_velocity_target(velocity)
                if scalar.ndim != 0 or not bool(torch.isfinite(scalar)):
                    raise RuntimeError('Declared flow-velocity scalar is not finite')
                row.update(scalar_value=float(scalar.detach().cpu()), status='backward_started')
                save()
                scalar.backward()
                state['backward_calls'] += 1
                contribution = alias_contribution(captured, layout.token_positions, logical)
                row['contribution'] = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in contribution.items()}
                row.update(captured_prefix_calls=captured.forward_calls, captured_backward_calls=captured.backward_calls)
            if any((p.grad is not None for p in params)):
                raise RuntimeError('Real frozen parameters accumulated gradients')
            row['status'] = 'complete'
            state['probes_completed'] += 1
            save()
            del scalar, velocity, captured, contribution
            gc.collect()
            torch.cuda.empty_cache()
            budget()
    state.update(parameter_flags_restored=[p.requires_grad for p in params] == old_flags, parameter_grads_present=sum((p.grad is not None for p in params)))
    if not state['parameter_flags_restored'] or state['parameter_grads_present']:
        raise RuntimeError('Parameter restoration failed')
    validate_counts(state)
    state.update(status='computation_complete', peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(), peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
    save()
