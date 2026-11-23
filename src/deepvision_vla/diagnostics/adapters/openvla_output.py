"""Native paper OUTPUT probe body; caller owns sealed inputs, CPU load and lease."""
from .openvla_gradients import (LAYERS, TARGET_NAME, TARGET_FORMULA, validate_layers,
    freeze_parameters, prepare_inputs, native_generation, teacher_sequence,
    action_logit_positions, compare_native_ce, validate_counts)
from dataclasses import asdict
import gc
import os
import time
from deepvision_vla.diagnostics.resources import require_available
from deepvision_vla.diagnostics.adapters.openvla_runtime import load_openvla_runtime
from deepvision_vla.diagnostics.resources import require_own_memory
from deepvision_vla.diagnostics.geometry import project_roi_boxes
from deepvision_vla.diagnostics.gradient_output_capture import capture_residual_output, token_gradcam
from deepvision_vla.diagnostics.resources import verify_cuda_pci

def run_native(args, sample, image, filtered, selected, state, save):
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

    def budget(minimum=8000000000, own=True):
        if time.time() > state['deadline_epoch']:
            raise TimeoutError('Original probe deadline expired')
        observed = require_available(args.gpu, selected['uuid'], sharing=args.allow_gpu_sharing, minimum_free_bytes=minimum, maximum_own_bytes=24000000000)
        if own:
            state['last_observed_own_gpu_bytes'] = require_own_memory(observed, os.getpid())
    budget(32000000000, False)
    runtime = args.loaded_adapter
    budget(32000000000, False)
    runtime.to('cuda:0')
    budget()
    validate_layers(LAYERS, runtime.layer_count)
    model = runtime.model
    layers = model.language_model.model.layers
    layout = runtime.visual_layout(sample['image_sha256'], image.size)
    roi = ()
    if filtered is not None:
        roi = project_roi_boxes(
            [x['xyxy_px'] for x in filtered['candidate_boxes']],
            image_sha256=sample['image_sha256'], layout=layout, membership='patch_center',
        )
        if not roi:
            raise ValueError('Screened ROI projects to no genuine visual patches')
    state.update(layout=asdict(layout), roi_tokens=list(roi), source_hashes=runtime.source_hashes, versions=runtime.versions, loading_info=runtime.loading_info)
    params = list(model.parameters())
    old_flags = [p.requires_grad for p in params]
    with freeze_parameters(params) as freeze_audit:
        state['parameter_freeze_audit'] = freeze_audit
        inputs, prompt, appended = prepare_inputs(runtime, image, sample['instruction'])
        state.update(prompt_tokens=len(prompt), empty_token_appended=appended, status='generating')
        save()
        budget()
        generated = native_generation(runtime, inputs, prompt, state)
        budget()
        save()
        full, labels = teacher_sequence(prompt, generated)
        teacher = {'input_ids': torch.tensor([full], dtype=inputs['input_ids'].dtype, device='cuda:0'), 'attention_mask': torch.cat((inputs['attention_mask'], torch.ones((1, 7), dtype=inputs['attention_mask'].dtype, device='cuda:0')), dim=1), 'pixel_values': inputs['pixel_values'], 'labels': torch.tensor([labels], dtype=inputs['input_ids'].dtype, device='cuda:0')}
        patches = len(layout.token_positions)
        positions = action_logit_positions(len(prompt), patches)
        expected_shape = (1, len(full) + patches, int(model.config.text_config.hidden_size))
        if state['generation']['projected_shapes'] != [[1, patches, expected_shape[2]]]:
            raise RuntimeError('Actual projected visual tensor differs from the declared layout')
        state.update(teacher_token_count=len(full), multimodal_shape=list(expected_shape), teacher_action_logit_positions=list(positions))
        save()
        for layer in LAYERS:
            budget()
            torch.manual_seed(42)
            row = {'layer': layer, 'status': 'forward_started', 'target_name': TARGET_NAME, 'target_formula': TARGET_FORMULA, 'hook_location': 'decoder_residual_output'}
            state['probes'].append(row)
            state['status'] = 'probing'
            save()
            observed_projector = []

            def count_forward(_module, _args):
                state['teacher_forward_calls'] += 1

            def observe_projector(_module, _args, output):
                state['teacher_projector_calls'] += 1
                observed_projector.append(list(output.shape))
            handles = [model.register_forward_pre_hook(count_forward), model.projector.register_forward_hook(observe_projector)]
            try:
                with capture_residual_output(layers[layer], expected_shape=expected_shape, target_name=TARGET_NAME) as captured:
                    outputs = model(**teacher, use_cache=False, return_dict=True)
                    if tuple(outputs.logits.shape[:2]) != expected_shape[:2] or observed_projector != [[1, patches, expected_shape[2]]]:
                        raise RuntimeError('Native teacher multimodal sequence/projector differs')
                    logits = outputs.logits[0, list(positions), :]
                    predicted = logits.argmax(-1).detach().cpu().tolist()
                    row.update(teacher_argmax_ids=predicted, argmax_matches_generated=predicted == generated)
                    if predicted != generated:
                        raise RuntimeError('Teacher causal argmax differs from native greedy generated IDs')
                    scalar = outputs.loss
                    manual = torch.nn.functional.cross_entropy(logits.float(), teacher['labels'][0, -7:])
                    if scalar is None or scalar.ndim != 0 or (not bool(torch.isfinite(scalar))):
                        raise RuntimeError('Native token CE is not a finite scalar')
                    row['ce_numeric_verification'] = compare_native_ce(float(scalar.detach().cpu()), float(manual.detach().cpu()))
                    save()
                    if not row['ce_numeric_verification']['within_bound']:
                        raise RuntimeError('Native seven-token CE disagrees with explicit shifted multimodal CE')
                    row.update(scalar_value=float(scalar.detach().cpu()), explicit_shifted_ce=float(manual.detach().cpu()), status='backward_started')
                    save()
                    scalar.backward()
                    state['backward_calls'] += 1
                    contribution = token_gradcam(captured, layout.token_positions, target_name=TARGET_NAME)
                    row['contribution'] = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in contribution.items()}
                    row.update(captured_forward_calls=captured.forward_calls, captured_backward_calls=captured.backward_calls)
                if any((p.grad is not None for p in params)):
                    raise RuntimeError('Frozen parameters accumulated gradients')
                row['status'] = 'complete'
                state['probes_completed'] += 1
                save()
                del scalar, manual, logits, outputs, captured
            finally:
                for handle in handles:
                    handle.remove()
            gc.collect()
            torch.cuda.empty_cache()
            budget()
    state['parameter_flags_restored'] = [p.requires_grad for p in params] == old_flags
    state['parameter_grads_present'] = sum((p.grad is not None for p in params))
    if not state['parameter_flags_restored'] or state['parameter_grads_present']:
        raise RuntimeError('Parameter freeze/restoration contract failed')
    validate_counts(state)
    runtime.assert_assets_unchanged()
    state.update(status='computation_complete', model_forward_calls=state['generation_model_forward_calls'] + state['teacher_forward_calls'], prediction_requests=1 + state['teacher_forward_calls'], peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(), peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
    save()
