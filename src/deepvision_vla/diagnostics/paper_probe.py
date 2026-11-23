"""Sealed one-source paper OUTPUT probe in a caller-supervised native environment.

Validation is the default and never loads a model. Execution requires a separate
bounded permit and a live owned parent; this command is not a GPU dispatcher.
Targets are native attribution operators, not physical-action correctness.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
from types import SimpleNamespace

from .adapters.qwen_h16_contracts import RUNTIME_EXPORT_FORMAT
from .factory_imports import configure_factory_imports
from .formal_execution_core import DurableJSONL
from .input_inventory import safe_payload_path
from .input_payloads import prepare_image, prepare_sample_inputs, screened_roi
from .provenance import sha256
from .resources import require_available, pci_identity
from .runtime_observation import actual_memory
from .source_manifest import verify_source_manifest
from .public_surface import release_paths
from ..runtime.gpu_lease import gpu_lease

MODELS = ('openvla', 'pi0', 'qwen_h16')
UUID = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z')
FACTORY_FIELDS = {
    'openvla': {'checkpoint', 'asset_receipt', 'official_inventory'},
    'pi0': {'checkpoint', 'lerobot_source_root', 'asset_receipt'},
    'qwen_h16': {'source_root', 'config_dir', 'checkpoint_root', 'statistics_path',
                 'source_sha256', 'processor_sha256'},
}


def sealed_metadata(path, expected):
    path = Path(path)
    if path.stat().st_size > 20_000_000 or sha256(path) != expected:
        raise ValueError('Metadata size/SHA256 differs')
    return json.loads(path.read_text(encoding='utf-8'))


def validate_configuration(config, permit, config_sha, *, now_epoch, now_monotonic):
    model = config.get('model')
    if config.get('prepared_not_authorized') is True or permit.get('prepared_not_authorized') is True:
        raise ValueError('Prepared examples are not an authorized execution scope')
    if config.get('schema') != 'deepvision_paper_probe_config_v1' or model not in MODELS:
        raise ValueError('Explicit single-model paper probe configuration required')
    if set(config.get('factory', {})) != FACTORY_FIELDS[model]:
        raise ValueError('Factory fields must match the genuine isolated model contract')
    if (permit.get('schema') != 'deepvision_paper_probe_permit_v1'
            or permit.get('config_sha256') != config_sha or permit.get('model') != model
            or permit.get('source_manifest_sha256') != config.get('source_manifest_sha256')):
        raise ValueError('Separate permit must bind exact config/source/model')
    gpu = permit.get('physical_gpu')
    if (type(gpu) is not int or not 0 <= gpu <= 7 or permit.get('exclusive_gpu') is not True
            or permit.get('allow_gpu_sharing') is not False
            or not UUID.fullmatch(permit.get('expected_gpu_uuid', ''))):
        raise ValueError('Bind one exclusive physical GPU, never implicit sharing')
    pci_identity(permit['expected_gpu_pci'])
    for key in ('budget_origin_epoch', 'budget_origin_monotonic', 'deadline_epoch', 'deadline_monotonic'):
        value = permit.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Original finite dual-clock budget required')
    if (not 0 < permit['deadline_epoch']-permit['budget_origin_epoch'] <= 1200
            or not 0 < permit['deadline_monotonic']-permit['budget_origin_monotonic'] <= 1200
            or abs((permit['deadline_epoch']-permit['budget_origin_epoch'])-
                   (permit['deadline_monotonic']-permit['budget_origin_monotonic'])) > .01
            or now_epoch < permit['budget_origin_epoch'] or now_monotonic < permit['budget_origin_monotonic']
            or now_epoch >= permit['deadline_epoch'] or now_monotonic >= permit['deadline_monotonic']):
        raise ValueError('Expired/inconsistent original budget; invocation never resets it')
    if (permit.get('maximum_own_gpu_bytes') != 24_000_000_000
            or permit.get('allocator_gpu_bytes') != 22_000_000_000
            or permit.get('minimum_forward_free_bytes') != 8_000_000_000
            or permit.get('external_watchdog_required') is not True
            or permit.get('maximum_rss_bytes') != 32_000_000_000
            or permit.get('cpu_threads') != 8):
        raise ValueError('Native GPU/RSS/CPU limits must remain explicit and unchanged')
    return model


def process_identity(pid):
    text = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')
    fields = text.rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(fields[19]),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text(encoding='utf-8').strip()}


def verify_parent(permit):
    declared = permit.get('parent')
    if not isinstance(declared, dict) or set(declared) != {'pid', 'start_ticks', 'boot_id'}:
        raise ValueError('A real supervised parent identity is mandatory')
    if type(declared['pid']) is not int or declared['pid'] != os.getppid() or process_identity(os.getppid()) != declared:
        raise RuntimeError('Expected outer supervisor is absent or changed')
    intent = sealed_metadata(permit['parent_intent'], permit['parent_intent_sha256'])
    if (intent.get('schema') != 'deepvision_probe_parent_intent_v1'
            or intent.get('parent') != declared
            or intent.get('config_sha256') != permit['config_sha256']
            or any(intent.get(key) != permit[key] for key in
                   ('budget_origin_epoch','budget_origin_monotonic','deadline_epoch','deadline_monotonic'))
            or intent.get('maximum_child_rss_bytes') != 32_000_000_000
            or intent.get('owned_process_group_only') is not True
            or intent.get('gpu_leases_owned_by_parent') is not False):
        raise ValueError('Actual external parent intent does not bind the bounded child scope')
    source = Path(intent['watchdog_source']).resolve(strict=True)
    if sha256(source) != intent['watchdog_source_sha256']:
        raise ValueError('Reviewed outer watchdog source changed')
    command = Path(f'/proc/{declared["pid"]}/cmdline').read_bytes()
    import hashlib
    if (hashlib.sha256(command).hexdigest() != intent['parent_cmdline_sha256']
            or str(source).encode() not in command.split(b'\0')):
        raise RuntimeError('Live parent command is not the declared watchdog program')


def validate_baseline_binding(binding, sample, baseline_sha, source_manifest_sha):
    required = {'schema', 'baseline_sha256', 'sample_id', 'image_sha256', 'arrays_sha256',
                'anchor', 'instruction', 'source_manifest_sha256', 'status', 'native_equivalence_verified'}
    if (set(binding) != required or binding['schema'] != 'deepvision_qwen_baseline_binding_v1'
            or binding['status'] != 'verified' or binding['native_equivalence_verified'] is not True
            or binding['baseline_sha256'] != baseline_sha
            or binding['source_manifest_sha256'] != source_manifest_sha
            or any(binding[key] != sample.get(key) for key in ('sample_id', 'image_sha256', 'arrays_sha256', 'anchor', 'instruction'))):
        raise ValueError('Qwen external baseline needs exact independently verified native/input/source binding')


def validate_factory_metadata_seals(config):
    factory = config['factory']
    required = {str(Path(factory[key]).resolve(strict=True)) for key in
                ('asset_receipt','official_inventory') if key in factory}
    if config['model'] == 'qwen_h16':
        checkpoint = Path(factory['checkpoint_root'])
        asset = sealed_metadata(config['asset_receipt'], config['asset_receipt_sha256'])
        paths = [checkpoint/'training_config.json', checkpoint/'portable/qwen_backbone/manifest.json',
                 checkpoint/'portable/shared_action_head/manifest.json']
        if asset.get('receipt_format') == RUNTIME_EXPORT_FORMAT:
            contracts = Path(config['package_source_root'])/'configs/diagnostics'
            paths.extend((Path(config['asset_receipt']), contracts/'qwen-h16-runtime-export-inventory.json',
                          contracts/'qwen-h16-vendor-seals.json'))
        else:
            paths.append(Path(factory['config_dir'])/'reference_source_manifest.json')
        required.update(str(path.resolve(strict=True)) for path in paths)
    seals = config.get('factory_metadata_seals')
    if not isinstance(seals, list) or not seals:
        raise ValueError('Exact native factory metadata seals are required')
    observed = {}
    for row in seals:
        path = str(Path(row['path']).resolve(strict=True))
        if path in observed:
            raise ValueError('Duplicate native metadata seal')
        if Path(path).stat().st_size > 20_000_000 or sha256(path) != row['sha256']:
            raise ValueError('Native factory metadata changed')
        observed[path] = row['sha256']
    if not required.issubset(observed):
        raise ValueError('Native factory receipt/inventory/manifest is not byte-bound')
    return observed


def validate_qwen_vendor_binding(config, cpu):
    """The qualified CPU receipt and processor reference fix vendor identity.

    A newly self-consistent config cannot borrow qualification from different
    source bytes. The genuine loader still checks the actual files afterwards.
    """
    from .adapters.qwen_h16 import CRITICAL_SOURCE_FILES
    factory = config['factory']
    qualified_sources = cpu.get('source_hashes')
    if (not isinstance(qualified_sources,dict) or not CRITICAL_SOURCE_FILES.issubset(qualified_sources)
            or factory.get('source_sha256') != qualified_sources):
        raise ValueError('Configured Qwen vendor source hashes differ from qualified CPU receipt')
    reference = (Path(factory['config_dir'])/'reference_source_manifest.json').resolve(strict=True)
    rows = [row for row in config.get('factory_metadata_seals', [])
            if Path(row['path']).resolve(strict=True) == reference]
    if len(rows) != 1:
        raise ValueError('Qwen processor reference needs one separate exact metadata seal')
    value = sealed_metadata(reference, rows[0]['sha256'])
    files = value.get('files')
    if (not isinstance(files,list) or not files or any(not isinstance(row,dict) or
            not isinstance(row.get('path'),str) or not isinstance(row.get('sha256'),str) for row in files)):
        raise ValueError('Complete sealed processor reference file map required')
    expected = {row['path']:row['sha256'] for row in files}
    if len(expected) != len(files) or factory.get('processor_sha256') != expected:
        raise ValueError('Configured Qwen processor hashes differ from sealed reference manifest')
    return {'source_hashes_equal_qualified_CPU_receipt':True,
            'processor_hashes_equal_sealed_reference':True,
            'processor_reference_sha256':rows[0]['sha256'],
            'actual_file_bytes_checked_again_by_native_loader':True}


def prepare_inputs(config):
    sample = sealed_metadata(config['sample'], config['sample_sha256'])
    root = Path(config['data_root'])
    image_path = safe_payload_path(root, sample['image'])
    arrays_path = safe_payload_path(root, sample['arrays'])
    if sha256(image_path) != sample['image_sha256'] or sha256(arrays_path) != sample['arrays_sha256']:
        raise ValueError('Sealed source image/NPZ bytes differ; no replacement allowed')
    if not re.fullmatch('[0-9a-f]{20}', sample.get('sample_id', '')):
        raise ValueError('Explicit source identity required')
    image = prepare_image(sample, image_path)
    filtered = screened_roi(sample)
    if not filtered['candidate_boxes']:
        raise ValueError('No original source ROI survives individual-box60 screening')
    state7 = audit = None
    if config['model'] == 'pi0':
        image, state7, audit = prepare_sample_inputs(sample, image_path, arrays_path)
    return sample, image, filtered, state7, audit


def model_metadata(config, sample):
    """Validate genuine model contracts without any model import/materialization."""
    model = config['model']
    result = {}
    if model == 'pi0':
        from .adapters.pi0_gradients import validate_cpu_receipt, verify_output_seam_source
        cpu = sealed_metadata(config['cpu_receipt'], config['cpu_receipt_sha256'])
        validate_cpu_receipt(cpu)
        proof = sealed_metadata(config['source_proof'], config['source_proof_sha256'])
        helper = Path(config['factory']['lerobot_source_root'])/'lerobot/common/policies/pi0/paligemma_with_expert.py'
        verify_output_seam_source(proof, helper)
        result.update(cpu=cpu, proof=proof)
    elif model == 'qwen_h16':
        from .adapters import qwen_h16
        from .adapters.qwen_h16_contracts import REVISION, RUNTIME_EXPORT_ID, matches_controlled_repository, validate_cpu_receipt
        from .adapters.qwen_h16_gradients import validate_baseline, prediction_float32_bytes

        asset = sealed_metadata(config['asset_receipt'], config['asset_receipt_sha256'])
        published = asset.get('receipt_format') == RUNTIME_EXPORT_FORMAT
        repo_matches = asset.get('repo') == RUNTIME_EXPORT_ID if published else matches_controlled_repository(asset.get('repo'))
        if asset.get('status') != 'complete' or not repo_matches or asset.get('revision') != REVISION:
            raise ValueError('Actual qwen_only acquisition identity is required')
        if published:
            from .prepare_launch import controlled_qwen_factory

            checkout = Path(config['package_source_root']).resolve(strict=True)
            inventory_file = checkout/'configs/diagnostics/qwen-h16-runtime-export-inventory.json'
            inventory = sealed_metadata(inventory_file, asset['inventory_sha256'])
            factory = config['factory']
            expected = controlled_qwen_factory(checkout, Path(factory['checkpoint_root']), inventory, asset,
                                               Path(factory['config_dir']), Path(factory['statistics_path']))
            if factory != expected:
                raise ValueError('Qwen attribution factory differs from the pinned export')
            if any(config.get(key) is not None for key in ('baseline', 'baseline_sha256',
                                                          'baseline_binding', 'baseline_binding_sha256',
                                                          'cpu_receipt', 'cpu_receipt_sha256')):
                raise ValueError('The published runtime probe creates its own native baseline')
            cpu = None
            vendor_binding = {'published_runtime_export_verified': True,
                              'baseline_mode': 'fresh_native_unmasked_prediction'}
        else:
            cpu = sealed_metadata(config['cpu_receipt'], config['cpu_receipt_sha256'])
            validate_cpu_receipt(cpu, sha256(qwen_h16.__file__))
            vendor_binding = validate_qwen_vendor_binding(config, cpu)
        required = {'training_config.json', 'portable/qwen_backbone/manifest.json',
                    'portable/qwen_backbone/model.safetensors', 'portable/shared_action_head/manifest.json',
                    'portable/shared_action_head/action_head.safetensors'}
        root = Path(config['factory']['checkpoint_root']).resolve(strict=True)
        snapshots = {}
        acquired = {}
        for row in asset['files']:
            prefix = 'qwen_only/checkpoint-5-10000/'
            if row['path'].startswith(prefix):
                name = row['path'][len(prefix):]
                path = safe_payload_path(root, name)
                stat = path.stat()
                if row.get('verified') is not True or stat.st_size != row['size']:
                    raise ValueError('Qwen component is unverified or changed')
                snapshots[str(path)] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
                acquired[name] = row
                if path.suffix != '.safetensors' and sha256(path) != row['sha256']:
                    raise ValueError('Qwen acquisition metadata bytes changed')
                required.discard(name)
        if required:
            raise ValueError('Missing genuine Qwen H16 components')
        for component, filename in (('qwen_backbone','model.safetensors'),('shared_action_head','action_head.safetensors')):
            saved = json.loads((root/'portable'/component/'manifest.json').read_text(encoding='utf-8'))
            expected = acquired['portable/'+component+'/'+filename]
            if saved.get('weights_file') != filename or saved.get('weights_sha256') != expected['sha256']:
                raise ValueError('Saved weights manifest differs from pinned acquisition receipt')
        if published:
            return {'cpu': None, 'baseline': None, 'snapshots': snapshots, 'vendor_binding': vendor_binding}
        baseline = sealed_metadata(config['baseline'], config['baseline_sha256'])
        binding = sealed_metadata(config['baseline_binding'], config['baseline_binding_sha256'])
        validate_baseline_binding(binding, sample, config['baseline_sha256'], config['source_manifest_sha256'])
        if baseline.get('schema') == 'deepvision_qwen_h16_native_baseline_v1':
            if (baseline.get('status') != 'complete' or baseline.get('lease_released') is not True
                    or not matches_controlled_repository(baseline.get('repo')) or baseline.get('revision') != REVISION
                    or baseline.get('sample_id') != sample['sample_id']
                    or baseline.get('instruction') != sample['instruction']
                    or baseline.get('model_forward_calls') != 1 or baseline.get('head_forward_calls') != 1
                    or not isinstance(baseline.get('layout', {}).get('adapter'), dict)):
                raise ValueError('A completed exact native full-H16 single-forward baseline is required')
            prediction_float32_bytes(baseline['baseline_normalized_h16'])
        else:
            # Legacy seven-call receipt is supported only with the same strict
            # external source/native-equivalence binding. No implicit promotion.
            validate_baseline(baseline, sample)
        result.update(cpu=cpu, baseline=baseline, snapshots=snapshots, vendor_binding=vendor_binding)
    return result


def cpu_factory(model, factory):
    """Real strict CPU materialization before GPU lease; no substitute backends."""
    import torch
    torch.set_num_threads(8)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU factory must start without any CUDA context')
    configure_factory_imports(model, factory)
    if model == 'openvla':
        from .adapters.openvla_runtime import load_openvla_runtime
        adapter = load_openvla_runtime(**factory)
        parameters = adapter.model.parameters()
    elif model == 'pi0':
        from .adapters.openpi_bridge import load_bridge_pi0
        adapter = load_bridge_pi0(**factory)
        parameters = adapter.policy.parameters()
    else:
        from .adapters.qwen_h16 import load_qwen_h16
        adapter = load_qwen_h16(**factory)
        parameters = list(adapter.model.parameters())+list(adapter.head.parameters())
    if torch.cuda.is_initialized() or any(p.device.type != 'cpu' for p in parameters):
        raise RuntimeError('Strict CPU factory initialized CUDA or materialized elsewhere')
    return adapter


def verify_probe_weights(config):
    """Reuse the formal executor's complete pinned component byte verification.

    A receipt's declared SHA and a strict tensor key/header load are not a hash
    of the actual large tensor payload. No framework import occurs here.
    """
    from .formal_diagnostic_executor import verify_factory_assets
    model = config['model']
    receipt = config['asset_receipt'] if model == 'qwen_h16' else config['factory']['asset_receipt']
    return verify_factory_assets({'model': model, 'factory_kwargs': config['factory'],
                                  'asset_receipt': receipt,
                                  'asset_receipt_prefix': 'qwen_only/checkpoint-5-10000' if model == 'qwen_h16' else ''})


def materialize_cpu_once(config, state, save):
    """Hash every selected weight before any native loading or UUID lease."""
    state['phase'] = 'verifying_actual_native_weight_bytes'; save()
    state['actual_weight_components_rehashed'] = verify_probe_weights(config); save()
    state['phase'] = 'cpu_loading'; state['model_load_attempts'] = 1; save()
    adapter = cpu_factory(config['model'], config['factory'])
    state['model_loads'] = 1
    return adapter


def run(config_path, config_sha, permit_path, permit_sha, *, execute=False):
    config = sealed_metadata(config_path, config_sha)
    permit = sealed_metadata(permit_path, permit_sha)
    model = validate_configuration(config, permit, config_sha, now_epoch=time.time(), now_monotonic=time.monotonic())
    manifest = sealed_metadata(config['source_manifest'], config['source_manifest_sha256'])
    if Path(config['package_source_root']).resolve(strict=True) != Path(__file__).resolve().parents[3]:
        raise ValueError('Source seals must bind the actual editable checkout in use')
    verify_source_manifest(config['package_source_root'], manifest)
    required_sources = set(release_paths(Path(config['package_source_root'])/'src'))
    if not required_sources.issubset({row['path'] for row in manifest['files']}):
        raise ValueError('Source manifest omits part of the public diagnostic closure')
    validate_factory_metadata_seals(config)
    sample, image, filtered, state7, input_audit = prepare_inputs(config)
    metadata = model_metadata(config, sample)
    if not execute:
        return {'status': 'validated_not_executed', 'model': model, 'model_loads': 0, 'model_calls': 0, 'GPU_calls': 0}
    verify_parent(permit)
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') not in ('', str(permit['physical_gpu'])):
        raise RuntimeError('Conflicting numeric GPU environment')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(permit['physical_gpu'])
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    selected = require_available(permit['physical_gpu'], permit['expected_gpu_uuid'], sharing=False, minimum_free_bytes=32_000_000_000)
    if pci_identity(selected['pci']) != pci_identity(permit['expected_gpu_pci']):
        raise RuntimeError('Selected physical PCI differs from permit')
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=False)
    with Path(config['once']).open('x', encoding='utf-8') as once:
        json.dump({'config_sha256': config_sha, 'permit_sha256': permit_sha, 'worker': process_identity(os.getpid())}, once)
        once.flush(); os.fsync(once.fileno())
    journal = DurableJSONL(output/'events.jsonl')
    state = {'status': 'starting', 'model': model, 'sample_id': sample['sample_id'],
             'image_sha256': sample['image_sha256'], 'arrays_sha256': sample['arrays_sha256'],
             'instruction': sample['instruction'], 'anchor': sample['anchor'],
             'config_sha256': config_sha, 'permit_sha256': permit_sha,
             'deadline_epoch': permit['deadline_epoch'], 'deadline_monotonic': permit['deadline_monotonic'],
             'model_load_attempts': 0, 'model_loads': 0, 'probes': [], 'probes_completed': 0,
             'backward_calls': 0, 'model_forward_calls': 0, 'head_forward_calls': 0,
             'generation_requests': 0, 'generation_model_forward_calls': 0, 'generation_projector_calls': 0,
             'teacher_forward_calls': 0, 'teacher_projector_calls': 0,
             'native_calls': {'embed_prefix': 0, 'denoise_step': 0}, 'lease_released': False,
             'gt_mse': None, 'gt_mapping_verified': False, 'optimizer_steps': 0,
             'paper_reproduction': False, 'original_paper_checkpoint_binding_verified': False}
    def save():
        if time.monotonic() >= permit['deadline_monotonic'] or time.time() >= permit['deadline_epoch']:
            raise TimeoutError('Original dual-clock probe deadline expired')
        verify_parent(permit)
        journal.append('progress', state=state.copy())
    def expired(_sig, _frame):
        raise TimeoutError('Original finite probe deadline expired')
    remaining = min(permit['deadline_epoch']-time.time(), permit['deadline_monotonic']-time.monotonic())
    if remaining <= 0: raise TimeoutError('No budget left before materialization')
    old_alarm = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    adapter = None
    args = SimpleNamespace(gpu=permit['physical_gpu'], allow_gpu_sharing=False, **{
        'checkpoint': Path(config['factory'].get('checkpoint', config['factory'].get('checkpoint_root', '.'))),
        'asset_receipt': Path(config.get('asset_receipt', config['factory'].get('asset_receipt', '.'))),
        'official_inventory': config['factory'].get('official_inventory'),
        'lerobot_source': Path(config['factory'].get('lerobot_source_root', '.')),
        'config_dir': Path(config['factory'].get('config_dir', '.'))})
    try:
        actual_memory(require_available=True)
        adapter = materialize_cpu_once(config, state, save); args.loaded_adapter = adapter
        actual_memory(); save()
        selected = require_available(args.gpu, permit['expected_gpu_uuid'], sharing=False, minimum_free_bytes=32_000_000_000)
        if pci_identity(selected['pci']) != pci_identity(permit['expected_gpu_pci']):
            raise RuntimeError('PCI changed after CPU load')
        with gpu_lease(selected['uuid']) as lease:
            state['lease'] = lease; save()
            try:
                if model == 'openvla':
                    from .adapters.openvla_output import run_native
                    run_native(args, sample, image, filtered, selected, state, save)
                elif model == 'pi0':
                    from .adapters.pi0_output import run_native
                    run_native(args, sample, image, state7, input_audit, metadata['cpu'], metadata['proof'], selected, state, save)
                else:
                    from .adapters.qwen_h16_output import run_native
                    run_native(args, sample, image, filtered, metadata['cpu'], metadata['baseline'], selected, state, save)
                    for name, expected in metadata['snapshots'].items():
                        stat = Path(name).stat()
                        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != expected:
                            raise RuntimeError('Qwen component changed during probe')
            finally:
                args.loaded_adapter = None; adapter = None
                gc.collect()
                torch = sys.modules.get('torch')
                if torch is not None and torch.cuda.is_initialized(): torch.cuda.empty_cache()
        verify_source_manifest(config['package_source_root'], manifest)
        state.update(status='complete', lease_released=True); save()
        with (output/'result.json').open('x', encoding='utf-8') as stream:
            json.dump(state, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
        return state
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        journal.append('failed_terminal', state=state.copy())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        journal.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'config-sha256', 'permit', 'permit-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--execute', action='store_true', help='Requires a separate bounded permit and real external watchdog')
    args = parser.parse_args(argv)
    result = run(args.config, args.config_sha256, args.permit, args.permit_sha256, execute=args.execute)
    print(json.dumps({'status': result['status'], 'model': result['model']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
