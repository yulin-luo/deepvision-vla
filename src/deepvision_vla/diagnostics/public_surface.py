"""Explicit diagnostic release surface and static import-closure verification.

This validates source without importing Torch, third-party model code or private
operational modules. Vendor execution remains guarded by each strict factory.
"""
import argparse
import ast
import importlib.util
import json
import hashlib
from pathlib import Path

PREFIX = 'deepvision_vla.diagnostics.'
MODULES = tuple(PREFIX + name for name in '''
__init__ bridge contracts controls dataset_reference_metrics
formal_diagnostic_executor formal_execution_core formal_plan formal_statistics
geometry gradient_capture gradient_output_capture interventions
openvla_processor_geometry pi0_preprocessing_geometry process_observation
qwen_processor_geometry random_controls resources roi_filter sampling
predictors factory_imports provenance runtime_observation preprocessing_limits
input_inventory input_payloads model_geometry paper_probe source_manifest public_surface
adapters.__init__ adapters.openpi_bridge adapters.openpi_bridge_config
adapters.openpi_bridge_helper_compat adapters.openpi_bridge_unused_head
adapters.openpi_contract adapters.openvla_runtime adapters.qwen_h16
adapters.qwen_h16_contracts adapters.pi0_gradients adapters.qwen_h16_gradients
adapters.openvla_gradients adapters.openvla_output adapters.pi0_output adapters.qwen_h16_output
'''.split()) + ('deepvision_vla.runtime.gpu_lease',)
FORBIDDEN = ('canary', 'role_pilot', 'gradient_probe', 'legacy_success_cache_export',
             'qwen_state_identity_runner', 'qwen_numeric_identity', 'qwen_candidate_metadata')
# Exact call-site AST allowlist, not permission for arbitrary dynamic imports.
# Each native factory verifies real source hashes before these vendor operations.
VENDOR_DYNAMIC_CONTRACTS = {
    PREFIX+'adapters.openpi_bridge': {'spec_from_file_location', 'exec_module'},
    PREFIX+'adapters.qwen_h16': {'spec_from_file_location', 'exec_module'},
    PREFIX+'adapters.openvla_runtime': {'from_pretrained_trust_remote_code'},
    PREFIX+'openvla_processor_geometry': {'from_pretrained_trust_remote_code'},
}
OPTIONAL_MODULES = tuple(PREFIX + name for name in (
    'reconstruct_bridge', 'fetch_bridge_shards', 'asset_receipts', 'formal_reporting', 'formal_reporting_plots', 'attribution_plot',
    '__main__', 'formal_single_model_watchdog', 'paper_probe_watchdog',
    'prepare_launch', 'formal_reporting_verify',
))


def module_file(src, module):
    return Path(src).joinpath(*module.split('.')).with_suffix('.py')


def import_targets(node, module):
    """Resolve both `from package import child` and ordinary local imports."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    package = module.rpartition('.')[0]
    base = importlib.util.resolve_name('.'*node.level+(node.module or ''), package) if node.level else node.module
    if not base:
        return []
    return [base] + [base+'.'+alias.name for alias in node.names if alias.name != '*']


def verify_public_surface(src):
    src = Path(src).resolve(strict=True)
    all_files = {'.'.join(path.relative_to(src).with_suffix('').parts): path
                 for path in (src/'deepvision_vla').rglob('*.py')}
    selected = MODULES+tuple(name for name in OPTIONAL_MODULES if name in all_files)
    allowed = set(selected)
    missing = [name for name in MODULES if name not in all_files]
    if missing:
        raise ValueError('Missing public source modules: '+','.join(missing))
    imports = []
    dynamic = []
    for module in selected:
        tree = ast.parse(all_files[module].read_text(encoding='utf-8'))
        import_aliases = {}
        for item in ast.walk(tree):
            if isinstance(item, ast.ImportFrom) and item.module in {'importlib', 'builtins'}:
                for alias in item.names:
                    import_aliases[alias.asname or alias.name] = alias.name
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and any(a.name == '*' for a in node.names):
                    raise ValueError('Wildcard import cannot establish public closure')
                for name in import_targets(node, module):
                    if any(part in name for part in FORBIDDEN):
                        raise ValueError('Public source depends on private operation: '+name)
                    if name in all_files and name not in allowed:
                        # Package __init__ files are dependency-free and are
                        # included explicitly during source staging.
                        if name not in {'deepvision_vla.__init__', 'deepvision_vla.runtime.__init__'}:
                            raise ValueError('Local dependency outside public allowlist: '+name)
                    imports.append({'module': module, 'line': node.lineno, 'target': name})
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
                name = import_aliases.get(name, name)
                if (isinstance(node.func,ast.Call) and isinstance(node.func.func,ast.Name)
                        and node.func.func.id=='getattr' and len(node.func.args)==2
                        and isinstance(node.func.args[1],ast.Constant)):
                    name=node.func.args[1].value
                remote = [k.value for k in node.keywords if k.arg=='trust_remote_code']
                if name=='from_pretrained' and remote:
                    if len(remote)!=1 or not isinstance(remote[0],ast.Constant):
                        raise ValueError('Dynamic remote-code policy is not statically explicit')
                    if remote[0].value is True:name='from_pretrained_trust_remote_code'
                sensitive = (isinstance(node.func, ast.Name) and name in {'__import__', 'exec', 'eval'}) or name in {'import_module', 'spec_from_file_location', 'exec_module'}
                sensitive = sensitive or name=='from_pretrained_trust_remote_code'
                if sensitive:
                    if name not in VENDOR_DYNAMIC_CONTRACTS.get(module, set()):
                        raise ValueError('Unreviewed dynamic import/execution call: '+module+':'+str(node.lineno))
                    dynamic.append({'module': module, 'line': node.lineno, 'call': name,
                                    'contract': 'unchanged_strict_factory_source_and_asset_seals_required'})
    return {'status': 'public_static_import_closure_verified', 'modules': len(selected),
            'imports': imports, 'vendor_dynamic_calls': dynamic,
            'runtime_or_cache_equivalence_approved': False}


def release_paths(src):
    verify_public_surface(src)
    selected = MODULES+tuple(name for name in OPTIONAL_MODULES if module_file(src,name).is_file())
    paths = [str(module_file(Path('src'), name)).replace('\\', '/') for name in selected]
    return sorted(paths + ['src/deepvision_vla/__init__.py', 'src/deepvision_vla/runtime/__init__.py'])


def stage_public_sources(src, output):
    """Export only reviewed diagnostic modules into a new source staging tree.

    This is not a complete replacement for the existing inference/RLBench repo.
    It is a checked overlay for a later reviewed merge, never a public push.
    """
    src = Path(src).resolve(strict=True)
    paths = release_paths(src)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for relative in paths:
        source = src.joinpath(*Path(relative).parts[1:]).resolve(strict=True)
        if not source.is_relative_to(src):
            raise ValueError('A release source resolves outside the checkout')
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        with target.open('xb') as stream:
            stream.write(raw)
        rows.append({'path': relative, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    manifest = {'schema': 'deepvision_diagnostic_source_overlay_v1', 'files': rows,
                'source_only': True, 'runtime_acceptance': 'pending',
                'historical_cache_equivalence_approved': False}
    with (output/'source_overlay_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True, type=Path)
    output = parser.add_mutually_exclusive_group()
    output.add_argument('--stage-to', type=Path, help='New local source-only overlay; never commit or push')
    output.add_argument('--source-manifest-to', type=Path, help='Create new seals for the complete public source closure')
    args = parser.parse_args(argv)
    if args.source_manifest_to:
        from .source_manifest import build_source_manifest
        result = build_source_manifest(args.src.resolve().parent, release_paths(args.src))
        with args.source_manifest_to.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,indent=2);stream.write('\n')
    else:
        result = stage_public_sources(args.src, args.stage_to) if args.stage_to else verify_public_surface(args.src)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
