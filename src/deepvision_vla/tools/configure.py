"""Bind a verified checkpoint to an image-only or full RLBench environment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from deepvision_vla.paths import ASSETS_ROOT, require_source_checkout
from deepvision_vla.runtime.checkpoint import (
    CheckpointBindingError, checkpoint_binding, load_inventory, verify_checkpoint,
    validate_checkpoint_binding,
)


def _legacy_receipt(path):
    """Only old, unversioned receipts are eligible for fresh hashing."""
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return False
    return isinstance(value, dict) and 'schema' not in value and 'seal' not in value


def _binding_for_configuration(checkpoint, output, supplied_receipt=None):
    candidate = Path(supplied_receipt).expanduser().resolve(strict=True) if supplied_receipt else checkpoint.parent / 'checkpoint_verification.json'
    if candidate.is_file() and not _legacy_receipt(candidate):
        return checkpoint_binding(checkpoint, candidate), 'reused_full_verification'
    # Legacy receipts do not prove a stable twenty-file binding. Hash once into
    # the new config output area, never mutate/replace an old receipt.
    fresh = output.with_name(output.stem + '-checkpoint-verification.json')
    if not fresh.exists():
        result = verify_checkpoint(
            checkpoint, fresh,
            progress=lambda name, ok: print(name, 'PASS' if ok else 'FAIL', flush=True),
        )
        if result['status'] != 'passed':
            raise CheckpointBindingError('Checkpoint verification failed; failed receipt preserved at ' + str(fresh))
    return checkpoint_binding(checkpoint, fresh), 'new_full_verification'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--env-prefix', required=True)
    parser.add_argument('--deps-root')
    parser.add_argument('--image-only', action='store_true')
    parser.add_argument('--native-library-path', action='append', default=[], metavar='DIRECTORY')
    parser.add_argument('--verification-receipt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sample-data')
    args = parser.parse_args()
    if not args.image_only and not args.deps_root:
        parser.error('--deps-root is required for RLBench; use --image-only for image inference')
    if args.image_only and args.native_library_path:
        parser.error('--native-library-path is only meaningful for an RLBench configuration')
    source = require_source_checkout()
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    prefix = Path(args.env_prefix).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().absolute()
    if output.exists():
        raise FileExistsError('A fresh config output is required')
    python = prefix / 'bin/python'
    if not python.is_file():
        raise FileNotFoundError('The selected environment has no bin/python: ' + str(prefix))
    native = {'rlbench_root': '', 'pyrep_root': '', 'coppeliasim_root': '',
              'host_library_path': '', 'native_library_paths': [], 'display_argv': []}
    if not args.image_only:
        dependencies = Path(args.deps_root).expanduser().resolve(strict=True)
        for name in ('PyRep', 'RLBench', 'CoppeliaSim'):
            if not (dependencies / name).is_dir():
                raise FileNotFoundError('Run deepvision prepare-rlbench first: ' + name)
        libraries = []
        for value in args.native_library_path:
            directory = Path(value).expanduser().resolve(strict=True)
            if not directory.is_dir():
                raise NotADirectoryError('Native library path is not a directory: ' + str(directory))
            libraries.append(str(directory))
        native.update(rlbench_root=str(dependencies / 'RLBench'),
                      pyrep_root=str(dependencies / 'PyRep'),
                      coppeliasim_root=str(dependencies / 'CoppeliaSim'),
                      host_library_path=str(prefix / 'lib'), native_library_paths=libraries,
                      display_argv=[str(python), '-m', 'deepvision_vla.runtime.display', '--'])
    profile = ASSETS_ROOT / 'rlbench_profile.json'
    if not profile.is_file():
        raise FileNotFoundError('Packaged RLBench profile missing')
    output.parent.mkdir(parents=True, exist_ok=True)
    binding, evidence_source = _binding_for_configuration(checkpoint, output, args.verification_receipt)
    expected, _ = load_inventory()
    view = output.parent / (output.stem + '-dino-runtime')
    if view.exists():
        if not (view / 'pytorch_model.bin').samefile(checkpoint / 'dinov3_model/pytorch_model.bin'):
            raise ValueError('Existing DINO view points to different weights')
        for name in ('config.json', 'preprocessor_config.json'):
            if (view / name).read_bytes() != (ASSETS_ROOT / 'dino_runtime' / name).read_bytes():
                raise ValueError('Existing DINO view configuration differs')
    else:
        subprocess.run([sys.executable, '-m', 'deepvision_vla.tools.prepare_dino',
                        '--checkpoint', str(checkpoint), '--output', str(view)], check=True)
    config = {'source_root': str(source), **binding, 'image_only': args.image_only,
              'dino_runtime_root': str(view), 'python_bin': str(python),
              'profile_path': str(profile), **native,
              'sample_data_path': str(Path(args.sample_data).expanduser().resolve(strict=True)) if args.sample_data else '',
              'checkpoint_small_file_sha256': {name: expected[name]['sha256']
                                                for name in ('training_config.json', 'stats_data.json')}}
    validate_checkpoint_binding(config)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(config, stream, indent=2)
    print(json.dumps({'config': str(output), 'image_only': args.image_only,
                      'checkpoint_verification': evidence_source,
                      'verification_receipt': binding['checkpoint_verification_receipt'],
                      'status': 'configured_not_yet_environment_or_rollout_validated'}))


if __name__ == '__main__':
    main()
