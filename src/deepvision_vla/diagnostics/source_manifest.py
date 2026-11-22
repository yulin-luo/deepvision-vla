"""Portable source seals; file relocation never approves historical NN reuse."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('Source paths must be nonempty POSIX-relative names')
    logical = PurePosixPath(relative)
    if logical.is_absolute() or '..' in logical.parts or ':' in relative:
        raise ValueError('Unsafe source path')
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*logical.parts).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Source file escapes declared checkout')
    return path


def build_source_manifest(root, relative_paths):
    paths = list(relative_paths)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Explicit nonempty unique source list required')
    return {'schema': 'deepvision_diagnostic_source_manifest_v1',
            'historical_cache_equivalence_approved': False,
            'files': [{'path': name, 'sha256': file_sha256(source_path(root, name))}
                      for name in sorted(paths)]}


def verify_source_manifest(root, manifest):
    if (manifest.get('schema') != 'deepvision_diagnostic_source_manifest_v1'
            or manifest.get('historical_cache_equivalence_approved') is not False):
        raise ValueError('Source seals do not independently approve historical cache reuse')
    rows = manifest.get('files')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Missing source files')
    actual = build_source_manifest(root, [row['path'] for row in rows])
    if actual != manifest:
        raise ValueError('Current checkout source bytes differ from declared manifest')
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--file', action='append', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_source_manifest(args.source_root, args.file)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
