"""Pure-CPU integrity/TOCTOU tests using twenty tiny synthetic files, not weights."""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from deepvision_vla.runtime import checkpoint as binding
from deepvision_vla.tools import configure


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CheckpointHardeningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='checkpoint-tiny-fixture-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cp = self.root / 'checkpoint'
        self.cp.mkdir()
        self.assets = self.root / 'assets'
        (self.assets / 'dino_runtime').mkdir(parents=True)
        actual = json.loads((binding.ASSETS_ROOT / 'checkpoint_inventory.origin.json').read_bytes())
        self.inventory = {}
        for index, name in enumerate(sorted(actual)):
            path = self.cp / name
            path.parent.mkdir(parents=True, exist_ok=True)
            content = ('fixture-' + str(index)).encode()
            if name.endswith('.json'):
                content = b'{}'
            path.write_bytes(content)
            self.inventory[name] = {'sha256': sha(path), 'size': len(content)}
        self.inventory_path = self.assets / 'checkpoint_inventory.origin.json'
        self.inventory_path.write_text(json.dumps(self.inventory), encoding='utf-8')
        (self.assets / 'rlbench_profile.json').write_text('{}')
        for name in ['config.json', 'preprocessor_config.json']:
            (self.assets / 'dino_runtime' / name).write_text('{}')
        self.receipt = self.root / 'verification.json'
        self.env = self.root / 'environment'
        (self.env / 'bin').mkdir(parents=True)
        (self.env / 'bin/python').write_text('not executed: synthetic test file')
        (self.env / 'lib').mkdir()
        self.deps = self.root / 'deps'
        for name in ['PyRep', 'RLBench', 'CoppeliaSim']:
            (self.deps / name).mkdir(parents=True)
        self.patch_assets = mock.patch.object(binding, 'ASSETS_ROOT', self.assets)
        self.patch_config_assets = mock.patch.object(configure, 'ASSETS_ROOT', self.assets)
        self.patch_assets.start(); self.patch_config_assets.start()
        self.addCleanup(self.patch_assets.stop); self.addCleanup(self.patch_config_assets.stop)

    def verified(self, receipt=None):
        target = receipt or self.receipt
        result = binding.verify_checkpoint(self.cp, target)
        self.assertEqual(result['status'], 'passed')
        return binding.checkpoint_binding(self.cp, target)

    def write_receipt(self, data, config=None):
        self.receipt.write_text(json.dumps(data))
        if config is not None:
            config['checkpoint_verification_receipt_sha256'] = sha(self.receipt)

    def fake_dino_view(self, argv, check):
        self.assertTrue(check)
        self.assertEqual(argv[1:3], ['-m', 'deepvision_vla.tools.prepare_dino'])
        output = Path(argv[argv.index('--output') + 1])
        output.mkdir()
        for name in ['config.json', 'preprocessor_config.json']:
            shutil.copy2(self.assets / 'dino_runtime' / name, output / name)
        target = self.cp / 'dinov3_model/pytorch_model.bin'
        if os.name == 'nt':
            os.link(target, output / 'pytorch_model.bin')
        else:
            (output / 'pytorch_model.bin').symlink_to(target)

    def run_configure(self, output, *extra):
        args = ['configure', '--checkpoint', str(self.cp), '--env-prefix', str(self.env), '--output', str(output), *extra]
        with mock.patch.object(sys, 'argv', args), mock.patch.object(configure.subprocess, 'run', side_effect=self.fake_dino_view):
            with contextlib.redirect_stdout(io.StringIO()):
                configure.main()
        return json.loads(output.read_bytes())

    def with_runtime_view(self, config):
        view = self.root / 'dino-runtime'
        self.fake_dino_view([sys.executable, '-m', 'deepvision_vla.tools.prepare_dino',
                             '--checkpoint', str(self.cp), '--output', str(view)], True)
        return dict(config, dino_runtime_root=str(view))

    def test_all_twenty_sha_read_once_runtime_reuse_is_stat_only(self):
        with mock.patch.object(binding, '_read_digest', wraps=binding._read_digest) as hashes:
            config = self.verified()
            self.assertEqual(hashes.call_count, 20)
            summary = binding.validate_checkpoint_binding(config)
            self.assertEqual(summary['weight_files_rehashed'], 0)
            self.assertEqual(hashes.call_count, 20)
        receipt = json.loads(self.receipt.read_bytes())
        self.assertEqual(receipt['schema_version'], 1)
        self.assertEqual(len(receipt['files']), 20)
        self.assertTrue(all(row['before'] == row['after'] == row['stat'] for row in receipt['files']))

    def test_weight_changed_after_verification_fails_without_rehash(self):
        config = self.verified()
        path = self.cp / 'action_head.pt'
        path.write_bytes(b'changed file')
        with mock.patch.object(binding, '_read_digest', side_effect=AssertionError('must not hash weights at runtime')):
            with self.assertRaisesRegex(binding.CheckpointBindingError, 'member changed'):
                binding.validate_checkpoint_binding(config)

    def test_runtime_view_valid_json_and_weight_alias_are_accepted_without_weight_hash(self):
        config = self.with_runtime_view(self.verified())
        with mock.patch.object(binding, '_read_digest', side_effect=AssertionError('must not rehash weights')):
            result = binding.validate_dino_runtime_view(config)
            self.assertTrue(result['weight_samefile'])
            self.assertEqual(binding.validate_checkpoint_binding(config)['file_count'], 20)

    def test_runtime_processor_change_is_rejected_by_regular_binding_gate(self):
        config = self.with_runtime_view(self.verified())
        (Path(config['dino_runtime_root']) / 'preprocessor_config.json').write_text('{"image_mean":[0,0,0]}')
        with self.assertRaisesRegex(binding.CheckpointBindingError, 'preprocessor_config.json'):
            binding.validate_checkpoint_binding(config)

    def test_runtime_model_config_change_is_rejected(self):
        config = self.with_runtime_view(self.verified())
        (Path(config['dino_runtime_root']) / 'config.json').write_text('{"image_size":128}')
        with self.assertRaisesRegex(binding.CheckpointBindingError, 'config.json'):
            binding.validate_dino_runtime_view(config)

    def test_runtime_weight_copy_is_not_accepted_as_same_checkpoint_alias(self):
        config = self.with_runtime_view(self.verified())
        alias = Path(config['dino_runtime_root']) / 'pytorch_model.bin'
        alias.unlink()
        alias.write_bytes((self.cp / 'dinov3_model/pytorch_model.bin').read_bytes())
        with self.assertRaisesRegex(binding.CheckpointBindingError, 'different checkpoint file'):
            binding.validate_checkpoint_binding(config)

    def test_same_bytes_replaced_inode_requires_reverification(self):
        config = self.verified()
        path = self.cp / 'action_head.pt'
        replacement = self.root / 'new-file'
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
        with self.assertRaises(binding.CheckpointBindingError):
            binding.validate_checkpoint_binding(config)

    def test_receipt_tamper_and_rebound_digest_still_fail_seal(self):
        config = self.verified()
        data = json.loads(self.receipt.read_bytes())
        data['verified_utc'] = 'changed'
        self.write_receipt(data)
        with self.assertRaisesRegex(binding.CheckpointBindingError, 'receipt bytes changed'):
            binding.validate_checkpoint_binding(config)
        config['checkpoint_verification_receipt_sha256'] = sha(self.receipt)
        with self.assertRaisesRegex(binding.CheckpointBindingError, 'unsealed'):
            binding.validate_checkpoint_binding(config)

    def test_non_pinned_sha_produces_failed_receipt(self):
        path = self.cp / 'action_head.pt'
        path.write_bytes(b'z' * path.stat().st_size)
        receipt = binding.verify_checkpoint(self.cp, self.receipt)
        self.assertEqual(receipt['status'], 'failed')
        self.assertTrue(self.receipt.exists())
        with self.assertRaises(binding.CheckpointBindingError):
            binding.checkpoint_binding(self.cp, self.receipt)

    def test_change_during_hash_is_rejected(self):
        read = binding._read_digest
        changed = False
        def race(stream):
            nonlocal changed
            digest = read(stream)
            if not changed:
                changed = True
                target = self.cp / 'action_head.pt'
                target.write_bytes(target.read_bytes() + b'changed')
            return digest
        with mock.patch.object(binding, '_read_digest', side_effect=race):
            receipt = binding.verify_checkpoint(self.cp, self.receipt)
        self.assertEqual(receipt['status'], 'failed')

    def test_previously_hashed_file_changed_before_finalization_fails(self):
        modified = False
        def progress(name, ok):
            nonlocal modified
            if name != 'action_head.pt' and not modified:
                modified = True
                target = self.cp / 'action_head.pt'
                target.write_bytes(target.read_bytes() + b'later')
        result = binding.verify_checkpoint(self.cp, self.receipt, progress=progress)
        self.assertEqual(result['status'], 'failed')
        row = next(r for r in result['files'] if r['path'] == 'action_head.pt')
        self.assertFalse(row['ok'])

    def test_inventory_change_and_wrong_checkpoint_root_fail_closed(self):
        config = self.verified()
        wrong = dict(config, checkpoint_root=str(self.root))
        with self.assertRaises(binding.CheckpointBindingError):
            binding.validate_checkpoint_binding(wrong)
        self.inventory_path.write_text(self.inventory_path.read_text() + '\n')
        with self.assertRaisesRegex(binding.CheckpointBindingError, 'inventory changed'):
            binding.validate_checkpoint_binding(config)

    def test_legacy_receipt_not_accepted_as_binding(self):
        self.receipt.write_text(json.dumps({'status': 'passed', 'files': []}))
        with self.assertRaises(binding.CheckpointBindingError):
            binding.checkpoint_binding(self.cp, self.receipt)

    def test_inventory_rejects_traversal_even_with_twenty_members(self):
        value = dict(self.inventory)
        value['../outside.pt'] = value.pop('action_head.pt')
        self.inventory_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(binding.CheckpointBindingError, 'Unsafe'):
            binding.load_inventory()

    def test_image_only_config_needs_no_native_paths_and_reuses_download_receipt(self):
        receipt = self.cp.parent / 'checkpoint_verification.json'
        self.verified(receipt)
        output = self.root / 'image-config.json'
        with mock.patch.object(binding, '_read_digest', side_effect=AssertionError('receipt must be reused')):
            config = self.run_configure(output, '--image-only')
        self.assertTrue(config['image_only'])
        for name in ['rlbench_root', 'pyrep_root', 'coppeliasim_root', 'host_library_path']:
            self.assertEqual(config[name], '')
        self.assertEqual(config['display_argv'], [])
        self.assertEqual(config['native_library_paths'], [])
        self.assertEqual(config['checkpoint_verification_receipt'], str(receipt.resolve()))
        binding.validate_checkpoint_binding(config)

    def test_legacy_config_receipt_gets_fresh_complete_verification_once(self):
        old = self.cp.parent / 'checkpoint_verification.json'
        old.write_text('{"status":"passed","files":[]}')
        original = old.read_bytes()
        output = self.root / 'fresh-config.json'
        with mock.patch.object(binding, '_read_digest', wraps=binding._read_digest) as hashes:
            config = self.run_configure(output, '--image-only')
        self.assertEqual(hashes.call_count, 20)
        self.assertEqual(old.read_bytes(), original)
        self.assertEqual(config['checkpoint_verification_receipt'], str(output.with_name('fresh-config-checkpoint-verification.json').resolve()))

    def test_full_configuration_binds_repeatable_overlay_library_paths(self):
        self.verified(self.cp.parent / 'checkpoint_verification.json')
        first = self.root / 'overlay-a'; second = self.root / 'overlay-b'
        first.mkdir(); second.mkdir()
        config = self.run_configure(self.root / 'full.json', '--deps-root', str(self.deps),
                                    '--native-library-path', str(first), '--native-library-path', str(second))
        self.assertFalse(config['image_only'])
        self.assertEqual(config['native_library_paths'], [str(first.resolve()), str(second.resolve())])
        self.assertEqual(config['display_argv'][1:3], ['-m', 'deepvision_vla.runtime.display'])

    def test_full_configuration_without_deps_rejects_before_verification(self):
        with self.assertRaises(SystemExit) as error:
            self.run_configure(self.root / 'invalid.json')
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / 'invalid-checkpoint-verification.json').exists())

    def test_component_directory_link_is_canonical_and_receipt_detects_retarget(self):
        # Linux uses a real directory symlink. Windows uses its unprivileged
        # directory-junction equivalent; no existing user directories touched.
        alias = self.root / 'linked-checkpoint'; alias.mkdir()
        target = self.cp / 'dinov3_model'
        link = alias / 'dinov3_model'
        def create_link(destination):
            if os.name == 'nt':
                environment = dict(os.environ, DV_TEST_LINK=str(link), DV_TEST_TARGET=str(destination))
                subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                                'New-Item -ItemType Junction -Path $env:DV_TEST_LINK -Target $env:DV_TEST_TARGET | Out-Null'],
                               env=environment, check=True, capture_output=True)
            else:
                link.symlink_to(destination, target_is_directory=True)
        create_link(target)
        for name in self.inventory:
            if name.startswith('dinov3_model/'):
                continue
            destination = alias / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(self.cp / name, destination)
        watched = binding.component_watch_paths(alias)
        self.assertEqual(watched[str((link / 'pytorch_model.bin').resolve())], 'finetuned_dino')
        self.assertEqual(watched[str((alias / 'action_head.pt').resolve())], 'action_head')
        receipt = self.root / 'linked-verification.json'
        self.assertEqual(binding.verify_checkpoint(alias, receipt)['status'], 'passed')
        config = binding.checkpoint_binding(alias, receipt)
        self.assertEqual(binding.validate_checkpoint_binding(config)['file_count'], 20)
        replacement = self.root / 'replacement-dino'
        shutil.copytree(target, replacement)
        if os.name == 'nt':
            os.rmdir(link)  # Remove this fixture's junction, never the target.
        else:
            link.unlink()
        create_link(replacement)
        with self.assertRaisesRegex(binding.CheckpointBindingError, 'member changed'):
            binding.validate_checkpoint_binding(config)

    def test_runtime_guards_do_not_use_optimizable_assert_statements(self):
        import deepvision_vla.paths as paths
        for name in ['runtime/checkpoint.py', 'runtime/strict_entry.py', 'tools/configure.py', 'tools/verify.py']:
            tree = ast.parse((paths.PACKAGE_ROOT / name).read_text(encoding='utf-8-sig'))
            self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)), name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
