"""CPU-only organization checks. Never construct a model or simulator."""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from deepvision_vla.runtime import checkpoint as checkpoint_integrity

from deepvision_vla import cli
from deepvision_vla.evaluation import runner
from deepvision_vla.evaluation import formal
from deepvision_vla.inference.image import image_dispatch_tree, read_request
from deepvision_vla.paths import ASSETS_ROOT, MODEL_RUNTIME_ROOT, PACKAGE_ROOT, REPOSITORY_ROOT
from deepvision_vla.runtime import reset
from deepvision_vla.tools.download import plan as download_plan


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LayoutTests(unittest.TestCase):
    def test_scientific_snapshot_is_complete_and_exact(self):
        manifest = json.loads((REPOSITORY_ROOT / 'tests/fixtures/scientific_source_manifest.json').read_text(encoding='utf-8'))
        actual = {p.relative_to(MODEL_RUNTIME_ROOT).as_posix(): sha(p) for p in MODEL_RUNTIME_ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        self.assertEqual(len(actual), 367)
        self.assertEqual(actual, manifest['sha256'])
        self.assertEqual(sha(MODEL_RUNTIME_ROOT / 'scripts_rlbench10_ablation/test_rlbench_no_gen_encoder.py'), runner.EVALUATOR_SHA)

    def test_public_helpers_parse(self):
        for path in PACKAGE_ROOT.rglob('*.py'):
            with self.subTest(path=path):
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    def test_all_public_help_commands_need_no_gpu_or_native(self):
        for command in [None, *cli.COMMANDS]:
            args = [sys.executable, '-m', 'deepvision_vla']
            if command:
                args.append(command)
            args.append('--help')
            with self.subTest(command=command):
                result = subprocess.run(args, text=True, capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_download_plan_keeps_exact_public_model_identity(self):
        plan = download_plan()
        self.assertEqual(plan['repo_id'], 'lyl010221-pku/deepvision-vla-rlbench10')
        self.assertEqual(plan['revision'], 'f0a3a4c6eaa07e7e939452ad9dec66aa7c252fe3')
        self.assertEqual(plan['required_files'], 20)
        self.assertEqual(plan['required_bytes'], 11391500062)
        self.assertEqual(len(set(plan['allow_patterns'])), 20)
        self.assertFalse(plan['publishes_or_changes_access'])

    def test_plan_dispatch_does_not_import_heavy_runtime(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            cli.main(['download', '--plan-only'])
        self.assertEqual(json.loads(output.getvalue()), download_plan())
        for module in ('torch', 'transformers', 'pyrep', 'rlbench'):
            self.assertNotIn(module, sys.modules)

    def test_request_preserves_instruction_and_blocks_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'image.png').write_bytes(b'fixture-not-decoded-by-request-reader')
            request = root / 'request.json'
            data = {'schema_version': 1, 'image': 'image.png', 'instruction': 'Put the green fruit on the plate.'}
            request.write_text(json.dumps(data))
            self.assertEqual(read_request(request)['instruction'], data['instruction'])
            self.assertEqual(read_request(request)['image'], str(root / 'image.png'))
            for extra in ({'gt_action': [0] * 7}, {'current_ee_position_xyz': [0, 0, float('nan')]}, {'instruction': '<action_0>'}):
                request.write_text(json.dumps({**data, **extra}))
                with self.assertRaises(ValueError):
                    read_request(request)

    def test_image_ast_changes_only_dispatcher_body(self):
        source = (MODEL_RUNTIME_ROOT / 'scripts_rlbench10_ablation/test_rlbench_no_gen_encoder.py').read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / 'request.json'
            request.write_text('{}')
            tree = image_dispatch_tree(source, request)
        old = ast.parse(source)
        for a, b in zip(old.body, tree.body):
            if isinstance(a, ast.FunctionDef) and a.name == 'run_dataset_predict':
                self.assertEqual(b.body[0].module, 'deepvision_vla.inference.image')
            else:
                self.assertEqual(ast.dump(a), ast.dump(b))

    def test_single_seed_reset_remains_strict(self):
        source = (MODEL_RUNTIME_ROOT / 'scripts_rlbench10_ablation/test_rlbench_no_gen_encoder.py').read_text(encoding='utf-8')
        transformed = reset.transform_evaluator(source)
        assignments = [node for node in ast.walk(transformed) if isinstance(node, ast.Assign)]
        retry = [n for n in assignments if any(isinstance(t, ast.Name) and t.id == 'MAX_RESET_RETRIES' for t in n.targets)]
        self.assertEqual([n.value.value for n in retry], [1])
        self.assertEqual(reset.POLICY, 'single_seed_episode_identity_v1')

    def build_fixture(self, root):
        protocol = runpy.run_path(str(MODEL_RUNTIME_ROOT / 'utils/rlbench_protocol.py'))
        active = protocol['load_profile_registry'](ASSETS_ROOT / 'rlbench_profile.json')['profiles']['hf-legacy-224-v1']
        cfg = dict(action_chunk=1, action_dim=7, action_query_contract='decoupled_per_dim_v1', action_query_count=7,
                   action_head_contract='decoupled_per_dim_trunk_shared_then_per_dim_linear_v1',
                   action_normalization_contract='quantile_linear_unclipped_v1', action_representation='xyz_delta_euler_absolute_v1',
                   image_aug_crop_scale=.9, rlbench_profile_definition_sha256=protocol['_profile_definition_sha256'](active))
        cp = root / 'checkpoint'
        cp.mkdir()
        (cp / 'training_config.json').write_text(json.dumps(cfg))
        (cp / 'stats_data.json').write_text('{}')
        # Tiny complete inventory fixtures exercise receipt/stat binding without
        # reading or manufacturing actual model weights. Production inventory
        # selection remains fixed; the override exists only under this test mock.
        release_names = json.loads((ASSETS_ROOT / 'checkpoint_inventory.origin.json').read_text(encoding='utf-8'))
        for name in release_names:
            path = cp / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(('CPU fixture: ' + name).encode())
        view = root / 'dino-view'
        view.mkdir()
        for name in ('config.json', 'preprocessor_config.json'):
            (view / name).write_bytes((ASSETS_ROOT / 'dino_runtime' / name).read_bytes())
        if os.name == 'nt':
            os.link(cp / 'dinov3_model/pytorch_model.bin', view / 'pytorch_model.bin')
        else:
            (view / 'pytorch_model.bin').symlink_to(cp / 'dinov3_model/pytorch_model.bin')
        inventory = {name: {'size': (cp / name).stat().st_size, 'sha256': sha(cp / name)}
                     for name in release_names}
        fixture_inventory = root / 'fixture-inventory.json'
        fixture_inventory.write_text(json.dumps(inventory))
        original_loader = checkpoint_integrity.load_inventory
        patcher = mock.patch.object(
            checkpoint_integrity, 'load_inventory',
            side_effect=lambda inventory_path=None: original_loader(inventory_path or fixture_inventory))
        patcher.start()
        self.addCleanup(patcher.stop)
        receipt = root / 'checkpoint-verification.json'
        verified = checkpoint_integrity.verify_checkpoint(cp, receipt)
        self.assertEqual(verified['status'], 'passed')
        binding = checkpoint_integrity.checkpoint_binding(cp, receipt)
        return {'source_root': str(MODEL_RUNTIME_ROOT), **binding,
                'dino_runtime_root': str(view), 'python_bin': sys.executable,
                'profile_path': str(ASSETS_ROOT / 'rlbench_profile.json'),
                'sample_data_path': str(cp / 'stats_data.json'),
                'checkpoint_small_file_sha256': {p: sha(cp / p)
                                               for p in ('training_config.json', 'stats_data.json')}}

    def test_runner_argv_contract_and_checkpoint_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.build_fixture(root)
            for mode in ('image', 'eval', 'infer'):
                args = runner.build(cfg, mode, 'close_box', 1, root / 'out')
                self.assertEqual(args[:3], [sys.executable, '-m', 'deepvision_vla.runtime.strict_entry'])
                self.assertEqual(args[args.index('--max-steps') + 1], '10')
                self.assertEqual(args[args.index('--eval-seed') + 1], '42')
                self.assertEqual(args[args.index('--need-to-sub') + 1], '0')
                self.assertEqual(args[args.index('--use_robot_state') + 1], '0')
            (root / 'checkpoint/stats_data.json').write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'stats_data.json'):
                runner.build(cfg, 'eval', 'close_box', 1, root / 'out')

    @unittest.skipUnless(os.name == 'posix', 'interpreter symlink is a Linux runtime contract')
    def test_python_executable_symlink_keeps_selected_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / 'python'
            link.symlink_to(sys.executable)
            self.assertEqual(runner.resolve({'python_bin': str(link)}, 'python_bin'), link)
            self.assertNotEqual(link, link.resolve())

    def test_formal_plan_retains_all_requests_and_helper_seals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = formal.plan(self.build_fixture(root), root / 'out', '2')
        self.assertEqual(plan['requested'], 500)
        self.assertEqual(len(plan['jobs']), 10)
        self.assertEqual(sum(len(job['episodes']) for job in plan['jobs']), 500)
        self.assertTrue(all(job['episodes'][-1]['seed'] == 42 + 49 * 1009 for job in plan['jobs']))
        self.assertFalse(plan['physical_state_identity_claim'])
        self.assertEqual(plan['automatic_retries'], 0)
        sealed_python = {key for key in plan['helper_sha256'] if key.endswith('.py')}
        actual_python = {str(p.relative_to(REPOSITORY_ROOT)) for p in PACKAGE_ROOT.rglob('*.py')}
        self.assertEqual(sealed_python, actual_python)


if __name__ == '__main__':
    unittest.main(verbosity=2)
