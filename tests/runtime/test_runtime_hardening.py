"""Runtime hardening regressions: stdlib CPU only, no model/native/GPU calls."""
from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import uuid

from deepvision_vla.evaluation import runner
from deepvision_vla.paths import PACKAGE_ROOT
from deepvision_vla.runtime.gpu_lease import gpu_lease, lease_path
from deepvision_vla.runtime.processes import cleanup_owned_session, session_identity

GPU_A = 'GPU-00000000-0000-0000-0000-000000000001'
GPU_B = 'GPU-00000000-0000-0000-0000-000000000002'
REPO = PACKAGE_ROOT.parents[1]
LINUX = os.name == 'posix' and Path('/proc').is_dir()
TEST_TEMP_ROOT = Path(os.environ.get('DEEPVISION_TEST_TEMP_ROOT', tempfile.gettempdir()))


def child_environment():
    return dict(os.environ, PYTHONPATH=str(PACKAGE_ROOT.parent), PYTHONDONTWRITEBYTECODE='1')


def checked_run(arguments, **kwargs):
    return subprocess.run([sys.executable, *arguments], env=child_environment(),
                          text=True, capture_output=True, timeout=15, **kwargs)


class RuntimeHardeningTests(unittest.TestCase):
    def test_formal_task_command_is_accepted_by_real_public_cli(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            root = Path(directory)
            command = runner.task_cli_command(sys.executable, 'eval', root/'missing.json',
                                              root/'not-created', '0', 'close_box', 50, 300)
            result = subprocess.run(command + ['--help'], env=child_environment(),
                                    text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--config', result.stdout)
            self.assertNotIn('unknown command', result.stderr)
            self.assertFalse((root/'not-created').exists())

    def test_gpu_selector_has_no_eight_card_server_assumption(self):
        command = runner.task_cli_command(sys.executable, 'eval', 'c', 'o', '12', 'close_box')
        self.assertEqual(command[command.index('--gpu') + 1], '12')
        self.assertEqual(runner.validate_gpu_selector(GPU_A), GPU_A)
        with self.assertRaises(ValueError):
            runner.validate_gpu_selector('0,1')

    def _runner_fixture(self, root):
        image = root/'input.png'; image.write_bytes(b'not decoded in this CPU launcher test')
        request = root/'request.json'
        request.write_text(json.dumps({'schema_version':1, 'image':str(image), 'instruction':'close box'}))
        config = {'source_root':str(root), 'rlbench_root':'/intentionally/missing/rlbench',
                  'pyrep_root':'/intentionally/missing/pyrep', 'coppeliasim_root':'/intentionally/missing/sim'}
        args = SimpleNamespace(output=str(root/'output'), mode='image', request=str(request),
                               task='close_box', episodes=1, gpu='0', wait_for_gpu=0, print_command=False)
        resource = {'index':'0', 'uuid':GPU_A, 'pci':'00000000:61:00.0', 'memory_mib':0,
                    'observation':'0, '+GPU_A+', 00000000:61:00.0, 0'}
        return config, args, resource

    def test_post_popen_receipt_errors_cleanup_before_releasing_lease(self):
        for fault in (OSError, KeyboardInterrupt):
            with self.subTest(fault=fault.__name__), tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
                root = Path(directory)
                config, args, resource = self._runner_fixture(root)
                original_popen = subprocess.Popen
                original_write = Path.write_text
                children = []
                events = []

                @contextmanager
                def lease(*unused, **ignored):
                    events.append('lease_acquired')
                    try:
                        if LINUX:
                            with gpu_lease(GPU_A, lock_root=root/'actual-local-locks') as held:
                                yield held
                        else:
                            yield {'path':'cpu-fixture-only','gpu_uuid':GPU_A}
                    finally:
                        events.append('lease_released_after_child_stopped' if all(p.poll() is not None for p in children)
                                      else 'lease_released_while_child_live')

                def spawn(*unused, **kwargs):
                    if os.name == 'nt':
                        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
                    child = original_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
                    children.append(child)
                    return child

                def write(path, data, *positional, **keywords):
                    if path.name == 'launcher_receipt.json':
                        value = json.loads(data)
                        if value.get('status') == 'running' and 'child_pid' in value:
                            raise fault('injected after Popen, before wait')
                    return original_write(path, data, *positional, **keywords)

                try:
                    with patch.object(runner, 'build', return_value=[sys.executable,'-c','unused','--']), \
                         patch.object(runner, 'observe_gpu', return_value=resource), \
                         patch.object(runner, 'require_idle_gpu'), \
                         patch.object(runner, 'gpu_lease', side_effect=lease), \
                         patch.object(runner.subprocess, 'Popen', side_effect=spawn), \
                         patch.object(Path, 'write_text', write):
                        with self.assertRaises(fault):
                            runner._execute(config, args)
                    self.assertEqual(len(children), 1)
                    self.assertIsNotNone(children[0].poll(), 'owned child survived launcher recording error')
                    self.assertEqual(events, ['lease_acquired','lease_released_after_child_stopped'])
                finally:
                    for child in children:
                        if child.poll() is None:
                            child.terminate()
                        child.wait(timeout=10)

    def test_image_and_infer_do_not_resolve_native_paths(self):
        for mode in ('image', 'infer'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
                config, args, resource = self._runner_fixture(Path(directory))
                args.mode = mode
                if mode == 'infer':
                    args.request = None
                args.print_command = True
                with patch.object(runner, 'build', return_value=[sys.executable,'-m','placeholder','--']), \
                     patch.object(runner, 'observe_gpu', return_value=resource), \
                     patch('deepvision_vla.runtime.native.require_native_dependencies') as native, \
                     patch.object(runner, 'gpu_lease') as lease:
                    with redirect_stdout(io.StringIO()):
                        runner._execute(config, args)
                native.assert_not_called()
                lease.assert_not_called()
                self.assertFalse(Path(args.output).exists())

    def test_native_dependency_failure_precedes_gpu_or_output(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            config, args, resource = self._runner_fixture(Path(directory))
            args.mode = 'eval'; args.request = None
            with patch('deepvision_vla.runtime.native.require_native_dependencies', side_effect=RuntimeError('missing libdbus')), \
                 patch.object(runner, 'build') as build, \
                 patch.object(runner, 'observe_gpu') as gpu, \
                 patch.object(runner, 'gpu_lease') as lease:
                with self.assertRaisesRegex(RuntimeError, 'missing libdbus'):
                    runner._execute(config, args)
            build.assert_not_called(); gpu.assert_not_called(); lease.assert_not_called()
            self.assertFalse(Path(args.output).exists())

    def test_eval_rejects_image_only_configuration_before_native_or_gpu(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            config, args, resource = self._runner_fixture(Path(directory))
            config['image_only'] = True
            args.mode = 'eval'; args.request = None
            with patch('deepvision_vla.runtime.native.require_native_dependencies') as native, \
                 patch.object(runner, 'build') as build, \
                 patch.object(runner, 'observe_gpu') as gpu, \
                 patch.object(runner, 'gpu_lease') as lease:
                with self.assertRaisesRegex(ValueError, 'image-only.*--deps-root'):
                    runner._execute(config, args)
            native.assert_not_called(); build.assert_not_called()
            gpu.assert_not_called(); lease.assert_not_called()
            self.assertFalse(Path(args.output).exists())

    def test_invalid_image_and_dataset_evidence_rejected_under_optimization(self):
        for command, mode in [('audit-image','image'), ('audit-inference','infer')]:
            with self.subTest(command=command), tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
                root=Path(directory); evidence=root/'evidence'; evidence.mkdir()
                (evidence/'launcher_receipt.json').write_text(json.dumps({'mode':mode,'returncode':1}))
                for flag in ([], ['-O']):
                    receipt=root/('receipt-'+str(len(flag))+'.json')
                    result=checked_run([*flag,'-m','deepvision_vla',command,
                                        '--output-root',str(evidence),'--receipt',str(receipt)])
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('invalid saved evidence', result.stderr)
                    self.assertFalse(receipt.exists())

    @unittest.skipUnless(LINUX, 'real flock requires Linux; execute this check on the validation server')
    def test_real_flock_excludes_same_gpu_and_releases_on_success_or_error(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            root=Path(directory)
            script=('import sys; from deepvision_vla.runtime.gpu_lease import gpu_lease,GPULeaseUnavailable; '
                    'gpu,root=sys.argv[1:];\n'
                    'try:\n'
                    ' with gpu_lease(gpu,lock_root=root): print("acquired")\n'
                    'except GPULeaseUnavailable: sys.exit(42)\n')
            def acquire(gpu):
                return checked_run(['-c',script,gpu,str(root)])
            with gpu_lease(GPU_A,lock_root=root):
                self.assertEqual(acquire(GPU_A).returncode,42)
                self.assertEqual(acquire(GPU_B).returncode,0)
            self.assertEqual(acquire(GPU_A).returncode,0)
            with self.assertRaisesRegex(RuntimeError,'fixture'):
                with gpu_lease(GPU_A,lock_root=root):
                    raise RuntimeError('fixture')
            self.assertEqual(acquire(GPU_A).returncode,0)

    @unittest.skipUnless(LINUX, 'secure real lock-file checks require Linux/POSIX')
    def test_symlink_lease_directory_and_file_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            root=Path(directory)
            target=root/'target'; target.mkdir(mode=0o700)
            linked=root/'directory-link'; linked.symlink_to(target,target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError,'symlink'):
                with gpu_lease(GPU_A,lock_root=linked):
                    self.fail('symlink directory was accepted')
            self.assertEqual(list(target.iterdir()),[])
            safe=root/'safe'; safe.mkdir(mode=0o700)
            unrelated=root/'unrelated.txt'; unrelated.write_text('preserve this file')
            lease_path(GPU_A,safe).symlink_to(unrelated)
            with self.assertRaisesRegex(RuntimeError,'symlink'):
                with gpu_lease(GPU_A,lock_root=safe):
                    self.fail('symlink file was accepted')
            self.assertEqual(unrelated.read_text(),'preserve this file')

    @unittest.skipUnless(LINUX, 'real session/descendant ownership check requires Linux /proc')
    def test_owned_descendants_stop_without_signalling_foreign_session(self):
        token=uuid.uuid4().hex
        env=dict(child_environment(), DEEPVISION_RUN_TOKEN=token)
        foreign_env=dict(child_environment(), DEEPVISION_RUN_TOKEN=uuid.uuid4().hex)
        code='import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])'
        owned=subprocess.Popen([sys.executable,'-c',code],env=env,start_new_session=True)
        owned_id=session_identity(owned)
        foreign=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],
                                 env=foreign_env,start_new_session=True)
        foreign_id=session_identity(foreign)
        try:
            owned.wait(timeout=10)
            result=cleanup_owned_session(owned,owned_id,token)
            self.assertEqual(result['status'],'released')
            self.assertIsNone(foreign.poll(), 'cleanup signalled an unrelated process')
        finally:
            cleanup_owned_session(owned,owned_id,token)
            cleanup_owned_session(foreign,foreign_id,foreign_env['DEEPVISION_RUN_TOKEN'])


if __name__=='__main__':
    unittest.main(verbosity=2)
