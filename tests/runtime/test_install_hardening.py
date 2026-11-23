"""Finite CPU regression tests for installer recovery and native preflight."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from deepvision_vla.runtime import native
from deepvision_vla.tools import prepare_rlbench
from deepvision_vla import cli


class Response:
    def __init__(self, fail=False): self.reads=0; self.fail=fail
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def read(self,n):
        self.reads+=1
        if self.reads==1:return b'abc'
        if self.fail:raise ConnectionError('fixture interruption')
        return b''


class InstallHardeningTests(unittest.TestCase):
    def test_interrupted_attempt_does_not_poison_shared_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.tar.gz';digest=hashlib.sha256(b'abc').hexdigest()
            legacy=Path(directory)/'source.tar.gz.partial';legacy.write_bytes(b'old-failed-attempt')
            with mock.patch.object(prepare_rlbench.urllib.request,'urlopen',return_value=Response(True)):
                with self.assertRaises(ConnectionError):prepare_rlbench.download('https://example.invalid/source',path,digest,3)
            failed=set(Path(directory).glob('*.partial'));self.assertEqual(len(failed),2)
            with mock.patch.object(prepare_rlbench.urllib.request,'urlopen',return_value=Response()):
                self.assertFalse(prepare_rlbench.download('https://example.invalid/source',path,digest,3))
            self.assertEqual(path.read_bytes(),b'abc')
            self.assertTrue(all(p.exists() for p in failed))

    def test_bad_existing_cache_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.tar.gz';path.write_bytes(b'bad')
            with mock.patch.object(prepare_rlbench.urllib.request,'urlopen') as network:
                with self.assertRaisesRegex(ValueError,'existing cache hash mismatch'):
                    prepare_rlbench.download('https://example.invalid/source',path,hashlib.sha256(b'abc').hexdigest(),3)
                network.assert_not_called()
            self.assertEqual(path.read_bytes(),b'bad')

    def test_concurrent_bad_cache_winner_is_not_clobbered(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.tar.gz'
            def winner(part,dest):path.write_bytes(b'bad-winner');raise FileExistsError()
            with mock.patch.object(prepare_rlbench.urllib.request,'urlopen',return_value=Response()),mock.patch.object(prepare_rlbench.os,'link',side_effect=winner):
                with self.assertRaisesRegex(ValueError,'concurrent cache hash mismatch'):
                    prepare_rlbench.download('https://example.invalid/source',path,hashlib.sha256(b'abc').hexdigest(),3)
            self.assertEqual(path.read_bytes(),b'bad-winner')
            self.assertEqual(len(list(Path(directory).glob('*.partial'))),1)

    def fixture(self,root):
        sim=root/'sim';(sim/'platforms').mkdir(parents=True)
        for rel in ['platforms/libqxcb.so','libQt5Core.so.5','libQt5Gui.so.5','libQt5Widgets.so.5']:(sim/rel).write_bytes(b'fixture-not-executed')
        extra=root/'extra';extra.mkdir();host=root/'host';host.mkdir()
        return {'coppeliasim_root':str(sim),'native_library_paths':[str(extra)],'host_library_path':str(host)}

    def test_image_only_doctor_imports_no_native_or_gpu(self):
        with mock.patch.object(native.subprocess,'run') as process:
            self.assertEqual(native.require_native_dependencies({'image_only':True})['status'],'not_required')
            self.assertEqual(native.native_environment({'image_only':True}),{})
            process.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json';path.write_text('{"image_only":true}')
            with contextlib.redirect_stdout(io.StringIO()) as output:cli.main(['doctor','--config',str(path)])
            self.assertEqual(json.loads(output.getvalue())['gpu_calls'],0)

    def test_missing_dbus_fails_before_model_with_same_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=self.fixture(Path(directory))
            missing=subprocess.CompletedProcess([],0,'libdbus-1.so.3 => not found\n','')
            with mock.patch.object(native.platform,'system',return_value='Linux'),mock.patch.object(native.subprocess,'run',return_value=missing) as process:
                with self.assertRaisesRegex(native.NativeDependencyError,'libdbus-1.so.3'):
                    native.require_native_dependencies(cfg)
                self.assertEqual(process.call_args.kwargs['env']['LD_LIBRARY_PATH'],native.native_environment(cfg)['LD_LIBRARY_PATH'])
                self.assertEqual(process.call_args.kwargs['timeout'],10)
            self.assertNotIn('PYTHONPATH',native.native_environment(cfg))

    def test_resolved_library_overlay_is_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=self.fixture(Path(directory))
            found=subprocess.CompletedProcess([],0,'libdbus-1.so.3 => /overlay/libdbus-1.so.3 (0x1234)\n','')
            with mock.patch.object(native.platform,'system',return_value='Linux'),mock.patch.object(native.subprocess,'run',return_value=found):
                self.assertEqual(native.require_native_dependencies(cfg)['status'],'ready')


if __name__=='__main__':unittest.main()
