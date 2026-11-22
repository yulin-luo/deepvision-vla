"""Portable attribution metadata and real CPU parent binding tests."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

from deepvision_vla.diagnostics import paper_probe_watchdog as watchdog
from deepvision_vla.diagnostics.source_manifest import build_source_manifest


class ProbeTerminalTest(unittest.TestCase):
    def setUp(self):
        self.sample = {"sample_id": "b"*20, "image_sha256": "c"*64,
            "arrays_sha256": "d"*64, "instruction": "put the cup on the plate", "anchor": 12}
        self.value = {"model": "pi0", "status": "complete", "model_loads": 1,
            "config_sha256": "e"*64, "permit_sha256": "f"*64, "lease_released": True,
            "probes_completed": 4, "backward_calls": 4, **self.sample,
            "probes": [{"logical_output_layer": layer, "status": "complete",
                "scalar_value": .25, "contribution": {
                    "display_normalization_applied": False, "visual_positions": [4, 5, 6],
                    "raw_signed": [-.3, 0, .4], "raw_relu": [0, 0, .4],
                    "channel_weights": [.1, -.2]}} for layer in (0, 6, 12, 16)]}

    def valid(self, value, sample=None):
        return watchdog.terminal_matches(value, config_sha="e"*64, permit_sha="f"*64,
            model="pi0", sample=self.sample if sample is None else sample)

    def test_exact_sample_and_all_original_layers_required(self):
        self.assertTrue(self.valid(self.value))
        self.assertFalse(self.valid(self.value, {**self.sample, "anchor": 13}))
        shifted = copy.deepcopy(self.value)
        shifted["probes"][1]["logical_output_layer"] = 7
        self.assertFalse(self.valid(shifted))
        self.assertFalse(self.valid({**self.value, "probes_completed": 3}))

    def test_display_normalization_and_malformed_maps_are_rejected(self):
        for field, bad in (
            ("display_normalization_applied", True),
            ("visual_positions", [4, 4, 5]),
            ("raw_signed", [-.3, float("nan"), .4]),
            ("raw_relu", [0, .4]),
            ("raw_relu", [0, .1, .4]),
            ("channel_weights", []),
        ):
            with self.subTest(field=field, bad=bad):
                altered = copy.deepcopy(self.value)
                altered["probes"][0]["contribution"][field] = bad
                self.assertFalse(self.valid(altered))

    def test_watchdog_reads_a_single_checksummed_json_object(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            watchdog.create_json(path, {"x": 1})
            digest = watchdog.file_sha256(path)
            self.assertEqual(watchdog.read_bound(path, digest), {"x": 1})
            with self.assertRaises(FileExistsError):
                watchdog.create_json(path, {"x": 2})
            path.write_text('{"x": 2}', encoding="utf8")
            with self.assertRaisesRegex(ValueError, "digest differs"):
                watchdog.read_bound(path, digest)


@unittest.skipUnless(sys.platform.startswith("linux"), "Requires real Linux /proc parent identity")
class LiveParentTest(unittest.TestCase):
    def test_launch_preflight_spawns_no_worker_and_preserves_output_paths(self):
        source = Path(watchdog.__file__).resolve()
        root = source.parents[3]
        files = [source, Path(watchdog.guard.__file__).resolve(), source.with_name("paper_probe.py")]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            manifest = folder / "source.json"
            watchdog.create_json(manifest, build_source_manifest(root, [p.relative_to(root).as_posix() for p in files]))
            sample = folder / "sample.json"
            watchdog.create_json(sample, {"sample_id": "b"*20, "image_sha256": "c"*64,
                "arrays_sha256": "d"*64, "instruction": "put the cup on the plate", "anchor": 12})
            config_path = folder / "config.json"
            config = {"schema": "deepvision_paper_probe_config_v1", "model": "openvla",
                "package_source_root": str(root), "source_manifest": str(manifest),
                "source_manifest_sha256": watchdog.file_sha256(manifest), "sample": str(sample),
                "sample_sha256": watchdog.file_sha256(sample), "output": str(folder / "worker"),
                "once": str(folder / "worker.once"), "factory": {
                    key: str(folder / key) for key in ("checkpoint", "asset_receipt", "official_inventory")}}
            watchdog.create_json(config_path, config)
            start = time.time()
            monotonic = time.monotonic()
            template = folder / "permit.json"
            watchdog.create_json(template, {"schema": "deepvision_paper_probe_permit_v1", "model": "openvla",
                "source_manifest_sha256": config["source_manifest_sha256"],
                "config_sha256": watchdog.file_sha256(config_path), "physical_gpu": 0,
                "exclusive_gpu": True, "allow_gpu_sharing": False,
                "expected_gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
                "expected_gpu_pci": "00000000:00:00.0", "budget_origin_epoch": start,
                "budget_origin_monotonic": monotonic, "deadline_epoch": start+60,
                "deadline_monotonic": monotonic+60, "maximum_own_gpu_bytes": 24_000_000_000,
                "allocator_gpu_bytes": 22_000_000_000, "minimum_forward_free_bytes": 8_000_000_000,
                "external_watchdog_required": True, "maximum_rss_bytes": 32_000_000_000, "cpu_threads": 8})
            nvidia = folder / "nvidia-smi"
            marker = folder / "gpu_was_queried"
            nvidia.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 1\n', encoding="utf8")
            nvidia.chmod(0o700)
            launch = folder / "launch.json"
            watchdog.create_json(launch, {"schema": "deepvision_paper_probe_launch_v1", "launch_authorized": False,
                "config": str(config_path), "config_sha256": watchdog.file_sha256(config_path),
                "permit_template": str(template), "permit_template_sha256": watchdog.file_sha256(template),
                "model_python": sys.executable, "model_python_sha256": watchdog.file_sha256(sys.executable),
                "nvidia_smi": str(nvidia), "nvidia_smi_sha256": watchdog.file_sha256(nvidia),
                "watchdog_sha256": watchdog.file_sha256(source), "parent_output": str(folder / "parent")})
            command = [sys.executable, str(source), "--launch-plan", str(launch),
                "--launch-plan-sha256", watchdog.file_sha256(launch)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {
                "status": "launch_metadata_validated", "model": "openvla", "child_attempts": 0})
            result = subprocess.run([*command, "--execute"], capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("did not authorize execution", result.stderr)
            for name in (marker, folder / "parent", Path(config["output"]), Path(config["once"])):
                self.assertFalse(name.exists(), name)

    def test_cpu_child_checks_actual_parent_and_rejects_altered_receipt(self):
        program = textwrap.dedent("""
            from pathlib import Path
            import json
            import subprocess
            import sys
            import time

            from deepvision_vla.diagnostics.paper_probe_watchdog import bind_live_permit

            location = Path(sys.argv[1])
            start = time.time()
            monotonic = time.monotonic()
            template = dict(config_sha256='a'*64, maximum_rss_bytes=32_000_000_000,
                budget_origin_epoch=start, budget_origin_monotonic=monotonic,
                deadline_epoch=start+30, deadline_monotonic=monotonic+30)
            permit, host = bind_live_permit(template, location, Path(__file__))
            child_code = '''import json, sys
            from deepvision_vla.diagnostics.paper_probe import verify_parent
            with open(sys.argv[1], encoding="utf8") as stream:
                verify_parent(json.load(stream))
            print("bound")'''
            command = [sys.executable, "-c", child_code, str(permit)]
            passed = subprocess.run(command, capture_output=True, text=True, timeout=10)
            assert passed.returncode == 0, passed.stderr
            assert passed.stdout.strip() == "bound"
            body = json.loads(permit.read_bytes())
            body["parent_intent_sha256"] = "0"*64
            permit.write_text(json.dumps(body), encoding="utf8")
            failed = subprocess.run(command, capture_output=True, text=True, timeout=10)
            assert failed.returncode != 0 and "SHA256 differs" in failed.stderr
            print("live isolated CPU parent bound, altered intent refused")
        """)
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "test_actual_probe_parent.py"
            script.write_text(program, encoding="utf8")
            process = subprocess.run([sys.executable, str(script), folder], text=True,
                capture_output=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("altered intent refused", process.stdout)


if __name__ == "__main__":
    unittest.main()
