import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from deepvision_vla.diagnostics import asset_receipts as assets


class AssetReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "snapshot"
        self.root.mkdir()
        source, weight = b"print('source')\n", b"\x00\x01weights\xff"
        (self.root / "model.py").write_bytes(source)
        (self.root / "model.safetensors").write_bytes(weight)
        self.inventory = {"model": "example/model", "revision": "1" * 40, "files": [
            {"path": "model.py", "size": len(source), "git_blob": hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()},
            {"path": "model.safetensors", "size": len(weight), "sha256": hashlib.sha256(weight).hexdigest(),
             "git_blob": "f" * 40}]}

    def receipt(self, inventory=None):
        raw = json.dumps(self.inventory if inventory is None else inventory, ensure_ascii=False).encode("utf-8")
        return assets.create_receipt(raw, self.root, hashlib.sha256(raw).hexdigest())

    def cli(self, output, sha=None):
        metadata = self.base / "inventory.json"
        metadata.write_text(json.dumps(self.inventory, ensure_ascii=False), encoding="utf-8")
        return assets.main(["--inventory", str(metadata), "--inventory-sha256", sha or assets.file_sha(metadata),
                            "--checkpoint", str(self.root), "--output", str(output)])

    def test_complete_content_and_source_git_hashes(self):
        result = self.receipt()
        self.assertEqual((result["status"], result["verified_files"], result["failed_files"]), ("complete", 2, 0))
        self.assertEqual([r["verified_by"] for r in result["files"]], ["git_blob", "sha256"])
        self.assertEqual(result["model_calls"], 0)
        self.assertNotIn("torch", sys.modules)

    def test_lfs_pointer_is_not_compared_as_data_git_blob(self):
        self.assertEqual(self.receipt()["files"][1]["verified_by"], "sha256")
        self.inventory["files"][1]["sha256"] = "0" * 64
        result = self.receipt()
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("content SHA256", result["files"][1]["error"])

    def test_missing_and_corrupt_files_each_retained(self):
        (self.root / "model.py").unlink()
        (self.root / "model.safetensors").write_bytes(b"x" * self.inventory["files"][1]["size"])
        result = self.receipt()
        self.assertEqual((len(result["files"]), result["failed_files"], result["verified_files"]), (2, 2, 0))
        self.assertTrue(all(not row["verified"] for row in result["files"]))

    def test_source_git_mismatch_is_not_accepted_as_new_pin(self):
        (self.root / "model.py").write_bytes(b"x" * self.inventory["files"][0]["size"])
        result = self.receipt()
        self.assertFalse(result["files"][0]["verified"])
        self.assertIn("Git blob", result["files"][0]["error"])

    def test_duplicate_paths_and_malformed_metadata_fail_before_file_reads(self):
        duplicate = copy.deepcopy(self.inventory)
        duplicate["files"].append(dict(duplicate["files"][0]))
        for bad in (duplicate, {**self.inventory, "revision": "main"},
                    {**self.inventory, "total_bytes": 1}, {**self.inventory, "model": None}):
            with patch.object(assets, "verify_one", side_effect=AssertionError("must not read assets")):
                with self.assertRaises(ValueError):
                    self.receipt(bad)

    def test_unsafe_paths_rejected(self):
        for name in ("../secret", "/absolute", "C:/file", "nested\\file", "./model.py", ".", "a\nfile"):
            inventory = copy.deepcopy(self.inventory)
            inventory["files"][0]["path"] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.receipt(inventory)

    def test_duplicate_json_keys_and_incorrect_inventory_seal_rejected(self):
        raw = b'{"model":"example/one","model":"example/two"}'
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            assets.create_receipt(raw, self.root, hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError):
            assets.create_receipt(raw, self.root, "0" * 64)

    def test_leaf_hf_symlinks_allowed_but_directory_redirects_rejected(self):
        content = (self.root / "model.py").read_bytes()
        target = self.base / "blob"
        target.write_bytes(content)
        leaf = self.root / "link.py"
        try:
            leaf.symlink_to(target)
        except OSError as error:
            self.skipTest("host does not permit symlinks: " + str(error))
        inventory = copy.deepcopy(self.inventory)
        inventory["files"][0]["path"] = "link.py"
        self.assertTrue(self.receipt(inventory)["files"][0]["verified"])
        directory = self.root / "outside"
        directory.symlink_to(self.base, target_is_directory=True)
        inventory["files"][0]["path"] = "outside/blob"
        self.assertFalse(self.receipt(inventory)["files"][0]["verified"])

    def test_same_sized_mutation_during_hashing_is_rejected(self):
        original_hash = hashlib.sha256
        path = self.root / "model.safetensors"
        item = self.inventory["files"][1]
        class MutatingHash:
            def __init__(self):
                self.real = original_hash()
            def update(self, data):
                self.real.update(data)
                path.write_bytes(b"z" * item["size"])
            def hexdigest(self):
                return self.real.hexdigest()
        with patch.object(assets.hashlib, "sha256", MutatingHash):
            with self.assertRaisesRegex(ValueError, "changed"):
                assets.verify_one(self.root, item)

    def test_path_and_fd_ctime_are_compared_within_each_api(self):
        original_stamp = assets._stamp
        def different_ctime_semantics(path):
            value = list(original_stamp(path))
            value[5] -= 10_000
            return tuple(value)
        with patch.object(assets, "_stamp", different_ctime_semantics):
            result = assets.verify_one(self.root, self.inventory["files"][0])
        self.assertTrue(result["verified"])

    def test_public_cli_utf8_and_no_overwrite(self):
        self.inventory["note"] = "déplier de gauche à droite"
        output = self.base / "receipt.json"
        self.assertEqual(self.cli(output), 0)
        original = output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.cli(output)
        self.assertEqual(output.read_bytes(), original)

    def test_failed_cli_receipt_never_has_complete_status(self):
        (self.root / "model.py").unlink()
        output = self.base / "failed-receipt.json"
        self.assertEqual(self.cli(output), 2)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "incomplete")

    def test_wrong_input_sha_writes_nothing(self):
        output = self.base / "receipt.json"
        with self.assertRaisesRegex(ValueError, "inventory SHA256"):
            self.cli(output, "0" * 64)
        self.assertFalse(output.exists())

    def test_stream_that_keeps_growing_is_bounded_to_size_plus_one(self):
        class GrowingStream:
            def __init__(self):
                self.requests = []
            def read(self, count):
                self.requests.append(count)
                return b"x" * count
        stream = GrowingStream()
        with self.assertRaisesRegex(ValueError, "grew beyond"):
            assets._bounded_hashes(stream, 19)
        self.assertEqual(stream.requests, [19, 1])

    def test_metadata_read_has_a_hard_byte_cap(self):
        path = self.base / "oversized.json"
        path.write_bytes(b"x" * 100)
        with patch.object(assets, "MAX_METADATA_BYTES", 32):
            with self.assertRaisesRegex(ValueError, "metadata exceeds"):
                assets.read_metadata(path)

    def test_explicit_qwen_component_receipt_matches_consumer_prefix(self):
        from deepvision_vla.diagnostics.adapters import qwen_h16_contracts

        repo = "example-org/qwen-component-fixture"
        fixture_hash = hashlib.sha256(repo.encode("utf-8")).hexdigest()
        policy = patch.object(qwen_h16_contracts, "REPOSITORY_NAME_SHA256", fixture_hash)
        policy.start()
        self.addCleanup(policy.stop)
        inventory = {"model": repo, "revision": assets.QWEN_REVISION,
                     "receipt_format": "qwen_h16_components_v1", "receipt_prefix": assets.QWEN_PREFIX, "files": []}
        for name in sorted(assets.QWEN_REQUIRED):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = ("synthetic component " + name).encode()
            path.write_bytes(data)
            inventory["files"].append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        runtime_export = {**inventory, "model": qwen_h16_contracts.RUNTIME_EXPORT_ID,
                          "receipt_format": qwen_h16_contracts.RUNTIME_EXPORT_FORMAT}
        for candidate in (inventory, runtime_export):
            with self.subTest(receipt_format=candidate["receipt_format"]):
                result = self.receipt(candidate)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["repo"], candidate["model"])
                self.assertEqual({row["path"] for row in result["files"]},
                                 {assets.QWEN_PREFIX + "/" + name for name in assets.QWEN_REQUIRED})
                for key, value in (("model", "other-org/unknown-model"), ("receipt_prefix", "guessed"),
                                   ("revision", "2" * 40), ("receipt_format", "snapshot_v1")):
                    with self.subTest(field=key), self.assertRaises(ValueError):
                        self.receipt({**candidate, key: value})


if __name__ == "__main__":
    unittest.main()
