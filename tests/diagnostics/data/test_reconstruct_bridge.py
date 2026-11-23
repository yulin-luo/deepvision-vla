import copy
import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from deepvision_vla.diagnostics import reconstruct_bridge as module


def vi(n):
    result = bytearray()
    while n > 127:
        result.append((n & 127) | 128)
        n >>= 7
    return bytes(result + bytes([n]))


def blob(field, value):
    return vi(field * 8 + 2) + vi(len(value)) + value


def feature(name, kind, values):
    if kind == 1:
        value_list = b"".join(blob(1, value) for value in values)
    elif kind == 2:
        value_list = blob(1, struct.pack("<" + "f" * len(values), *values))
    else:
        value_list = blob(1, b"".join(vi(v) for v in values))
    return blob(1, blob(1, name.encode()) + blob(2, blob(kind, value_list)))


def fixture(instruction="pick object"):
    source = "/upstream/dataset/example/out.npy"
    sid = hashlib.sha256((source + "::9").encode()).hexdigest()[:20]
    image = io.BytesIO()
    Image.new("RGB", (256, 256), (20, 40, 80)).save(image, format="JPEG")
    pixels = image.getvalue()
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    actions[0, 0] = -0.0
    states = actions + np.float32(0.5)
    archive = io.BytesIO()
    np.savez_compressed(archive, actions=actions, states=states)
    entries = [feature("episode_metadata/file_path", 1, [source.encode()]),
               feature("episode_metadata/episode_id", 3, [9]),
               feature("steps/action", 2, actions.ravel().tolist()),
               feature("steps/observation/state", 2, states.ravel().tolist()),
               feature("steps/is_first", 3, [1, 0, 0]),
               feature("steps/is_last", 3, [0, 0, 1]),
               feature("steps/observation/image_0", 1, [pixels] * 3),
               feature("steps/language_instruction", 1, [instruction.encode("utf-8")] * 3)]
    entries.extend(feature(f"episode_metadata/{name}", 3, [value]) for name, value in (
        ("has_image_0", 0), ("has_image_1", 0), ("has_language", 1),
        ("has_image_2", 0), ("has_image_3", 0),
    ))
    raw = blob(1, b"".join(entries))
    length = struct.pack("<Q", len(raw))
    record = length + struct.pack("<I", module.masked_crc(length)) + raw + struct.pack("<I", module.masked_crc(raw))
    name = "bridge_dataset-train.tfrecord-00001-of-01024"
    row = dict(sample_id=sid, ordinal=1, upstream_episode_file_path=source, episode_id=9,
               shard=name, record_offset=0, record_sha256=hashlib.sha256(raw).hexdigest(),
               anchor=1, steps=3, camera="image_0", instruction=instruction,
               image_sha256=hashlib.sha256(pixels).hexdigest(),
               arrays_sha256=hashlib.sha256(archive.getvalue()).hexdigest())
    index = dict(schema="deepvision_bridge_source_index_v2", dataset="example/bridge", revision="fixed", sources=[row])
    checks = dict(dataset="example/bridge", revision="fixed", shards=[dict(name=name, bytes=len(record), sha256=hashlib.sha256(record).hexdigest())])
    return index, checks, record, raw, pixels, actions, states


class ReconstructTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.index, self.checks, self.record, self.raw, self.image, self.actions, self.states = fixture()
        self.shards = self.root / "shards"
        self.shards.mkdir()
        self.path = self.shards / self.index["sources"][0]["shard"]
        self.path.write_bytes(self.record)

    def test_crc_known_vector(self):
        self.assertEqual(module.crc32c(b"123456789"), 0xe3069283)

    def test_exact_image_and_full_float_bytes(self):
        result = module.reconstruct(self.index, self.checks, self.shards, self.root / "out")
        self.assertEqual((result["requested"], result["reconstructed"], result["failed"]), (1, 1, 0))
        sid = self.index["sources"][0]["sample_id"]
        self.assertEqual((self.root / "out/images" / (sid + ".jpg")).read_bytes(), self.image)
        with np.load(self.root / "out/arrays" / (sid + ".npz"), allow_pickle=False) as data:
            self.assertEqual(data["actions"].tobytes(), self.actions.tobytes())
            self.assertEqual(data["states"].tobytes(), self.states.tobytes())
        self.assertFalse(result["raw_arrays_are_model_GT"])

    def test_shard_reader_uses_verified_open_bytes_when_path_inode_metadata_changes(self):
        original_stat = Path.stat
        changes = iter(range(1, 100))

        def variable_path_stat(path, *args, **kwargs):
            value = original_stat(path, *args, **kwargs)
            if path == self.path:
                fields = {name: getattr(value, name) for name in dir(value) if name.startswith("st_")}
                fields["st_ino"] += next(changes)
                return SimpleNamespace(**fields)
            return value

        with patch.object(Path, "stat", variable_path_stat):
            result = module.reconstruct(self.index, self.checks, self.shards, self.root / "stable-bytes")
        self.assertEqual((result["reconstructed"], result["failed"]), (1, 0))

    def test_record_crc_and_sha(self):
        for pos in (8, 20, -1):
            corrupt = bytearray(self.record)
            corrupt[pos] ^= 1
            self.path.write_bytes(corrupt)
            with self.assertRaises(ValueError):
                module.read_record(self.path, 0, hashlib.sha256(self.raw).hexdigest())
        self.path.write_bytes(self.record)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            module.read_record(self.path, 0, "0" * 64)

    def test_shard_checksum_failure_retains_all_requests(self):
        self.path.write_bytes(self.record + b"corrupt")
        other = copy.deepcopy(self.index["sources"][0])
        other.update(sample_id="a" * 20, ordinal=2)
        self.index["sources"].append(other)
        result = module.reconstruct(self.index, self.checks, self.shards, self.root / "out")
        ledger = [json.loads(line) for line in (self.root / "out/outcomes.jsonl").read_text().splitlines()]
        self.assertEqual((result["requested"], result["failed"], len(ledger)), (2, 2, 2))
        self.assertTrue(all(row["status"] == "failed" for row in ledger))

    def test_identity_and_anchor_never_reselected(self):
        fields = module.decode_example(self.raw)
        for key, value in (("sample_id", "f" * 20), ("anchor", 0), ("instruction", "different instruction")):
            row = dict(self.index["sources"][0], **{key: value})
            with self.assertRaises(ValueError):
                module.extract_record(fields, row)

    def test_index_without_plaintext_upstream_path_rebuilds_exact_original_bytes(self):
        self.index["schema"] = "deepvision_bridge_source_index_v3"
        row = self.index["sources"][0]
        old_path = row.pop("upstream_episode_file_path")
        row["upstream_episode_path_sha256"] = hashlib.sha256(old_path.encode("utf-8")).hexdigest()
        out = self.root / "redacted-index-output"
        result = module.reconstruct(self.index, self.checks, self.shards, out)
        self.assertEqual((result["reconstructed"], result["failed"]), (1, 0))
        self.assertEqual(module.digest(out / "arrays" / (row["sample_id"] + ".npz")), row["arrays_sha256"])
        fields = module.decode_example(self.raw)
        with self.assertRaisesRegex(ValueError, "episode identity mismatch"):
            module.extract_record(fields, {**row, "upstream_episode_path_sha256": "0" * 64})
        row["upstream_episode_file_path"] = old_path
        with self.assertRaisesRegex(ValueError, "identity or sequence metadata"):
            module.validate_index(self.index, self.checks)

    def test_npz_archive_mismatch_is_failure_with_emitted_files_retained(self):
        self.index["sources"][0]["arrays_sha256"] = "0" * 64
        result = module.reconstruct(self.index, self.checks, self.shards, self.root / "out")
        self.assertEqual((result["reconstructed"], result["failed"]), (0, 1))
        ledger = json.loads((self.root / "out/outcomes.jsonl").read_text())
        self.assertIn("NPZ SHA256 mismatch", ledger["error"])
        self.assertEqual(len(list((self.root / "out/arrays").iterdir())), 1)

    def test_runtime_inventory_preserves_pinned_bytes_and_rejects_bad_seal(self):
        from deepvision_vla.diagnostics import input_inventory

        row = self.index["sources"][0]
        source_id = row["sample_id"]
        expected = module.runtime_inventory_row(
            row, module.decode_example(self.raw), Path(source_id + ".jpg"), Path(source_id + ".npz"))
        serialized = json.dumps([expected], indent=2).encode("utf-8")
        self.index["source_inventory_sha256"] = hashlib.sha256(serialized).hexdigest()
        verified = self.root / "verified-runtime"
        with patch.object(input_inventory, "SOURCES", 1):
            result = module.reconstruct(self.index, self.checks, self.shards, verified, runtime_inventory=True)
        self.assertEqual((verified / "inventory.json").read_bytes(), serialized)
        self.assertEqual(result["runtime_inventory"]["data_files_rehashed"], 2)
        self.assertEqual(result["runtime_inventory"]["source_manifest_sha256"],
                         module.digest(verified / "manifest.jsonl"))

        failed = self.root / "wrong-runtime-seal"
        self.index["source_inventory_sha256"] = "0" * 64
        with patch.object(input_inventory, "SOURCES", 1):
            result = module.reconstruct(self.index, self.checks, self.shards, failed, runtime_inventory=True)
        self.assertIn("SHA256 differs", result["runtime_inventory_error"])
        self.assertFalse((failed / "transfer_receipt.json").exists())
        self.assertTrue((failed / "outcomes.jsonl").is_file())

    def test_output_never_overwritten(self):
        out = self.root / "out"
        out.mkdir()
        (out / "keep").write_text("original")
        with self.assertRaises(FileExistsError):
            module.reconstruct(self.index, self.checks, self.shards, out)
        self.assertEqual((out / "keep").read_text(), "original")

    def test_path_and_duplicate_identity_rejected_before_output(self):
        for name in ("../escape", "C:/absolute", "folder/shard"):
            bad = copy.deepcopy(self.checks)
            bad["shards"][0]["name"] = name
            with self.assertRaises(ValueError):
                module.reconstruct(self.index, bad, self.shards, self.root / "out")
        duplicate = copy.deepcopy(self.index)
        duplicate["sources"].append(dict(duplicate["sources"][0], ordinal=2))
        with self.assertRaises(ValueError):
            module.reconstruct(duplicate, self.checks, self.shards, self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_protobuf_truncation_overflow_and_duplicate_keys(self):
        for data in (b"\x0a\x05x", b"\x0a" + b"\xff" * 10, b"\x00"):
            with self.assertRaises(ValueError):
                module.decode_example(data)
        entry = feature("duplicate", 1, [b"x"])
        with self.assertRaisesRegex(ValueError, "duplicate feature"):
            module.decode_example(blob(1, entry + entry))

    def test_declared_metadata_hash_checked_before_writes(self):
        index = self.root / "index.json"
        checks = self.root / "checks.json"
        index.write_text(json.dumps(self.index))
        checks.write_text(json.dumps(self.checks))
        argv = ["--source-index", str(index), "--source-index-sha256", "0" * 64,
                "--shard-checksums", str(checks), "--shard-checksums-sha256", module.digest(checks),
                "--shard-dir", str(self.shards), "--output", str(self.root / "out")]
        with self.assertRaisesRegex(ValueError, "metadata SHA256"):
            module.main(argv)
        self.assertFalse((self.root / "out").exists())

    def test_non_ascii_instruction_preserved_by_public_cli(self):
        index, checks, record, *_ = fixture("déplier le tissu de gauche à droite")
        self.path.write_bytes(record)
        index_path, checks_path = self.root / "index.json", self.root / "checks.json"
        index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        checks_path.write_text(json.dumps(checks), encoding="utf-8")
        result = module.main(["--source-index", str(index_path), "--source-index-sha256", module.digest(index_path),
                              "--shard-checksums", str(checks_path), "--shard-checksums-sha256", module.digest(checks_path),
                              "--shard-dir", str(self.shards), "--output", str(self.root / "out")])
        self.assertEqual(result, 0)
        summary = json.loads((self.root / "out/summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["reconstructed"], summary["failed"]), (1, 0))


if __name__ == "__main__":
    unittest.main()
