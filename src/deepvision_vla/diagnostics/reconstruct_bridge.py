"""Rebuild a fixed Bridge panel from local, checksum-bound TFRecord shards.

This exports original raw arrays, not model-specific ground-truth actions.
No downloads, source selection, ROI filtering or model imports occur here.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import io
import json
import os
import re
import struct
from pathlib import Path

MAX_RECORD_BYTES = 256 * 2**20
HEX = re.compile(r"[0-9a-f]{64}\Z")
SID = re.compile(r"[0-9a-f]{20}\Z")
SHARD = re.compile(r"bridge_dataset-train\.tfrecord-\d{5}-of-01024\Z")


def digest(path):
    with Path(path).open("rb") as stream:
        return digest_stream(stream)


def digest_stream(stream):
    stream.seek(0)
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(2**20), b""):
        h.update(block)
    return h.hexdigest()


def crc32c(data):
    try:
        import google_crc32c
    except ImportError:
        value = 0xffffffff
        for byte in data:
            value ^= byte
            for _ in range(8):
                value = (value >> 1) ^ (0x82f63b78 if value & 1 else 0)
        return value ^ 0xffffffff
    return google_crc32c.value(data)


def masked_crc(data):
    value = crc32c(data)
    return (((value >> 15) | (value << 17)) + 0xa282ead8) & 0xffffffff


def _varint(data, pos):
    value = 0
    for shift in range(0, 70, 7):
        if pos >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[pos]
        pos += 1
        if shift == 63 and byte > 1:
            raise ValueError("overflowing protobuf varint")
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
    raise ValueError("overflowing protobuf varint")


def _wire(data):
    pos, result = 0, {}
    while pos < len(data):
        tag, pos = _varint(data, pos)
        field, kind = tag >> 3, tag & 7
        if not field:
            raise ValueError("invalid protobuf field zero")
        if kind == 0:
            value, pos = _varint(data, pos)
        else:
            if kind == 2:
                length, pos = _varint(data, pos)
            elif kind in (1, 5):
                length = {1: 8, 5: 4}[kind]
            else:
                raise ValueError("unsupported protobuf wire type")
            if length > len(data) - pos:
                raise ValueError("truncated protobuf field")
            value = data[pos:pos + length]
            pos += length
        result.setdefault(field, []).append((kind, value))
    return result


def _one(fields, key, kind):
    entries = fields.get(key, [])
    if len(entries) != 1 or entries[0][0] != kind:
        raise ValueError("missing, duplicate or incorrectly typed protobuf field")
    return entries[0][1]


def decode_example(data):
    """Decode the three TensorFlow Example feature types without TensorFlow."""
    features = _wire(_one(_wire(data), 1, 2))
    result = {}
    for kind, entry in features.get(1, []):
        if kind != 2:
            raise ValueError("invalid feature map entry")
        fields = _wire(entry)
        name = _one(fields, 1, 2).decode("utf-8")
        if name in result:
            raise ValueError("duplicate feature name")
        feature = _wire(_one(fields, 2, 2))
        if len(feature) != 1:
            raise ValueError("ambiguous feature type")
        feature_type = next(iter(feature))
        values = _wire(_one(feature, feature_type, 2)).get(1, [])
        decoded = []
        for wire_type, value in values:
            if feature_type == 1 and wire_type == 2:
                decoded.append(value)
            elif feature_type == 2 and wire_type in (2, 5):
                if len(value) % 4:
                    raise ValueError("misaligned float feature")
                decoded.extend(struct.unpack("<" + "f" * (len(value) // 4), value))
            elif feature_type == 3 and wire_type in (0, 2):
                if wire_type == 0:
                    numbers = [value]
                else:
                    numbers, pos = [], 0
                    while pos < len(value):
                        number, pos = _varint(value, pos)
                        numbers.append(number)
                decoded.extend(n if n < 2**63 else n - 2**64 for n in numbers)
            else:
                raise ValueError("invalid feature value wire type")
        if feature_type not in (1, 2, 3):
            raise ValueError("unsupported feature type")
        result[name] = decoded
    return result


def read_record(path, offset, expected_sha):
    with Path(path).open("rb") as stream:
        return read_record_stream(stream, offset, expected_sha)


def read_record_stream(stream, offset, expected_sha):
    if type(offset) is not int or offset < 0 or not HEX.fullmatch(expected_sha):
        raise ValueError("invalid source record locator")
    stream.seek(offset)
    header = stream.read(12)
    if len(header) != 12 or masked_crc(header[:8]) != struct.unpack("<I", header[8:])[0]:
        raise ValueError("TFRecord length CRC mismatch")
    length = struct.unpack("<Q", header[:8])[0]
    if not 0 < length <= MAX_RECORD_BYTES:
        raise ValueError("TFRecord length outside bound")
    raw, tail = stream.read(length), stream.read(4)
    if len(raw) != length or len(tail) != 4 or masked_crc(raw) != struct.unpack("<I", tail)[0]:
        raise ValueError("TFRecord payload CRC mismatch")
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("source record SHA256 mismatch")
    return decode_example(raw)


def extract_record(features, row):
    import numpy as np
    from PIL import Image
    source = features["episode_metadata/file_path"]
    episode = features["episode_metadata/episode_id"]
    if len(source) != 1 or len(episode) != 1:
        raise ValueError("ambiguous source episode identity")
    source = source[0].decode("utf-8")
    identity = source + "::" + str(episode[0])
    if "upstream_episode_path_sha256" in row:
        path_matches = hashlib.sha256(source.encode("utf-8")).hexdigest() == row["upstream_episode_path_sha256"]
    else:
        path_matches = source == row["upstream_episode_file_path"]
    if (not path_matches or episode[0] != row["episode_id"]
            or hashlib.sha256(identity.encode()).hexdigest()[:20] != row["sample_id"]):
        raise ValueError("source episode identity mismatch")
    actions = np.asarray(features["steps/action"], dtype="<f4").reshape(-1, 7)
    states = np.asarray(features["steps/observation/state"], dtype="<f4").reshape(-1, 7)
    count = len(actions)
    if (count != row["steps"] or count <= 1 or len(states) != count
            or not np.isfinite(actions).all() or not np.isfinite(states).all()):
        raise ValueError("invalid raw action/state sequence")
    first, last = features["steps/is_first"], features["steps/is_last"]
    images, instructions = features["steps/observation/image_0"], features["steps/language_instruction"]
    if not (len(first) == len(last) == len(images) == len(instructions) == count):
        raise ValueError("inconsistent source sequence lengths")
    anchor = (count - 1) // 2
    if first[0] != 1 or last[-1] != 1 or anchor != row["anchor"]:
        raise ValueError("source sequence boundary or anchor mismatch")
    if instructions[anchor].decode("utf-8").strip() != row["instruction"]:
        raise ValueError("source instruction mismatch")
    image = images[anchor]
    if hashlib.sha256(image).hexdigest() != row["image_sha256"]:
        raise ValueError("source image SHA256 mismatch")
    with Image.open(io.BytesIO(image)) as decoded:
        decoded.load()
        if decoded.size != (256, 256):
            raise ValueError("source image dimensions mismatch")
    return image, actions, states


def validate_index(index, shards):
    schema = index.get("schema")
    if schema not in {"deepvision_bridge_source_index_v2", "deepvision_bridge_source_index_v3"} or not index.get("sources"):
        raise ValueError("invalid fixed source index")
    if index.get("dataset") != shards.get("dataset") or index.get("revision") != shards.get("revision"):
        raise ValueError("dataset revision binding mismatch")
    checks = {}
    for item in shards["shards"]:
        name = item["name"]
        if (not SHARD.fullmatch(name) or name in checks or not HEX.fullmatch(item["sha256"])
                or type(item["bytes"]) is not int or item["bytes"] <= 0):
            raise ValueError("invalid shard checksum entry")
        checks[name] = item
    seen = set()
    for ordinal, row in enumerate(index["sources"], 1):
        sid = row["sample_id"]
        if not SID.fullmatch(sid) or sid in seen or row["ordinal"] != ordinal:
            raise ValueError("duplicate or unordered source identity")
        if (row["shard"] not in checks or type(row["record_offset"]) is not int
                or row["record_offset"] < 0 or not HEX.fullmatch(row["record_sha256"])
                or not HEX.fullmatch(row["image_sha256"]) or not HEX.fullmatch(row["arrays_sha256"])):
            raise ValueError("invalid source locator")
        if schema == "deepvision_bridge_source_index_v3":
            path_hash = row.get("upstream_episode_path_sha256")
            path_is_valid = ("upstream_episode_file_path" not in row and isinstance(path_hash, str)
                             and bool(HEX.fullmatch(path_hash)))
        else:
            path_is_valid = (isinstance(row.get("upstream_episode_file_path"), str)
                             and "upstream_episode_path_sha256" not in row)
        if (type(row["episode_id"]) is not int or type(row["steps"]) is not int
                or row["steps"] <= 1 or type(row["anchor"]) is not int
                or row["anchor"] != (row["steps"] - 1) // 2 or row["camera"] != "image_0"
                or not isinstance(row["instruction"], str) or not row["instruction"].strip()
                or not path_is_valid):
            raise ValueError("invalid source identity or sequence metadata")
        seen.add(sid)
    return checks


def shard_stamp(stream):
    stat = os.fstat(stream.fileno())
    return stat.st_size, stat.st_mtime_ns


def verify_open_shard(stream, item):
    size, _ = shard_stamp(stream)
    if size != item["bytes"] or digest_stream(stream) != item["sha256"]:
        raise ValueError("local shard size/SHA256 mismatch")
    return shard_stamp(stream)


@contextmanager
def open_verified_shard(path, item):
    # Verify and read one opened file, even on mounts with changing path inode metadata.
    with path.open("rb") as stream:
        yield stream, verify_open_shard(stream, item)


def runtime_inventory_row(row, features, image_path, arrays_path):
    # Preserve the original field order so the established inventory byte seal
    # and all row-bound geometry/annotation receipts remain usable.
    raw_flags = {}
    for name in ("has_image_0", "has_image_1", "has_language", "has_image_2", "has_image_3"):
        field = f"episode_metadata/{name}"
        value, = features[field]
        if type(value) is not int or value not in (0, 1):
            raise ValueError("invalid original episode flag")
        raw_flags[field] = value
    original = {
        "sample_id": row["sample_id"],
        "source_path": features["episode_metadata/file_path"][0].decode("utf-8"),
        "episode_id": row["episode_id"],
        "shard": row["shard"],
        "record_offset": row["record_offset"],
        "record_sha256": row["record_sha256"],
        "roi_status": "unannotated",
        "human_verified": False,
        "action_semantics": "raw_tfds_7d_unverified",
        "status": "ready",
        "instruction": row["instruction"],
        "steps": row["steps"],
        "anchor": row["anchor"],
        "camera": row["camera"],
        "image_path": f"images/{image_path.name}",
        "image_sha256": row["image_sha256"],
        "arrays_path": f"arrays/{arrays_path.name}",
        "width": 256,
        "height": 256,
        "raw_metadata_flags": raw_flags,
    }
    return {
        "ordinal": row["ordinal"],
        "sample_id": row["sample_id"],
        "image_sha256": row["image_sha256"],
        "image_size": [256, 256],
        "anchor": row["anchor"],
        "instruction": row["instruction"],
        "original_source": original,
        "arrays_sha256": row["arrays_sha256"],
    }


def write_runtime_inventory(output, inventory, assets, expected_sha, *, utf8_unescaped=False):
    from .input_inventory import validate_linked_inventory

    raw_inventory = json.dumps(inventory, indent=2, ensure_ascii=not utf8_unescaped).encode("utf-8")
    inventory_sha = hashlib.sha256(raw_inventory).hexdigest()
    if inventory_sha != expected_sha:
        raise ValueError("reconstructed runtime inventory SHA256 differs from the sealed index")
    original = [row["original_source"] for row in inventory]
    manifest_format = {"ensure_ascii":False,"separators":(",", ":")} if utf8_unescaped else {}
    raw_manifest = "".join(json.dumps(row, **manifest_format) + "\n" for row in original).encode("utf-8")
    manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
    receipt = {"status": "complete", "samples": len(inventory), "data_files_rehashed": len(assets),
               "inventory_sha256": inventory_sha, "source_manifest_sha256": manifest_sha,
               "model_calls": 0, "gpu_calls": 0}
    validate_linked_inventory(inventory, original, assets, receipt, inventory_sha, manifest_sha)
    # Write the receipt last. A partially written inventory cannot authorize a run.
    for name, data in (
        ("inventory.json", raw_inventory),
        ("manifest.jsonl", raw_manifest),
        ("files.json", json.dumps(assets, indent=2).encode("utf-8")),
        ("transfer_receipt.json", json.dumps(receipt, indent=2).encode("utf-8")),
    ):
        with (output / name).open("xb") as stream:
            stream.write(data)
    return receipt


def reconstruct(index, shards, shard_dir, output, *, runtime_inventory=False):
    import numpy as np
    checks = validate_index(index, shards)
    inventory, assets = [], []
    runtime_format = index.get("runtime_inventory_json_format")
    if runtime_format not in (None, "utf8_unescaped_indent2"):
        raise ValueError("Unknown pinned runtime inventory JSON format")
    if runtime_inventory:
        from .input_inventory import SOURCES

        if len(index["sources"]) != SOURCES or not HEX.fullmatch(index.get("source_inventory_sha256", "")):
            raise ValueError("runtime inventory requires the complete original panel and its inventory SHA256")
    output, shard_dir = Path(output), Path(shard_dir).resolve(strict=True)
    output.mkdir(exist_ok=False)
    (output / "images").mkdir()
    (output / "arrays").mkdir()
    verified, failures = {}, []
    success = 0
    with ExitStack() as shard_handles, (output / "outcomes.jsonl").open("x", encoding="utf-8") as ledger:
        for row in index["sources"]:
            result = {"sample_id": row["sample_id"], "ordinal": row["ordinal"]}
            try:
                name = row["shard"]
                item = checks[name]
                if name not in verified:
                    path = (shard_dir / name).resolve(strict=True)
                    if path.parent != shard_dir:
                        raise ValueError("shard escapes declared input directory")
                    verified[name] = shard_handles.enter_context(open_verified_shard(path, item))
                stream, stamp = verified[name]
                if shard_stamp(stream) != stamp:
                    verified[name] = stream, verify_open_shard(stream, item)
                features = read_record_stream(stream, row["record_offset"], row["record_sha256"])
                image, actions, states = extract_record(features, row)
                image_path = output / "images" / (row["sample_id"] + ".jpg")
                arrays_path = output / "arrays" / (row["sample_id"] + ".npz")
                with image_path.open("xb") as stream:
                    stream.write(image)
                with arrays_path.open("xb") as stream:
                    np.savez_compressed(stream, actions=actions, states=states)
                arrays_sha = digest(arrays_path)
                if arrays_sha != row["arrays_sha256"]:
                    raise ValueError("reconstructed NPZ SHA256 mismatch: expected " + row["arrays_sha256"]
                                     + "; actual " + arrays_sha + "; emitted files retained")
                image_sha = digest(image_path)
                if image_sha != row["image_sha256"]:
                    raise ValueError("reconstructed image SHA256 differs after writing")
                if runtime_inventory:
                    record = runtime_inventory_row(row, features, image_path, arrays_path)
                    assets.extend({"path": path.relative_to(output).as_posix(),
                                   "bytes": path.stat().st_size, "sha256": checksum}
                                  for path, checksum in ((image_path, image_sha), (arrays_path, arrays_sha)))
                    inventory.append(record)
                result.update(status="reconstructed", image_sha256=image_sha,
                              arrays_sha256=arrays_sha, action_semantics="raw_tfds_7d_unverified")
                success += 1
            except Exception as error:
                result.update(status="failed", error_type=type(error).__name__, error=str(error))
                failures.append(row["sample_id"])
            ledger.write(json.dumps(result, ensure_ascii=False) + "\n")
            ledger.flush()
    summary = dict(requested=len(index["sources"]), reconstructed=success, failed=len(failures),
                   missing=0, failed_source_ids=failures, model_calls=0, network_downloads=0,
                   raw_arrays_are_model_GT=False, numpy_version=np.__version__)
    if runtime_inventory and not failures:
        try:
            summary["runtime_inventory"] = write_runtime_inventory(
                output, inventory, assets, index["source_inventory_sha256"],
                utf8_unescaped=runtime_format == "utf8_unescaped_indent2")
        except (ValueError, OSError) as error:
            summary["runtime_inventory_error"] = str(error)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", required=True, type=Path)
    parser.add_argument("--source-index-sha256", required=True)
    parser.add_argument("--shard-checksums", required=True, type=Path)
    parser.add_argument("--shard-checksums-sha256", required=True)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-inventory", action="store_true",
                        help="Also rebuild the fixed full-panel input inventory for prepare plan and run")
    args = parser.parse_args(argv)
    for path, expected in ((args.source_index, args.source_index_sha256),
                           (args.shard_checksums, args.shard_checksums_sha256)):
        if not HEX.fullmatch(expected) or digest(path) != expected:
            raise ValueError("input metadata SHA256 mismatch")
    index = json.loads(args.source_index.read_text(encoding="utf-8"))
    shards = json.loads(args.shard_checksums.read_text(encoding="utf-8"))
    summary = reconstruct(index, shards, args.shard_dir, args.output,
                          runtime_inventory=args.runtime_inventory)
    print(json.dumps(summary))
    return int(bool(summary["failed"] or summary.get("runtime_inventory_error")))


if __name__ == "__main__":
    raise SystemExit(main())
