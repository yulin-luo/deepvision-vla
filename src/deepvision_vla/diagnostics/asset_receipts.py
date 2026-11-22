"""Verify already-downloaded snapshot files against a pinned inventory."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

from .adapters.qwen_h16_contracts import RUNTIME_EXPORT_FORMAT, RUNTIME_EXPORT_ID, matches_controlled_repository

SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GIT = re.compile(r"[0-9a-f]{40}\Z")
REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
MODEL = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
MAX_METADATA_BYTES = 20_000_000
QWEN_REVISION = "b131353d58b77daf2b6308e5ea4ea0aa4a417a79"
QWEN_PREFIX = "qwen_only/checkpoint-5-10000"
QWEN_REQUIRED = {"training_config.json", "portable/qwen_backbone/manifest.json",
                 "portable/qwen_backbone/model.safetensors", "portable/shared_action_head/manifest.json",
                 "portable/shared_action_head/action_head.safetensors"}


def _bounded_hashes(stream, size):
    if type(size) is not int or size < 0:
        raise ValueError("Exact expected read size required")
    sha = hashlib.sha256()
    git = hashlib.sha1(("blob " + str(size) + "\0").encode("ascii"))
    remaining = size
    while remaining:
        block = stream.read(min(2**20, remaining))
        if not block:
            raise ValueError("File ended before expected byte size")
        remaining -= len(block)
        sha.update(block)
        git.update(block)
    if stream.read(1):
        raise ValueError("File grew beyond the pinned byte budget")
    return sha.hexdigest(), git.hexdigest()


def file_sha(path, expected_size=None):
    with Path(path).open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size if expected_size is None else expected_size
        return _bounded_hashes(stream, size)[0]


def read_metadata(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("Source inventory metadata exceeds size bound")
    return raw


def validate_inventory(inventory):
    if (not isinstance(inventory, dict) or not isinstance(inventory.get("model"), str)
            or not MODEL.fullmatch(inventory["model"])):
        raise ValueError("Explicit snapshot repository required")
    if not isinstance(inventory.get("revision"), str) or not REVISION.fullmatch(inventory["revision"]):
        raise ValueError("Pinned immutable revision required")
    files = inventory.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Nonempty complete expected file inventory required")
    seen, total = set(), 0
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Invalid inventory file entry")
        name = item["path"]
        logical = PurePosixPath(name)
        if (not name or not logical.parts or logical.is_absolute() or str(logical) != name or ".." in logical.parts
                or any(c in name for c in '\\:<>|?*') or any(ord(c) < 32 for c in name) or name in seen):
            raise ValueError("Unsafe, duplicate or noncanonical snapshot path")
        if type(item.get("size")) is not int or item["size"] < 0:
            raise ValueError("Exact nonnegative expected byte size required")
        content, source = item.get("sha256"), item.get("git_blob")
        if content is not None and (not isinstance(content, str) or not SHA256.fullmatch(content)):
            raise ValueError("Invalid expected content SHA256")
        if source is not None and (not isinstance(source, str) or not GIT.fullmatch(source)):
            raise ValueError("Invalid expected Git blob")
        if content is None and source is None:
            raise ValueError("Every expected file needs a publisher checksum")
        seen.add(name)
        total += item["size"]
    if "total_bytes" in inventory and (type(inventory["total_bytes"]) is not int or inventory["total_bytes"] != total):
        raise ValueError("Expected total does not equal the complete file inventory")
    flavor = inventory.get("receipt_format", "snapshot_v1")
    prefix = inventory.get("receipt_prefix", "")
    if flavor in {"qwen_h16_components_v1", RUNTIME_EXPORT_FORMAT}:
        public = flavor == RUNTIME_EXPORT_FORMAT
        model = inventory["model"]
        identity_matches = model == RUNTIME_EXPORT_ID if public else matches_controlled_repository(model)
        files_match = seen == QWEN_REQUIRED if public else QWEN_REQUIRED <= seen
        if (not identity_matches or inventory["revision"] != QWEN_REVISION
                or prefix != QWEN_PREFIX or not files_match):
            raise ValueError("Explicit pinned Qwen H16 component identity and prefix required")
        weight_paths = {name for name in seen if name.endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt"))}
        if weight_paths != {"portable/qwen_backbone/model.safetensors", "portable/shared_action_head/action_head.safetensors"}:
            raise ValueError("Qwen H16 requires exactly its backbone and shared action head weights")
    elif (flavor != "snapshot_v1" or prefix or inventory["model"] == RUNTIME_EXPORT_ID
            or matches_controlled_repository(inventory["model"])):
        raise ValueError("Unknown receipt format or implicit Qwen prefix")
    return total


def _stamp(path):
    link, target = path.lstat(), path.resolve(strict=True)
    stat = target.stat()
    if not target.is_file():
        raise ValueError("Snapshot entry is not a regular file")
    return (str(target), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
            stat.st_ctime_ns, link.st_dev, link.st_ino, link.st_size, link.st_mtime_ns)


def verify_one(root, item):
    path = root.joinpath(*PurePosixPath(item["path"]).parts)
    # HF leaf links may point into its blob cache. Directory redirects are not
    # part of that layout and must not let an inventory escape its input root.
    path.parent.resolve(strict=True).relative_to(root)
    before = _stamp(path)
    if before[3] != item["size"]:
        raise ValueError("Snapshot file byte size mismatch")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != before[1:5]:
            raise ValueError("Snapshot link changed before reading")
        sha, git = _bounded_hashes(stream, item["size"])
        ended = os.fstat(stream.fileno())
    # On Windows path.stat and fstat may expose different ctime semantics.
    # Compare each API's own before/after ctime, without weakening the shared
    # inode, size and mtime binding between the path and the opened descriptor.
    fd_before = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
    fd_after = (ended.st_dev, ended.st_ino, ended.st_size, ended.st_mtime_ns, ended.st_ctime_ns)
    if _stamp(path) != before or fd_after != fd_before:
        raise ValueError("Snapshot file changed during verification")
    if item.get("sha256") is not None:
        # A Hub LFS Git blob identifies its pointer, not the downloaded data.
        if sha != item["sha256"]:
            raise ValueError("Pinned content SHA256 mismatch")
        method = "sha256"
    else:
        if git != item["git_blob"]:
            raise ValueError("Pinned source Git blob mismatch")
        method = "git_blob"
    # Cached/coarse filesystem metadata alone can miss a same-sized write.
    # Reopen the path and require a second identical content read; no sleeps,
    # timestamp tolerance or substitution of a newly observed expected hash.
    if file_sha(path, item["size"]) != sha or _stamp(path) != before:
        raise ValueError("Snapshot file changed between verification passes")
    return {"path": item["path"], "size": item["size"], "sha256": sha,
            "verified": True, "verified_by": method, "content_passes": 2}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON metadata key")
        result[key] = value
    return result


def create_receipt(inventory_bytes, root, inventory_sha):
    if (not isinstance(inventory_bytes, bytes) or len(inventory_bytes) > MAX_METADATA_BYTES
            or not isinstance(inventory_sha, str) or not SHA256.fullmatch(inventory_sha)
            or hashlib.sha256(inventory_bytes).hexdigest() != inventory_sha):
        raise ValueError("Exact bounded source inventory bytes/SHA256 required")
    inventory = json.loads(inventory_bytes.decode("utf-8"), object_pairs_hook=_unique_object)
    total = validate_inventory(inventory)
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Local snapshot directory required")
    rows = []
    for item in inventory["files"]:
        try:
            row = verify_one(root, item)
        except (OSError, ValueError, RuntimeError) as error:
            row = {"path": item["path"], "size": item["size"], "verified": False,
                   "error_type": type(error).__name__, "error": str(error)}
        if inventory.get("receipt_format") in {"qwen_h16_components_v1", RUNTIME_EXPORT_FORMAT}:
            row["path"] = QWEN_PREFIX + "/" + row["path"]
        rows.append(row)
    failed = sum(not row["verified"] for row in rows)
    return {"schema": "deepvision_diagnostic_snapshot_receipt_v1",
            "status": "complete" if not failed else "incomplete",
            "model": inventory["model"], "revision": inventory["revision"],
            "repo": inventory["model"], "receipt_format": inventory.get("receipt_format", "snapshot_v1"),
            "inventory_sha256": inventory_sha, "files": rows,
            "expected_files": len(rows), "verified_files": len(rows)-failed, "failed_files": failed,
            "total_expected_bytes": total, "total_verified_bytes": sum(row["size"] for row in rows if row["verified"]),
            "model_loads": 0, "model_calls": 0, "downloads": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Receipt output already exists; keep it and choose a new path")
    raw = read_metadata(args.inventory)
    if (not SHA256.fullmatch(args.inventory_sha256)
            or hashlib.sha256(raw).hexdigest() != args.inventory_sha256):
        raise ValueError("Pinned source inventory SHA256 mismatch")
    receipt = create_receipt(raw, args.checkpoint, args.inventory_sha256)
    if read_metadata(args.inventory) != raw:
        raise RuntimeError("Source inventory changed during verification")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: receipt[key] for key in ("status", "expected_files", "verified_files", "failed_files")}))
    return 0 if receipt["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
