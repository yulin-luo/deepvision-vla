"""Pinned checkpoint verification and inexpensive, fail-closed reuse gates.

A full verification reads every pinned byte once. Runtime validation checks the
sealed receipt and file identities without hashing the multi-GB model again.
Receipts are local integrity evidence, not signatures or a defence against a
malicious local account able to forge both config and receipt.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time

from deepvision_vla.paths import ASSETS_ROOT

SCHEMA = "deepvision.checkpoint-verification"
SCHEMA_VERSION = 1
POLICY = "full_sha256_stable_stat_v1"
REVERIFY = "Run deepvision verify with a fresh receipt, then regenerate the configuration."


class CheckpointBindingError(ValueError):
    """The requested model no longer matches its complete verification evidence."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inventory(inventory_path=None):
    path = Path(inventory_path) if inventory_path else ASSETS_ROOT / "checkpoint_inventory.origin.json"
    raw = path.read_bytes()
    inventory = json.loads(raw)
    if not isinstance(inventory, dict) or len(inventory) != 20:
        raise CheckpointBindingError("The release requires exactly twenty inventory members.")
    for name, entry in inventory.items():
        member = PurePosixPath(name)
        if (not name or member.is_absolute() or ".." in member.parts or "\\" in name or ":" in name
                or not isinstance(entry, dict)
                or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
                or isinstance(entry.get("size"), bool)
                or not isinstance(entry.get("size"), int) or entry["size"] <= 0):
            raise CheckpointBindingError("Unsafe or unpinned inventory member: " + name)
    return inventory, hashlib.sha256(raw).hexdigest()


def _stat_fields(value):
    return {"size": value.st_size, "mtime_ns": value.st_mtime_ns,
            "ctime_ns": value.st_ctime_ns, "ino": value.st_ino,
            "dev": value.st_dev, "mode": value.st_mode}


def file_identity(path):
    logical = Path(path).absolute()
    resolved = logical.resolve(strict=True)
    target = resolved.stat()
    if not stat.S_ISREG(target.st_mode):
        raise CheckpointBindingError("Expected a regular checkpoint file: " + str(logical))
    link = logical.lstat()
    return {"resolved_path": str(resolved), "target": _stat_fields(target),
            "entry": _stat_fields(link),
            "symlink_target": os.readlink(logical) if stat.S_ISLNK(link.st_mode) else None}


def _read_digest(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4 * 2**20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _hash_stable_file(path):
    before = file_identity(path)
    with Path(path).open("rb") as stream:
        opened_before = _stat_fields(os.fstat(stream.fileno()))
        # Windows path stat and descriptor fstat can expose slightly different
        # creation-time values as ctime. Compare handle identity here; ctime is
        # still independently required stable before/after on BOTH interfaces.
        handle_fields = ("size", "mtime_ns", "ino", "dev", "mode")
        if any(opened_before[key] != before["target"][key] for key in handle_fields):
            raise CheckpointBindingError("Checkpoint changed while opening: " + str(path))
        digest = _read_digest(stream)
        opened_after = _stat_fields(os.fstat(stream.fileno()))
    after = file_identity(path)
    if before != after or opened_before != opened_after:
        raise CheckpointBindingError("Checkpoint changed during SHA256 verification: " + str(path))
    return digest, before, after


def _payload_seal(receipt):
    payload = {key: value for key, value in receipt.items() if key != "seal"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_checkpoint(checkpoint_root, receipt_path, *, inventory_path=None, progress=None):
    """Hash all twenty files once and persist a sealed passed/failed receipt.

    Existing receipts are never overwritten. A changed/missing/non-pinned file
    produces a failed receipt; callers must not configure or load it.
    """
    root = Path(checkpoint_root).expanduser().resolve(strict=True)
    output = Path(receipt_path).expanduser().absolute()
    if output.exists():
        raise FileExistsError("A fresh verification receipt is required: " + str(output))
    inventory, inventory_sha = load_inventory(inventory_path)
    rows = []
    for name, expected in sorted(inventory.items()):
        row = {"path": name, "expected_size": expected["size"],
               "expected_sha256": expected["sha256"], "ok": False}
        try:
            digest, before, after = _hash_stable_file(root / name)
            row.update(size=before["target"]["size"], sha256=digest,
                       before=before, after=after, stat=after)
            row["ok"] = row["size"] == expected["size"] and digest == expected["sha256"]
            if not row["ok"]:
                row["error"] = "pinned_size_or_sha256_mismatch"
        except (OSError, ValueError) as error:
            row["error"] = type(error).__name__ + ": " + str(error)
        rows.append(row)
        if progress:
            progress(name, row["ok"])
    # Catch a previously checked file changing while later files were hashed.
    for row in rows:
        if row["ok"]:
            try:
                if file_identity(root / row["path"]) != row["stat"]:
                    raise CheckpointBindingError("File changed before verification completed")
            except (OSError, ValueError) as error:
                row.update(ok=False, error=type(error).__name__ + ": " + str(error))
    receipt = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION,
               "policy": POLICY, "status": "passed" if all(r["ok"] for r in rows) else "failed",
               "checkpoint_root": str(root), "inventory_sha256": inventory_sha,
               "files": rows, "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    receipt["seal"] = {"algorithm": "sha256", "payload_sha256": _payload_seal(receipt)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def is_sealed_receipt(path):
    """Distinguish legacy unsealed receipts; do not treat this as validation."""
    try:
        data = json.loads(Path(path).read_bytes())
        return (isinstance(data, dict) and data.get("schema") == SCHEMA
                and data.get("schema_version") == SCHEMA_VERSION and "seal" in data)
    except (OSError, ValueError):
        return False


def checkpoint_binding(checkpoint_root, receipt_path, *, inventory_path=None):
    """Build config fields only after checking an existing sealed receipt."""
    _, inventory_sha = load_inventory(inventory_path)
    receipt = Path(receipt_path).expanduser().resolve(strict=True)
    binding = {"checkpoint_root": str(Path(checkpoint_root).expanduser().resolve(strict=True)),
               "checkpoint_verification_receipt": str(receipt),
               "checkpoint_verification_receipt_sha256": file_sha256(receipt),
               "checkpoint_inventory_sha256": inventory_sha}
    validate_checkpoint_binding(binding, inventory_path=inventory_path)
    return binding


def validate_checkpoint_binding(config, *, inventory_path=None):
    """Check receipt, pinned hashes and current file stats; never rehash weights."""
    required = ("checkpoint_root", "checkpoint_verification_receipt",
                "checkpoint_verification_receipt_sha256", "checkpoint_inventory_sha256")
    if any(not config.get(key) for key in required):
        raise CheckpointBindingError("Missing complete checkpoint verification binding. " + REVERIFY)
    inventory, inventory_sha = load_inventory(inventory_path)
    if config["checkpoint_inventory_sha256"] != inventory_sha:
        raise CheckpointBindingError("Checkpoint inventory changed. " + REVERIFY)
    expected_receipt_sha = config["checkpoint_verification_receipt_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_receipt_sha)):
        raise CheckpointBindingError("Invalid verification receipt digest. " + REVERIFY)
    try:
        root = Path(config["checkpoint_root"]).expanduser().resolve(strict=True)
        path = Path(config["checkpoint_verification_receipt"]).expanduser().resolve(strict=True)
        receipt_before = file_identity(path)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_receipt_sha:
            raise CheckpointBindingError("Verification receipt bytes changed")
        receipt = json.loads(raw)
        if (receipt.get("schema") != SCHEMA or receipt.get("schema_version") != SCHEMA_VERSION
                or receipt.get("policy") != POLICY or receipt.get("status") != "passed"
                or receipt.get("checkpoint_root") != str(root)
                or receipt.get("inventory_sha256") != inventory_sha
                or receipt.get("seal") != {"algorithm": "sha256", "payload_sha256": _payload_seal(receipt)}):
            raise CheckpointBindingError("Receipt is unsealed, failed, stale, or belongs to another checkpoint")
        rows = receipt.get("files")
        if (not isinstance(rows, list) or len(rows) != len(inventory)
                or len({row.get("path") for row in rows}) != len(inventory)
                or {row.get("path") for row in rows} != set(inventory)):
            raise CheckpointBindingError("Receipt does not cover the exact twenty-file inventory")
        for row in rows:
            expected = inventory[row["path"]]
            if (row.get("ok") is not True or row.get("size") != expected["size"]
                    or row.get("sha256") != expected["sha256"]
                    or row.get("expected_size") != expected["size"]
                    or row.get("expected_sha256") != expected["sha256"]
                    or row.get("before") != row.get("after") or row.get("after") != row.get("stat")):
                raise CheckpointBindingError("Unverified checkpoint member: " + row["path"])
            if file_identity(root / row["path"]) != row["stat"]:
                raise CheckpointBindingError("Checkpoint member changed: " + row["path"])
        if config.get("dino_runtime_root"):
            validate_dino_runtime_view(config)
        if file_identity(path) != receipt_before or file_sha256(path) != expected_receipt_sha:
            raise CheckpointBindingError("Verification receipt changed during validation")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        raise CheckpointBindingError(str(error) + ". " + REVERIFY) from error
    return {"checkpoint_root": str(root), "file_count": len(rows),
            "receipt_sha256": expected_receipt_sha, "inventory_sha256": inventory_sha,
            "weight_files_rehashed": 0, "policy": POLICY}


def validate_dino_runtime_view(config_or_root, checkpoint_root=None):
    """Bind runtime processor/model JSON and its weight alias to the release.

    This reads only two small JSON files and stats the weight alias. The full
    checkpoint receipt remains responsible for the original weight bytes.
    """
    if isinstance(config_or_root, dict):
        view_value = config_or_root.get("dino_runtime_root")
        checkpoint_value = config_or_root.get("checkpoint_root")
    else:
        view_value, checkpoint_value = config_or_root, checkpoint_root
    if not view_value or not checkpoint_value:
        raise CheckpointBindingError("DINO runtime view and checkpoint root are required; regenerate the configuration.")
    try:
        view = Path(view_value).expanduser().resolve(strict=True)
        checkpoint = Path(checkpoint_value).expanduser().resolve(strict=True)
        digests = {}
        for name in ("config.json", "preprocessor_config.json"):
            path = view / name
            before = file_identity(path)
            actual = path.read_bytes()
            expected = (ASSETS_ROOT / "dino_runtime" / name).read_bytes()
            if actual != expected or file_identity(path) != before:
                raise CheckpointBindingError("DINO runtime view changed or differs from release: " + name)
            digests[name] = hashlib.sha256(actual).hexdigest()
        weight = view / "pytorch_model.bin"
        before = file_identity(weight)
        if not weight.samefile(checkpoint / "dinov3_model/pytorch_model.bin"):
            raise CheckpointBindingError("DINO runtime weight alias points to a different checkpoint file")
        if file_identity(weight) != before:
            raise CheckpointBindingError("DINO runtime weight alias changed during validation")
    except (OSError, ValueError, TypeError) as error:
        raise CheckpointBindingError(str(error) + ". Regenerate the DINO view/configuration; do not edit runtime processor files.") from error
    return {"status": "passed", "small_file_sha256": digests,
            "weight_samefile": True, "weight_files_rehashed": 0}


def component_watch_paths(checkpoint_root):
    """Canonical keys support both HF cache leaf links and local DINO views."""
    root = Path(checkpoint_root)
    return {str((root / relative).resolve(strict=True)): name for relative, name in (
        ("action_head.pt", "action_head"), ("dinov3_aligner.pt", "aligner"),
        ("dinov3_model/pytorch_model.bin", "finetuned_dino"))}
