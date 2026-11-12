"""Versioned H1 RLBench action representations and immutable data provenance.

Relative Euler means componentwise wrap(target_xyz_euler - current_xyz_euler).
It is deliberately NOT a quaternion/SO(3) relative rotation. Positions already
contain deltas in both contracts. Angles are radians, SciPy extrinsic ``xyz``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ABSOLUTE_EULER = "xyz_delta_euler_absolute_v1"
RELATIVE_EULER = "xyz_delta_euler_relative_wrapped_v1"
ACTION_REPRESENTATIONS = (ABSOLUTE_EULER, RELATIVE_EULER)
SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_content_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_representation_name(representation: str) -> str:
    if representation not in ACTION_REPRESENTATIONS:
        raise ValueError(f"Unknown RLBench action representation: {representation!r}")
    return representation


def wrap_euler(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Euler values must be finite")
    return np.remainder(values + np.pi, 2.0 * np.pi) - np.pi


def _actions(values: Any, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] != 7:
        raise ValueError(f"{label} must have final dimension 7, got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return array


def encode_relative_euler_actions(actions: Any, current_states: Any) -> np.ndarray:
    """Convert absolute Euler targets to wrapped componentwise Euler deltas.

    Shapes must match exactly (..., 7); no implicit temporal broadcasting.
    Returns a copy preserving floating input dtype and xyz/gripper values.
    """
    actions = _actions(actions, "actions")
    states = _actions(current_states, "current_states")
    if actions.shape != states.shape:
        raise ValueError(f"Action/state shapes must match: {actions.shape} != {states.shape}")
    output = actions.copy()
    output[..., 3:6] = wrap_euler(
        actions[..., 3:6].astype(np.float64) - states[..., 3:6].astype(np.float64)
    )
    return output


def decode_action_for_execution(action: Any, current_state: Any, representation: str) -> np.ndarray:
    """Restore absolute Euler for one denormalized H1 action before EEpose.

    This does not add xyz to current position or threshold the gripper. Those
    existing evaluator steps remain unchanged. A/default is an exact copy.
    The caller must pass the live [xyz, xyz-Euler, gripper] state for B even
    when proprioception is excluded from the model input (robot_state=0).
    """
    validate_representation_name(representation)
    action = _actions(action, "action")
    if action.shape != (7,):
        raise ValueError(f"Execution decoder is H1 only, got {action.shape}")
    output = action.copy()
    if representation == ABSOLUTE_EULER:
        return output
    state = _actions(current_state, "current_state")
    if state.shape != (7,):
        raise ValueError(f"Execution state must be [7], got {state.shape}")
    output[3:6] = wrap_euler(state[3:6].astype(np.float64) + action[3:6].astype(np.float64))
    return output


def _validate_manifest_content(manifest: Mapping[str, Any], representation: str) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "action_representation": representation,
        "source_action_representation": ABSOLUTE_EULER,
        "action_chunk": 1,
        "action_dim": 7,
        "rotation_convention": "scipy_extrinsic_xyz_radians",
        "relative_rotation_definition": ("wrap(target_euler_minus_current_euler)_componentwise"
                                         if representation == RELATIVE_EULER else "not_applicable_absolute_euler"),
        "position_definition": "source_xyz_delta_unchanged",
        "gripper_definition": "source_value_unchanged",
        "statistics_scope": ("converted_train_json_all_rows_only" if representation == RELATIVE_EULER
                              else "source_train_json_statistics_unchanged"),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Action manifest {key}: {manifest.get(key)!r} != {value!r}")
    for key in ("source_data_sha256", "source_stats_sha256", "converted_data_sha256",
                "converted_stats_sha256", "converted_stats_content_sha256"):
        value = manifest.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Missing/invalid action manifest hash: {key}")


def _validate_stats_representation(stats: Mapping[str, Any], representation: str) -> None:
    if not isinstance(stats, dict) or len(stats) != 1:
        raise ValueError("Expected one dataset in action statistics")
    dataset = next(iter(stats.values()))
    declared = dataset.get("action_representation", ABSOLUTE_EULER)
    if declared != representation:
        raise ValueError(f"Stats representation {declared!r} != {representation!r}")


def _manifest_metadata(manifest: Mapping[str, Any], path: str | None, file_sha: str | None) -> dict:
    metadata = {
        "action_representation": manifest["action_representation"],
        "action_representation_manifest_path": path,
        "action_representation_manifest_sha256": file_sha,
        "action_representation_manifest_content_sha256": json_content_sha256(manifest),
        "action_representation_provenance": dict(manifest),
    }
    for key in ("source_data_sha256", "source_stats_sha256", "converted_data_sha256",
                "converted_stats_sha256", "converted_stats_content_sha256"):
        metadata[f"action_representation_{key}"] = manifest[key]
    return metadata


def validate_action_representation_manifest(
    data_path: str | Path,
    stats_path: str | Path,
    representation: str = ABSOLUTE_EULER,
    manifest_path: str | Path | None = None,
) -> dict:
    """Validate training data/stats before training; relative always fails closed."""
    validate_representation_name(representation)
    stats = _read_json(stats_path)
    _validate_stats_representation(stats, representation)
    if representation == ABSOLUTE_EULER and not manifest_path:
        return {"action_representation": representation,
                "action_representation_validation_mode": "legacy_absolute_no_manifest"}
    if not manifest_path:
        raise ValueError("Relative Euler training requires an action representation manifest")
    path = Path(manifest_path).resolve()
    manifest = _read_json(path)
    _validate_manifest_content(manifest, representation)
    for actual_path, key in ((data_path, "converted_data_sha256"),
                             (stats_path, "converted_stats_sha256")):
        if sha256_file(actual_path) != manifest[key]:
            raise ValueError(f"Action representation {key} mismatch: {actual_path}")
    if json_content_sha256(stats) != manifest["converted_stats_content_sha256"]:
        raise ValueError("Converted statistics JSON content hash mismatch")
    return _manifest_metadata(manifest, str(path), sha256_file(path))


def validate_saved_action_representation(training_config: Mapping[str, Any], stats_path: str | Path) -> dict:
    """Validate saved stats using embedded provenance; original data need not exist.

    Checkpoint saving may change whitespace/key order. Therefore saved statistics
    are bound by canonical JSON content, not the original stats file byte hash.
    """
    nested = training_config.get("data_provenance", {})
    metadata = training_config if "action_representation_provenance" in training_config else nested
    representation = training_config.get("action_representation", metadata.get("action_representation", ABSOLUTE_EULER))
    validate_representation_name(representation)
    stats = _read_json(stats_path)
    _validate_stats_representation(stats, representation)
    if representation == ABSOLUTE_EULER and not metadata.get("action_representation_provenance"):
        return {"action_representation": representation,
                "action_representation_validation_mode": "legacy_absolute_no_manifest"}
    manifest = metadata.get("action_representation_provenance")
    if not isinstance(manifest, dict):
        raise ValueError("Relative checkpoint is missing embedded action representation provenance")
    expected_sha = metadata.get("action_representation_manifest_content_sha256")
    if json_content_sha256(manifest) != expected_sha:
        raise ValueError("Saved action representation manifest content hash mismatch")
    _validate_manifest_content(manifest, representation)
    if json_content_sha256(stats) != manifest["converted_stats_content_sha256"]:
        raise ValueError("Saved statistics content does not match action representation manifest")
    return _manifest_metadata(manifest, metadata.get("action_representation_manifest_path"),
                              metadata.get("action_representation_manifest_sha256"))
