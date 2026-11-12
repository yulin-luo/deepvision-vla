"""Versioned RLBench data and simulator protocol resolution."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_PROFILE_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "rlbench_env_profiles.json"
)

TRAINING_PROTOCOLS = {"corrected_v1", "paper_history_v1"}
SIGLIP_TOKEN_CONTRACTS = {"published_legacy_225", "corrected_256"}


class RLBenchProtocolError(ValueError):
    pass


def resolve_siglip_token_contract(contract: str) -> Dict[str, Any]:
    contract = str(contract)
    if contract not in SIGLIP_TOKEN_CONTRACTS:
        raise RLBenchProtocolError(
            "SigLIP token contract must be one of "
            f"{sorted(SIGLIP_TOKEN_CONTRACTS)}, got {contract!r}"
        )
    if contract == "published_legacy_225":
        return {
            "siglip_token_contract": contract,
            "expert_num_prefix_tokens": 1,
            "expert_patch_grid_contract": "legacy_floor_square",
            "source_patch_tokens": 256,
            "spatially_ranked_patch_tokens": 225,
        }
    return {
        "siglip_token_contract": contract,
        "expert_num_prefix_tokens": 0,
        "expert_patch_grid_contract": "strict_square",
        "source_patch_tokens": 256,
        "spatially_ranked_patch_tokens": 256,
    }


def training_protocol_metadata(protocol: str) -> Dict[str, Any]:
    protocol = str(protocol)
    if protocol not in TRAINING_PROTOCOLS:
        raise RLBenchProtocolError(
            f"training protocol must be one of {sorted(TRAINING_PROTOCOLS)}, got {protocol!r}"
        )
    if protocol == "paper_history_v1":
        return {
            "training_protocol": protocol,
            "dataloader_rng_protocol": "torch_global_no_explicit_generator",
            "augmentation_rng_protocol": "python_global_worker_seeded",
            "scheduler_protocol": "historical_pre_shard_world_divide",
        }
    return {
        "training_protocol": protocol,
        "dataloader_rng_protocol": "explicit_torch_generator_per_rank",
        "augmentation_rng_protocol": "explicit_python_rng_per_rank_worker",
        "scheduler_protocol": "prepared_dataloader_length",
    }


def compute_historical_training_schedule(
    *,
    unprepared_dataloader_length: int,
    n_epochs: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> Tuple[int, int]:
    """Reproduce the pre-shard scheduler horizon in the paper-era trainer."""

    unprepared_dataloader_length = int(unprepared_dataloader_length)
    n_epochs = int(n_epochs)
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    world_size = int(world_size)
    for name, value in (
        ("unprepared_dataloader_length", unprepared_dataloader_length),
        ("n_epochs", n_epochs),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
        ("world_size", world_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    scheduler_steps = (
        unprepared_dataloader_length * n_epochs
    ) // gradient_accumulation_steps // world_size
    if scheduler_steps <= 0:
        raise ValueError(
            "Historical training configuration produces zero scheduler steps: "
            f"unprepared_batches_per_epoch={unprepared_dataloader_length}, "
            f"n_epochs={n_epochs}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}, "
            f"world_size={world_size}"
        )
    return unprepared_dataloader_length, scheduler_steps


def policy_checkpoint_evidence(model_path: os.PathLike[str] | str) -> Tuple[str, ...]:
    """Return robot-policy artifacts reachable from an initialization path."""

    path = Path(model_path).expanduser().resolve()
    roots = [path]
    if path.name == "tfmr":
        roots.append(path.parent)

    evidence = []
    for root in roots:
        for name in (
            "training_config.json",
            "action_head.pt",
            "stats_data.json",
            "dinov3_aligner.pt",
            "siglip_aligner.pt",
        ):
            candidate = root / name
            if candidate.exists():
                evidence.append(str(candidate))
    return tuple(sorted(set(evidence)))


def require_base_model_initialization(model_path: os.PathLike[str] | str) -> None:
    """Reject RLBench/OXE policy checkpoints for paper-from-base training."""

    evidence = policy_checkpoint_evidence(model_path)
    if evidence:
        rendered = "\n  - ".join(evidence)
        raise RLBenchProtocolError(
            "RLBench4 paper reproduction forbids robot-policy checkpoint initialization. "
            "Initialize from the Qwen3-VL base directory instead. Found:\n  - "
            f"{rendered}"
        )


def compute_training_schedule(
    *,
    prepared_dataloader_length: int,
    n_epochs: int,
    gradient_accumulation_steps: int,
) -> Tuple[int, int]:
    prepared_dataloader_length = int(prepared_dataloader_length)
    n_epochs = int(n_epochs)
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    if prepared_dataloader_length <= 0:
        raise ValueError(
            "prepared_dataloader_length must be positive, "
            f"got {prepared_dataloader_length}"
        )
    if n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive, got {n_epochs}")
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive, "
            f"got {gradient_accumulation_steps}"
        )

    optimizer_steps = math.ceil(
        prepared_dataloader_length / gradient_accumulation_steps
    ) * n_epochs
    if optimizer_steps <= 0:
        raise ValueError(
            "Training configuration produces zero optimizer steps: "
            f"prepared_batches_per_epoch={prepared_dataloader_length}, "
            f"n_epochs={n_epochs}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}"
        )
    return prepared_dataloader_length, optimizer_steps


def sha256_file(path: os.PathLike[str] | str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def resolve_dataset_image_path(
    raw_path: str,
    data_path: os.PathLike[str] | str,
    data_root: str = "",
) -> Tuple[str, str]:
    raw_path = os.path.expanduser(str(raw_path))
    data_root = os.path.expanduser(str(data_root)) if data_root else ""
    if data_root and ":" in data_root:
        old_prefix, new_prefix = data_root.split(":", 1)
        resolved = raw_path.replace(os.path.expanduser(old_prefix), os.path.expanduser(new_prefix))
        return os.path.abspath(resolved), "prefix-replace"
    if os.path.isabs(raw_path):
        return os.path.abspath(raw_path), "absolute"
    if data_root:
        return os.path.abspath(os.path.join(data_root, raw_path)), "relative-data-root"
    return (
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(data_path)), raw_path)),
        "relative-json-dir",
    )


def normalize_dataset_image_paths(
    data: Sequence[MutableMapping[str, Any]],
    data_path: os.PathLike[str] | str,
    data_root: str = "",
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in data:
        if not isinstance(item, MutableMapping):
            continue
        for field in ("input_image", "output_image"):
            value = item.get(field)
            if value is None:
                continue
            original_is_list = isinstance(value, list)
            values = value if original_is_list else [value]
            resolved_values = []
            for raw_path in values:
                if not isinstance(raw_path, str):
                    resolved_values.append(raw_path)
                    continue
                resolved, mode = resolve_dataset_image_path(raw_path, data_path, data_root)
                resolved_values.append(resolved)
                counts[mode] = counts.get(mode, 0) + 1
            if field == "input_image":
                item[field] = resolved_values
            else:
                item[field] = resolved_values if original_is_list else resolved_values[0]
    return dict(sorted(counts.items()))


def _as_hashes(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    return tuple(str(item).lower() for item in value if item)


def _profile_definition_sha256(profile: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        profile,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_profile_registry(
    profile_path: os.PathLike[str] | str | None = None,
) -> Dict[str, Any]:
    path = Path(profile_path or DEFAULT_PROFILE_PATH).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RLBenchProtocolError(f"Cannot load RLBench profile registry {path}: {exc}") from exc

    if payload.get("schema_version") != 1:
        raise RLBenchProtocolError(
            f"Unsupported RLBench profile schema in {path}: {payload.get('schema_version')!r}"
        )
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise RLBenchProtocolError(f"RLBench profile registry has no profiles: {path}")

    for profile_id, profile in profiles.items():
        if not isinstance(profile, dict):
            raise RLBenchProtocolError(f"Profile {profile_id!r} must be an object")
        simulator = profile.get("simulator", {})
        camera_resolution = simulator.get("camera_resolution")
        if not isinstance(camera_resolution, int) or camera_resolution <= 0:
            raise RLBenchProtocolError(
                f"Profile {profile_id!r} has invalid simulator.camera_resolution={camera_resolution!r}"
            )
        offsets = simulator.get("task_base_yaw_offsets", {})
        if not isinstance(offsets, dict):
            raise RLBenchProtocolError(
                f"Profile {profile_id!r} simulator.task_base_yaw_offsets must be an object"
            )
        for task_name, offset in offsets.items():
            if not isinstance(task_name, str) or not math.isfinite(float(offset)):
                raise RLBenchProtocolError(
                    f"Profile {profile_id!r} has invalid yaw offset {task_name!r}: {offset!r}"
                )

    return {
        "schema_version": payload["schema_version"],
        "profiles": profiles,
        "profile_file": str(path),
        "profile_file_sha256": sha256_file(path),
    }


def _candidate_profiles(
    profiles: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> Dict[str, set[str]]:
    evidence: Dict[str, set[str]] = {}
    for provenance_key, fingerprint_key in (
        ("data_sha256", "data_sha256"),
        ("stats_sha256", "stats_sha256"),
    ):
        value = provenance.get(provenance_key)
        if not value:
            continue
        value = str(value).lower()
        matched = {
            profile_id
            for profile_id, profile in profiles.items()
            if value in _as_hashes(profile.get("fingerprints", {}).get(fingerprint_key))
        }
        evidence[provenance_key] = matched
    return evidence


def _path_hint_candidates(
    profiles: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> set[str]:
    path_values = [
        provenance.get("data_path"),
        provenance.get("data_path_abs"),
        provenance.get("data_path_resolved"),
        provenance.get("stats_path"),
        provenance.get("stats_path_resolved"),
    ]
    haystack = "\n".join(str(value) for value in path_values if value)
    if not haystack:
        return set()
    return {
        profile_id
        for profile_id, profile in profiles.items()
        if any(
            str(hint) in haystack
            for hint in profile.get("fingerprints", {}).get("path_hints", [])
        )
    }


def _intersect_candidates(
    evidence: Mapping[str, set[str]],
) -> Optional[Tuple[set[str], str]]:
    if not evidence:
        return None
    rendered = {key: sorted(value) for key, value in evidence.items()}
    unknown = sorted(key for key, value in evidence.items() if not value)
    if unknown:
        raise RLBenchProtocolError(
            "Unknown RLBench provenance fingerprint(s) for "
            f"{unknown}: {rendered}"
        )
    candidates = set.intersection(*(set(value) for value in evidence.values()))
    if not candidates:
        raise RLBenchProtocolError(
            f"RLBench provenance fingerprints disagree across profiles: {rendered}"
        )
    return candidates, "+".join(sorted(evidence))


def resolve_profile(
    requested_profile: str,
    provenance: Mapping[str, Any],
    *,
    profile_path: os.PathLike[str] | str | None = None,
    allow_unknown: bool = False,
    allow_mismatch: bool = False,
) -> Dict[str, Any]:
    registry = load_profile_registry(profile_path)
    profiles = registry["profiles"]
    requested = str(requested_profile or "auto")
    recorded = provenance.get("rlbench_data_profile") or provenance.get(
        "resolved_rlbench_data_profile"
    )
    hash_evidence = _candidate_profiles(profiles, provenance)
    fingerprint_resolution = _intersect_candidates(hash_evidence)
    fingerprint_candidates = fingerprint_resolution[0] if fingerprint_resolution else set()
    fingerprint_source = fingerprint_resolution[1] if fingerprint_resolution else ""

    if requested != "auto":
        if requested not in profiles:
            raise RLBenchProtocolError(
                f"Unknown requested RLBench profile {requested!r}; available={sorted(profiles)}"
            )
        conflicts = []
        if recorded and recorded != requested:
            conflicts.append(f"checkpoint records {recorded!r}")
        if fingerprint_candidates and requested not in fingerprint_candidates:
            conflicts.append(
                f"fingerprints resolve to {sorted(fingerprint_candidates)!r} "
                f"via {fingerprint_source}"
            )
        if conflicts and not allow_mismatch:
            raise RLBenchProtocolError(
                f"Requested RLBench profile {requested!r} conflicts with " + "; ".join(conflicts)
            )
        resolved = requested
        source = "explicit"
    elif recorded:
        if recorded not in profiles:
            raise RLBenchProtocolError(
                f"Checkpoint records unknown RLBench profile {recorded!r}; available={sorted(profiles)}"
            )
        if fingerprint_candidates and recorded not in fingerprint_candidates and not allow_mismatch:
            raise RLBenchProtocolError(
                f"Checkpoint profile {recorded!r} conflicts with fingerprint candidates "
                f"{sorted(fingerprint_candidates)!r} via {fingerprint_source}"
            )
        resolved = str(recorded)
        source = "checkpoint"
    elif fingerprint_candidates:
        if len(fingerprint_candidates) != 1:
            raise RLBenchProtocolError(
                "RLBench provenance fingerprints are ambiguous across profiles: "
                f"{sorted(fingerprint_candidates)} via {fingerprint_source}"
            )
        resolved = next(iter(fingerprint_candidates))
        source = f"fingerprint:{fingerprint_source}"
    else:
        path_candidates = _path_hint_candidates(profiles, provenance)
        if len(path_candidates) == 1:
            resolved = next(iter(path_candidates))
            source = "path-hint"
        elif len(path_candidates) > 1:
            raise RLBenchProtocolError(
                f"RLBench path hints are ambiguous across profiles: {sorted(path_candidates)}"
            )
        elif allow_unknown:
            return {
                "requested_profile": requested,
                "resolved_profile": "legacy-unversioned",
                "resolution_source": "unknown-allowed",
                "profile": None,
                "profile_file": registry["profile_file"],
                "profile_file_sha256": registry["profile_file_sha256"],
                "schema_version": registry["schema_version"],
            }
        else:
            raise RLBenchProtocolError(
                "Cannot resolve RLBench profile from checkpoint/data provenance. "
                "Pass --rlbench-env-profile/--rlbench-data-profile explicitly for a known protocol. "
                f"data_sha256={provenance.get('data_sha256')!r}, "
                f"stats_sha256={provenance.get('stats_sha256')!r}, "
                f"data_path={provenance.get('data_path')!r}"
            )

    current_definition_sha256 = _profile_definition_sha256(profiles[resolved])
    recorded_definition_sha256 = provenance.get("rlbench_profile_definition_sha256")
    if (
        recorded_definition_sha256
        and str(recorded_definition_sha256).lower() != current_definition_sha256
        and not allow_mismatch
    ):
        raise RLBenchProtocolError(
            f"RLBench profile {resolved!r} definition changed since checkpoint creation: "
            f"checkpoint={recorded_definition_sha256}, current={current_definition_sha256}"
        )

    return {
        "requested_profile": requested,
        "resolved_profile": resolved,
        "resolution_source": source,
        "profile": profiles[resolved],
        "profile_definition_sha256": current_definition_sha256,
        "profile_file": registry["profile_file"],
        "profile_file_sha256": registry["profile_file_sha256"],
        "schema_version": registry["schema_version"],
    }


def resolve_task_runtime(
    profile_resolution: Mapping[str, Any],
    task_name: str,
    *,
    explicit_yaw_offset: float | None = None,
    camera_resolution_override: int | None = None,
) -> Dict[str, Any]:
    profile = profile_resolution.get("profile")
    if profile is None:
        if explicit_yaw_offset is None or camera_resolution_override is None:
            raise RLBenchProtocolError(
                "An unversioned RLBench profile requires explicit yaw and camera resolution overrides"
            )
        profile_yaw = 0.0
        profile_camera = int(camera_resolution_override)
    else:
        simulator = profile["simulator"]
        profile_yaw = float(simulator.get("default_task_base_yaw_offset", 0.0))
        profile_yaw = float(simulator.get("task_base_yaw_offsets", {}).get(task_name, profile_yaw))
        profile_camera = int(simulator["camera_resolution"])

    if explicit_yaw_offset is not None and not math.isfinite(float(explicit_yaw_offset)):
        raise RLBenchProtocolError(
            f"Explicit task yaw offset must be finite, got {explicit_yaw_offset!r}"
        )
    if camera_resolution_override is not None and int(camera_resolution_override) <= 0:
        raise RLBenchProtocolError(
            f"Camera resolution override must be positive, got {camera_resolution_override!r}"
        )

    return {
        "task_name": task_name,
        "profile_task_base_yaw_offset": profile_yaw,
        "explicit_task_base_yaw_offset": explicit_yaw_offset,
        "resolved_task_base_yaw_offset": (
            float(explicit_yaw_offset) if explicit_yaw_offset is not None else profile_yaw
        ),
        "profile_camera_resolution": profile_camera,
        "camera_resolution_override": camera_resolution_override,
        "resolved_camera_resolution": (
            int(camera_resolution_override)
            if camera_resolution_override is not None
            else profile_camera
        ),
    }


def profile_metadata(profile_resolution: Mapping[str, Any]) -> Dict[str, Any]:
    profile = profile_resolution.get("profile") or {}
    data = profile.get("data", {})
    simulator = profile.get("simulator", {})
    return {
        "rlbench_profile_schema_version": profile_resolution.get("schema_version"),
        "rlbench_profile_file": profile_resolution.get("profile_file"),
        "rlbench_profile_file_sha256": profile_resolution.get("profile_file_sha256"),
        "rlbench_profile_definition_sha256": profile_resolution.get(
            "profile_definition_sha256"
        ),
        "rlbench_profile_requested": profile_resolution.get("requested_profile"),
        "rlbench_data_profile": profile_resolution.get("resolved_profile"),
        "rlbench_profile_resolution_source": profile_resolution.get("resolution_source"),
        "rlbench_native_image_resolution": data.get("native_image_resolution"),
        "rlbench_image_path_contract": data.get("image_path_contract"),
        "rlbench_rotation_encoding": data.get("rotation_encoding"),
        "rlbench_profile_camera_resolution": simulator.get("camera_resolution"),
    }


def checkpoint_provenance(
    training_config: Mapping[str, Any],
    checkpoint_stats_path: os.PathLike[str] | str,
) -> Dict[str, Any]:
    provenance = dict(training_config)
    actual_stats_sha256 = sha256_file(checkpoint_stats_path)
    configured_checkpoint_sha256 = provenance.get("checkpoint_stats_sha256")
    if configured_checkpoint_sha256 and actual_stats_sha256:
        if str(configured_checkpoint_sha256).lower() != actual_stats_sha256.lower():
            raise RLBenchProtocolError(
                "Checkpoint stats_data.json hash differs from training_config checkpoint_stats_sha256: "
                f"actual={actual_stats_sha256}, configured={configured_checkpoint_sha256}"
            )
    if actual_stats_sha256:
        provenance["checkpoint_stats_sha256"] = actual_stats_sha256
        provenance.setdefault("stats_sha256", actual_stats_sha256)
    return provenance
