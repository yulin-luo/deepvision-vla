"""Prepare a local launch for a BridgeData V2 diagnostic model."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import formal_execution_core as core
from .adapters.qwen_h16_contracts import RUNTIME_EXPORT_FORMAT
from .asset_receipts import QWEN_PREFIX, validate_inventory
from .formal_single_model_watchdog import BINDING_PATHS, local_filesystem, preflight
from .resources import device_identity
from .source_manifest import build_source_manifest

CHECKPOINTS = {
    "openvla": (
        "openvla/openvla-7b",
        "4ea2f6a3ce32a05f9ff8d11efcd4b1dd50a8440551341e23a01b2bfdae631c94",
    ),
    "pi0": (
        "HaomingSong/lerobot-pi0-bridge",
        "a5127810e0db99cc48469b28493216f64f65c8396a768caafafa933c51fe96a7",
    ),
}
PI0_NOISE_SHA = "911edc48185d2fb26a3f351c9571330a0cda906c4f641e09f17f099f006ef99a"


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def read_json(path):
    return json.loads(path.read_bytes())


def find_executable(path):
    # Resolving a venv's python symlink would discard the venv.
    path = Path(os.path.abspath(path))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Executable not found: {path}")
    return path


def validate_model_python(executable, versions):
    query = (
        "import importlib.metadata as m, json, sys; "
        "print(json.dumps({name: m.version(name) for name in sys.argv[1:]}))"
    )
    actual = json.loads(subprocess.check_output(
        [str(executable), "-c", query, *versions], timeout=30
    ))
    if actual != versions:
        raise ValueError(f"Model environment versions differ: {actual!r}")


def controlled_qwen_factory(checkout, checkpoint, inventory, receipt, processor, statistics):
    """Bind the original or re-exported H16 components to the same native policy."""
    contract = read_json(checkout / "configs/diagnostics/qwen-h16-vendor-seals.json")
    expected = dict(contract["checkpoint_component_sha256"])
    validate_inventory(inventory)
    flavor = inventory.get("receipt_format")
    if flavor == RUNTIME_EXPORT_FORMAT:
        expected["training_config.json"] = contract["runtime_export_training_config_sha256"]
        if receipt.get("receipt_format") != flavor:
            raise ValueError("Use the Qwen runtime export asset receipt")
    elif flavor != "qwen_h16_components_v1":
        raise ValueError("Use a supported Qwen H16 component inventory")
    if {item["path"]: item["sha256"] for item in inventory["files"]} != expected:
        raise ValueError("The inventory does not bind the pinned Qwen H16 components")
    prefix = QWEN_PREFIX + "/"
    receipt_files = {row["path"]: (row["size"], row["sha256"], row.get("verified"))
                     for row in receipt["files"]}
    intended_receipt = {prefix + row["path"]: (row["size"], row["sha256"], True)
                        for row in inventory["files"]}
    if (len(receipt_files) != len(receipt["files"]) or receipt_files != intended_receipt):
        raise ValueError("The Qwen H16 receipt differs from the approved component inventory")
    vendor = checkout / "model_runtime"
    if core.file_sha(statistics) != contract["statistics_sha256"]:
        raise ValueError("Qwen H16 action statistics differ from the pinned checkpoint")
    for directory, checksums in ((vendor, contract["source_sha256"]), (processor, contract["processor_sha256"])):
        for name, expected_sha in checksums.items():
            if core.file_sha(directory / name) != expected_sha:
                raise ValueError(f"Qwen H16 runtime or processor file differs: {name}")
    return {
        "source_root": str(vendor),
        "config_dir": str(processor),
        "checkpoint_root": str(checkpoint),
        "statistics_path": str(statistics),
        "source_sha256": contract["source_sha256"],
        "processor_sha256": contract["processor_sha256"],
    }


def prepare_launch(args):
    if sys.platform != "linux" or not shutil.which("nvidia-smi"):
        raise ValueError("Use Linux with nvidia-smi for launch preparation")
    checkout = Path(__file__).resolve().parents[3]
    model = args.model
    if (model == "pi0") != (args.lerobot is not None):
        raise ValueError("Supply --lerobot only for the Bridge Pi0 policy")
    qwen = model == "qwen_h16"
    qwen_paths = (args.qwen_processor, args.qwen_statistics)
    if (qwen and not all(qwen_paths)) or (not qwen and any(qwen_paths)):
        raise ValueError("Supply --qwen-processor and --qwen-statistics only for Qwen H16")
    panel = args.panel.resolve(strict=True)
    inputs = args.inputs.resolve(strict=True)
    metadata = args.metadata.resolve(strict=True)
    plan = args.plan.resolve(strict=True)
    work = args.output.resolve()
    local = args.node_local.resolve()
    parent = local / "watchdog"
    child = work / "result"
    if (local == work or local.is_relative_to(work) or work.is_relative_to(local)):
        raise ValueError("Choose separate working and node-local output directories")
    if local.exists() or work.exists():
        raise ValueError("Choose new working and node-local output directories")
    local_filesystem(local)
    checkpoint = args.checkpoint.resolve(strict=True)
    receipt_path = args.asset_receipt.resolve(strict=True)
    inventory_path = args.checkpoint_inventory.resolve(strict=True)
    actual_inventory_sha = core.file_sha(inventory_path)
    inventory = read_json(inventory_path)
    expected_repo, expected_inventory_sha = (
        (inventory["model"], actual_inventory_sha) if qwen else CHECKPOINTS[model])
    receipt = read_json(receipt_path)
    if (expected_inventory_sha != actual_inventory_sha or receipt.get("status") != "complete"
            or receipt.get("model") != expected_repo or receipt.get("revision") != core.REVISIONS[model]
            or receipt.get("inventory_sha256") != expected_inventory_sha):
        raise ValueError("Create a complete asset receipt for the pinned checkpoint first")

    head_python = find_executable(sys.executable)
    model_python = find_executable(args.model_python)
    smi = find_executable(shutil.which("nvidia-smi"))
    gpu = device_identity(args.gpu)
    runtime = metadata / f"{model}.runtime.json"
    codec = metadata / f"{model}.codec.json"
    references = metadata / "pi0.references.json"
    native_report = read_json(metadata / "READBACK.json")
    report_checksums = native_report["payload_sha256"]
    checked_metadata = [runtime, codec, metadata / "empty-history.json"]
    if model == "pi0":
        checked_metadata.append(references)
    for path in checked_metadata:
        if core.file_sha(path) != report_checksums[path.name]:
            raise ValueError(f"Runtime metadata differs: {path.name}")
    validate_model_python(model_python, read_json(runtime)["versions"])
    manifest = read_json(plan / "manifest.json")
    core.validate_manifest(manifest, model)
    if manifest["inventory_sha256"] != native_report["source_inventory_sha256"]:
        raise ValueError("The request plan uses a different Bridge panel")
    if (manifest.get("pilot_source_ordinals") is None
            and manifest["requests_jsonl_sha256"] != native_report["portable_request_sha256"]):
        raise ValueError("The full request plan differs from the verified portable request plan")
    rows = manifest["prospective_workload"][model]
    if rows["prospective_distinct_computations"] <= 0:
        raise ValueError("The request plan does not contain eligible predictions")
    factory = {"checkpoint": str(checkpoint), "asset_receipt": str(receipt_path)}
    if model == "openvla":
        factory["official_inventory"] = str(inventory_path)
    elif model == "pi0":
        factory["lerobot_source_root"] = str(args.lerobot.resolve(strict=True))
    else:
        factory = controlled_qwen_factory(
            checkout, checkpoint, inventory, receipt,
            args.qwen_processor.resolve(strict=True), args.qwen_statistics.resolve(strict=True))

    source_files = (checkout / "src" / "deepvision_vla").rglob("*.py")
    source = build_source_manifest(checkout, [p.relative_to(checkout).as_posix() for p in source_files])
    files = {row["path"]: row["sha256"] for row in source["files"]}
    work.mkdir(parents=True, exist_ok=False)
    local.mkdir(parents=True, exist_ok=False)
    source_manifest = write_json(work / "source_manifest.json", {"files": files})
    scientific_seals = write_json(work / "scientific_source_seals.json", [
        {"path": str(checkout / name), "sha256": sha} for name, sha in files.items()
    ])
    config = {
        "schema": "formal_diagnostic_executor_config_v1",
        "model": model,
        "factory_kwargs": factory,
        "asset_receipt_prefix": QWEN_PREFIX if qwen else "",
        "asset_receipt": str(receipt_path),
        "plan_manifest": str(plan / "manifest.json"),
        "requests": str(plan / "requests.jsonl"),
        "inventory": str(panel / "inventory.json"),
        "data_root": str(panel),
        "profiles": str(inputs / "profiles.json"),
        "annotations": str(inputs / "annotations.json"),
        "geometries": str(inputs / "geometries.jsonl"),
        "scientific_source_seals": str(scientific_seals),
        "runtime_receipt": str(runtime),
        "output_codec": str(codec),
        "history_manifest": str(metadata / "empty-history.json"),
        "output": str(child),
        "once_file": str(work / "worker.once.json"),
        "attempt_registry": str(work / "attempts"),
    }
    if model == "pi0":
        config["dataset_references"] = str(references)
    config_path = write_json(work / "config.json", config)
    bindings = {key: core.file_sha(config[path] if path else config_path)
                for key, path in BINDING_PATHS.items()}
    for key, sha in manifest["evidence_input_sha256"].items():
        if bindings.get(key + "_sha256") != sha:
            raise ValueError(f"The plan and supplied evidence differ: {key}")
    if (bindings["inventory_sha256"] != manifest["inventory_sha256"]
            or bindings["requests_jsonl_sha256"] != manifest["requests_jsonl_sha256"]):
        raise ValueError("Panel or request bytes differ from the declared plan")

    scope = f"bridge-{model}-{bindings['config_sha256'][:16]}"
    now = time.time()
    seconds = args.time_limit
    if seconds is None:
        seconds = 1200 if manifest.get("pilot_source_ordinals") else 19 * 3600
    permit = {
        "schema": core.SCHEMA, "model": model, "seed": 42,
        "permit_id": scope, "scope_id": scope,
        "execution_authorized": True, "reviewed_bank_qualified": True,
        "history_coverage_review_complete": True, "single_writer_scope": True,
        "bindings": bindings,
        "budget_origin_epoch": now,
        "cpu_deadline_epoch": now + min(1200, seconds),
        "gpu_deadline_epoch": now + seconds,
        "absolute_deadline_epoch": now + seconds + 180,
        "max_total_distinct_attempts": rows["prospective_distinct_computations"],
        "noise_identity_sha256": PI0_NOISE_SHA if model == "pi0" else core.canonical_sha(
            {"kind": "deterministic_no_external_noise", "seed": 42}),
        "metric_reference_sha256": core.file_sha(references) if model == "pi0" else None,
        "resources": {
            "physical_gpu": args.gpu, "gpu_uuid": gpu["uuid"], "gpu_pci": gpu["pci"],
            "exclusive_gpu": True, "outer_watchdog_required": True, "cpu_threads": 8,
            "allocator_bytes": 22_000_000_000, "observed_gpu_bytes_cap": 24_000_000_000,
            "free_gpu_bytes_min": 8_000_000_000, "load_free_gpu_bytes_min": 32_000_000_000,
            "cpu_rss_hwm_cap": 32_000_000_000, "cpu_mem_available_min": 24_000_000_000,
        },
    }
    core.validate_permit(permit, bindings, now=now)
    permit_path = write_json(work / "permit.json", permit)
    launch = {
        "schema": "formal_single_model_watchdog_launch_v1", "model": model,
        "launch_authorized": True, "source_root": str(checkout),
        "parent_output": str(parent), "parent_once_file": str(local / "watchdog.once.json"),
        "watchdog_sha256": files["src/deepvision_vla/diagnostics/formal_single_model_watchdog.py"],
        "natural_exit_grace_seconds": 5, "term_grace_seconds": 15, "kill_grace_seconds": 5,
        "poll_seconds": 1, "phase_json_probe_timeout_seconds": 3,
    }
    for name, path_key, path in (
        ("config", "config_path", config_path),
        ("permit", "permit_path", permit_path),
        ("source_manifest", "source_manifest_path", source_manifest),
        ("python_executable", "python_executable", model_python),
        ("helper_python_executable", "helper_python_executable", head_python),
        ("nvidia_smi_executable", "nvidia_smi_executable", smi),
    ):
        launch[path_key] = str(path)
        launch[name + "_sha256"] = core.file_sha(path)
    launch_path = write_json(work / "launch.json", launch)
    digest = core.file_sha(launch_path)
    _, _, _, proof = preflight(launch_path, digest)
    return {
        "status": "launch_validated_not_executed",
        "model": model, "gpu": args.gpu,
        "launch_plan": str(launch_path), "launch_plan_sha256": digest,
        "requests": proof["requested"], "eligible": proof["eligible"],
        "available_gpu_memory": gpu["free_bytes"], "gpu_processes_observed": len(gpu["processes"]),
        "model_calls": 0, "watchdog_started": False,
    }


def positive_seconds(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Time limit must be positive")
    return number


def main(argv=None):
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--model", required=True, choices=core.REVISIONS)
    for name in ("panel", "inputs", "metadata", "plan", "checkpoint", "asset-receipt",
                 "checkpoint-inventory", "model-python", "output", "node-local"):
        command.add_argument("--" + name, required=True, type=Path)
    command.add_argument("--lerobot", type=Path, help="Pinned LeRobot checkout for Bridge Pi0")
    command.add_argument("--qwen-processor", type=Path, help="Verified local Qwen H16 processor directory")
    command.add_argument("--qwen-statistics", type=Path, help="Verified Qwen H16 action statistics")
    command.add_argument("--gpu", required=True, choices=range(8), type=int)
    command.add_argument("--time-limit", type=positive_seconds, metavar="SECONDS")
    args = command.parse_args(argv)
    try:
        receipt = prepare_launch(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        command.exit(2, f"{command.prog}: {error}\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
