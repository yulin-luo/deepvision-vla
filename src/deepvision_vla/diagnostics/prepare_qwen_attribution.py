"""Prepare a native attribution probe from a published Qwen H16 drop launch."""

import argparse
from pathlib import Path
import json
import time

from .adapters.qwen_h16_contracts import RUNTIME_EXPORT_FORMAT
from .formal_plan import TASK_ROI_BOX_ROLES, _annotation
from .formal_single_model_watchdog import local_filesystem
from .paper_probe import model_metadata, validate_factory_metadata_seals
from .paper_probe_watchdog import preflight, read_bound
from .prepare_launch import read_json, write_json
from .provenance import sha256
from .public_surface import release_paths
from .resources import device_identity
from .source_manifest import build_source_manifest


def prepare(args):
    launch = read_json(args.prepared_launch)
    if (launch.get("schema") != "formal_single_model_watchdog_launch_v1"
            or launch.get("model") != "qwen_h16"):
        raise ValueError("Use a prepared Qwen H16 BridgeV2 drop launch")
    checkout = Path(__file__).resolve().parents[3]
    if Path(launch["source_root"]).resolve(strict=True) != checkout:
        raise ValueError("Prepare the drop and attribution launches in the same checkout")
    drop = read_bound(Path(launch["config_path"]), launch["config_sha256"])
    prior = read_bound(Path(launch["permit_path"]), launch["permit_sha256"])
    bindings = prior["bindings"]
    if bindings["config_sha256"] != launch["config_sha256"] or drop.get("model") != "qwen_h16":
        raise ValueError("The original permit did not bind this prepared drop configuration")

    inventory_path = Path(drop["inventory"])
    if sha256(inventory_path) != bindings["inventory_sha256"]:
        raise ValueError("Inventory has changed")
    inventory = read_json(inventory_path)
    annotations_path = Path(drop["annotations"])
    if sha256(annotations_path) != bindings["annotations_sha256"]:
        raise ValueError("Annotations have changed")
    if not 1 <= args.ordinal <= len(inventory):
        raise ValueError("Ordinal falls outside the fixed Bridge panel")
    row = inventory[args.ordinal - 1]
    if row.get("ordinal") != args.ordinal:
        raise ValueError("Inventory ordinal order differs")
    annotation, = [item for item in read_json(annotations_path) if item["sample_id"] == row["sample_id"]]
    manifest = read_bound(Path(drop["plan_manifest"]), bindings["plan_manifest_sha256"])
    reasons, roles, _ = _annotation(row, annotation, manifest["bank_revision"])
    if reasons:
        raise ValueError(f"The selected sample is not qualified for the reviewed task ROI: {', '.join(reasons)}")
    original = row["original_source"]
    sample = {key: row[key] for key in (
        "ordinal", "sample_id", "image_sha256", "arrays_sha256", "image_size", "instruction", "anchor",
    )}
    sample.update(image=original["image_path"], arrays=original["arrays_path"],
                  roi_boxes=[r["xyxy_px"] for r in roles if r["role"] in TASK_ROI_BOX_ROLES])

    asset_path = Path(drop["asset_receipt"])
    asset = read_bound(asset_path, bindings["asset_receipt_sha256"])
    if asset.get("receipt_format") != RUNTIME_EXPORT_FORMAT:
        raise ValueError("Use the published Qwen H16 runtime export with this command")
    factory = drop["factory_kwargs"]
    checkpoint = Path(factory["checkpoint_root"])
    contracts = checkout / "configs/diagnostics"
    seals = [asset_path, checkpoint / "training_config.json",
             checkpoint / "portable/qwen_backbone/manifest.json",
             checkpoint / "portable/shared_action_head/manifest.json",
             contracts / "qwen-h16-runtime-export-inventory.json", contracts / "qwen-h16-vendor-seals.json"]

    out = args.output.resolve()
    node_local = args.node_local.resolve()
    if out.exists() or node_local.exists() or out == node_local or out in node_local.parents or node_local in out.parents:
        raise ValueError("Choose separate, new probe and node-local output directories")
    local_filesystem(node_local)
    out.mkdir(parents=True, exist_ok=False)
    source_manifest_path = write_json(out / "source_manifest.json",
        build_source_manifest(checkout, release_paths(checkout / "src")))
    sample_path = write_json(out / "sample.json", sample)
    config_path = write_json(out / "config.json", {
        "schema": "deepvision_paper_probe_config_v1", "model": "qwen_h16",
        "package_source_root": str(checkout),
        "source_manifest": str(source_manifest_path), "source_manifest_sha256": sha256(source_manifest_path),
        "sample": str(sample_path), "sample_sha256": sha256(sample_path),
        "data_root": drop["data_root"], "factory": factory,
        "asset_receipt": str(asset_path), "asset_receipt_sha256": bindings["asset_receipt_sha256"],
        "factory_metadata_seals": [{"path": str(path), "sha256": sha256(path)} for path in seals],
        "output": str(out / "result"), "once": str(out / "worker.once.json"),
    })
    config = read_json(config_path)
    validate_factory_metadata_seals(config)
    model_metadata(config, sample)
    gpu = device_identity(args.gpu)
    now, mono = time.time(), time.monotonic()
    permit = read_json(contracts / "paper-probe-permit.example.json")
    permit.update(prepared_not_authorized=False, model="qwen_h16", config_sha256=sha256(config_path),
                  source_manifest_sha256=config["source_manifest_sha256"], physical_gpu=args.gpu,
                  expected_gpu_uuid=gpu["uuid"], expected_gpu_pci=gpu["pci"],
                  budget_origin_epoch=now, budget_origin_monotonic=mono,
                  deadline_epoch=now+1200, deadline_monotonic=mono+1200)
    permit_path=write_json(out / "permit-template.json", permit)
    launch_path=write_json(out / "launch.json", {
        "schema":"deepvision_paper_probe_launch_v1", "launch_authorized":True,
        "config":str(config_path), "config_sha256":sha256(config_path),
        "permit_template":str(permit_path), "permit_template_sha256":sha256(permit_path),
        "model_python":launch["python_executable"], "model_python_sha256":launch["python_executable_sha256"],
        "nvidia_smi":launch["nvidia_smi_executable"], "nvidia_smi_sha256":launch["nvidia_smi_executable_sha256"],
        "watchdog_sha256":sha256(checkout/"src/deepvision_vla/diagnostics/paper_probe_watchdog.py"),
        "parent_output":str(node_local),
    })
    digest=sha256(launch_path)
    preflight(launch_path,digest)
    return {"status":"qwen_attribution_validated_not_executed","model":"qwen_h16",
            "ordinal":args.ordinal,"launch_plan":str(launch_path),"launch_plan_sha256":digest,"model_calls":0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-launch",type=Path,required=True)
    parser.add_argument("--ordinal",type=int,required=True)
    parser.add_argument("--gpu",type=int,choices=range(8),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--node-local",type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args(argv)),indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
