"""Check BridgeV2 action references independently of model predictions."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from .bridge import bridge_orig_targets
from .pi0_base import CHECKPOINT_FILES, RUNTIME_FILES, digest, read, write


def targets(states, actions, anchor):
    """Derive the first-step Qwen label and four OpenVLA-convention reference rows."""
    if states.shape != actions.shape or states.ndim != 2 or states.shape[1] != 7:
        raise ValueError("Expected matching raw Bridge state and action arrays [steps, 7]")
    reached = bridge_orig_targets(states, actions)
    if not 1 <= anchor < len(states) - 1:
        raise ValueError("The anchor has no valid same-frame action label")
    bridge = np.asarray([row["action"] for row in reached[anchor - 1:anchor + 3]], dtype=np.float32)
    qwen = bridge[0].copy()
    qwen[6] = actions[anchor, 6]
    return bridge, qwen


def index_rows(rows):
    index = {row["sample_id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate source identifier")
    return index


def audit(runtime, evidence, *, qwen_terminals=None, qwen_receipt=None):
    panel = runtime / "panel"
    inventory = panel / "inventory.json"
    inventory_sha = digest(inventory)
    if inventory_sha != RUNTIME_FILES[inventory.name]:
        raise ValueError("Use the pinned BridgeV2 v55 inventory")
    rows = index_rows(read(inventory))
    refs = read(runtime / "metadata/pi0.references.json")
    if refs["model"] != "pi0" or (refs["native_horizon"], refs["native_dimensions"]) != (4, 7):
        raise ValueError("Expected the archived Bridge-adapted four-step reference")
    pi0 = index_rows(refs["references"])
    if pi0.keys() != rows.keys():
        raise ValueError("The Bridge-adapted reference and v55 panel use different sources")

    stored = None
    evidence_sha = None
    if evidence is not None:
        evidence_sha = digest(evidence)
        with evidence.open(encoding="utf8") as file:
            stored = index_rows([json.loads(line) for line in file if line.strip()])
        if stored.keys() != rows.keys():
            raise ValueError("Qwen evidence and v55 inventory use different sources")

    qwen_gripper_differs = 0
    targets_by_id = {}
    for sample_id, row in rows.items():
        source = row["original_source"]
        raw = (panel / source["arrays_path"]).read_bytes()
        if sha256(raw).hexdigest() != row["arrays_sha256"]:
            raise ValueError(f"Bridge arrays changed for {sample_id}")
        with np.load(BytesIO(raw), allow_pickle=False) as arrays:
            legacy, qwen = targets(arrays["states"], arrays["actions"], row["anchor"])
        ref = pi0[sample_id]
        if (ref["arrays_sha256"] != row["arrays_sha256"] or ref["anchor"] != row["anchor"]
                or ref["status"] != "verified_numeric_reference"
                or ref["converted_frame_indices"] != list(range(row["anchor"] - 1, row["anchor"] + 3))
                or legacy.shape != (4, 7)
                or not np.array_equal(legacy, np.asarray(ref["native_reference_horizon"], dtype=np.float32))):
            raise ValueError(f"Four-step Bridge reference mismatch for {sample_id}")

        if stored is not None:
            q = stored[sample_id]
            expected = {
                "ordinal": row["ordinal"], "source_anchor": row["anchor"],
                "source_image_sha256": row["image_sha256"], "source_arrays_sha256": row["arrays_sha256"],
                "status": "exact_stored_h1",
            }
            if (any(q.get(key) != value for key, value in expected.items())
                    or not np.array_equal(qwen, np.asarray(q["h1"], dtype=np.float32))):
                raise ValueError(f"Stored same-frame Qwen label mismatch for {sample_id}")
        qwen_gripper_differs += int(qwen[-1] != legacy[0, -1])
        targets_by_id[sample_id] = (row["ordinal"], qwen)

    result = {
        "inventory_sha256": inventory_sha,
        "fixed_sources": len(rows),
        "bridge_adapted_pi0_h4_equals_openvla_reference": len(rows),
        "qwen_h1_stored_numeric_matches": None if stored is None else len(rows),
        "qwen_stored_evidence_sha256": evidence_sha,
        "raw_and_openvla_command_gripper_differ": qwen_gripper_differs,
        "historical_paper_binding_verified": False,
    }

    if qwen_terminals is not None:
        result["qwen_h1_replay"] = replay_qwen_h1(qwen_terminals, qwen_receipt, targets_by_id)
    return result


def replay_qwen_h1(terminal_path, merge_receipt, targets_by_id):
    receipt = read(merge_receipt)
    if receipt["model"] != "qwen_h16":
        raise ValueError("Expected a Qwen H16 merge receipt")
    file_hash = sha256()
    scores = {}
    with terminal_path.open("rb") as file:
        for line in file:
            file_hash.update(line)
            terminal = json.loads(line)["terminal"]
            region = terminal["region"]
            if region not in {"baseline", "primary_task_roi"}:
                continue
            sample_id = terminal["sample_id"]
            ordinal, target = targets_by_id[sample_id]
            if terminal["ordinal"] != ordinal or terminal["model"] != "qwen_h16":
                raise ValueError(f"Prediction identity mismatch for {sample_id}")
            prediction = np.asarray(terminal["prediction"]["native"], dtype=np.float64)
            if prediction.shape != (16, 7) or not np.isfinite(prediction).all():
                raise ValueError(f"Expected a finite native H16 prediction for {sample_id}")
            cell = ("baseline", None) if region == "baseline" else (
                terminal["layer"], float(terminal["keep_ratio"])
            )
            group = scores.setdefault(cell, {})
            if sample_id in group:
                raise ValueError(f"Duplicate prediction for {sample_id} at {cell}")
            squared = np.square(prediction[0] - target)
            group[sample_id] = [float(squared.mean()), float(squared[:6].mean()), float(squared[6])]
    if file_hash.hexdigest() != receipt["merged_terminals_sha256"]:
        raise ValueError("The Qwen terminal ledger does not match its merge receipt")

    layers, ratios = (0, 12, 24, 34), tuple(point / 10 for point in range(11))
    expected = {("baseline", None)} | {(layer, keep) for layer in layers for keep in ratios}
    if scores.keys() != expected or any(group.keys() != targets_by_id.keys() for group in scores.values()):
        raise ValueError("The prediction ledger must cover the same v55 sources for all 44 points")
    unmasked = np.mean(list(scores["baseline", None].values()), axis=0)
    groups = {
        str(layer): [float(np.mean([entry[0] for entry in scores[layer, keep].values()]))
                     for keep in ratios]
        for layer in layers
    }
    return {
        "prediction_ledger_sha256": file_hash.hexdigest(),
        "sources": len(targets_by_id),
        "primary_masked_points": len(targets_by_id) * len(layers) * len(ratios),
        "unmasked_native_7d_mse": float(unmasked[0]),
        "unmasked_motion_6d_mse": float(unmasked[1]),
        "unmasked_gripper_mse": float(unmasked[2]),
        "keep_ratios": list(ratios),
        "layer_means": groups,
        "layer0_minus_layer34_full_mask": groups["0"][0] - groups["34"][0],
    }


def base_contract(checkpoint):
    paths = [checkpoint / name for name in ("config.json", "policy_preprocessor.json")]
    for path in paths:
        if digest(path) != CHECKPOINT_FILES[path.name]:
            raise ValueError(f"Expected pinned official Pi0 file: {path.name}")
    config, processor = map(read, paths)
    normalizer, = [step["config"] for step in processor["steps"]
                   if step["registry_name"] == "normalizer_processor"]
    return {
        "state_shape": config["input_features"]["observation.state"]["shape"],
        "action_shape": config["output_features"]["action"]["shape"],
        "horizon": config["chunk_size"],
        "preprocessor_normalizer_features": normalizer["features"],
        "bridge_7d_to_native_32d_mapping_verified": False,
        "gt_mse_available": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--qwen-stored-evidence", type=Path)
    parser.add_argument("--pi0-base-checkpoint", type=Path)
    parser.add_argument("--qwen-terminals", type=Path)
    parser.add_argument("--qwen-merge-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists")
    if (args.qwen_terminals is None) != (args.qwen_merge_receipt is None):
        parser.error("Provide --qwen-terminals and --qwen-merge-receipt together")
    result = audit(args.runtime, args.qwen_stored_evidence,
                   qwen_terminals=args.qwen_terminals, qwen_receipt=args.qwen_merge_receipt)
    if args.pi0_base_checkpoint:
        result["official_pi0_base"] = base_contract(args.pi0_base_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
