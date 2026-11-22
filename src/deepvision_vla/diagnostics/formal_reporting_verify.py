"""Recompute episode-level report statistics from released metric rows."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .formal_reporting import (
    cluster_bootstrap,
    cluster_sensitivity,
    error_value,
    points,
    registered_layer_contrast,
    registered_semantic_contrasts,
)
from .formal_reporting_plots import load_verified_report, report_artifact
from .formal_statistics import (
    ANALYSIS_SEED,
    BOOTSTRAP_RESAMPLES,
    build_paired_panel,
    summarize_paired_panel,
)


def load_metric_rows(path, descriptor):
    panels = defaultdict(list)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            size += len(line)
            row = json.loads(line)
            panels[(row["model"], row["group"] + "|" + row["metric"])].append(row)
    if size != descriptor["bytes"] or digest.hexdigest() != descriptor["sha256"]:
        raise ValueError("Per-source metrics differ from the saved artifact manifest")
    return panels


def checked_panel(model, panel, rows):
    requested = panel["requested_source_ids"]
    if [row["sample_id"] for row in rows] != requested:
        raise ValueError("Metric ledger does not preserve the complete source order")
    records = {}
    qualified = []
    frozen = points(model, panel["regions"])
    for row in rows:
        source = row["sample_id"]
        if row["experimental_unit"] != "source_episode" or type(row["qualified"]) is not bool:
            raise ValueError(f"Invalid episode record for {source}")
        if row["qualified"]:
            values = row["points"]
            if set(values) != set(frozen):
                raise ValueError(f"Missing or unexpected point for {source}")
            baseline = error_value(row["baseline"])
            deltas = {
                name: value - baseline if baseline is not None and error_value(value) is not None else None
                for name, value in values.items()
            }
            if row["signed_baseline_differences"] != deltas:
                raise ValueError(f"Saved baseline differences disagree for {source}")
            qualified.append(source)
            records[source] = {"baseline": row["baseline"], "points": values}
        elif any(row[field] is not None for field in ("baseline", "points", "signed_baseline_differences")):
            raise ValueError(f"Unexpected metric on an unqualified episode: {source}")
    result = build_paired_panel(records, requested, qualified, frozen)
    summary = summarize_paired_panel(result)
    paired = set(result["paired_source_ids"])
    if any(row["paired"] is not (row["sample_id"] in paired) for row in rows):
        raise ValueError("Saved paired cohort differs from the complete metric cohort")
    if any(panel[name] != value for name, value in summary.items()):
        raise ValueError("Saved coverage, curves, or bootstrap intervals disagree")
    return records, result["paired_source_ids"]


def verify_shard_intervals(panel, records, source_ids, inventory):
    saved = panel["container_sensitivity"]
    if saved["field"] != "original_source.shard":
        raise ValueError("The exported source index supports shard sensitivity only")
    shard_groups = cluster_sensitivity(panel, inventory, saved["field"])
    if shard_groups["status"] != "container_index_ready":
        raise ValueError("Source index cannot reproduce the saved shard partition")
    absolute = [[records[source]["points"][point] for point in panel["points"]]
                for source in source_ids]
    differences = [[value - records[source]["baseline"] for value in values]
                   for source, values in zip(source_ids, absolute)]
    expected = {
        "field": shard_groups["field"],
        "effective_clusters": shard_groups["effective_clusters"],
        "absolute_error": cluster_bootstrap(absolute, shard_groups["cluster_indices"]),
        "signed_baseline_difference": cluster_bootstrap(differences, shard_groups["cluster_indices"]),
    }
    if saved != expected:
        raise ValueError("Saved shard resampling intervals disagree")


def read_shard_index(path, expected_sha256, report):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Source index SHA256 differs")
    index = json.loads(data)
    if index["source_inventory_sha256"] != report["input_seals"]["inventory"]:
        raise ValueError("Source index describes a different report inventory")
    rows = index["sources"]
    if (len(rows) != report["requested_source_identities"]
            or [row["ordinal"] for row in rows] != list(range(1, len(rows) + 1))):
        raise ValueError("Source index does not contain every indexed episode")
    return [{"sample_id": row["sample_id"], "original_source": {"shard": row["shard"]}}
            for row in rows]


def verify_summaries(report, rows_by_panel, inventory=None):
    """Recompute registered curves plus the paired layer and control contrasts."""
    if (report["analysis_seed"] != ANALYSIS_SEED
            or report["bootstrap_resamples"] != BOOTSTRAP_RESAMPLES
            or report["experimental_unit"] != "source_episode"):
        raise ValueError("The report uses a different analysis protocol")
    expected = {(model, key) for model, result in report["models"].items()
                for key in result["panels"]}
    if not expected or set(rows_by_panel) != expected:
        raise ValueError("Per-source metrics and saved report list different panels")

    total_panels = total_shallow_deep = total_semantic = total_shard_panels = 0
    for model, result in report["models"].items():
        shallow_deep = []
        semantic = []
        for key, panel in result["panels"].items():
            records, source_ids = checked_panel(model, panel, rows_by_panel[(model, key)])
            metric = panel["metric"]
            group = panel["group"]
            shallow_deep.append(registered_layer_contrast(
                model, metric, group, panel["regions"], records, source_ids))
            if group == "semantic":
                semantic.extend(registered_semantic_contrasts(model, metric, records, source_ids))
            if inventory is not None:
                if [row["sample_id"] for row in inventory] != panel["requested_source_ids"]:
                    raise ValueError("Source index does not preserve the report's episode order")
                if "field" in panel.get("container_sensitivity", {}):
                    verify_shard_intervals(panel, records, source_ids, inventory)
                    total_shard_panels += 1
            total_panels += 1
        if shallow_deep != result["shallow_deep"] or semantic != result["semantic_contrasts"]:
            raise ValueError(f"Saved paired contrasts disagree for {model}")
        total_shallow_deep += len(shallow_deep)
        total_semantic += len(semantic)
    return {
        "status": "saved_episode_metrics_recomputed",
        "models": list(report["models"]), "panels": total_panels,
        "layer_contrasts": total_shallow_deep, "semantic_contrasts": total_semantic,
        "shard_sensitivity_panels": total_shard_panels,
        "episode_rows": sum(len(rows) for rows in rows_by_panel.values()),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "analysis_seed": ANALYSIS_SEED,
        "synthetic_fixture_only": report["input_seals"].get("synthetic_fixture_only") is True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source-index", type=Path, help="Verify any saved shard-resampling intervals")
    parser.add_argument("--source-index-sha256")
    args = parser.parse_args(argv)
    if (args.source_index is None) != (args.source_index_sha256 is None):
        parser.error("--source-index and --source-index-sha256 must be supplied together")
    path = args.report.resolve(strict=True)
    report, manifest, report_sha = load_verified_report(path)
    ledger = path.parent / "per_source_metrics.jsonl"
    descriptor = report_artifact(manifest, ledger.name)
    rows = load_metric_rows(ledger, descriptor)
    inventory = read_shard_index(args.source_index, args.source_index_sha256, report) if args.source_index else None
    result = verify_summaries(report, rows, inventory)
    print(json.dumps({**result, "report_sha256": report_sha,
                      "per_source_metrics_sha256": descriptor["sha256"]}))


if __name__ == "__main__":
    main()
