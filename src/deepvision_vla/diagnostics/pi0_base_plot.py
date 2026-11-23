"""Render the official Pi0 base diagnostic from saved or freshly computed rows."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .pi0_base import LAYERS, KEEP, MODEL_SHA, REVISION, TOKENIZER_SHA

COLORS = ("#0072B2", "#B74C00", "#008477", "#815399")


def read(path):
    return json.loads(path.read_bytes())


def from_run(root, count):
    plan_file = root / "COHORT150.json"
    plan = read(plan_file)
    assert plan["revision"] == REVISION and plan["model_sha256"] == MODEL_SHA
    plan_sha = hashlib.sha256(plan_file.read_bytes()).hexdigest()
    rows = []
    for source in plan["sources"][:count]:
        row = read(root / "outputs" / f'{source["sample_id"]}.json')
        assert row["sample_id"] == source["sample_id"] and row["plan_sha256"] == plan_sha
        assert row["model_sha256"] == MODEL_SHA and row["tokenizer_sha256"] == TOKENIZER_SHA
        rows.append(row)
    return rows


def summarize(rows):
    ids = [r["sample_id"] for r in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("The inputs must contain distinct BridgeData V2 episodes")
    data = np.asarray([[r["layers"][str(layer)] for layer in LAYERS] for r in rows], dtype=np.float64)
    if (data.shape != (len(rows), 4, 11) or not np.isfinite(data).all() or (data < 0).any()
            or (data[:, :, -1] != 0).any() or not all(r["keep1_bit_exact_same_batch"] for r in rows)):
        raise ValueError("Missing or invalid masking measurements")
    shards = defaultdict(list)
    for index, row in enumerate(rows):
        shards[row["shard"]].append(index)
    groups = [shards[name] for name in sorted(shards)]
    draw = np.random.default_rng(42).integers(len(groups), size=(4000, len(groups)))
    sums = np.stack([data[group].sum(axis=0) for group in groups])
    sizes = np.array([len(group) for group in groups])
    bootstrap = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)[:, None, None]
    lo, hi = np.quantile(bootstrap, [.025, .975], axis=0)
    paired = data[:, 0, 0] - data[:, -1, 0]
    paired_bootstrap = bootstrap[:, 0, 0] - bootstrap[:, -1, 0]
    return {
        "model": "lerobot/pi0_base", "revision": REVISION, "episodes": len(rows), "shards": len(shards),
        "layers": list(LAYERS), "nominal_keep": [k / 10 for k in KEEP],
        "metric": "normalized prediction drift", "mean": data.mean(axis=0).tolist(),
        "ci95": [lo.tolist(), hi.tolist()],
        "full_mask_l0_minus_l16": {
            "mean": float(paired.mean()), "ci95": np.quantile(paired_bootstrap, [.025, .975]).tolist(),
        },
    }


def render(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    with matplotlib.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 10,
        "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False,
    }):
        fig, axis = plt.subplots(figsize=(6.8, 4.0))
        fig.subplots_adjust(left=.14, right=.97, bottom=.215, top=.73)
        x = summary["nominal_keep"]
        lo, hi = summary["ci95"]
        for i, (layer, color) in enumerate(zip(LAYERS, COLORS, strict=True)):
            axis.fill_between(x, lo[i], hi[i], color=color, alpha=.14, linewidth=0)
            axis.plot(x, summary["mean"][i], color=color, marker="o", markersize=3.2,
                      linewidth=1.65, label=f"Layer {layer}")
        axis.set_xlim(-.012, 1.012)
        axis.set_xticks([0, .2, .4, .6, .8, 1])
        axis.set_xlabel("Nominal ROI retention")
        axis.set_ylabel("Normalized prediction drift")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=.16)
        fig.text(.14, .955, "Visual masking in official π₀ base", fontsize=16, weight="bold", va="top")
        fig.text(.14, .85, f"{summary['episodes']} BridgeData V2 episodes", color="#475569")
        fig.legend(*axis.get_legend_handles_labels(), ncol=4, loc="lower left",
                   bbox_to_anchor=(.125, .747), columnspacing=1.3, fontsize=9)
        fig.text(.14, .06, "Shading shows pointwise 95% shard-bootstrap intervals.", fontsize=8.5, color="#475569")
        fig.savefig(output / "pi0-base-bridgev2.pdf", facecolor="white")
        fig.savefig(output / "pi0-base-bridgev2.png", facecolor="white", dpi=240)
        plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument("--saved", type=Path)
    data.add_argument("--run", type=Path)
    p.add_argument("--count", type=int, default=150)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.count <= 150:
        p.error("--count must be between 1 and 150")
    if args.saved:
        saved = read(args.saved)
        if saved["revision"] != REVISION or tuple(saved["layers"]) != LAYERS:
            p.error("Use the saved official Pi0 base curves")
        rows = saved["observations"]
    else:
        rows = from_run(args.run, args.count)
    summary = summarize(rows)
    render(summary, args.output)
    print(json.dumps(summary["full_mask_l0_minus_l16"]))


if __name__ == "__main__":
    main()
