# Reproduce the BridgeData V2 diagnostics

Use the Hugging Face dataset for the 1,500 annotated frames, runtime panel
and archived per-episode results. For π₀, start with the [official base checkpoint
and 150-episode reproduction](PI0_BASE.md).

## Get the data

Use the [BridgeV2 ROI dataset](https://huggingface.co/datasets/lyl010221-pku/deepvision-vla-bridgev2-roi) at its pinned v55 revision.
From this checkout on Linux, with the Hugging Face CLI, Python 3.12 and uv:

```bash
set -e
export DV_CODE="$PWD"
export DV_WORK="$HOME/deepvision-bridgev2-v55"
export DV_DATASET=lyl010221-pku/deepvision-vla-bridgev2-roi
export DV_DATASET_REVISION=0f4641b190ee15fe896c6dd207c694231d9d3530
export DV_HF_FILES="$DV_WORK/hf"
mkdir -p "$DV_HF_FILES"
hf download "$DV_DATASET" refs/source_index.json results/bridgev2-v55-results.tar.gz \
  --repo-type dataset --revision "$DV_DATASET_REVISION" --local-dir "$DV_HF_FILES"
```

The dataset viewer reads `data/evaluation.parquet`. Download that file
separately when you only need the images and ROI labels.

## Verify and redraw the archived 1,500-episode results on CPU

```bash
set -e
mkdir -p "$DV_WORK/saved-results"
RESULTS="$DV_HF_FILES/results/bridgev2-v55-results.tar.gz"
echo "8c9f1e66181bf5d00d954c624a9f1efab16af03a3913ee066c97b0c6ab581507  $RESULTS" | sha256sum -c -
tar -xzf "$RESULTS" -C "$DV_WORK/saved-results"

uv venv --python 3.12 "$DV_WORK/report-env"
export DV_REPORT_PYTHON="$DV_WORK/report-env/bin/python"
uv pip install --python "$DV_REPORT_PYTHON" -r "$DV_CODE/requirements/diagnostics-analysis.txt"
uv pip install --python "$DV_REPORT_PYTHON" --no-deps -e "$DV_CODE"

INDEX="$DV_HF_FILES/refs/source_index.json"
INDEX_SHA=6d6224cafbeb9485fbdc21c6b83afab35d0495b4db76b9533b93ee1ecf5b9417
REPORT="$DV_WORK/saved-results/report.json"
"$DV_REPORT_PYTHON" -m deepvision_vla.diagnostics plot verify \
  --report "$REPORT" --source-index "$INDEX" --source-index-sha256 "$INDEX_SHA"
"$DV_REPORT_PYTHON" -m deepvision_vla.diagnostics plot redraw \
  --report "$REPORT" --overview --output "$DV_WORK/overview"
```

The verifier recomputes 12,000 source-metric rows across eight panels.
Open `overview/bridge_primary_overview.pdf` under `DV_WORK` for the three
archived curves. Omit `--overview` and choose a new output directory for
the individual plots. The [attribution example](../../examples/bridgev2-attribution/README.md)
redraws the saved three-model Grad-CAM maps from their raw token values.

## Prepare the runtime panel for OpenVLA and QwenVLA-OFT

The runtime input contains the exact 1,500 images and action/state arrays,
ROI and geometry inputs, and model metadata.

```bash
hf download "$DV_DATASET" runtime/bridgev2-v55-runtime.tar.gz \
  --repo-type dataset --revision "$DV_DATASET_REVISION" --local-dir "$DV_HF_FILES"
```

```bash
set -e
RUNTIME="$DV_HF_FILES/runtime/bridgev2-v55-runtime.tar.gz"
echo "95bf776862b91961b463e5c3f56fb98c3129b074821cbebcd77fa9bdd3baa8fa  $RUNTIME" | sha256sum -c -
mkdir "$DV_WORK/runtime"
tar -xzf "$RUNTIME" -C "$DV_WORK/runtime"
(cd "$DV_WORK/runtime" && sha256sum -c SHA256SUMS)
export PANEL="$DV_WORK/runtime/panel"
export INPUTS="$DV_WORK/runtime/inputs"
export NATIVE_METADATA="$DV_WORK/runtime/metadata"

INVENTORY_SHA=$(sha256sum "$PANEL/inventory.json" | cut -d' ' -f1)
ORDER_SHA=$("$DV_REPORT_PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["source_id_order_sha256"])' "$INPUTS/PROVENANCE.json")
export PLAN="$DV_WORK/pilot-plan"
"$DV_REPORT_PYTHON" -m deepvision_vla.diagnostics prepare plan \
  --inventory "$PANEL/inventory.json" --annotations "$INPUTS/annotations.json" \
  --geometries "$INPUTS/geometries.jsonl" --profiles "$INPUTS/profiles.json" \
  --expected-inventory-sha256 "$INVENTORY_SHA" --expected-id-order-sha256 "$ORDER_SHA" \
  --inventory-revision bridge1500-v55 --bank-revision bridge1500-ai-reviewed-role-candidates-v55 \
  --include-interaction-in-primary \
  --pilot-source-ordinals "$DV_CODE/configs/diagnostics/bridgev2-pilot-ordinals.json" \
  --output "$PLAN"
```

The pilot plans 208 primary requests per model for 16 verified images. Use the
[Pi0 base guide](PI0_BASE.md) for π₀.
Use the full-panel command in `inputs/README.md` for all 1,500 images
and the semantic controls. The same guide documents reconstruction
from the 47 original BridgeData shards.

## Get a checkpoint and run inference

- [OpenVLA](install/OPENVLA_ENVIRONMENT.md) uses the original public revision.
- [Official Pi0 base](PI0_BASE.md) has its own inference and plotting commands.
- [QwenVLA-OFT H16](QWEN_H16_ENVIRONMENT.md) uses the exact backbone and action head for this reproduction, along with their processor and normalization statistics.

For OpenVLA or QwenVLA-OFT, obtain the asset receipt and follow [`prepare launch` and `run`](PUBLIC_CORE.md#run-a-prepared-plan).
Use [formal reporting](FORMAL_REPORTING.md) to aggregate completed runs.
OpenVLA reports action MSE. Qwen H16 and official Pi0 base report prediction drift.
