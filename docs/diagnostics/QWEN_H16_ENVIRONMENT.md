# Qwen-only H16 diagnostic environment

Use the public QwenVLA-OFT H16 runtime export in a dedicated Linux Python 3.10 environment. The checkpoint contains a Qwen3-VL-4B backbone and a shared seven-channel action head with 16 action queries.

```bash
python -m venv /path/to/new/qwen-h16-env
source /path/to/new/qwen-h16-env/bin/activate
python -m pip install -r requirements/diagnostics-qwen-h16.txt
python -m pip install -e . --no-deps
```

Install `transformers==4.57.0` for the Python dependencies. Set the model
factory's `source_root` to `model_runtime/` from the DeepVision-VLA checkout.
The loader verifies the vendor hashes in `configs/diagnostics/qwen-h16-vendor-seals.json`. Use the processor and action statistics supplied with the export.

## Obtain the H16 runtime checkpoint

Get the [QwenVLA-OFT H16 runtime export](https://huggingface.co/lyl010221-pku/deepvision-vla-qwenvla-oft-h16) of `checkpoint-5-10000` at its pinned Hub revision. It contains the original
Qwen3-VL-4B backbone and shared action-head weights, the matching processor,
action normalization statistics, and the configuration used for inference.

```bash
export DV_QWEN_MODEL=lyl010221-pku/deepvision-vla-qwenvla-oft-h16
export DV_QWEN_HUB_REVISION=3be6577ad4455ccbc28494c32bfbea291dd9a88e
export DV_QWEN_CHECKPOINT="$DV_WORK/checkpoints/qwenvla-oft-h16"
hf download "$DV_QWEN_MODEL" --revision "$DV_QWEN_HUB_REVISION" \
  --local-dir "$DV_QWEN_CHECKPOINT"

export CHECKPOINT_INVENTORY="$DV_CODE/configs/diagnostics/qwen-h16-runtime-export-inventory.json"
QWEN_INDEX_SHA=$(sha256sum "$CHECKPOINT_INVENTORY" | cut -d' ' -f1)
export ASSET_RECEIPT="$DV_WORK/qwen-h16-asset-receipt.json"
python -m deepvision_vla.diagnostics.asset_receipts \
  --inventory "$CHECKPOINT_INVENTORY" --inventory-sha256 "$QWEN_INDEX_SHA" \
  --checkpoint "$DV_QWEN_CHECKPOINT" --output "$ASSET_RECEIPT"
```

Follow the [common launch command](PUBLIC_CORE.md#run-a-prepared-plan) with
`--model qwen_h16` and `--checkpoint "$DV_QWEN_CHECKPOINT"`. Use
`--qwen-processor "$DV_QWEN_CHECKPOINT/processor"` and
`--qwen-statistics "$DV_QWEN_CHECKPOINT/action_statistics.json"` and pass
the new component receipt. The inventory records the source checkpoint
revision, while the Hub revision identifies the downloadable export.

## Recompute a Grad-CAM example with the published Qwen checkpoint

Follow the [three-model attribution guide](ATTRIBUTION.md#qwenvla-oft-h16) for the Qwen launch and CPU redraw. See [Bridge action alignment](BRIDGE_ACTION_ALIGNMENT.md) for the action references and Qwen H1 replay.
