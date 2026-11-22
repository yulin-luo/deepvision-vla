# Qwen-only H16 diagnostic environment

The diagnostic uses a recovered Qwen3-VL-4B Qwen-only pretraining checkpoint
with 16 per-step shared queries and a shared seven-channel head. The recovered
weights have yet to be linked to the historical paper run.

Use a dedicated Linux Python 3.10 environment. A fresh Python 3.10.20 install
passed dependency and import checks with the pinned versions in
`requirements/diagnostics-qwen-h16.txt`. The pinned backbone and shared action
head also passed strict CPU loading with the matching processor and
normalization assets. A single-GPU run in the fresh environment completed
208 new predictions over the fixed 16-source pilot. Native and normalized
outputs matched the earlier pilot bit for bit.

```bash
python -m venv /path/to/new/qwen-h16-env
source /path/to/new/qwen-h16-env/bin/activate
python -m pip install -r requirements/diagnostics-qwen-h16.txt
python -m pip install -e . --no-deps
```

Install `transformers==4.57.0` for the Python dependencies. Set the model
factory's `source_root` to `model_runtime/` from the DeepVision-VLA checkout.
The public runtime at commit `6fdc5b6125ba265c3f2714fcf721e6d20dd8c274` contains
the matching Qwen modeling, processor, and shared-head files. A GPU replay
with the controlled checkpoint reproduced all four saved layer contribution
maps using this runtime. The loader checks the version and file hashes in
`configs/diagnostics/qwen-h16-vendor-seals.json`. Preserve the repository's
[third-party notices](../../THIRD_PARTY_NOTICES.md).

The probe also requires configured vendor hashes to equal the qualified CPU
receipt's source map, and configured processor hashes to equal the separately
sealed processor reference manifest. A different self-consistent configuration
cannot borrow an earlier CPU qualification. The native loader still verifies
the actual source/processor file bytes before using them.

The real factory additionally needs:

- The actual Qwen-only H16 backbone and shared action-head components, each with
  its producer manifest and exact weights SHA; no random/foundation substitute.
- The matching processor/tokenizer/config assets and their complete file seals.
- Original112-channel row-major H16 statistics, preserving continuous quantiles
  and gripper normalization mask. Native decoding thresholds gripper at0.5 and
  uses unclipped q01/q99 for continuous channels; it is not a Bridge frame map.
- The saved training contract: `per_step_shared_v1`, 16 action queries, 7 channels,
  single image, no robot-state input, no DINO, and explicit224 pre-resize. The
  actual processor may upscale internally; visual geometry is observed rather
  than assumed from224 pixels.


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

## Run with the original controlled components


Place the backbone, shared head and their component manifests under the
checkpoint directory. Use the pinned local inventory and its SHA256 in the
[asset verification command](ASSETS.md). Keep the corresponding processor
directory and normalization statistics as separate inputs.

Follow the [common run guide](PUBLIC_CORE.md#run-a-prepared-plan). Select
`--model qwen_h16` and point `--model-python` to the Qwen environment.
Set `--qwen-processor` to the processor directory and `--qwen-statistics` to
the statistics file. Use the native metadata bundle that includes
`qwen_h16.codec.json` and `qwen_h16.runtime.json`.

Native inference requires access to the recovered components and the matching
vendor code. For BridgeData V2, the diagnostic reports the change from the
unmasked prediction in native and normalized action spaces. Reproducing the
paper's single-step ground-truth MSE requires the historical seven-channel
labels and their source-frame binding.
