# Diagnostic source layout and use

The public diagnostic core separates model-native prediction, input geometry,
attribution targets and finite execution controls. It uses the actual model
factories in independent environments; importing the core does not load Torch,
weights or CUDA. Existing RLBench inference and evaluation entry points are
unchanged.

The diagnostic commands share an entry point. Use `--help` after each command
to see its required configuration and input files.

```bash
python -m deepvision_vla.diagnostics prepare fetch --help
python -m deepvision_vla.diagnostics prepare data --help
python -m deepvision_vla.diagnostics prepare plan --help
python -m deepvision_vla.diagnostics prepare launch --help
python -m deepvision_vla.diagnostics run --help
python -m deepvision_vla.diagnostics plot aggregate --help
python -m deepvision_vla.diagnostics plot redraw --help
python -m deepvision_vla.diagnostics plot attribution --help
```

`prepare fetch` downloads the exact shards required by the release index.
`prepare data` rebuilds the fixed Bridge panel from verified local shards.
`prepare plan` builds a request inventory without running a model.
For an engineering pilot, `prepare plan --include-interaction-in-primary
--pilot-source-ordinals FILE` takes a JSON array of 16 previously reviewed
ordinals in the original panel order. The plan keeps the 1500-source identity
but emits requests only for those 16 inputs. Each must pass the primary ROI
and processor-geometry checks. The pilot tests full, half, and zero retention
at all four layers. The population run uses the full eleven-ratio grid.
`run` uses the Linux watchdog and validates the supplied launch plan by
default. With `--execute`, it supervises one configured model run using the
sealed source, runtime, assets and execution permit. Run the parent with a
Python build that provides Linux pidfds, such as the verified Python 3.12
environment. The model runs in its own configured Python environment.

`plot aggregate` generates an analysis report from completed runs.
`plot redraw` reads a saved report and its artifact manifest and recreates the
plots without rebuilding model inputs or loading a policy.

`plot attribution` redraws the four native layer maps from a completed
model attribution result and its original JPEG. Pass one `--result` per model.
All results must use the same sample, image and instruction.

```bash
python -m deepvision_vla.diagnostics plot attribution \
  --image source.jpg \
  --result openvla/result.json --result pi0/result.json --result qwen/result.json \
  --output attribution
```

The overlay uses each processor's recorded crop and token grid. The default
`--scale model` shows all four layers of each model on their shared positive
maximum. Add `--scale layer` and use a new output directory to see the spatial
pattern on each layer's own min–max scale. Flat layers are labeled and rendered
in grayscale in that view. The result JSON retains the raw signed and positive
values. The render receipt records the scale, original value ranges and file
checksums.

## Run a prepared plan

Run `prepare launch` in the Python 3.12 Linux supervisor environment from an
editable checkout. Install the model in its
own environment and create the checkpoint receipt with the model's
installation guide. Use the v55 input bundle, the native runtime metadata
bundle, and a plan built from those inputs. Set `LOCAL_RUN` to a new
node-local directory and `RUN` to a separate new directory for the run
configuration and model results.

```bash
python -m deepvision_vla.diagnostics prepare launch \
  --model openvla --gpu "$GPU" \
  --panel "$PANEL" --inputs "$INPUTS" --metadata "$NATIVE_METADATA" --plan "$PLAN" \
  --model-python "$MODEL_PYTHON" --checkpoint "$CHECKPOINT" \
  --checkpoint-inventory "$CHECKPOINT_INVENTORY" --asset-receipt "$ASSET_RECEIPT" \
  --output "$RUN" --node-local "$LOCAL_RUN"

LAUNCH_SHA=$(sha256sum "$RUN/launch.json" | cut -d' ' -f1)
python -m deepvision_vla.diagnostics run \
  --launch-plan "$RUN/launch.json" --launch-plan-sha256 "$LAUNCH_SHA" --execute
```

For π₀, follow the [official Pi0 base guide](PI0_BASE.md). For Qwen H16, select
`--model qwen_h16` and supply `--qwen-processor "$QWEN_PROCESSOR"` and
`--qwen-statistics "$QWEN_STATISTICS"`. The [Qwen environment guide](QWEN_H16_ENVIRONMENT.md)
describes the controlled checkpoint and processor assets. The preparation
command checks the launch and creates its local configuration without
importing the model. Execute it on the selected free GPU while the local
run window is current. For an engineering pilot, set `PLAN` to the plan
with the fixed 16-source selection. To repeat a run, use new output
directories so previous predictions and failures remain available.

## BridgeData V2 inputs

The panel uses the 256×256 RLDS/TFDS `bridge_dataset/1.0.0` training split.
`prepare fetch` reads the release index and downloads the 47 required shards
from the Berkeley archive. Use `--source hf` to retrieve the same checksum-pinned
bytes from the published Hugging Face mirror. The command verifies size and
SHA256, including for files already in the destination directory.

```bash
python -m deepvision_vla.diagnostics prepare fetch \
  --source-index source_index.json --source-index-sha256 "$INDEX_SHA256" \
  --shard-checksums shard_checksums.json --shard-checksums-sha256 "$SHARDS_SHA256" \
  --shard-dir /path/to/bridge-shards
```

Use the same index and shard directory to rebuild the full runtime input:

```bash
python -m deepvision_vla.diagnostics prepare data \
  --source-index source_index.json --source-index-sha256 "$INDEX_SHA256" \
  --shard-checksums shard_checksums.json --shard-checksums-sha256 "$SHARDS_SHA256" \
  --shard-dir /path/to/bridge-shards --output bridge-panel --runtime-inventory
```

For the fixed 1,500-source panel, `--runtime-inventory` generates
`inventory.json`, `manifest.jsonl`, `files.json` and `transfer_receipt.json`
from the archived records and verified payloads. Use
`bridge-panel/inventory.json` with `prepare plan --inventory` and the
`bridge-panel` directory as the runtime `data_root`. The index pins the exact
inventory bytes, preserving existing sample, geometry and annotation
bindings. The release index stores hashes of upstream episode paths. Any
path metadata in the local inventory is read from the original public records.

The source frames and original trajectory data are from BridgeData V2 by Walke
et al. (CoRL 2023), distributed under CC BY 4.0. The DeepVision-VLA panel records
the selected episodes and anchors, with derived ROI labels and contribution-map
overlays. [BridgeData V2 project and attribution](https://rail-berkeley.github.io/bridgedata/)
· [Official TFDS archive](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/)
· [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Prediction and masking

`predictors.ModelOnce` owns one strict CPU model load, actual processor geometry,
the native full action chunk and counters. It delegates to the official OpenVLA
runtime, the pinned third-party Bridge Pi0 policy, or the Qwen-only H16 adapter.
Horizon is respectively 1, 4 and 16, with seven physical/stored channels per
step. No chunk is padded or converted to H1. Pi noise is fixed as in the native
contract; Qwen retains its sixteen trained action queries. Masks and paired
controls are generated by the existing geometry/sampling/control modules.

`formal_plan`, `formal_execution_core` and `formal_diagnostic_executor` preserve
requested/blocked/attempted/unknown/failed/missing records and exact computation
identities. A data-only plan does not grant execution. A source relocation also
does not authorize reuse of historical predictions: their native/source/runtime
equivalence needs a separate independent binding.

The report uses official Bridge-orig H1×7 action MSE for OpenVLA, source-bound
H4×7 dataset-reference MSE for Bridge Pi0, and H16×7 drift relative to the
unmasked prediction for QwenVLA-OFT. Qwen drift is reported in native and
normalized action space. The archived Qwen dropout script compares a single
decoded action to a stored seven-channel label. The exact label producer and
its frame binding remain unverified for the current Qwen checkpoint and panel.

## Grad-CAM

Follow the [three-model attribution guide](ATTRIBUTION.md) to recompute the maps from the released checkpoints. Use the [attribution example](../../examples/bridgev2-attribution/README.md) to redraw the saved maps on CPU.

See [Bridge action alignment](BRIDGE_ACTION_ALIGNMENT.md) for the model-specific action references and Qwen first-step replay.
