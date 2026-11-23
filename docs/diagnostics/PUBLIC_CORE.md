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
`paper_probe` result and its original JPEG. Pass one `--result` per model.
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

## Parameterized paper-layer attribution

`paper_probe` accepts a sealed source descriptor instead of a built-in book
image. The descriptor contains sample ID, exact instruction, JPEG/NPZ paths and
SHA256, image size, anchor, and source-pixel ROI boxes. It does not translate,
normalize or repair instructions. Use a new output and once path for every
independently authorized scope; a failed or unknown attempt is retained.

The native operators differ by model:

- OpenVLA observes actual greedy generation of seven action tokens, then probes
  decoder OUTPUT layers 0/10/20/30 with native teacher-forced token CE. The
  explicit seven-token CE is only a machine-precision consistency check.
- Bridge Pi0 probes logical OUTPUT layers 0/6/12/16 at their source-proven next
  layernorm-input alias. The scalar is one spatial flow-velocity step at fixed
  noise and time, not final ten-step action correctness.
- Qwen H16 probes OUTPUT layers 0/12/24/34 using the full normalized 16×7 action
  head scalar. It requires a same-source, same-model, full native baseline and
  an independently verified baseline/input/source binding. An old canary's
  identity is never silently generalized to a different image.

All three preserve native parameter flags, frozen parameter gradients, hooks,
actual layouts and forward/backward counters. These attribution scalars are not
interchangeable GT errors. Qwen has no accepted physical Bridge GT conversion.
Pi source-bound dataset references remain distinct from global physical GT.

Prepare config/permit from the examples under `configs/diagnostics`; their null
fields deliberately cannot execute. Each factory metadata file needs an exact
seal. Build a complete checkout source manifest using `source_manifest`; the
probe checks its public dependency coverage, not just one entry file.

```bash
python -m deepvision_vla.diagnostics.public_surface --src ./src
python -m deepvision_vla.diagnostics.public_surface \
  --src ./src --source-manifest-to /path/to/new/source-manifest.json
python -m deepvision_vla.diagnostics.paper_probe \
  --config /path/to/probe.json --config-sha256 "$CONFIG_SHA256" \
  --permit /path/to/permit.json --permit-sha256 "$PERMIT_SHA256"
```

The default command validates source/input/metadata only. `--execute` additionally
requires an exclusive physical UUID/PCI, original dual-clock deadline (at most
1200 seconds), a real live external parent identity/command/source proof, and
the existing memory/CPU/lease limits. The CPU factory loads before the GPU UUID
lease, with resource checks before and after loading.
An unchanged formal-executor guard hashes every actual selected weight payload
against its sealed receipt before CPU loading or UUID acquisition. A saved
receipt, matching byte size or valid tensor header alone cannot satisfy this
guard; same-size payload corruption fails before any model load.
The formal curve runner and the attribution probe use separate supervisors.
Fill the attribution launch-plan example with sealed paths and leave the three
parent fields in the permit template empty. Set the original dual-clock budget
on the target host. Use a new node-local directory for the parent, separate
from the probe output and once file.

```bash
export PYTHONPATH=/path/to/editable/deepvision-vla/src
python /path/to/editable/deepvision-vla/src/deepvision_vla/diagnostics/paper_probe_watchdog.py \
  --launch-plan /path/to/probe-launch.json --launch-plan-sha256 "$LAUNCH_SHA256"
```

This command checks the sealed launch metadata and exits before spawning a
worker. A launch with an authorized plan and `--execute` binds the permit to the
live Linux supervisor. It records the worker log, raw probe receipt, and the
supervisor's independent process and GPU release check.

## Source overlay for a reviewed release

Private pilots, canaries, research trace tools and cache-export operations remain
in the private development history. `public_surface` checks all direct/local
imports, named dynamic import calls and the explicit source allowlist. It can
prepare a source-only overlay into a new directory:

```bash
python -m deepvision_vla.diagnostics.public_surface \
  --src ./src --stage-to /path/to/new/diagnostics-overlay
```

This is an overlay for a later reviewed merge into the existing codebase, not a
complete replacement for its inference package and not a publish command. Its
manifest does not assert runtime acceptance or historical cache equivalence.
Approved annotation/source indexes and the separately reviewed offline Bridge
reconstructor are additional release inputs. Sanitizing an annotation release
creates new hashes and requires its own review; old qualification flags and
hashes cannot be relabelled.
