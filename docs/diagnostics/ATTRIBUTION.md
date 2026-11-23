# Recompute the three-model BridgeV2 attribution example

Use [the selected RGB frame](../../examples/bridgev2-attribution/selected/source.jpg), ordinal **1318** of the verified BridgeV2 v55 runtime. This frame was selected for readability from the 27-frame rerun. [The paper's figure](https://arxiv.org/html/2603.15618v1#S3.SS2) shows a separate example.

Set `DV_CODE` to the root of this checkout in each model's environment and run the commands from that root. Use an idle NVIDIA GPU and a different output path for each run. The three model environments are separate. The CPU plotting environment is shared.

## OpenVLA

Use the environment, pinned checkpoint and asset receipt in the [OpenVLA guide](install/OPENVLA_ENVIRONMENT.md).

```bash
export DV_CODE="$PWD"
export PYTHONPATH="$DV_CODE/src"
export IMAGE="$DV_CODE/examples/bridgev2-attribution/selected/source.jpg"
export INSTRUCTION=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["instruction"])' \\
  "$DV_CODE/examples/bridgev2-attribution/selected/source.json")

python -m deepvision_vla.diagnostics attribute openvla \
  --image "$IMAGE" --instruction "$INSTRUCTION" \
  --checkpoint "$DV_WORK/checkpoints/openvla-7b" \
  --asset-receipt "$DV_WORK/receipts/openvla-assets.json" \
  --official-inventory "$DV_CODE/docs/diagnostics/install/openvla-inventory.json" \
  --gpu "$GPU" --output "$DV_WORK/attribution/openvla.json"
```

The native forward generates seven action tokens. The gradient target is their teacher-forced cross-entropy. Each layer verifies the actual visual slots, causal logit offsets and agreement between teacher-forced and native greedy token IDs.

## Official Pi0 base

Use the model and tokenizer from [Official Pi0 base](PI0_BASE.md). Set `PI0_WORK` to the same root passed to `fetch` and `prepare`. Preparing the Bridge state calibration does not require running the masking sweep.

```bash
export PYTHONPATH="$DV_CODE/src"
python -m deepvision_vla.diagnostics attribute pi0-base \
  --root "$PI0_WORK" --ordinal 1318 --gpu "$GPU" \
  --output "$DV_WORK/attribution/pi0-base.json"
```

The probe uses one native denoising step at time 0.5 with fixed noise. Its scalar is the mean squared native flow output. The Bridge state input is the documented EEF/gripper proxy. The runner checks the camera mask, 16×16 visual grid, and that the captured forward matches the unhooked forward. See [Bridge action alignment](BRIDGE_ACTION_ALIGNMENT.md) for the metric scope.

## QwenVLA-OFT H16

Follow the [Qwen H16 environment and launch guide](QWEN_H16_ENVIRONMENT.md) to obtain the public checkpoint and prepare its drop launch in this checkout. Set `QWEN_RUN` to the prepared run directory. This attribution probe uses the qualified ROI for ordinal 1318 and computes a fresh baseline before capturing gradients through the native H16 action head.

```bash
export PYTHONPATH="$DV_CODE/src"
export QWEN_PROBE="$DV_WORK/attribution/qwen-probe-1318"

python -m deepvision_vla.diagnostics attribute qwen-h16 \
  --prepared-launch "$QWEN_RUN/launch.json" --ordinal 1318 --gpu "$GPU" \
  --output "$QWEN_PROBE" --node-local /tmp/deepvision-qwen-attribution-1318

PROBE_SHA=$(sha256sum "$QWEN_PROBE/launch.json" | cut -d' ' -f1)
python "$DV_CODE/src/deepvision_vla/diagnostics/paper_probe_watchdog.py" \
  --launch-plan "$QWEN_PROBE/launch.json" --launch-plan-sha256 "$PROBE_SHA" --execute
```

Use a Linux supervisor Python with `os.pidfd_open` and `signal.pidfd_send_signal` for the last command. The prepared launch still uses the pinned model Python.

## Combine all three outputs on CPU

Use the plotting Python from the [BridgeV2 quickstart](QUICKSTART.md).

```bash
"$DV_REPORT_PYTHON" -m deepvision_vla.diagnostics plot attribution \
  --image "$DV_CODE/examples/bridgev2-attribution/selected/source.jpg" \
  --result "$DV_WORK/attribution/openvla.json" \
  --result "$DV_WORK/attribution/pi0-base.json" \
  --result "$QWEN_PROBE/result/result.json" \
  --output "$DV_WORK/attribution/three-model-1318"
```

The renderer checks image SHA256, instruction, layer positions and visual token order. It writes PNG, PDF and a render receipt. Each model uses its own documented gradient target. Compare strengths within each model using the shared-per-model display scale.
