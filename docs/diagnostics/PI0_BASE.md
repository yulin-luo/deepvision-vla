# Reproduce visual masking with official Pi0 base

Use [Physical Intelligence's Pi0 base in LeRobot format](https://huggingface.co/lerobot/pi0_base/tree/6e245399ad403ccf02b8112b73cfdc2633068f38). We evaluated it on 150 fixed BridgeData V2 episodes across 46 source shards. The experiment uses the published base weights without additional Bridge fine-tuning.

[Download the curve (PDF)](../../examples/bridgev2-pi0-base/pi0-base-bridgev2.pdf) · [Per-episode measurements](../../examples/bridgev2-pi0-base/per-source.json)

## Redraw the published curve on CPU

From the repository root with Python 3.12 and uv:

```bash
uv venv --python 3.12 .venv-report
uv pip install --python .venv-report/bin/python -r requirements/diagnostics-analysis.txt
export PYTHONPATH="$PWD/src"
.venv-report/bin/python -m deepvision_vla.diagnostics.pi0_base_plot \
  --saved examples/bridgev2-pi0-base/per-source.json --output pi0-base-redraw
```

## Run the 150-episode diagnostic on one GPU

Use Linux x86-64 and an NVIDIA driver compatible with PyTorch CUDA 13.0. The tested environment uses Python 3.12, LeRobot 0.7.0, and PyTorch 2.12.1. The checkpoint is approximately 14 GB. The validated run used less than 18 GB of allocated GPU memory.

```bash
set -e
export PYTHONPATH="$PWD/src"
uv venv --python 3.12 .venv-pi0-base
export PI0_PYTHON="$PWD/.venv-pi0-base/bin/python"
uv pip install --python "$PI0_PYTHON" \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
  -r requirements/diagnostics-pi0-base.txt

export DV_WORK="$HOME/deepvision-pi0-base"
"$PI0_PYTHON" -m deepvision_vla.diagnostics.pi0_base fetch --root "$DV_WORK"

hf download lyl010221-pku/deepvision-vla-bridgev2-roi runtime/bridgev2-v55-runtime.tar.gz \
  --repo-type dataset --revision 0f4641b190ee15fe896c6dd207c694231d9d3530 --local-dir "$DV_WORK/hf"
RUNTIME="$DV_WORK/hf/runtime/bridgev2-v55-runtime.tar.gz"
echo "95bf776862b91961b463e5c3f56fb98c3129b074821cbebcd77fa9bdd3baa8fa  $RUNTIME" | sha256sum -c -
mkdir "$DV_WORK/runtime"
tar -xzf "$RUNTIME" -C "$DV_WORK/runtime"

"$PI0_PYTHON" -m deepvision_vla.diagnostics.pi0_base prepare \
  --root "$DV_WORK" --runtime "$DV_WORK/runtime"

GPU=0  # Choose an idle GPU.
GPU_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
CUDA_VISIBLE_DEVICES="$GPU" "$PI0_PYTHON" -m deepvision_vla.diagnostics.pi0_base run \
  --root "$DV_WORK" --gpu-uuid "$GPU_UUID" --count 16
CUDA_VISIBLE_DEVICES="$GPU" "$PI0_PYTHON" -m deepvision_vla.diagnostics.pi0_base run \
  --root "$DV_WORK" --gpu-uuid "$GPU_UUID" --start 16 --count 150

.venv-report/bin/python -m deepvision_vla.diagnostics.pi0_base_plot \
  --run "$DV_WORK" --output pi0-base-rerun
```

The runner verifies the pinned checkpoint and tokenizer. The sample list is deterministic and independent of model output. The runner masks visual token outputs at layers 0, 6, 12, and 16 at 11 retention levels, reusing the same mask ordering and denoising noise within each episode. At full retention it checks that the outputs match the unmasked reference bit for bit.

We feed the available Bridge RGB camera and mask the missing wrist views. Bridge end-effector pose and gripper values serve as a state proxy, standardized on the other 1,350 episodes. We report squared drift of the native normalized 50 × 32 output from the same-input, same-noise unmasked reference. **The curve measures visual masking sensitivity. Bridge joint-state and action correspondence for the base checkpoint remain unverified, so this run does not report action GT-MSE.**

With full masking, the mean layer 0 minus layer 16 drift was 0.2430 with a 95% source-shard bootstrap interval of [0.2176, 0.2681].
