# Pi0 Bridge diagnostic environment

This adapter uses the **third-party Bridge-adapted Pi0 checkpoint** published by
Haoming Song. Set the paths in the [installation overview](README.md).

Use a dedicated **Linux Python 3.10** environment. The diagnostic run environment
used Python 3.10.20, PyTorch 2.12.1+cu130, torchvision 0.27.1+cu130 and
Transformers 4.48.1.

## Install

```bash
python3.10 -m venv "$DV_WORK/pi0-env"
source "$DV_WORK/pi0-env/bin/activate"
python -m pip install -r "$DV_INSTALL/requirements-diagnostics-pi0.txt"
python -m pip install -e "$DV_CODE" --no-deps
python -m pip check
```

For GPU runs, the CUDA 13.0 wheels require a compatible NVIDIA driver. The
pinned LeRobot source imports `datasets`, `imageio`, `jsonlines`, and `pytest`
during initialization.

## Obtain the historical LeRobot implementation

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/huggingface/lerobot.git "$DV_WORK/lerobot-pi0"
git -C "$DV_WORK/lerobot-pi0" sparse-checkout init --cone
git -C "$DV_WORK/lerobot-pi0" sparse-checkout set lerobot
git -C "$DV_WORK/lerobot-pi0" checkout --detach \
  5d24ce31606eb3e48a0f644530f928c5f119a724
export PYTHONPATH="$DV_WORK/lerobot-pi0:$DV_CODE/src"
```

Keep the complete `lerobot` package at that commit, including its original
Pi0 helper and conversion script. The offline diagnostic imports from the
checkout directly.

[Pinned LeRobot source and Apache-2.0 license](https://github.com/huggingface/lerobot/tree/5d24ce31606eb3e48a0f644530f928c5f119a724).

## Obtain the Bridge model

```bash
hf download HaomingSong/lerobot-pi0-bridge \
  --revision 93c2840a2c68c337da0d472a2f52d1fd8d62132a \
  --local-dir "$DV_WORK/checkpoints/pi0-bridge"
```

The complete snapshot is about 6.52 GB. It includes the author's model and
config source, tokenizer, weights and saved normalization buffers.

```bash
python -m deepvision_vla.diagnostics.asset_receipts \
  --inventory "$DV_INSTALL/pi0-bridge-inventory.json" \
  --inventory-sha256 a5127810e0db99cc48469b28493216f64f65c8396a768caafafa933c51fe96a7 \
  --checkpoint "$DV_WORK/checkpoints/pi0-bridge" \
  --output "$DV_WORK/receipts/pi0-assets.json"
```

The successful receipt reports `complete` and 11 verified files. Use a fresh
receipt filename when repeating the check.

The author's model card declares MIT and lists `google/paligemma-3b-pt-224`
as its base model. Retain LeRobot's Apache-2.0 notices and follow the upstream
model and dataset terms. [Pinned checkpoint](https://huggingface.co/HaomingSong/lerobot-pi0-bridge/tree/93c2840a2c68c337da0d472a2f52d1fd8d62132a) · [PaliGemma model terms](https://huggingface.co/google/paligemma-3b-pt-224) · [BridgeData licensing](https://rail-berkeley.github.io/bridgedata/)

## Narrow compatibility with the released artifact

The adapter uses the author's trained weights, forward methods, and pinned
upstream helper. It applies three compatibility checks.

1. The checkpoint's saved `device=cpu` and `use_amp=false` are retained as
   runtime metadata. The genuine policy configuration is parsed and must
   round-trip exactly. The parser rejects unknown fields.
2. A constructor-only wrapper repairs the historical helper's broken explicit
   subconfiguration branches. It uses genuine Transformers configuration
   classes and restores the upstream constructor afterwards.
3. The artifact omits the expert's unused vocabulary projection. The adapter
   removes that detached, unused module before strict loading.
   The complete remaining state is checked against the safetensors header.

The adapter pins these operations to the source and artifact checksums.

## Inputs and prediction

The Bridge state input is `[x,y,z,roll,pitch,yaw,0,gripper]`. The seventh
slot is the documented pad. The eighth slot keeps the measured gripper value,
including finite sensor overshoot. The model returns a complete 4×7 action
chunk. Comparisons use the same native `[1,4,32]` noise tensor and denoising
settings.

Image preprocessing preserves the native resize-with-padding geometry. The
selected prefix layers are 0, 6, 12 and 16. This implementation uses a manual
layer loop, and the hooks follow its residual and normalization path.

## Import check

Run the import check with the environment active and `PYTHONPATH` set as above.

```bash
CUDA_VISIBLE_DEVICES='' python - <<'PY'
import transformers
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.pi0 import paligemma_with_expert
from deepvision_vla.diagnostics.adapters.openpi_bridge import load_bridge_pi0
assert transformers.__version__ == '4.48.1'
print('Pi0 imports ready')
PY
```

This checks imports before model loading. Retain the file inventory and
verified asset receipt with the model configuration. Source-bound action
references require the documented normalization and timing alignment.
