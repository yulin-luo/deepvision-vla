# OpenVLA diagnostic environment

Use a dedicated **Linux Python 3.10** environment for OpenVLA and set the
paths in the [installation overview](README.md). OpenVLA uses Transformers
4.40.1.

The pins come from the OpenVLA adapter environment used for the diagnostic runs.
It used Python 3.10.20, PyTorch 2.12.1+cu130, torchvision 0.27.1+cu130,
Transformers 4.40.1, tokenizers 0.19.1 and timm 0.9.10. For GPU runs, the CUDA
13.0 wheels need a compatible NVIDIA driver.

## Install

Create and activate the environment, then install the diagnostic code from
your checkout.

```bash
python3.10 -m venv "$DV_WORK/openvla-env"
source "$DV_WORK/openvla-env/bin/activate"
python -m pip install -r "$DV_INSTALL/requirements-diagnostics-openvla.txt"
python -m pip install -e "$DV_CODE" --no-deps
python -m pip check
```

## Obtain the pinned checkpoint and model source

```bash
hf download openvla/openvla-7b \
  --revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f \
  --local-dir "$DV_WORK/checkpoints/openvla-7b"
```

The pinned snapshot contains the checkpoint, tokenizer, processor, and Prismatic
model source. The recorded snapshot is about 15.09 GB.

```bash
python -m deepvision_vla.diagnostics.asset_receipts \
  --inventory "$DV_INSTALL/openvla-inventory.json" \
  --inventory-sha256 4ea2f6a3ce32a05f9ff8d11efcd4b1dd50a8440551341e23a01b2bfdae631c94 \
  --checkpoint "$DV_WORK/checkpoints/openvla-7b" \
  --output "$DV_WORK/receipts/openvla-assets.json"
```

The successful receipt reports `complete` and 17 verified files. Use a fresh
receipt filename when repeating the check.

The official model card declares MIT. Preserve upstream model/code notices and
consult the model card for the underlying pretrained components' terms.
BridgeData inputs have a separate dataset license. [Pinned model snapshot](https://huggingface.co/openvla/openvla-7b/tree/47a0ec7fc4ec123775a391911046cf33cf9ed83f) · [OpenVLA source](https://github.com/openvla/openvla)

## Adapter behavior

`deepvision_vla.diagnostics.adapters.openvla_runtime` uses the pinned processor
and `AutoModelForVision2Seq`, eager attention and BF16. Checkpoint-local Python
code is enabled only after the snapshot identity is checked. Missing,
unexpected or mismatched loading keys are rejected.

Prediction uses the original seven action tokens and `bridge_orig` statistics.
The diagnostic decoder blocks are 0, 10, 20 and 30. Interventions apply to native
image-prefill tokens. The runner accounts for cached autoregressive steps
separately. The processor's resize-naive path determines the 256-token visual
grid.

The adapter takes the checkpoint path, verified asset receipt and pinned
official inventory. Retain the receipt with the model-specific diagnostic
configuration.

## Import check

This checks the library pins and adapter import on CPU.

```bash
CUDA_VISIBLE_DEVICES='' python - <<'PY'
import timm
import tokenizers
import transformers
from deepvision_vla.diagnostics.adapters.openvla_runtime import load_openvla_runtime
assert transformers.__version__ == '4.40.1'
assert timm.__version__ == '0.9.10'
assert tokenizers.__version__ == '0.19.1'
print('OpenVLA imports ready')
PY
```

For a model run, supply native input and ROI geometry, a fixed request plan and
an independent resource budget. These checks cover the imports and snapshot
bytes.

