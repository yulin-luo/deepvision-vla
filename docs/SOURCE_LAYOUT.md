# Source guide

## Method implementation

- [Qwen3-VL integration](../model_runtime/transformers/models/qwen3_vl/modeling_qwen3_vl.py): modified attention and vision-language interaction for VL-MoT and AGVP.
- [DeepVision model helpers](../model_runtime/deepvision/models/): multimodal processing, action tokenizer and action head.
- [Reference evaluator](../model_runtime/scripts_rlbench10_ablation/test_rlbench_no_gen_encoder.py): model construction, image preprocessing, action prediction and execution decoding.
- [RLBench protocol](../model_runtime/utils/rlbench_protocol.py): dataset, camera and simulator profile conventions.

## Public interface

- [Image inference](../src/deepvision_vla/inference/image.py): RGB-plus-instruction requests and prediction output.
- [Evaluation](../src/deepvision_vla/evaluation/runner.py): single-task evaluation on a selected GPU.
- [Checkpoint loading](../src/deepvision_vla/runtime/strict_entry.py): model loading and parameter checks before prediction.
- [Setup tools](../src/deepvision_vla/tools/): checkpoint download, verification and configuration.

Run `deepvision --help` to see the available commands. `python -m deepvision_vla` provides the same interface using your selected Python interpreter.

## Package layout

`src/deepvision_vla` contains the user-facing interface and runtime orchestration. `model_runtime` groups the model implementation, modified Transformers and the LIFT3D helper subset. Keeping their relative layout preserves the runtime's existing imports.

Use an editable installation from a retained source checkout. The `model_runtime` directory must remain next to `pyproject.toml`; moving only the installed package to another machine is insufficient.
