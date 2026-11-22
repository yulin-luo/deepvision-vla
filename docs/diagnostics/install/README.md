# BridgeV2 model installation

This package contains installation instructions, version pins, and checkpoint inventories for two BridgeV2 diagnostic policies. Use a DeepVision-VLA checkout that contains `deepvision_vla.diagnostics.asset_receipts`. Each policy has its own Linux x86-64 Python 3.10 environment.

- [OpenVLA](OPENVLA_ENVIRONMENT.md) uses the official OpenVLA-7B checkpoint and the model code in its snapshot.
- [Bridge Pi0](PI0_ENVIRONMENT.md) uses Haoming Song's Bridge-adapted Pi0 checkpoint and a pinned Hugging Face LeRobot checkout.

Before choosing a guide, set these paths in a Bash shell. Use a workspace with room for the new environment and model snapshot.

```bash
export DV_CODE=/path/to/deepvision-vla
export DV_INSTALL=/path/to/this-install-package
export DV_WORK=/path/to/bridge-diagnostics-work
mkdir -p "$DV_WORK"
(cd "$DV_INSTALL" && sha256sum -c SHA256SUMS)
```

The guides create an isolated environment, use the official pinned download, and verify the checkpoint against the supplied inventory. `pip check` checks installed dependency metadata. The CPU-only import checks confirm library and adapter imports. The asset receipt reads and verifies the actual local files. Simulator setup is unnecessary for these checks.

The pins come from the Linux environments used for the diagnostic runs. OpenVLA and Bridge Pi0 both passed installation in fresh Python 3.10 environments, checkpoint verification, and the 16-source pilot. Each model executed 208 new requests and reproduced the previous pilot's action values exactly. The dependency sources are the official package indexes. [PyTorch CUDA 13.0 wheel index](https://download.pytorch.org/whl/cu130/) · [PyPI](https://pypi.org/)

## Qwen H16

The matching Qwen3-VL model, processor and shared action-head source files are present under `model_runtime/` in the candidate DeepVision-VLA checkout. The Qwen adapter passed import and dependency checks in a fresh Python 3.10 environment with Transformers 4.57.0 and tokenizers 0.22.2. The H16 checkpoint, component manifests, processor assets, and 112-channel normalization statistics are required for model execution and have a separate access path. The saved H16 drift and attribution arrays support offline analysis. The public RLBench-10 checkpoint is for RLBench inference and evaluation.

## Diagnostic inputs

For a BridgeV2 run, the verified model snapshot is one input. The diagnostic planner also needs the frozen Bridge inventory, annotations, processor geometries and model profiles. The runtime then needs its model config, environment receipt, launch plan and fresh execution permit. See `docs/diagnostics/PUBLIC_CORE.md` in the source checkout for the Bridge action references and diagnostic commands.
