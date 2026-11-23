# BridgeV2 model installation

This package contains installation instructions, version pins, and checkpoint inventories for the BridgeV2 diagnostics. Use a dedicated environment for each model.

- [OpenVLA](OPENVLA_ENVIRONMENT.md) uses the official OpenVLA-7B checkpoint and the model code in its snapshot.
- [Official Pi0 base](../PI0_BASE.md) uses the LeRobot checkpoint in a Linux Python 3.12 environment.

Before choosing a guide, set these paths in a Bash shell. Use a workspace with room for the new environment and model snapshot.

```bash
export DV_CODE=/path/to/deepvision-vla
export DV_INSTALL=/path/to/this-install-package
export DV_WORK=/path/to/bridge-diagnostics-work
mkdir -p "$DV_WORK"
(cd "$DV_INSTALL" && sha256sum -c SHA256SUMS)
```

The guides create an isolated environment, use the official pinned download, and verify the checkpoint against the supplied inventory. `pip check` checks installed dependency metadata. The CPU-only import checks confirm library and adapter imports. The asset receipt reads and verifies the actual local files. Simulator setup is unnecessary for these checks.

The pins come from the Linux environments used for the diagnostic runs. OpenVLA passed checkpoint verification and a 16-source pilot in a fresh Python 3.10 environment. The official Pi0 base guide documents its separate 150-episode run. The dependency sources are the official package indexes. [PyTorch CUDA 13.0 wheel index](https://download.pytorch.org/whl/cu130/) · [PyPI](https://pypi.org/)

## Qwen H16

The matching Qwen3-VL model, processor and shared action-head source files are present under `model_runtime/` in the candidate DeepVision-VLA checkout. The Qwen adapter passed import and dependency checks in a fresh Python 3.10 environment with Transformers 4.57.0 and tokenizers 0.22.2. The H16 checkpoint, component manifests, processor assets, and 112-channel normalization statistics are required for model execution and have a separate access path. The saved H16 drift and attribution arrays support offline analysis. The public RLBench-10 checkpoint is for RLBench inference and evaluation.

## Diagnostic inputs

For a BridgeV2 run, the verified model snapshot is one input. The diagnostic planner also needs the frozen Bridge inventory, annotations, processor geometries and model profiles. The runtime then needs its model config, environment receipt, launch plan and fresh execution permit. See `docs/diagnostics/PUBLIC_CORE.md` in the source checkout for the Bridge action references and diagnostic commands.
