# BridgeV2 model installation

Use a separate Linux environment for each model.

- [OpenVLA](OPENVLA_ENVIRONMENT.md) uses the pinned OpenVLA-7B snapshot.
- [Official π₀ base](PI0_BASE.md) uses the pinned LeRobot checkpoint.
- [QwenVLA-OFT H16](QWEN_H16_ENVIRONMENT.md) uses the public Qwen runtime export.

Set the working paths in Bash before following the model guide.

```bash
export DV_CODE=/path/to/deepvision-vla
export DV_INSTALL="$DV_CODE/docs/diagnostics/install/files"
export DV_WORK=/path/to/bridge-diagnostics-work
mkdir -p "$DV_WORK"
(cd "$DV_INSTALL" && sha256sum -c SHA256SUMS)
```

See the [BridgeV2 quickstart](../QUICKSTART.md) for the runtime panel and [checkpoint verification](ASSETS.md) for the asset receipt.
