# Verifying diagnostic checkpoints

Follow the model's installation guide to download the pinned snapshot and obtain its published file inventory. The [model guide index](README.md) covers OpenVLA, official π₀ base, and QwenVLA-OFT H16.

For OpenVLA and the [QwenVLA-OFT H16 runtime export](QWEN_H16_ENVIRONMENT.md#obtain-the-h16-runtime-checkpoint), verify the files with:

```bash
python -m deepvision_vla.diagnostics.asset_receipts \
  --inventory model-inventory.json \
  --inventory-sha256 "$MODEL_INVENTORY_SHA256" \
  --checkpoint checkpoints/model \
  --output receipts/model-assets.json
```

Use the published inventory and its expected SHA256. The receipt identifies any missing or changed files. Pass it to the [launch preparation command](../guides/PUBLIC_CORE.md#run-a-prepared-plan). The [π₀ base guide](PI0_BASE.md) has its own download and verification command.
