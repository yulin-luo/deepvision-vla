# Verifying local diagnostic checkpoints

Download the model snapshot at the revision listed in its environment guide,
then verify it against the matching published file inventory. Keep OpenVLA,
Bridge Pi0 and Qwen H16 in their separate environments.

With the environment activated:

```bash
python -m deepvision_vla.diagnostics.asset_receipts \
  --inventory model-inventory.json \
  --inventory-sha256 "$MODEL_INVENTORY_SHA256" \
  --checkpoint checkpoints/model \
  --output receipts/model-assets.json
```

Use the inventory and checksum supplied for that pinned model, rather than
generating expected checksums from whichever files are currently on disk.
The command hashes local files only; it does not download weights or load a
neural network. It supports ordinary Hugging Face snapshot leaf symlinks.
Verification uses two streaming content reads as well as file-identity checks,
so it does not rely only on filesystem timestamps to detect changing files.
Each read is bounded by the pinned file size plus one trailing-byte probe.

Every expected file appears in the receipt. Missing, changed or corrupt files
produce an `incomplete` receipt and a nonzero exit status. Existing receipts
are preserved; choose a new output path after repairing an asset. The model
adapter additionally checks its source, configuration and loading contract.

For large Hub LFS files, the inventory's SHA256 identifies the downloaded
content. The Git blob may identify the LFS pointer instead; those two hashes
are not interchangeable. Small source files without a content SHA256 are
verified against their pinned Git blob.

OpenVLA and Bridge Pi0 use the default `snapshot_v1` inventory format. Qwen H16
uses the explicitly named `qwen_h16_components_v1` format for the separately
trained qwen-only checkpoint at revision
`b131353d58b77daf2b6308e5ea4ea0aa4a417a79`. The local inventory supplied with the controlled checkpoint binds
`receipt_prefix` to `qwen_only/checkpoint-5-10000`. Supply that checkpoint
directory as `--checkpoint`; inventory file paths are relative to it, while
the receipt includes the prefix required by the Qwen consumer. This format
requires the training configuration, backbone/head manifests and exactly
their two weight files. It does not convert the RLBench H1 or joint Full32
model into the Qwen-only H16 model.

The receipt does not change the model's license, make a private repository
public, or establish action/reference alignment. Follow the source and model
terms linked in each model's environment guide.
