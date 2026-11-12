# RLBench-10 checkpoint

[Download on Hugging Face](https://huggingface.co/lyl010221-pku/deepvision-vla-rlbench10)

- **Revision:** `f0a3a4c6eaa07e7e939452ad9dec66aa7c252fe3`
- **Checkpoint directory:** `checkpoint-299-39300`
- **Contents:** 20 required files, 11,391,500,062 bytes

This is the complete RLBench-10-fine-tuned policy: backbone/tokenizer, saved DINOv3, aligner, action head, configuration and action statistics. It can be used directly for inference; the separate embodied-pretraining checkpoint is not required.

## Download and verify

```bash
deepvision download --plan-only

deepvision download --output-root /new/assets/rlbench10-model \
  --cache /your/asset-cache/huggingface
```

No Hugging Face login is required. The downloader retrieves the pinned revision and verifies all required bytes. It writes `checkpoint_verification.json` outside the checkpoint directory. Reuse that receipt when generating a configuration:

```bash
deepvision configure \
  --checkpoint /new/assets/rlbench10-model/checkpoint-299-39300 \
  --verification-receipt /new/assets/rlbench10-model/checkpoint_verification.json \
  --env-prefix /your/env --image-only --output /new/image-config.json
```

## Reuse an existing checkpoint

Verify the existing files once; no copy or download is needed:

```bash
deepvision verify --checkpoint /your/checkpoint-299-39300 \
  --receipt /new/checkpoint-verification.json
```

Pass that receipt to `configure --verification-receipt`. If the option is omitted, configuration performs a full verification and records a new receipt alongside the configuration. A failed or unsealed old receipt is not accepted as proof of current file identity.

The receipt binds the full pinned inventory to stable file identities observed before and after hashing. Configuration records the receipt identity, and launches check that neither the receipt nor the selected files changed. If weights or links change, verify again and generate a new configuration. This avoids hashing the 11 GB bundle before every episode while detecting changes to an existing copy.

Use a new config/receipt path rather than overwriting earlier records. The checkpoint's original provenance paths are metadata, not required server locations. See the [model card](MODEL_CARD.md) for model and action conventions, and [third-party notices](../THIRD_PARTY_NOTICES.md) for component licences.
