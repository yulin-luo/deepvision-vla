# RLBench-10 checkpoint

[Download on Hugging Face](https://huggingface.co/lyl010221-pku/deepvision-vla-rlbench10)

- **Revision:** `f0a3a4c6eaa07e7e939452ad9dec66aa7c252fe3`
- **Checkpoint directory:** `checkpoint-299-39300`
- **Download size:** 11.39 GB

This checkpoint is fine-tuned on RLBench-10 and includes all components needed for inference. No separate base-model download is required.

## Download and verify

```bash
deepvision download --plan-only

deepvision download --output-root /new/assets/rlbench10-model \
  --cache /your/asset-cache/huggingface
```

No Hugging Face login is required. The downloader checks file integrity and writes `checkpoint_verification.json` next to the checkpoint directory. Pass that file when configuring the model:

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

Pass the generated file to `configure --verification-receipt`. Without this option, configuration verifies the checkpoint again.

After changing checkpoint files or their links, run verification again and generate a new configuration. Unchanged checkpoints can reuse the verification file across runs.

See the [model card](MODEL_CARD.md) for model specifications and action conventions, and [third-party notices](../THIRD_PARTY_NOTICES.md) for licenses.
