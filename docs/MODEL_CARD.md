# DeepVision-VLA — RLBench-10 model card

[Model repository](https://huggingface.co/lyl010221-pku/deepvision-vla-rlbench10) · [Paper](https://arxiv.org/abs/2603.15618)

## Model

The released checkpoint is an embodied-pretrained DeepVision-VLA policy fine-tuned on RLBench-10. It includes the backbone/tokenizer, DINOv3, aligner, action head, configuration and action statistics. The embodied-pretraining checkpoint will be released separately.

See [checkpoint setup](CHECKPOINT_DELIVERY.md) for the download revision and directory.

## Model specifications

| Setting | Value |
| --- | --- |
| RLBench fine-tuning | 300 epochs / 39,300 updates |
| Training seed | 42 |
| Learning rate | 2e-5 |
| DINOv3 during fine-tuning | Full fine-tuning |
| Action horizon | 1 |
| Action queries | 7 decoupled queries |
| Observation | 224 × 224 front-view RGB + instruction |
| Preprocessing | Matched crop 0.9; Qwen input 256 × 256; DINOv3 input 512 × 512 |
| Robot-state conditioning | None |
| Action output | World-frame EEF XYZ delta, absolute extrinsic XYZ Euler angles in radians, and gripper |
| Action normalization | Saved q01/q99 statistics, without clipping |

## Inputs and outputs

The policy predicts seven action values from an image and instruction. An optional measured EEF position is used only to decode an absolute execution target by adding the XYZ delta once; it is not an input to the policy.

## Usage

Use the supplied [image inference](IMAGE_INFERENCE.md) or [RLBench evaluation](EVALUATION.md) interface with the matching checkpoint statistics and simulator profile. This checkpoint targets RLBench tasks; real-robot deployment and other task suites require their own interfaces and validation. Image inference predicts an action and does not execute it.

## Licences

Qwen-derived weights retain Apache 2.0 terms. DINOv3-derived weights retain the separate DINOv3 licence. Source code and simulator components have their own applicable terms; see [third-party notices](../THIRD_PARTY_NOTICES.md) and the model repository's `LICENSES/` directory.
