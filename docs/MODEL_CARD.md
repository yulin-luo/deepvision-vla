# DeepVision-VLA — RLBench-10 model card

[Model repository](https://huggingface.co/lyl010221-pku/deepvision-vla-rlbench10) · [Paper](https://arxiv.org/abs/2603.15618)

## Model

The released checkpoint is a complete RLBench-10 downstream policy. It initializes from Full32 embodied pretraining and is then fine-tuned on RLBench for 300 epochs / 39,300 updates, with training seed 42, learning rate 2e-5, shared crop 0.9 and full DINOv3 fine-tuning. It uses seven decoupled action queries and an action horizon of one.

The model repository revision is `f0a3a4c6eaa07e7e939452ad9dec66aa7c252fe3`; the checkpoint lives under `checkpoint-299-39300`. All required components are included in the twenty-file bundle. The general embodied-pretraining checkpoint is planned as a separate release.

## Inputs and outputs

Input is a 224×224 front-view RGB observation and an instruction. The matched preprocessing applies crop 0.9 and prepares a 256-resolution Qwen input and a 512-resolution DINOv3 input. Robot state is not used as policy conditioning.

The output has seven values: world-frame EEF XYZ delta, absolute extrinsic XYZ Euler angles in radians, and gripper. Saved quantile statistics use unclipped normalization. An optional measured EEF position is used only to decode an absolute execution target, adding the predicted XYZ delta once.

## Usage

Use the supplied [image inference](IMAGE_INFERENCE.md) or [RLBench evaluation](EVALUATION.md) interface with the matching checkpoint statistics and simulator profile. This checkpoint targets RLBench tasks; real-robot deployment and other task suites require their own interfaces and validation. Image inference predicts an action and does not execute it.

## Licences

Qwen-derived weights retain Apache 2.0 terms. DINOv3-derived weights retain the separate DINOv3 licence. Source code and simulator components have their own applicable terms; see [third-party notices](../THIRD_PARTY_NOTICES.md) and the model repository's `LICENSES/` directory.
