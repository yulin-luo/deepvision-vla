<div align="center">
  <h1>👁️ DeepVision-VLA</h1>
  <h3>Look Before Acting: Enhancing Vision Foundation Representations<br>for Vision-Language-Action Models</h3>
  <p><strong>ACM Multimedia 2026</strong></p>
  <p>Yulin Luo · Hao Chen · Zhuangzhe Wu · Bowen Sui · Jiaming Liu<br>Chenyang Gu · Zhuoyang Liu · Peng Jia · Pheng-Ann Heng · Shanghang Zhang</p>
  <p>Peking University · The Chinese University of Hong Kong · Simplexity Robotics</p>
  <p>
    <a href="https://arxiv.org/abs/2603.15618"><img src="https://img.shields.io/badge/arXiv-2603.15618-B31B1B?style=flat&amp;logo=arxiv&amp;logoColor=white" alt="Paper"></a>
    <a href="https://deepvision-vla.github.io/"><img src="https://img.shields.io/badge/Project-Website-2C7A7B?style=flat&amp;logo=githubpages&amp;logoColor=white" alt="Project website"></a>
    <a href="https://huggingface.co/lyl010221-pku/deepvision-vla-rlbench10"><img src="https://img.shields.io/badge/Checkpoint-RLBench--10-FFD21E?style=flat&amp;logo=huggingface&amp;logoColor=black" alt="RLBench-10 checkpoint"></a>
    <a href="https://doi.org/10.1145/3767308.3835774"><img src="https://img.shields.io/badge/ACM_MM-2026-7E57C2?style=flat" alt="ACM Multimedia 2026"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/Code-Apache_2.0-2563EB?style=flat" alt="Source code licence"></a>
  </p>
  <p><a href="#overview">Overview</a> · <a href="#quick-start">Quick start</a> · <a href="#checkpoint">Checkpoint</a> · <a href="#evaluation">Evaluation</a> · <a href="#citation">Citation</a></p>
</div>

## 📰 News

- **2026.11.11** — Initial release of the inference code, RLBench-10 evaluation pipeline, and [RLBench-10 fine-tuned checkpoint](https://huggingface.co/lyl010221-pku/deepvision-vla-rlbench10).

Training code and embodied-pretraining checkpoints are planned for a later release. Future releases will be announced here.

<a name="overview"></a>

## 🔍 Overview

**Why do VLAs understand the instruction but still miss the object?** Visual information enters early, yet task-relevant evidence can become underused in the deep layers that generate actions. DeepVision-VLA brings high-resolution visual evidence back into those layers.

<p align="center">
  <img src="assets/teaser.png" alt="DeepVision-VLA overview: deep visual grounding, action-guided selection, and paper-reported robot results" width="960">
</p>

- **Diagnose visual grounding.** Layer-wise ROI masking and attention analysis reveal where task-relevant visual evidence weakens.
- **VL-MoT: refresh deep visual evidence.** Couple multi-level DINOv3 features with the VLA backbone through shared attention while retaining separate processing pathways.
- **AGVP: keep action-relevant tokens.** Use shallow action-to-vision attention to select the high-resolution visual tokens passed to deep layers.

<p align="center">
  <img src="assets/method.png" alt="DeepVision-VLA architecture, Action-Guided Visual Pruning, and vision-language shared attention" width="960">
</p>

The figures illustrate the method and results reported in the paper. This release provides **image inference and single-GPU RLBench-10 evaluation** with the downstream checkpoint below.

## 🧭 Start here

- **Set up the environment:** [Linux installation and simulator dependencies](docs/INSTALL_LINUX.md)
- **Get the model:** [Checkpoint download and verification](docs/CHECKPOINT_DELIVERY.md) · [Model card](docs/MODEL_CARD.md)
- **Predict an action:** [Image + instruction inference](docs/IMAGE_INFERENCE.md)
- **Run the benchmark:** [RLBench tasks, protocol and result summaries](docs/EVALUATION.md)
- **Explore the implementation:** [Source guide](docs/SOURCE_LAYOUT.md)

<a name="quick-start"></a>

## 🚀 Quick start

### 1. Install the environment

```bash
git clone https://github.com/yulin-luo/deepvision-vla.git
cd deepvision-vla
```

Follow the Python-environment section of [the Linux setup guide](docs/INSTALL_LINUX.md) to install the pinned environment and editable package. Native simulator dependencies are needed only for RLBench evaluation. The guide defines `DV_WORK`, `DV_ASSETS`, and `DV_PY`; the commands below use those same paths.

The reference environment uses **Python 3.10.20, PyTorch 2.12.1 + CUDA 13.0, and the modified Transformers 4.57.0 implementation**. Keep this source checkout alongside the editable package.

```bash
"$DV_PY" -m deepvision_vla --help
```

### 2. Download the RLBench-10 checkpoint

```bash
"$DV_PY" -m deepvision_vla download \
  --output-root "$DV_ASSETS/rlbench10-model" \
  --cache "$DV_ASSETS/cache/huggingface"

export DV_CHECKPOINT="$DV_ASSETS/rlbench10-model/checkpoint-299-39300"
export DV_VERIFICATION="$DV_ASSETS/rlbench10-model/checkpoint_verification.json"
```

The downloader retrieves the pinned model revision and checks all 20 required files. No Hugging Face login or separate base-model download is needed.

### 3. Configure image inference

For image inference alone, no PyRep, RLBench or CoppeliaSim directories are required:

```bash
"$DV_PY" -m deepvision_vla configure \
  --checkpoint "$DV_CHECKPOINT" \
  --verification-receipt "$DV_VERIFICATION" \
  --env-prefix "$DV_WORK/env" --image-only \
  --output "$DV_WORK/image-config.json"
```

The downloader verifies the complete checkpoint once. Configuration binds that verification to the selected files; subsequent launches reject changed files or a changed receipt. For an existing model copy, follow [checkpoint verification](docs/CHECKPOINT_DELIVERY.md).
### 4. Run image inference

Create a request JSON beside an **uncropped 224 × 224 front-view RGB image**:

```json
{
  "schema_version": 1,
  "image": "input.png",
  "instruction": "close box"
}
```

Save the request as, for example, `$DV_WORK/request.json` and place `input.png` beside it. Select an available physical GPU:

```bash
"$DV_PY" -m deepvision_vla image \
  --config "$DV_WORK/image-config.json" \
  --request "$DV_WORK/request.json" \
  --task close_box --gpu 0 \
  --output "$DV_WORK/runs/image-001"
```

The output is saved to `results/image_prediction/prediction.json` under the run directory. It contains **XYZ delta + absolute XYZ Euler rotation + gripper**. An optional measured current EEF position enables decoding an absolute execution target; it is not a policy input. Image inference predicts an action without executing it. See the [input/output specification](docs/IMAGE_INFERENCE.md).

Use a **new output directory** for every run.

<a name="checkpoint"></a>

## 🤗 Checkpoint

[**DeepVision-VLA RLBench-10**](https://huggingface.co/lyl010221-pku/deepvision-vla-rlbench10) is already fine-tuned on RLBench. It includes the backbone/tokenizer, saved DINOv3, aligner, action head, configuration and action statistics.

- **Pinned revision:** `f0a3a4c6eaa07e7e939452ad9dec66aa7c252fe3`
- **Checkpoint directory:** `checkpoint-299-39300`
- **Action horizon:** 1, with seven decoupled action queries
- **Image processing:** matched crop 0.9; 256-resolution Qwen input and 512-resolution DINOv3 input
- **Action convention:** world-frame EEF XYZ delta, absolute extrinsic XYZ Euler angles in radians, and gripper

The embodied-pretraining checkpoint is a separate planned release. It is not needed to run this RLBench checkpoint. See [checkpoint setup](docs/CHECKPOINT_DELIVERY.md) and the [model card](docs/MODEL_CARD.md).

<a name="evaluation"></a>

## 🧪 RLBench evaluation

The evaluator supports **10 tasks**: closing a box, fridge and laptop; placing a phone and wine bottle; sweeping; taking a frame and umbrella; watering plants; and lowering a toilet seat. Exact task identifiers and the pinned protocol are listed in [the evaluation guide](docs/EVALUATION.md).

Before evaluating, complete the native-dependency section of [Linux installation](docs/INSTALL_LINUX.md), then generate a full configuration and check its libraries:

```bash
"$DV_PY" -m deepvision_vla configure \
  --checkpoint "$DV_CHECKPOINT" \
  --verification-receipt "$DV_VERIFICATION" \
  --env-prefix "$DV_WORK/env" --deps-root "$DV_WORK/deps" \
  --output "$DV_WORK/rlbench-config.json"

"$DV_PY" -m deepvision_vla doctor --config "$DV_WORK/rlbench-config.json"
```

The native check uses the same dynamic-library paths as evaluation and reports missing Qt/X11 dependencies before a GPU/model is started. The pinned environment includes DBus. An image-only configuration cannot be used for simulation.

### One task

```bash
"$DV_PY" -m deepvision_vla eval \
  --config "$DV_WORK/rlbench-config.json" \
  --task close_box --episodes 50 --gpu 0 \
  --output "$DV_WORK/runs/close-box-001"
```

Use `--episodes 1` for a single-episode integration check. The protocol uses variation 0, front-view RGB, seed `42 + episode_id × 1009`, at most 10 policy steps, and one reset attempt per episode identity.

### Summarize saved results

```bash
"$DV_PY" -m deepvision_vla audit-rollouts \
  --output-root "$DV_WORK/runs/close-box-001" \
  --task close_box --episodes 50 \
  --receipt "$DV_WORK/runs/close-box-001-summary.json"
```

The summary separates requested episodes, actual attempts, successes, model failures, failures before policy entry, and missing outcomes. Reading a saved result does not replay the model or simulator.

### All ten tasks

```bash
"$DV_PY" -m deepvision_vla evaluate-all \
  --config "$DV_WORK/rlbench-config.json" --gpu 0 \
  --output "$DV_WORK/runs/rlbench10-001"
```

This explicitly runs 50 episodes per task, for 500 total requests. It does not resume or overwrite an existing run. An incomplete run is not reported as a complete benchmark score.

## 🗂️ Repository structure

```text
deepvision-vla/
├── assets/                  # Paper overview and method figures
├── configs/                 # Example configuration
├── docs/                    # Installation, model and usage guides
├── src/deepvision_vla/
│   ├── inference/           # Image requests and prediction adapter
│   ├── evaluation/          # Task runner and result summaries
│   ├── runtime/             # Strict loading, reset and headless display
│   ├── tools/               # Download, verification and setup
│   └── assets/              # Model inventory and runtime profiles
├── model_runtime/           # Model implementation and pinned dependencies
├── locks/                   # Exact Linux environment artifacts
├── licenses/                # Upstream licences
├── tests/                   # CPU regression tests
└── pyproject.toml
```

See [the source guide](docs/SOURCE_LAYOUT.md) for VL-MoT, AGVP and action-head implementation locations.

<a name="citation"></a>

## 📝 Citation

If you use DeepVision-VLA in your research, please cite the ACM Multimedia paper:

```bibtex
@inproceedings{luo2026deepvision,
  title = {Look Before Acting: Enhancing Vision Foundation Representations for Vision-Language-Action Models},
  author = {Luo, Yulin and Chen, Hao and Wu, Zhuangzhe and Sui, Bowen and Liu, Jiaming and Gu, Chenyang and Liu, Zhuoyang and Jia, Peng and Heng, Pheng-Ann and Zhang, Shanghang},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year = {2026},
  doi = {10.1145/3767308.3835774}
}
```

## 📜 Licence and acknowledgements

The source code is distributed under [Apache 2.0](LICENSE), with the applicable upstream notices preserved. Qwen-derived weights, DINOv3-derived weights, RLBench and CoppeliaSim retain their respective terms; see [third-party notices](THIRD_PARTY_NOTICES.md).

We thank the authors of [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [DINOv3](https://github.com/facebookresearch/dinov3), [OpenVLA](https://github.com/openvla/openvla), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [LIFT3D](https://github.com/PKU-HMI-Lab/LIFT3D) for their open-source contributions.
