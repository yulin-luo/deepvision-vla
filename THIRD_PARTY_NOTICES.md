# Release attribution and asset boundaries

DeepVision-VLA includes components with separate source, model and simulator terms. Preserve the applicable upstream copyright and licence notices when redistributing them.

- **DeepVision source:** the existing source distribution's Apache2.0 text is at `model_runtime/LICENSE`. Existing author attribution remains in `model_runtime/NOTICE`.
- **Modified Transformers4.57.0:** the exact vendored implementation is under `model_runtime/transformers/`, with its Apache2.0 text at `model_runtime/transformers/LICENSE`.
- **Janus-derived model helpers:** `model_runtime/deepvision/LICENSE-CODE` preserves the upstream MIT terms.
- **OpenVLA action discretizer / OpenVLA-OFT action head:** full MIT notices are included at `licenses/OpenVLA_MIT_LICENSE.txt` and `licenses/OpenVLA_OFT_MIT_LICENSE.txt`.
- **LIFT3D helper subset:** MIT text is included at `model_runtime/third_party/lift3d/LICENSE`. Official environment reference: [PKU-HMI-Lab/LIFT3D](https://github.com/PKU-HMI-Lab/LIFT3D), checked at commitb0cff859f737775c7b0255e21a83e97c76b4153c. Its full training/3D baseline stack is not required by this evaluation package.

## Model assets

The [official Qwen3-VL-4B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) declares Apache2.0. The RLBench-10 bundle includes downstream-modified Qwen weights rather than merely a downloadable adapter.

DINOv3 code and weights have a separate [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), not the repository's Apache2.0 license. A full copy is included at `licenses/DINOv3_LICENSE.md` for the DINOv3-derived component. Its redistribution and use conditions continue to apply to the saved fine-tuned DINO component. The code repository contains no weights. The separately published RLBench-10 checkpoint retains these terms; this package does not accept licences on behalf of users or claim unrestricted rights for the entire mixed bundle.

The RLBench-10 model-host revision and delivery path are pinned in [checkpoint delivery](docs/CHECKPOINT_DELIVERY.md). Do not label the combined checkpoint simply Apache2.0 or replace it with a different public DINO model. Preserve the verified twenty-file checkpoint inventory.

## Simulator dependencies fetched from upstream

`deepvision prepare-rlbench` downloads pinned upstream sources/runtime rather than committing the simulator binaries or task scenes into Git. Users must read and comply with the applicable upstream terms before obtaining or using them.

- **PyRep:** MIT, retained in the pinned upstream source's `LICENSE`.
- **RLBench:** its own [software licence agreement](https://github.com/CSCSX/RLBench/blob/841fd5c550e3ed6c196c40cb6ea636be3735ced4/LICENSE), not MIT. Wine code/scene are separately pinned from the PerAct RLBench fork and retain the upstream terms. Check the downloaded license, including its use/distribution restrictions.
- **CoppeliaSim Edu4.1.0:** the tested binary is an educational distribution. The [official licensing page](https://manual.coppeliarobotics.com/en/licensing.htm) restricts Edu eligibility and commercial use; the downloaded archive includes its own version-specific license agreement. The installer does not bypass license eligibility, remove watermarks or substitute an unvalidated simulator build. Users outside Edu eligibility need appropriate licensing; that runtime is not claimed tested here.

Actual pinned native license SHA256 values: PyRep`16502b86b7de4fd2d999eb055745e06cf93a93002e1e0b36e09c3180155fe81c`; RLBench`8a06aa559abfdbb29502a1670144d862bab8683ae588affaefb78bd546b37fab`.

DBus, GLib and PCRE2 are fetched as pinned conda-forge runtime packages. Their own package licence files and upstream terms continue to apply; they are not relicensed under this repository's Apache source licence. Native binary packages are not committed to this source distribution.
