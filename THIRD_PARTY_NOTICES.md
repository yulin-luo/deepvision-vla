# Third-party licenses

DeepVision-VLA includes components with separate source, model and simulator terms. Preserve the applicable upstream copyright and licence notices when redistributing them.

- **DeepVision source:** Apache 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
- **Modified Transformers 4.57.0:** included under `model_runtime/transformers/`; see its [Apache 2.0 license](model_runtime/transformers/LICENSE).
- **Janus-derived model helpers:** `model_runtime/deepvision/LICENSE-CODE` preserves the upstream MIT terms.
- **OpenVLA action discretizer / OpenVLA-OFT action head:** full MIT notices are included at `licenses/OpenVLA_MIT_LICENSE.txt` and `licenses/OpenVLA_OFT_MIT_LICENSE.txt`.
- **LIFT3D helper subset:** see the [MIT license](model_runtime/third_party/lift3d/LICENSE). The RLBench setup follows the [LIFT3D environment reference](https://github.com/PKU-HMI-Lab/LIFT3D/tree/b0cff859f737775c7b0255e21a83e97c76b4153c).

## Model weights

Qwen-derived weights use Apache 2.0; see the [Qwen3-VL-4B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).

DINOv3-derived code and weights retain the [DINOv3 License](licenses/DINOv3_LICENSE.md), including its use and redistribution conditions. The repository's Apache 2.0 license does not replace those terms.

The [RLBench-10 checkpoint](docs/CHECKPOINT_DELIVERY.md) includes Qwen- and DINOv3-derived components; their respective licenses apply.

## Simulator dependencies

`deepvision prepare-rlbench` downloads simulator dependencies from upstream. Their licenses apply separately:

- **PyRep:** MIT, retained in the pinned upstream source's `LICENSE`.
- **RLBench:** see its [software license agreement](https://github.com/CSCSX/RLBench/blob/841fd5c550e3ed6c196c40cb6ea636be3735ced4/LICENSE). The wine task code and scene come from the PerAct RLBench fork and retain the upstream terms.
- **CoppeliaSim Edu 4.1.0:** educational use is subject to the [CoppeliaSim licensing terms](https://manual.coppeliarobotics.com/en/licensing.htm) and the agreement included in the downloaded archive. Use requires Edu eligibility or an appropriate alternative license.

Conda dependencies, including DBus, GLib and PCRE2, retain their upstream licenses, available in the installed packages.
