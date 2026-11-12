# One image and an instruction, without labels

The RLBench-10 checkpoint predicts an action from one image and an instruction. Use the complete checkpoint and the environment described in [installation](INSTALL_LINUX.md); a training dataset and simulator are not needed for image inference.

First install the Python environment in [Linux installation](INSTALL_LINUX.md), obtain the [checkpoint and verification receipt](CHECKPOINT_DELIVERY.md), and configure the image-only interface:

```bash
deepvision configure --checkpoint /your/checkpoint-299-39300 \
  --verification-receipt /existing/checkpoint-verification.json \
  --env-prefix /your/env --image-only --output /new/image-config.json
```

This mode does not require native simulator directories. To run RLBench later, create a separate full configuration with `--deps-root`; do not reuse the image-only configuration for simulation.

Create a request JSON next to your own input image:

```json
{
  "schema_version": 1,
  "image": "input.png",
  "instruction": "close box"
}
```

`image` resolves relative to the request file. Supply the original uncropped 224×224 front-view RGB image for this trained profile; the scientific preprocessing applies the matched crop 0.9 and model-specific image processing. Another camera/input protocol is not silently substituted.

```bash
python -m deepvision_vla image --config /new/image-config.json --output /new/image-output \
  --gpu 0 --wait-for-gpu 300  --task close_box --request /your/request.json
python -m deepvision_vla audit-image --output-root /existing/image-output \
  --receipt /new/image-summary.json
```

Use one actually available card; the launcher checks compute/graphics occupancy and actual CUDA PCI identity. The task option selects the pinned model/profile context. **The image API preserves your explicit instruction** and does not replace it using the RLBench task-name overrides. This does not mean paraphrase/compositional performance has been measured.

The output `results/image_prediction/prediction.json` contains one finite 7D denormalized vector:

`[world_delta_x, world_delta_y, world_delta_z, absolute_roll, absolute_pitch, absolute_yaw, gripper]`

Rotation is absolute extrinsic XYZ Euler in radians. Normalization uses the saved unclipped q01/q99 contract. Gripper decoding comes from the original scientific function; no new clipping or threshold is introduced in this API.

Optionally include `current_ee_position_xyz` as a JSON list of three **measured world-frame position values in meters**. It is used only after prediction to add the XYZ delta once. The receipt then also includes `execution_xyz_xyzw_grip`, an 8D command with absolute world position and the original Euler-to-xyzw codec. Position is not a policy-conditioning input; no robot-state tokens are sent to this checkpoint. If no measured position is provided, the decoded execution command remains null.

The API never executes an action. It reports simulator episodes0 and RLBench success rate null. It rejects GT/unknown request fields and preserves existing output directories instead of overwriting them.
