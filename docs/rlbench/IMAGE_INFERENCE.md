# Image inference

The RLBench-10 checkpoint predicts an action from one image and an instruction. Image inference does not require a training dataset or simulator.

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

`image` is relative to the request file. Supply an uncropped 224×224 front-view RGB image. The pipeline applies the matched crop 0.9 and prepares inputs for Qwen and DINOv3.

```bash
python -m deepvision_vla image --config /new/image-config.json --output /new/image-output \
  --gpu 0 --wait-for-gpu 300  --task close_box --request /your/request.json
python -m deepvision_vla audit-image --output-root /existing/image-output \
  --receipt /new/image-summary.json
```

Choose an available GPU. `--task` selects the RLBench task profile. The instruction in your request is passed directly to the model.

The output `results/image_prediction/prediction.json` contains a seven-dimensional action vector in physical units:

`[world_delta_x, world_delta_y, world_delta_z, absolute_roll, absolute_pitch, absolute_yaw, gripper]`

Rotation is absolute extrinsic XYZ Euler in radians. Action decoding uses the checkpoint's saved q01/q99 statistics without clipping.

Optionally include `current_ee_position_xyz`: three measured world-frame coordinates in meters. The decoder adds the predicted XYZ delta once and returns `execution_xyz_xyzw_grip`, containing absolute position, an xyzw quaternion, and gripper. The measured position is used only for decoding, not as a policy input. Without it, this field is null.

Image inference returns a prediction without executing it. Use a new output directory for each request.
