# RLBench evaluation

## Tasks

The evaluator supports:

- `close_box`
- `close_fridge`
- `close_laptop_lid`
- `phone_on_base`
- `place_wine_at_rack_location`
- `sweep_to_dustpan`
- `take_frame_off_hanger`
- `take_umbrella_out_of_umbrella_stand`
- `toilet_seat_down`
- `water_plants`

## Run a task

Complete [installation](INSTALL_LINUX.md) and [checkpoint setup](CHECKPOINT_DELIVERY.md), then select an available physical GPU:

```bash
deepvision eval --config /your/config.json --output /new/close-box-results \
  --gpu 0 --task close_box --episodes 50
```

A one-episode environment check uses the same command with `--episodes 1`. Each run requires a new output directory.

The checkpoint uses variation 0, a 224×224 front RGB observation, matched crop 0.9, and no robot-state conditioning. Each episode permits at most 10 policy steps. The evaluation seed is `42 + episode_id * 1009`; the reset adapter makes one reset attempt for each episode identity. The selected model profile is `hf-legacy-224-v1`.

## Actions

The seven predicted values are world-frame EEF XYZ deltas, absolute extrinsic XYZ Euler angles in radians, and a gripper value. Saved q01/q99 statistics provide unclipped action normalization. At execution, the XYZ delta is added once to the current EEF position and rotation is converted to xyzw. Use the supplied decoder and checkpoint statistics together.

## Summarize results

```bash
deepvision audit-rollouts --output-root /existing/close-box-results \
  --task close_box --episodes 50 --receipt /new/close-box-summary.json
```

The summary reports requested episodes, actual model attempts, successes, model failures, failures before policy entry, and missing outcomes. Reset or infrastructure failures remain in the request count, and partial runs are marked incomplete. This command summarizes saved files without rerunning evaluation.

## Evaluate all ten tasks

```bash
deepvision evaluate-all --config /your/config.json \
  --output /new/rlbench10-results --gpu 0
```

This runs 50 episodes per task, for 500 requests on one GPU. It stops if a task does not produce a complete result. Episodes are not automatically retried or replaced with different seeds.
