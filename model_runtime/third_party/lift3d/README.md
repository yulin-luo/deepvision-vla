# Vendored LIFT3D (minimal subset for RLBench evaluation)

Source: https://github.com/PKU-HMI-Lab/LIFT3D @ `b0cff859f737775c7b0255e21a83e97c76b4153c` (main, 2026-07)

Only the modules imported by `scripts_rlbench10_ablation/test_rlbench_no_gen_encoder.py`
are vendored:

- `lift3d/envs/rlbench_env.py` — **project-pinned snapshot** (NOT upstream main): this is the
  exact env-glue version used for the paper's experiments. Known deltas vs upstream:
  `self.text = min(descriptions, key=len)` (upstream uses `max`; training instructions follow
  the *shortest* description, so this must stay `min`), extra `get_rgb_wrist`/
  `get_rgb_left_shoulder`, empty-pointcloud guard.
- `lift3d/envs/evaluator.py`, `lift3d/helpers/{common,gymnasium,graphics}.py` — from upstream
  main. `graphics.py` is patched to lazily import `open3d`/`pytorch3d` (only needed for point
  clouds; single-view RGB eval uses `EEpose` only).

RLBench itself must be the LIFT3D-pinned fork:
`https://github.com/CSCSX/RLBench` branch `lift3d` @ `841fd5c550e3ed6c196c40cb6ea636be3735ced4`
(plus `place_wine_at_rack_location` ported from MohitShridhar/RLBench peract branch —
see `docs/RLBENCH_REPRODUCTION.md`).

Usage: add this directory to `PYTHONPATH`, e.g.
`export PYTHONPATH=/path/to/deepvision-vla/third_party/lift3d:$PYTHONPATH`

LIFT3D license: see `LICENSE` in this directory.
