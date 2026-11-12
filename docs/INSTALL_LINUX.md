# Linux installation

This guide prepares the shared environment for image inference and single-GPU RLBench evaluation. Use a local SSD for the Python environment and executable simulator libraries; checkpoint assets can live on a larger data disk.

The reference environment is **Ubuntu 24.04.2, Python 3.10.20, PyTorch 2.12.1 + CUDA 13.0, and NVIDIA driver 580.159.04**. The model uses a modified Transformers 4.57.0 implementation included in this repository.

## Prerequisites and storage

Use Linux x86_64 with `git`, `curl`, `tar`, and a C/C++ compiler (for the PyRep CFFI build). Native evaluation additionally needs the pinned X11/Qt shared-library closure below. This setup does not edit global graphics or display settings.

Choose an environment prefix with enough room for native extraction, and an asset/cache directory that can hold the 11.39 GB model plus downloads. Avoid a nearly full root partition. Prefer local SSD for executable environments and simulator libraries; weights may remain on a sufficiently fast shared filesystem. All commands target new task-specific directories.

## 1. Python environment

Run from the repository root. Replace the two example storage paths before running these commands:

```bash
export DV_PACKAGE="$PWD"
export DV_WORK=/your/local-ssd/deepvision-runtime
export DV_ASSETS=/your/data-disk/deepvision-assets

mkdir -p "$DV_WORK/tools" "$DV_WORK/tmp" "$DV_ASSETS/cache"
export TMPDIR="$DV_WORK/tmp" TEMP="$DV_WORK/tmp" TMP="$DV_WORK/tmp"
export PIP_CACHE_DIR="$DV_ASSETS/cache/pip"
export MAMBA_ROOT_PREFIX="$DV_ASSETS/cache/mamba"
```

Install the pinned micromamba binary and create the environment:

```bash
curl -fL https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.10.0-0.tar.bz2 \
  -o "$DV_ASSETS/cache/micromamba-2.10.0.tar.bz2"
printf '%s  %s\n' 9a97976244929bbae28fb1fae504e74a3e7e6d1932870e3a03321f01b7fbd07b \
  "$DV_ASSETS/cache/micromamba-2.10.0.tar.bz2" | sha256sum -c -
tar -xjf "$DV_ASSETS/cache/micromamba-2.10.0.tar.bz2" \
  -C "$DV_WORK/tools" bin/micromamba

"$DV_WORK/tools/bin/micromamba" create -y -p "$DV_WORK/env" \
  --file "$DV_PACKAGE/locks/conda-linux-64.lock"

export DV_PY="$DV_WORK/env/bin/python"
"$DV_PY" -m pip install -r "$DV_PACKAGE/requirements.txt"
"$DV_PY" -m pip install --no-deps --no-build-isolation -e "$DV_PACKAGE"
"$DV_PY" -m pip check
"$DV_PY" -m deepvision_vla --help
```

The explicit lock fixes 74 conda artifacts (the original 71 plus DBus, GLib and PCRE2), including Xvfb 21.1.25 and Mesa 26.3. `environment.yml` is the readable dependency recipe; use the explicit lock above for the pinned installation. Pip direct dependencies are pinned in `requirements.txt`.

Transformers 4.57.0 is explicitly pinned because the model depends on its modified implementation in `model_runtime/transformers`. The upstream PyPI version is marked yanked for a packaging issue; the exact version pin is intentional. Do not substitute a newer Transformers release or overwrite the vendored implementation.

## 2. RLBench native dependencies

Read the [third-party terms](../THIRD_PARTY_NOTICES.md) before obtaining the simulator. The helper downloads fixed upstream revisions and checks their hashes:

```bash
"$DV_PY" -m deepvision_vla prepare-rlbench \
  --output-root "$DV_WORK/deps" --cache "$DV_ASSETS/cache/native"

export COPPELIASIM_ROOT="$DV_WORK/deps/CoppeliaSim"
"$DV_PY" -m pip install --no-deps --no-build-isolation \
  -e "$DV_WORK/deps/PyRep" -e "$DV_WORK/deps/RLBench"
"$DV_PY" -m pip check
```

A C compiler such as gcc is needed to build the PyRep CFFI extension. Keep `--no-deps` when installing RLBench: its upstream setup otherwise requests an unpinned Git dependency.

The helper prepares:

- **PyRep:** `8f420be8064b1970aae18a9cfbc978dfb15747ef`
- **RLBench:** CSCSX fork, `841fd5c550e3ed6c196c40cb6ea636be3735ced4`
- **CoppeliaSim:** Edu 4.1.0, Ubuntu 20.04 archive; Edu eligibility and use conditions apply
- **Wine task code and scene:** PerAct RLBench fork, `ad991951bc53e4f3b73b803a75cf4b7d55295cf7`

The [LIFT3D environment reference](https://github.com/PKU-HMI-Lab/LIFT3D) informed the RLBench fork and task conventions. Its complete training/3D-baseline stack is not needed here. Keep the uniform point-cloud fallback used by this environment; PyTorch3D is not required. W&B is forced offline by the evaluation launcher.

## 3. Check native libraries before using a GPU

The exact environment lock adds `dbus=1.16.2`, `libglib=2.90.1`, and `pcre2=10.47`. They supply the DBus dependency of Qt's XCB platform plugin. Keep these compatible builds together rather than replacing the simulator's Qt files.

After generating your config, run this CPU-only check before evaluation:

```bash
"$DV_PY" -m deepvision_vla doctor --config "$DV_WORK/config.json"
```

`doctor` runs `ldd` with the same native search paths used by the evaluator. It identifies missing shared libraries/files and exits nonzero before model, GPU or simulator startup. Image-only configs report native dependencies as not required. This dependency check does not execute an RLBench episode.

For an existing environment that must stay read-only, keep verified native libraries in independent overlay directories and pass repeatable `configure --native-library-path /your/overlay/lib` arguments. Include every required overlay library directory, then run `doctor`; do not overwrite another environment's libraries. A new installation should use the complete lock above.

Interrupted native downloads retain uniquely named `.partial` attempts. Retry with a fresh output directory and the same cache: completed verified files are reused, while failed attempts are preserved. An existing completed cache file with a wrong hash is never silently replaced. Inspect/isolate that artifact or choose a fresh cache directory after reviewing the error.

## 4. Continue with the model and configuration

Return to [Quick start — checkpoint download](../README.md#2-download-the-rlbench-10-checkpoint) to download the model and generate your configuration. The checkpoint helper also verifies the required file hashes.

Keep `DV_WORK`, `DV_ASSETS`, and `DV_PY` set in the shell used for the remaining commands. To use the shorter `deepvision` console command instead of `"$DV_PY" -m deepvision_vla`, add this environment to the current shell's path:

```bash
export PATH="$DV_WORK/env/bin:$PATH"
```

The configuration helper records the local library roots, camera profile and task-local authenticated Xvfb launcher. You do not need to edit global display settings or install a cluster launcher. Use a free physical GPU and a new output directory for each run.
