# Linux installation

This guide covers image inference and single-GPU RLBench evaluation.

The reference environment is **Ubuntu 24.04.2, Python 3.10.20, PyTorch 2.12.1 + CUDA 13.0, and NVIDIA driver 580.159.04**. The model uses a modified Transformers 4.57.0 implementation included in this repository.

## Prerequisites and storage

Use Linux x86_64 with `git`, `curl`, `tar`, and a C/C++ compiler for PyRep. Simulator evaluation also needs the X11/Qt libraries included in the environment below.

Allow space for the 11.39 GB checkpoint, environment and simulator downloads. Prefer a local SSD for the environment and simulator; weights can live on a larger data disk.

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

The lock file pins the conda dependencies, including Xvfb, Mesa and DBus. `environment.yml` lists the dependency recipe; use the lock file above to install the tested versions. Python dependencies are pinned in `requirements.txt`.

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

The setup follows the [LIFT3D environment reference](https://github.com/PKU-HMI-Lab/LIFT3D). PyTorch3D is not required. W&B runs offline during evaluation.

## 3. Check simulator libraries

The environment includes DBus, GLib and PCRE2, which are required by the simulator's Qt/XCB libraries.

After generating your config, run this CPU-only check before evaluation:

```bash
"$DV_PY" -m deepvision_vla doctor --config "$DV_WORK/rlbench-config.json"
```

`doctor` checks for missing shared libraries before loading the model or starting an episode.

If an existing environment uses additional library directories, supply each one with `configure --native-library-path /your/library/directory`, then run `doctor`. New installations should use the complete lock file above.

If a download is interrupted, rerun setup with a new output directory and the same cache. Verified downloads are reused. For a checksum error, inspect the reported file or use a new cache directory.

## 4. Continue with the model and configuration

Return to [Quick start — checkpoint download](../README.md#2-download-the-rlbench-10-checkpoint) to download the model and generate your configuration. The checkpoint helper also verifies the required file hashes.

Keep `DV_WORK`, `DV_ASSETS`, and `DV_PY` set in the shell used for the remaining commands. To use the shorter `deepvision` console command instead of `"$DV_PY" -m deepvision_vla`, add this environment to the current shell's path:

```bash
export PATH="$DV_WORK/env/bin:$PATH"
```

Configuration sets up the simulator paths and headless display. Use an available GPU and a new output directory for each run.
