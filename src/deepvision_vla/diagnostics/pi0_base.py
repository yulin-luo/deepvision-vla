"""Official LeRobot Pi0 base visual masking on a fixed BridgeV2 subset."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen
from unittest.mock import patch

import numpy as np

LAYERS = (0, 6, 12, 16)
KEEP = tuple(range(11))
TASK_ROLES = {"manipulated_object", "robot_arm", "goal_context_anchor"}
MODEL_SHA = "8229fd9a7c3c2aafc1e223567b61b5fe3e25eef873bb4233928dbee4bd836303"
REVISION = "6e245399ad403ccf02b8112b73cfdc2633068f38"
SEED = "deepvision-v55-official-pi0-base-10pct-seed42"
CHECKPOINT_FILES = {
    "policy_preprocessor.json": "7bae7494caaf887fbcf5e09052d72bed5ef09272eb296202708298802a02a2c5",
    "config.json": "700c59d206885faff307d63272558fb9ad9a6b9feecdae55626ed9e3f594feb9",
    "model.safetensors": "8229fd9a7c3c2aafc1e223567b61b5fe3e25eef873bb4233928dbee4bd836303"
}
RUNTIME_FILES = {
    "inventory.json": "4155dc0f1df1dde0efd312848490ba3aea701b82645e5a8f7a915a1c79c4f3a6",
    "annotations.json": "c49ff1fc2f2d1c5f5732e77b721069791f26e8f682fa95f55a7a691aca71c77d",
    "geometries.jsonl": "5fff0a0070ad724bbcbc27f020999f54d5e0c50f277c7b3c19aca6d6b367fcbf"
}
TOKENIZER_SHA = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x", encoding="utf8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def stable(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def state_at(panel, row):
    path = panel / row["original_source"]["arrays_path"]
    assert digest(path) == row["arrays_sha256"], row["sample_id"]
    with np.load(path, allow_pickle=False) as arrays:
        values = arrays["states"][row["anchor"]].astype(np.float64)
    assert values.shape == (7,) and np.isfinite(values).all()
    return values


def project(roles):
    centers = np.arange(16) * 16 + 8
    xx, yy = np.meshgrid(centers, centers)
    inside = np.zeros((16, 16), dtype=bool)
    for role in roles:
        if role["role"] not in TASK_ROLES:
            continue
        left, top, right, bottom = role["xyxy_px"]
        inside |= (left <= xx) & (xx < right) & (top <= yy) & (yy < bottom)
    return np.flatnonzero(inside).tolist()


def fetch(root):
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        "lerobot/pi0_base", revision=REVISION, local_dir=root / "checkpoint",
        allow_patterns=list(CHECKPOINT_FILES), token=False,
    )
    tokenizer = root / "paligemma_tokenizer.model"
    if not tokenizer.exists():
        with urlopen(TOKENIZER_URL, timeout=30) as stream:
            data = stream.read(6_000_000)
        if hashlib.sha256(data).hexdigest() != TOKENIZER_SHA:
            raise ValueError("The upstream PaliGemma tokenizer differs from the verified file")
        with tokenizer.open("xb") as stream:
            stream.write(data)
    if digest(tokenizer) != TOKENIZER_SHA:
        raise ValueError("The tokenizer differs from the verified file")
    verify_checkpoint(root / "checkpoint", announce=True)


def verify_checkpoint(checkpoint, *, announce=False):
    for name, expected in CHECKPOINT_FILES.items():
        if digest(checkpoint / name) != expected:
            raise ValueError(f"The downloaded Pi0 base file differs: {name}")
        if announce:
            print(f"Verified {name}", flush=True)


def prepare(root, runtime):
    root.mkdir(parents=True, exist_ok=True)
    runtime = runtime.resolve(strict=True)
    panel = runtime / "panel"
    inventory = panel / "inventory.json"
    annotations = runtime / "inputs/annotations.json"
    geometry = runtime / "inputs/geometries.jsonl"
    rows = read(inventory)
    assert len(rows) == 1500 and len({r["sample_id"] for r in rows}) == 1500
    input_hashes = {p.name: digest(p) for p in (inventory, annotations, geometry)}
    if input_hashes != RUNTIME_FILES:
        raise ValueError("Use the checksum-verified BridgeData V2 v55 runtime panel")
    annotated = {r["sample_id"]: r for r in read(annotations)}
    groups = defaultdict(list)
    for row in rows:
        groups[row["original_source"]["shard"]].append(row)
    quota = {name: len(group) // 10 for name, group in groups.items()}
    order = sorted(groups, key=lambda name: (-(len(groups[name]) % 10), stable(SEED, name)))
    for name in order[:150 - sum(quota.values())]:
        quota[name] += 1
    subset = [row for name, group in groups.items()
              for row in sorted(group, key=lambda r: stable(SEED, r["sample_id"]))[:quota[name]]]
    subset.sort(key=lambda r: stable(SEED + "-pilot", r["sample_id"]))
    assert len(subset) == 150
    chosen = {row["sample_id"] for row in subset}
    pi0_geometries = {}
    with geometry.open(encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            if row["model"] == "pi0" and row["sample_id"] in chosen:
                pi0_geometries[row["sample_id"]] = row
    assert set(pi0_geometries) == chosen

    calibration = [row for row in rows if row["sample_id"] not in chosen]
    calibration_state = np.stack([state_at(panel, row) for row in calibration])
    mean, std = calibration_state.mean(axis=0), calibration_state.std(axis=0)
    assert len(calibration) == 1350 and np.all(std > 1e-6)
    sources = []
    for row in subset:
        sid = row["sample_id"]
        layout = pi0_geometries[sid]["layout"]
        assert layout["grid_shape"] == [16, 16]
        assert layout["source_size"] == [256, 256] and layout["crop_xyxy"] == [0, 0, 256, 256]
        assert layout["content_xyxy"] == [0, 0, 224, 224]
        assert annotated[sid]["image_sha256"] == layout["image_sha256"] == row["image_sha256"]
        indices = project(annotated[sid]["roles"])
        assert indices, sid
        # Apply exactly the same token order at every depth.
        indices.sort(key=lambda index: stable(SEED + sid, str(index)))
        sources.append({**row, "roi": indices, "state": state_at(panel, row).tolist()})
    plan = {
        "status": "frozen_before_model_calls",
        "model": "lerobot/pi0_base",
        "revision": REVISION,
        "model_sha256": MODEL_SHA,
        "selection": "One tenth of v55. Largest remainder quotas by original shard; SHA-ranked within shards.",
        "seed": SEED,
        "runtime": str(runtime),
        "inputs_sha256": input_hashes,
        "layers": list(LAYERS),
        "nominal_keep": [k / 10 for k in KEEP],
        "camera": "Observed Bridge image_0 as base camera; absent wrist views masked by LeRobot.",
        "state": "EEF pose and gripper proxy for visual masking. Native joint-state mapping is unverified.",
        "state_calibration": {
            "source_count": len(calibration),
            "source_ids_sha256": stable(SEED, "|".join(sorted(row["sample_id"] for row in calibration))),
            "mean": mean.tolist(), "std": std.tolist(),
        },
        "metric": "Normalized native 50 x 32 action drift against same-input same-noise unmasked output. No GT-MSE.",
        "sources": sources,
    }
    target = root / "COHORT150.json"
    assert not target.exists()
    write(target, plan)
    print(json.dumps({"status": plan["status"], "n": len(subset), "shards": len(set(r["original_source"]["shard"] for r in subset)),
                      "roi_token_count_min": min(len(r["roi"]) for r in sources), "roi_token_count_max": max(len(r["roi"]) for r in sources),
                      "sha256": digest(target)}), flush=True)


def load_policy(root):
    import torch
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from safetensors.torch import load_file

    checkpoint = root / "checkpoint"
    verify_checkpoint(checkpoint)
    config = PI0Config.from_pretrained(checkpoint, local_files_only=True)
    config.device = "cpu"
    assert config.chunk_size == 50 and config.max_state_dim == config.max_action_dim == 32
    assert config.dtype in ("float32", torch.float32) and config.num_inference_steps == 10
    policy = PI0Policy(config)
    # Reuse upstream remapping, but require a strict load to succeed.
    weights = load_file(checkpoint / "model.safetensors")
    remapped = policy._fix_pytorch_state_dict_keys(weights, config)
    state = {(k if k.startswith("model.") else "model." + k): v for k, v in remapped.items()}
    assert len(state) == len(remapped)
    result = policy.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    print(json.dumps({"status": "strict_official_weights_loaded", "state_entries": len(state)}), flush=True)
    del state, remapped, weights
    return policy.eval()


def prepared_batch(policy, tokenizer, source, plan, device):
    import torch
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
    from PIL import Image

    panel = Path(plan["runtime"]) / "panel"
    image = panel / source["original_source"]["image_path"]
    assert digest(image) == source["image_sha256"]
    with Image.open(image) as decoded:
        assert decoded.size == (256, 256)
        rgb = np.asarray(decoded.convert("RGB")).copy()
    image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255
    prompt = source["instruction"].strip().replace("_", " ").replace("\n", " ")
    ids = (tokenizer.encode(prompt, add_bos=True) + tokenizer.encode("\n"))[:48]
    mask = [True] * len(ids) + [False] * (48 - len(ids))
    ids += [0] * (48 - len(ids))
    cal = plan["state_calibration"]
    proxy = ((np.asarray(source["state"]) - cal["mean"]) / cal["std"]).astype(np.float32)
    return {
        "observation.images.base_0_rgb": image_tensor,
        OBS_STATE: torch.from_numpy(proxy).unsqueeze(0).to(device),
        OBS_LANGUAGE_TOKENS: torch.tensor([ids], device=device, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([mask], device=device, dtype=torch.bool),
    }


def batched(tensor, size):
    return tensor.expand(size, *tensor.shape[1:]).contiguous()


def measure(policy, tokenizer, source, plan, batch_size, audit):
    import torch
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    batch = prepared_batch(policy, tokenizer, source, plan, "cuda")
    model = policy.model
    layers = model.paligemma_with_expert.paligemma.model.language_model.layers
    assert len(layers) == 18
    images, masks = policy._preprocess_images(batch)
    lang, language_mask = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    prefix = model.embed_prefix(images, masks, lang, language_mask)
    tokens, mask, _ = prefix
    assert tokens.shape[:2] == (1, 3 * 256 + 48)
    assert bool(mask[0, :256].all()) and not bool(mask[0, 256:768].any())
    state = policy.prepare_state(batch)
    noise_seed = int(stable(SEED + "-noise", source["sample_id"])[:15], 16)
    noise = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(noise_seed)).to("cuda")
    native_args = ([batched(x, batch_size) for x in images], [batched(x, batch_size) for x in masks],
                   batched(lang, batch_size), batched(language_mask, batch_size), batched(state, batch_size))
    cached_prefix = tuple(batched(x, batch_size) for x in prefix)
    batch_noise = batched(noise, batch_size)
    pool = source["roi"]
    outputs = {}
    hook_calls = 0

    def invoke(layer=None, keeps=None):
        nonlocal hook_calls
        def mask_hook(_module, _inputs, output):
            nonlocal hook_calls
            raw = output[0] if isinstance(output, tuple) else output
            assert raw.shape[:2] == (batch_size, 816)
            masked = raw.clone()
            for i, k in enumerate(keeps):
                dropped = pool[k:]
                masked[i, dropped, :] = 0
                if audit:
                    untouched = [p for p in range(816) if p not in dropped]
                    assert torch.equal(masked[i, untouched, :], raw[i, untouched, :])
                    assert torch.count_nonzero(masked[i, dropped, :]).item() == 0
            hook_calls += 1
            return (masked, *output[1:]) if isinstance(output, tuple) else masked
        with patch.object(model, "embed_prefix", return_value=cached_prefix):
            if layer is None:
                return model.sample_actions(*native_args, noise=batch_noise)
            handle = layers[layer].register_forward_hook(mask_hook)
            try:
                return model.sample_actions(*native_args, noise=batch_noise)
            finally:
                handle.remove()

    reference = invoke()
    assert reference.shape == (batch_size, 50, 32)
    assert torch.isfinite(reference).all()
    if audit:
        # Prove the prefix cache and native unmasked inference agree within the
        # float32 tolerance for different vision batch sizes.
        ordinary = model.sample_actions(*native_args, noise=batch_noise)
        torch.testing.assert_close(ordinary, reference, atol=1e-5, rtol=1e-5)
    mask_counts = [(len(pool) * point + 5) // 10 for point in KEEP]
    counts = sorted(set(mask_counts) - {len(pool)})
    for layer in LAYERS:
        measured = {len(pool): 0.0}
        for start in range(0, len(counts), batch_size - 1):
            keep = counts[start:start + batch_size - 1]
            complete = keep + [len(pool)] * (batch_size - len(keep))
            prediction = invoke(layer, complete)
            assert torch.isfinite(prediction).all()
            assert torch.equal(prediction[len(keep):], reference[len(keep):]), (source["sample_id"], layer)
            distance = (prediction - reference).float().square().mean(dim=(1, 2)).cpu().tolist()
            measured.update(zip(keep, distance[:len(keep)], strict=True))
        outputs[str(layer)] = [measured[k] for k in mask_counts]
    assert hook_calls == len(LAYERS) * ((len(counts) + batch_size - 2) // (batch_size - 1))
    return {
        "sample_id": source["sample_id"], "ordinal": source["ordinal"], "shard": source["original_source"]["shard"],
        "roi_tokens": len(pool), "actual_keep_counts": mask_counts, "noise_seed": noise_seed,
        "layers": outputs, "keep1_bit_exact_same_batch": True, "hook_calls": hook_calls,
    }


def run(root, start, count, batch, gpu_uuid):
    import torch
    from sentencepiece import SentencePieceProcessor
    from deepvision_vla.runtime.gpu_lease import gpu_lease

    assert 0 <= start < count <= 150 and batch >= 2
    plan_path = root / "COHORT150.json"
    plan = read(plan_path)
    plan_sha = digest(plan_path)
    if plan["model_sha256"] != MODEL_SHA or plan["revision"] != REVISION:
        raise ValueError("The run plan identifies a different checkpoint")
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    device_row = subprocess.check_output(["nvidia-smi","-i",gpu_uuid,"--query-gpu=uuid,memory.used","--format=csv,noheader,nounits"],text=True).strip().split(",")
    assert device_row[0] == gpu_uuid and int(device_row[1]) < 100
    processes = subprocess.check_output(["nvidia-smi","-i",gpu_uuid,"--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
    assert not processes, "Choose an idle GPU"
    index, pci = [x.strip() for x in subprocess.check_output(
        ["nvidia-smi","-i",gpu_uuid,"--query-gpu=index,pci.bus_id","--format=csv,noheader"],text=True).strip().split(",")]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != index:
        raise ValueError(f"Set CUDA_VISIBLE_DEVICES={index} for the selected GPU")
    from deepvision_vla.diagnostics.resources import verify_cuda_pci
    tokenizer_path = root / "paligemma_tokenizer.model"
    tokenizer_sha = digest(tokenizer_path)
    if tokenizer_sha != TOKENIZER_SHA:
        raise ValueError("Use the verified PaliGemma tokenizer from fetch")
    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
    with gpu_lease(gpu_uuid):
        policy = load_policy(root)
        assert torch.cuda.device_count() == 1
        assert not subprocess.check_output(["nvidia-smi","-i",gpu_uuid,"--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
        verify_cuda_pci(pci)
        policy.to("cuda")
        for index, source in enumerate(plan["sources"][start:count], start=start):
            target = out / f"{source['sample_id']}.json"
            if target.exists():
                saved = read(target)
                assert saved["plan_sha256"] == plan_sha and saved["batch_size"] == batch
                assert saved["tokenizer_sha256"] == tokenizer_sha and saved["model_sha256"] == MODEL_SHA
                continue
            start_time = time.monotonic()
            with torch.inference_mode():
                result = measure(policy, tokenizer, source, plan, batch, audit=index == 0)
            result.update(plan_sha256=plan_sha, batch_size=batch, device=gpu_uuid,
                          tokenizer_sha256=tokenizer_sha, model_sha256=MODEL_SHA,
                          seconds=time.monotonic()-start_time, cuda_peak_bytes=torch.cuda.max_memory_allocated())
            write(target, result)
            print(json.dumps({k: result[k] for k in ("ordinal","hook_calls","seconds","cuda_peak_bytes")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--start", type=int, default=0, help="Zero-based first cohort row.")
    parser.add_argument("--count", type=int, default=16, help="Exclusive cohort row limit.")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    if args.action == "fetch":
        fetch(args.root)
    elif args.action == "prepare":
        if args.runtime is None:
            parser.error("prepare requires --runtime")
        prepare(args.root, args.runtime)
    else:
        if not args.gpu_uuid:
            parser.error("run requires --gpu-uuid")
        run(args.root, args.start, args.count, args.batch, args.gpu_uuid)


if __name__ == "__main__":
    main()
