"""Recompute a fixed-step Grad-CAM map with official LeRobot Pi0 base."""

import argparse
from dataclasses import asdict
import gc
import os
from pathlib import Path

from .adapters.openvla_gradients import freeze_parameters
from .geometry import VisualLayout
from .gradient_output_capture import capture_residual_output, token_gradcam
from .pi0_base import (
    LAYERS, MODEL_SHA, REVISION, RUNTIME_FILES, SEED, TOKENIZER_SHA,
    digest, load_policy, prepared_batch, read, stable, state_at, write,
)
from .resources import require_available, verify_cuda_pci
from deepvision_vla.runtime.gpu_lease import gpu_lease

TARGET_NAME = "pi0_base_single_flow_step_native_velocity_squared_mean"


def source_from_runtime(plan, ordinal):
    if plan["model_sha256"] != MODEL_SHA or plan["revision"] != REVISION:
        raise ValueError("Prepare the plan for the pinned official Pi0 base weights")
    runtime = Path(plan["runtime"])
    panel = runtime / "panel"
    inventory = panel / "inventory.json"
    if digest(inventory) != RUNTIME_FILES[inventory.name]:
        raise ValueError("Use the verified BridgeData V2 v55 runtime inventory")
    matches = [row for row in read(inventory) if row["ordinal"] == ordinal]
    if len(matches) != 1:
        raise ValueError(f"Expected one BridgeV2 row for ordinal {ordinal}")
    source = matches[0]
    return {**source, "state": state_at(panel, source).tolist()}


def measure(policy, tokenizer, source, plan):
    import torch
    from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks, prepare_attention_masks_4d
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    flow = policy.model
    language_model = flow.paligemma_with_expert.paligemma.model.language_model
    layers = language_model.layers
    if len(layers) != 18 or list(source["image_size"]) != [256, 256]:
        raise ValueError("Expected the verified 18-layer decoder and square Bridge camera image")
    batch = prepared_batch(policy, tokenizer, source, plan, "cuda")
    with torch.no_grad():
        images, masks = policy._preprocess_images(batch)
        prefix, pads, attention = flow.embed_prefix(
            images, masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        )
        state = policy.prepare_state(batch)
    positions = tuple(range(256))
    if (tuple(prefix.shape[:2]) != (1, 816) or not bool(pads[0, :256].all())
            or bool(pads[0, 256:768].any())):
        raise RuntimeError("Expected one observed 16 x 16 camera grid followed by two masked views")
    layout = VisualLayout(
        image_sha256=source["image_sha256"],
        source_size=(256, 256),
        crop_xyxy=(0, 0, 256, 256),
        canvas_size=(224, 224),
        content_xyxy=(0, 0, 224, 224),
        grid_shape=(16, 16),
        token_positions=positions,
    )
    noise_seed = int(stable(SEED + "-noise", source["sample_id"])[:15], 16)
    noise = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(noise_seed)).to("cuda")
    time = torch.full((1,), 0.5, device="cuda", dtype=torch.float32)
    mask = prepare_attention_masks_4d(make_att_2d_masks(pads, attention))
    position_ids = torch.cumsum(pads, dim=1) - 1
    language_model.config._attn_implementation = "eager"

    def velocity():
        _, cache = flow.paligemma_with_expert.forward(
            attention_mask=mask, position_ids=position_ids,
            past_key_values=None, inputs_embeds=[prefix, None], use_cache=True,
        )
        return flow.denoise_step(
            state=state, prefix_pad_masks=pads, past_key_values=cache, x_t=noise, timestep=time
        )

    with torch.no_grad():
        reference = velocity()
    if reference.shape != (1, 50, 32) or not bool(torch.isfinite(reference).all()):
        raise RuntimeError("Expected the finite native Pi0 flow output [1, 50, 32]")

    rows = []
    with freeze_parameters(policy.parameters()):
        for layer in LAYERS:
            with capture_residual_output(
                layers[layer], expected_shape=tuple(prefix.shape), target_name=TARGET_NAME
            ) as captured:
                prediction = velocity()
                torch.testing.assert_close(prediction.detach(), reference, atol=1e-5, rtol=1e-5)
                scalar = prediction.float().square().mean()
                scalar.backward()
                contribution = token_gradcam(captured, positions, target_name=TARGET_NAME)
                rows.append({
                    "layer": layer, "status": "complete", "target_name": TARGET_NAME,
                    "scalar_value": float(scalar.detach().cpu()),
                    "captured_forward_calls": captured.forward_calls,
                    "captured_backward_calls": captured.backward_calls,
                    "contribution": {
                        key: value.tolist() if isinstance(value, torch.Tensor) else value
                        for key, value in contribution.items()
                    },
                })
            del scalar, prediction, captured
            print(f"Pi0 base layer {layer}: verified output and backward", flush=True)
    return {
        "model": "pi0_base", "model_id": "lerobot/pi0_base",
        "model_revision": REVISION, "model_sha256": MODEL_SHA,
        "image_sha256": source["image_sha256"], "instruction": source["instruction"],
        "sample_id": source["sample_id"], "ordinal": source["ordinal"],
        "layout": asdict(layout), "probes": rows, "noise_seed": noise_seed,
        "target_formula": "mean native padded flow velocity squared at t=0.5",
        "state_input": "standardized Bridge EEF pose and gripper proxy",
        "unhooked_velocity_matches": True, "gt_mse": None,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Pi0 base fetch/prepare root")
    parser.add_argument("--ordinal", type=int, required=True, help="BridgeV2 runtime sample ordinal")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output already exists")

    from sentencepiece import SentencePieceProcessor

    plan_path = args.root / "COHORT150.json"
    plan = read(plan_path)
    source = source_from_runtime(plan, args.ordinal)
    tokenizer_path = args.root / "paligemma_tokenizer.model"
    if digest(tokenizer_path) != TOKENIZER_SHA:
        raise ValueError("Use the verified PaliGemma tokenizer")
    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
    device = require_available(args.gpu, minimum_free_bytes=32_000_000_000)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with gpu_lease(device["uuid"]):
        policy = load_policy(args.root)
        try:
            require_available(args.gpu, device["uuid"], minimum_free_bytes=32_000_000_000)
            verify_cuda_pci(device["pci"])
            policy.to("cuda")
            result = measure(policy, tokenizer, source, plan)
        finally:
            del policy
            gc.collect()
            import torch

            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()

    result.update(status="complete", lease_released=True, state_plan_sha256=digest(plan_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
