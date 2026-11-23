"""Run native Grad-CAM on one image with the pinned OpenVLA checkpoint."""

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import gc
import json
import time

from .adapters.openvla_runtime import MODEL_ID, MODEL_REVISION, load_openvla_runtime
from .adapters.openvla_output import run_native
from .resources import require_available
from deepvision_vla.runtime.gpu_lease import gpu_lease


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset-receipt", type=Path, required=True)
    parser.add_argument("--official-inventory", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.instruction.strip():
        parser.error("--instruction must contain text")
    if args.output.exists():
        parser.error("--output already exists")
    run(args)
    return 0


def run(args):
    from PIL import Image

    image_bytes = args.image.read_bytes()
    with Image.open(BytesIO(image_bytes)) as decoded:
        image = decoded.convert("RGB")
    sample = {"image_sha256": sha256(image_bytes).hexdigest(), "instruction": args.instruction.strip()}
    selected = require_available(args.gpu, minimum_free_bytes=32_000_000_000)
    state = {
        **sample,
        "model": "openvla",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "status": "starting",
        "lease_released": False,
        "deadline_epoch": time.time() + 1200,
        "probes": [],
        "probes_completed": 0,
        "backward_calls": 0,
        "generation_requests": 0,
        "generation_model_forward_calls": 0,
        "generation_projector_calls": 0,
        "teacher_forward_calls": 0,
        "teacher_projector_calls": 0,
    }
    native = SimpleNamespace(gpu=args.gpu, allow_gpu_sharing=False, loaded_adapter=None)

    def check_deadline():
        if time.time() >= state["deadline_epoch"]:
            raise TimeoutError("Single-image attribution exceeded its time limit")

    with gpu_lease(selected["uuid"]):
        try:
            native.loaded_adapter = load_openvla_runtime(
                args.checkpoint, args.asset_receipt, args.official_inventory
            )
            check_deadline()
            run_native(native, sample, image, None, selected, state, check_deadline)
        finally:
            native.loaded_adapter = None
            gc.collect()
            import torch

            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()

    state.update(status="complete", lease_released=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as file:
        json.dump(state, file, indent=2, allow_nan=False)
        file.write("\n")
    print(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
