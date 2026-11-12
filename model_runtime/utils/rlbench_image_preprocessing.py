"""Shared RLBench image preprocessing contracts for training and evaluation."""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as torchvision_F


IMAGE_AUG_SCOPES = {"qwen_only", "shared"}
DINO_EVAL_SOURCES = {"raw", "shared_native", "qwen_resized"}
EVAL_IMAGE_PROTOCOLS = {
    "matched",
    "matched_no_crop",
    "paper_bug",
    "split_legacy",
}


def load_rgb_image(image_or_path: Any) -> Image.Image:
    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB")
    return Image.open(image_or_path).convert("RGB")


def validate_crop_scale(crop_scale: float) -> float:
    crop_scale = float(crop_scale)
    if not 0.0 < crop_scale <= 1.0:
        raise ValueError(f"crop_scale must be in (0, 1], got {crop_scale}")
    return crop_scale


def random_crop_resize_color_jitter(
    image_or_path: Any,
    *,
    crop_scale: float = 0.9,
    rng: Any = random,
) -> Image.Image:
    """Apply the historical RLBench augmentation without changing its RNG order."""

    crop_scale = validate_crop_scale(crop_scale)
    image = load_rgb_image(image_or_path)
    width, height = image.size
    crop_size = max(1, int(min(width, height) * crop_scale))
    left = rng.randint(0, width - crop_size)
    top = rng.randint(0, height - crop_size)

    image = torchvision_F.crop(image, top, left, crop_size, crop_size)
    image = torchvision_F.resize(image, (height, width))
    image = torchvision_F.adjust_brightness(image, 1.0 + rng.uniform(-0.2, 0.2))
    image = torchvision_F.adjust_contrast(image, rng.uniform(0.8, 1.2))
    image = torchvision_F.adjust_saturation(image, rng.uniform(0.8, 1.2))
    image = torchvision_F.adjust_hue(image, rng.uniform(-0.05, 0.05))
    return image


def prepare_training_image_views(
    image_or_path: Any,
    *,
    image_aug: int,
    image_aug_scope: str,
    crop_scale: float,
    rng: Any = random,
) -> Tuple[Image.Image, Image.Image]:
    if image_aug_scope not in IMAGE_AUG_SCOPES:
        raise ValueError(
            f"image_aug_scope must be one of {sorted(IMAGE_AUG_SCOPES)}, got {image_aug_scope!r}"
        )
    raw_image = load_rgb_image(image_or_path)
    if int(image_aug):
        qwen_native = random_crop_resize_color_jitter(
            raw_image,
            crop_scale=crop_scale,
            rng=rng,
        )
    else:
        qwen_native = raw_image
    dino_native = qwen_native if int(image_aug) and image_aug_scope == "shared" else raw_image
    return qwen_native, dino_native


def center_crop_resize(
    image_or_path: Any,
    *,
    crop_scale: float,
) -> Tuple[Image.Image, Dict[str, float | int]]:
    image = load_rgb_image(image_or_path)
    width, height = image.size
    if crop_scale <= 0.0 or crop_scale >= 1.0:
        crop_size = min(width, height)
        return image, {
            "crop_scale": 0.0 if crop_scale <= 0.0 else 1.0,
            "crop_size": crop_size,
            "effective_linear_scale": 1.0,
        }

    crop_scale = validate_crop_scale(crop_scale)
    crop_size = max(1, int(min(width, height) * crop_scale))
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    cropped = image.crop((left, top, left + crop_size, top + crop_size)).resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )
    return cropped, {
        "crop_scale": crop_scale,
        "crop_size": crop_size,
        "effective_linear_scale": min(width, height) / crop_size,
    }


def resize_qwen_image(image_or_path: Any, qwen_image_size: Optional[int]) -> Image.Image:
    image = load_rgb_image(image_or_path)
    if qwen_image_size is None:
        return image
    size = int(qwen_image_size)
    if size <= 0:
        raise ValueError(f"qwen_image_size must be positive, got {size}")
    return torchvision_F.resize(
        image,
        (size, size),
        interpolation=InterpolationMode.BICUBIC,
    )


def resolve_eval_dino_source(
    *,
    image_aug: int,
    image_aug_scope: str,
    dino_raw_override: Optional[bool],
) -> str:
    if image_aug_scope not in IMAGE_AUG_SCOPES:
        raise ValueError(
            f"image_aug_scope must be one of {sorted(IMAGE_AUG_SCOPES)}, got {image_aug_scope!r}"
        )
    if dino_raw_override is True:
        return "raw"
    if dino_raw_override is False:
        return "qwen_resized"
    if int(image_aug) and image_aug_scope == "shared":
        return "shared_native"
    return "raw"


def resolve_eval_image_protocol(
    *,
    protocol: str,
    image_aug: int,
    image_aug_scope: str,
    crop_scale: float,
    qwen_image_size: Optional[int],
    dinov3_image_size: Optional[int],
    use_dinov3: int,
    profile_camera_resolution: int,
) -> Dict[str, float | int | str]:
    """Resolve camera, deterministic crop, and DINO image source together."""

    protocol = str(protocol)
    if protocol not in EVAL_IMAGE_PROTOCOLS:
        raise ValueError(
            f"protocol must be one of {sorted(EVAL_IMAGE_PROTOCOLS)}, got {protocol!r}"
        )
    if image_aug_scope not in IMAGE_AUG_SCOPES:
        raise ValueError(
            f"image_aug_scope must be one of {sorted(IMAGE_AUG_SCOPES)}, "
            f"got {image_aug_scope!r}"
        )
    profile_camera_resolution = int(profile_camera_resolution)
    if profile_camera_resolution <= 0:
        raise ValueError(
            "profile_camera_resolution must be positive, "
            f"got {profile_camera_resolution}"
        )
    if qwen_image_size is not None and int(qwen_image_size) <= 0:
        raise ValueError(f"qwen_image_size must be positive, got {qwen_image_size}")
    if dinov3_image_size is not None and int(dinov3_image_size) <= 0:
        raise ValueError(f"dinov3_image_size must be positive, got {dinov3_image_size}")

    matched_crop = validate_crop_scale(crop_scale) if int(image_aug) else 0.0
    if protocol == "paper_bug":
        camera_resolution = (
            int(dinov3_image_size or profile_camera_resolution)
            if int(use_dinov3)
            else int(qwen_image_size or profile_camera_resolution)
        )
        center_crop_scale = 0.0
        dino_source = "qwen_resized" if int(use_dinov3) else "raw"
    elif protocol == "matched_no_crop":
        camera_resolution = profile_camera_resolution
        center_crop_scale = 0.0
        dino_source = "raw"
    elif protocol == "split_legacy":
        camera_resolution = profile_camera_resolution
        center_crop_scale = matched_crop
        dino_source = "raw"
    else:
        camera_resolution = profile_camera_resolution
        center_crop_scale = matched_crop
        dino_source = resolve_eval_dino_source(
            image_aug=image_aug,
            image_aug_scope=image_aug_scope,
            dino_raw_override=None,
        )

    return {
        "eval_image_protocol": protocol,
        "camera_resolution": camera_resolution,
        "center_crop_scale": center_crop_scale,
        "dino_source": dino_source,
        "qwen_image_size": int(qwen_image_size) if qwen_image_size is not None else 0,
        "profile_camera_resolution": profile_camera_resolution,
    }


def prepare_eval_image_views(
    image_or_path: Any,
    *,
    crop_scale: float,
    qwen_image_size: Optional[int],
    dino_source: str,
) -> Tuple[Image.Image, Image.Image, Dict[str, float | int | str]]:
    if dino_source not in DINO_EVAL_SOURCES:
        raise ValueError(
            f"dino_source must be one of {sorted(DINO_EVAL_SOURCES)}, got {dino_source!r}"
        )

    raw_image = load_rgb_image(image_or_path)
    shared_native, metadata = center_crop_resize(raw_image, crop_scale=crop_scale)
    qwen_image = resize_qwen_image(shared_native, qwen_image_size)

    if dino_source == "raw":
        dino_image = raw_image
    elif dino_source == "shared_native":
        dino_image = shared_native
    else:
        dino_image = qwen_image

    metadata = dict(metadata)
    metadata.update(
        {
            "dino_source": dino_source,
            "raw_width": raw_image.width,
            "raw_height": raw_image.height,
            "qwen_width": qwen_image.width,
            "qwen_height": qwen_image.height,
            "dino_width": dino_image.width,
            "dino_height": dino_image.height,
        }
    )
    return qwen_image, dino_image, metadata
