"""Matched distinct-token controls; insufficient regions remain unavailable."""

import hashlib
import json
import random

from .geometry import project_roi_boxes


def background_positions(*, layout, image_sha256, task_positions, membership):
    """Exclude task tokens and padding; this defines a spatial background proxy.

    The complement alone does not prove semantic irrelevance. Record the ROI
    and review policy with any background-control result.
    """
    width, height = layout.source_size
    visible = project_roi_boxes([(0, 0, width, height)], image_sha256=image_sha256,
                                layout=layout, membership=membership)
    task = set(task_positions)
    if not task.issubset(layout.token_positions):
        raise ValueError("Task positions are outside the model visual-token layout")
    return tuple(position for position in visible if position not in task)


def matched_control_mask(pool, *, required_count, sample_id, region, seed=42):
    """Select exactly the ROI's dropped-token count, without replacement.

    Ordering depends on sample/region/seed only, never layer or keep ratio.
    Different counts therefore use nested masks across ratios and depths.
    ``region`` identifies the supplied region; it does not annotate objects.
    """
    pool = tuple(pool)
    if (type(required_count) is not int or required_count < 0
            or type(seed) is not int or not sample_id or not region
            or len(set(pool)) != len(pool)
            or any(type(x) is not int or x < 0 for x in pool)):
        raise ValueError("Explicit distinct visual positions and a nonnegative budget are required")
    result = {"region": region, "requested_distinct_tokens": required_count,
              "available_distinct_tokens": len(pool), "sample_id": sample_id, "seed": seed,
              "sampling": "without_replacement_nested_by_sample_region_seed"}
    if required_count > len(pool):
        return {**result, "status": "insufficient_distinct_tokens", "positions": None}
    order = sorted(pool)
    key = json.dumps([sample_id, region, seed], separators=(",", ":"))
    random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), "big")).shuffle(order)
    positions = tuple(sorted(order[-required_count:])) if required_count else ()
    return {**result, "status": "ready", "positions": positions}
