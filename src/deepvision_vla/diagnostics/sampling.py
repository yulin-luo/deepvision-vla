"""Nested, ratio-independent random ordering for a single-layer ROI mask."""

import hashlib
import json
import math
import random


def dropped_positions(positions, *, keep_ratio, sample_id, layer, seed=42):
    if not math.isfinite(keep_ratio) or not 0 <= keep_ratio <= 1:
        raise ValueError("keep_ratio must be in [0, 1]")
    positions = tuple(positions)
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("ROI positions must be nonempty and unique")
    if any(type(x) is not int or x < 0 for x in positions):
        raise ValueError("ROI positions must be nonnegative integers")
    if not sample_id or type(layer) is not int or layer < 0 or type(seed) is not int:
        raise ValueError("Sample, layer and seed must be explicit")
    # The same subset is used at different depths so layer comparisons do not
    # accidentally change which visual evidence was removed.
    key = json.dumps([str(sample_id), seed], separators=(",", ":"))
    order = sorted(positions)
    random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), "big")).shuffle(order)
    count = round(len(order) * keep_ratio)
    return tuple(sorted(order[count:]))
