"""Versioned box-area screening; raw detections and all sample IDs are retained."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


MAX_AREA_FRACTION = 0.60
FILTER_REVISION = "source-image-box-area-gt60-v1"


def filter_candidate_annotation(annotation):
    """Reject individual source-pixel boxes exceeding 60%, without certifying ROI.

    Exactly 60% is retained. The limit applies to each box, not the union of
    several boxes. Invalid/out-of-image boxes are rejected separately, never
    clipped into valid ones. Empty results remain in the sample inventory.
    """
    if annotation.get("coordinate_space") != "source_image_pixels":
        raise ValueError("Area filtering requires original-image pixel coordinates")
    size = annotation.get("image_size")
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(type(x) is not int or x <= 0 for x in size)):
        raise ValueError("Original image width and height are required")
    width, height = size
    source = annotation.get("source", {})
    if source.get("image_sha256", annotation.get("image_sha256")) != annotation.get("image_sha256"):
        raise ValueError("ROI and source image identities differ")
    if (source.get("width", width), source.get("height", height)) != (width, height):
        raise ValueError("Conflicting source-image dimensions")
    boxes = annotation.get("candidate_boxes")
    if not isinstance(boxes, list):
        raise ValueError("Explicit candidate_boxes list is required")
    if "area_filter" in annotation or "raw_candidate_boxes" in annotation:
        raise ValueError("Already filtered input; use the immutable raw annotation")
    kept, rejected, indices = [], [], []
    for index, detection in enumerate(boxes):
        box = detection.get("xyxy_px") if isinstance(detection, dict) else None
        valid = (isinstance(box, (list, tuple)) and len(box) == 4
                 and all(type(x) in (int, float) and math.isfinite(x) for x in box)
                 and 0 <= box[0] < box[2] <= width
                 and 0 <= box[1] < box[3] <= height)
        area = (box[2]-box[0])*(box[3]-box[1]) if valid else None
        reason = ("invalid_source_pixel_box" if not valid else
                  "box_area_exceeds_60_percent" if area > MAX_AREA_FRACTION*width*height else None)
        if reason:
            rejected.append({"source_box_index": index, "reason": reason,
                             "box_area_fraction": area/(width*height) if valid else None,
                             "source_detection": copy.deepcopy(detection)})
        else:
            kept.append(copy.deepcopy(detection))
            indices.append(index)
    result = copy.deepcopy(annotation)
    result.update(raw_candidate_boxes=copy.deepcopy(boxes), candidate_boxes=kept,
                  rejected_candidates=rejected, retained_source_box_indices=indices,
                  area_filter={"revision": FILTER_REVISION,
                               "max_area_fraction": MAX_AREA_FRACTION,
                               "comparison": "strictly_greater_than",
                               "coordinate_space": "source_image_pixels",
                               "scope": "individual_box_not_union",
                               "input_box_count": len(boxes), "kept_box_count": len(kept),
                               "rejected_box_count": len(rejected)},
                  roi_filter_status="candidates_remain" if kept else "no_valid_roi_after_area_filter",
                  paper_roi_ready=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Output must not overwrite raw annotations")
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.input.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate source sample identities")
    results = [filter_candidate_annotation(row) for row in rows]
    summary = {"revision": FILTER_REVISION, "max_area_fraction": MAX_AREA_FRACTION,
               "requested": len(rows), "recorded": len(results),
               "source_sha256": hashlib.sha256(raw).hexdigest(),
               "input_boxes": sum(len(row["candidate_boxes"]) for row in rows),
               "kept_boxes": sum(len(row["candidate_boxes"]) for row in results),
               "rejected_boxes": sum(len(row["rejected_candidates"]) for row in results),
               "rows_with_candidates": sum(bool(row["candidate_boxes"]) for row in results),
               "rows_losing_all_candidates": sum(bool(old["candidate_boxes"]) and not new["candidate_boxes"]
                                                  for old, new in zip(rows, results)),
               "detector_calls": 0, "vla_calls": 0, "gpu_calls": 0}
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "annotations.jsonl").open("x", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    summary["output_sha256"] = hashlib.sha256((args.output / "annotations.jsonl").read_bytes()).hexdigest()
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != summary["source_sha256"]:
        raise RuntimeError("Source annotations changed during filtering")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
