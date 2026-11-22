"""Project original-image ROI boxes into a particular model's visual tokens."""

from dataclasses import dataclass
import math
import re


def _box(box):
    values = tuple(float(x) for x in box)
    if len(values) != 4 or not all(math.isfinite(x) for x in values):
        raise ValueError("A box needs four finite xyxy coordinates")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("A box must have positive area")
    return values


@dataclass(frozen=True)
class VisualLayout:
    image_sha256: str
    source_size: tuple[int, int]
    crop_xyxy: tuple[float, float, float, float]
    canvas_size: tuple[int, int]
    content_xyxy: tuple[float, float, float, float]
    grid_shape: tuple[int, int]
    token_positions: tuple[int, ...]

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{64}", self.image_sha256):
            raise ValueError("The layout must identify the source image")
        for pair in (self.source_size, self.canvas_size, self.grid_shape):
            if len(pair) != 2 or any(type(x) is not int or x <= 0 for x in pair):
                raise ValueError("Sizes and grid dimensions must be positive integers")
        crop, content = _box(self.crop_xyxy), _box(self.content_xyxy)
        sw, sh = self.source_size
        cw, ch = self.canvas_size
        if not (0 <= crop[0] < crop[2] <= sw and 0 <= crop[1] < crop[3] <= sh):
            raise ValueError("Crop lies outside the source image")
        if not (0 <= content[0] < content[2] <= cw and 0 <= content[1] < content[3] <= ch):
            raise ValueError("Resized image lies outside the model canvas")
        rows, cols = self.grid_shape
        if len(self.token_positions) != rows * cols:
            raise ValueError("Token positions must match the actual visual grid")
        if any(type(x) is not int or x < 0 for x in self.token_positions):
            raise ValueError("Token positions must be nonnegative integers")
        if len(set(self.token_positions)) != len(self.token_positions):
            raise ValueError("Visual token positions cannot overlap")


def project_roi_boxes(boxes, *, image_sha256, layout, membership="positive_intersection"):
    """Project boxes using an explicit cell-membership rule.

    Boxes are original-image pixels. The supplied layout describes the actual
    model processor's crop, resize, padding and visual sequence positions.
    The original adapter canaries used positive cell intersection. Recovered
    historical ROI code uses inclusive patch-center membership instead; callers
    reproducing that rule must request ``patch_center`` explicitly. A small
    box containing no patch center stays empty, without a nearest-token fallback.
    """
    if membership not in {"positive_intersection", "patch_center"}:
        raise ValueError("Unknown ROI token-membership rule")
    if image_sha256 != layout.image_sha256:
        raise ValueError("ROI and model layout refer to different images")
    if not boxes:
        raise ValueError("No ROI boxes; do not substitute full-image masking")
    x0, y0, x1, y1 = layout.crop_xyxy
    u0, v0, u1, v1 = layout.content_xyxy
    mapped = []
    for box in boxes:
        a, b, c, d = _box(box)
        a, b, c, d = max(a, x0), max(b, y0), min(c, x1), min(d, y1)
        if a >= c or b >= d:
            continue
        mapped.append((u0+(a-x0)*(u1-u0)/(x1-x0),
                       v0+(b-y0)*(v1-v0)/(y1-y0),
                       u0+(c-x0)*(u1-u0)/(x1-x0),
                       v0+(d-y0)*(v1-v0)/(y1-y0)))
    rows, cols = layout.grid_shape
    width, height = layout.canvas_size
    positions = []
    for row in range(rows):
        for col in range(cols):
            left, top = col*width/cols, row*height/rows
            right, bottom = (col+1)*width/cols, (row+1)*height/rows
            if membership == "patch_center":
                cx, cy = (left+right)/2, (top+bottom)/2
                selected = any(a <= cx <= c and b <= cy <= d for a, b, c, d in mapped)
            else:
                selected = any(min(right, c) > max(left, a) and min(bottom, d) > max(top, b)
                               for a, b, c, d in mapped)
            if selected:
                positions.append(layout.token_positions[row*cols+col])
    return tuple(positions)
