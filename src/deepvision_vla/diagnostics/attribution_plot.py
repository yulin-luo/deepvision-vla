"""Redraw token Grad-CAM on the original BridgeV2 image from completed probe results."""

import argparse
from dataclasses import fields
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from textwrap import wrap

from .formal_reporting import LAYERS
from .formal_reporting_plots import MODEL_LABELS, _alignment
from .geometry import VisualLayout
from .model_geometry import actual_visual_layout


TARGETS = {
    "openvla": ("teacher_forced_self_generated_action_token_ce", "Action token CE"),
    "pi0": ("single_flow_step_spatial_velocity_squared_mean", "Flow velocity"),
    "qwen_h16": ("normalized_h16_action_head_l2_mean", "H16 action head"),
}


def load_probe(payload, image_hash, image_size):
    import numpy as np

    probe = json.loads(payload)
    model = probe.get("model")
    if model not in TARGETS or probe.get("status") != "complete" or probe.get("lease_released") is not True:
        raise ValueError("A completed, released native attribution result is required")
    if probe["image_sha256"] != image_hash:
        raise ValueError("The probe and original image SHA256 differ")

    metadata = probe["layout"]
    if model == "qwen_h16":
        layout = actual_visual_layout(
            {"image_sha256": image_hash, "image_size": list(image_size)}, metadata["adapter"])
        if tuple(metadata["grid_shape"]) != tuple(layout.grid_shape):
            raise ValueError("Stored and reconstructed visual grids differ")
    else:
        layout = VisualLayout(**{field.name: metadata[field.name] for field in fields(VisualLayout)})
    if tuple(layout.source_size) != tuple(image_size) or layout.image_sha256 != image_hash:
        raise ValueError("The visual layout does not match the original image")

    expected_target = TARGETS[model][0]
    layers = []
    scores = []
    for row in probe["probes"]:
        contribution = row["contribution"]
        if row["status"] != "complete" or row["target_name"] != expected_target:
            raise ValueError("Native attribution target or layer status changed")
        if tuple(contribution["visual_positions"]) != tuple(layout.token_positions):
            raise ValueError("Attribution token ordering differs from the image grid")
        signed = np.asarray(contribution["raw_signed"], dtype=float)
        relu = np.asarray(contribution["raw_relu"], dtype=float)
        if (signed.shape != (len(layout.token_positions),) or not np.isfinite(signed).all()
                or not np.array_equal(relu, np.maximum(signed, 0))):
            raise ValueError("Expected finite token values and their exact positive part")
        layers.append(row["logical_output_layer"] if model == "pi0" else row["layer"])
        scores.append(relu.reshape(layout.grid_shape))
    if tuple(layers) != LAYERS[model]:
        raise ValueError("Expected the four fixed native layer positions in order")
    return model, probe, layout, scores


def source_overlay(scores, layout):
    """Map source-pixel centers to native visual cells without interpolation."""
    import numpy as np

    x0, y0, x1, y1 = layout.crop_xyxy
    u0, v0, u1, v1 = layout.content_xyxy
    width, height = layout.source_size
    canvas_width, canvas_height = layout.canvas_size
    rows, cols = layout.grid_shape
    xs = np.arange(width) + .5
    ys = np.arange(height) + .5
    visible_x = (xs >= x0) & (xs < x1)
    visible_y = (ys >= y0) & (ys < y1)
    token_x = ((u0 + (xs[visible_x] - x0) * (u1-u0) / (x1-x0)) * cols / canvas_width).astype(int)
    token_y = ((v0 + (ys[visible_y] - y0) * (v1-v0) / (y1-y0)) * rows / canvas_height).astype(int)
    overlay = np.zeros((height, width), dtype=float)
    overlay[np.ix_(visible_y, visible_x)] = scores[np.ix_(token_y, token_x)]
    return overlay


def normalized_positive_scores(values, model_peak, scale):
    """Map positive token values to the figure's display scale."""
    if scale == "model":
        return values / model_peak if model_peak else values
    if scale != "layer":
        raise ValueError(f"Unknown attribution scale: {scale}")
    low, high = values.min(), values.max()
    spread = high - low
    return (values - low) / spread if spread else values - low


def draw(image, bundles, *, scale="model"):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    source = np.asarray(image.convert("L"))
    instruction = " ".join(bundles[0][1]["instruction"].split())
    caption = wrap(f"BridgeV2 diagnostic example  |  {instruction}", width=68)
    header_extra = .16*(len(caption)-1)
    fig, axes = plt.subplots(len(bundles), 4, figsize=(7.2, 1.48*len(bundles)+1.28+header_extra),
                             squeeze=False)
    height = fig.get_figheight()
    fig.subplots_adjust(left=.17, right=.97, top=1-(.73+header_extra)/height,
                        bottom=.83/height, wspace=.07, hspace=.33)
    fig.text(.17, 1-.22/height, "Visual contribution across layers",
             fontsize=15, fontweight="bold", va="top")
    fig.text(.17, 1-.44/height, "\n".join(caption),
             fontsize=9, color="#475569", va="top")
    cmap = mpl.colormaps["inferno"]
    peaks = {}
    for (model, _, layout, scores), row in zip(bundles, axes, strict=True):
        peak = float(max(values.max() for values in scores))
        peaks[model] = peak
        for layer, values, axis in zip(LAYERS[model], scores, row, strict=True):
            intensity = source_overlay(normalized_positive_scores(values, peak, scale), layout)
            axis.imshow(source, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axis.imshow(intensity, cmap=cmap, vmin=0, vmax=1, alpha=.82*np.sqrt(intensity),
                        interpolation="nearest")
            flat = " (flat)" if scale == "layer" and values.min() == values.max() else ""
            axis.set_title(f"Layer {layer}{flat}", fontsize=8, pad=4)
            axis.set_axis_off()
        row[0].text(-.09, .55, MODEL_LABELS[model], transform=row[0].transAxes,
                    ha="right", va="center", fontsize=9, fontweight="bold")
        row[0].text(-.09, .43, TARGETS[model][1], transform=row[0].transAxes,
                    ha="right", va="center", fontsize=7, color="#475569")
    scale_axis = fig.add_axes([.30, .41/fig.get_figheight(), .62, .12/fig.get_figheight()])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap),
                            cax=scale_axis, orientation="horizontal", ticks=(0, .5, 1))
    colorbar.ax.tick_params(length=2, labelsize=7)
    colorbar.outline.set_visible(False)
    scale_label = {
        "model": "Positive Grad-CAM / maximum across the four layers of each model",
        "layer": "Positive Grad-CAM rescaled to each layer's own minimum and maximum",
    }
    colorbar.set_label(scale_label[scale], fontsize=8)
    return fig, axes, scale_axis, peaks


def plot(image_path, result_paths, output, *, scale="model"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    original = Path(image_path).read_bytes()
    image_hash = sha256(original).hexdigest()
    image = Image.open(BytesIO(original)).convert("RGB")
    records = {}
    input_hashes = {}
    for path in result_paths:
        payload = Path(path).read_bytes()
        bundle = load_probe(payload, image_hash, image.size)
        model = bundle[0]
        if model in records:
            raise ValueError(f"Multiple results for {model}")
        records[model] = bundle
        input_hashes[model] = sha256(payload).hexdigest()
    bundles = [records[model] for model in LAYERS if model in records]
    if len({(p["sample_id"], p["instruction"]) for _, p, _, _ in bundles}) != 1:
        raise ValueError("Attribution results must use the same source and instruction")
    out = Path(output)
    out.mkdir(exist_ok=False)
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        fig, axes, _, peaks = draw(image, bundles, scale=scale)
        try:
            alignment = _alignment(fig, list(axes.flat), columns=axes.shape[1])
            for extension in ("pdf", "png"):
                fig.savefig(out / f"attribution.{extension}", dpi=300, facecolor="white")
        finally:
            plt.close(fig)
    ranges = {
        model: {str(layer): {"min": float(values.min()), "max": float(values.max())}
                for layer, values in zip(LAYERS[model], scores, strict=True)}
        for model, _, _, scores in bundles
    }
    receipt = {"scope": "single_source_native_attribution",
               "source_image_sha256": image_hash, "result_sha256": input_hashes,
               "native_positive_peaks": peaks, "native_positive_ranges_by_layer": ranges,
               "color_scale": scale, "alignment": alignment,
               "artifacts": {name: sha256((out / name).read_bytes()).hexdigest()
                             for name in ("attribution.pdf", "attribution.png")}}
    (out / "RENDER_RECEIPT.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", choices=("model", "layer"), default="model",
                        help="Shared model maximum (default) or per-layer min–max")
    args = parser.parse_args(argv)
    print(json.dumps(plot(args.image, args.result, args.output, scale=args.scale), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
