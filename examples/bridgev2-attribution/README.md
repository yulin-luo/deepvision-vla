# BridgeData V2 contribution maps

<img src="https://deepvision-vla.github.io/static/images/diagnosis-final.png" alt="Paper Figure 2. Layer-wise ROI masking curves and contribution maps for OpenVLA, Pi0, and QwenVLA-OFT" width="880" />

[Figure 2 from the paper](https://arxiv.org/html/2603.15618v1#S3.SS2) · [Full-resolution figure](../../assets/bridgev2-paper-figure2.png)

## Runnable attribution example

The saved arrays below use the cloth-folding frame. They can be redrawn on CPU for all three models.

[OpenVLA and QwenVLA-OFT preview PDF](attribution.pdf) · [Render manifest](RENDER_RECEIPT.json)

This example contains the original image and the signed and positive token Grad-CAM values for four layers of each model. All three models read the same BridgeData V2 frame and the instruction “fold the cloth from the right to the left side.”

- OpenVLA layers 0, 10, 20, and 30 use cross-entropy on the model's generated action tokens.
- Bridge π₀ layers 0, 6, 12, and 16 use the squared spatial flow velocity at one fixed denoising step.
- QwenVLA-OFT layers 0, 12, 24, and 34 use the mean squared output of the full normalized H16 action head.

The figure divides the positive Grad-CAM values by the maximum across the four layers of each model. The maps describe this one example. The models use different attribution targets and the plot scale is shared only within each model.

## Redraw

From a checkout with `deepvision_vla.diagnostics`, use a Python environment with NumPy 2.3.5, Matplotlib 3.10.7, and Pillow 13.0.0. The command reads the saved contribution arrays and runs on CPU. Run the commands below from the repository root.

```bash
export PYTHONPATH="$PWD/src"
EXAMPLE_DIR="$PWD/examples/bridgev2-attribution"
FIGURE_DIR="$PWD/bridgev2-attribution-redraw"
(cd "$EXAMPLE_DIR" && sha256sum -c SHA256SUMS)
python -m deepvision_vla.diagnostics plot attribution \
  --image "$EXAMPLE_DIR/source.jpg" \
  --result "$EXAMPLE_DIR/openvla.json" \
  --result "$EXAMPLE_DIR/pi0.json" \
  --result "$EXAMPLE_DIR/qwen_h16.json" \
  --output "$FIGURE_DIR"
```

For the per-layer min–max view, run the same command with `--scale layer` and a
new output directory. This version shows where positive contribution is located
within each map. Use the shared-scale view to compare strengths across a
model's layers.

The output contains `attribution.png`, `attribution.pdf`, and `RENDER_RECEIPT.json`. The frame locator and image digest are in `source.json`.
For a new QwenVLA-OFT Grad-CAM probe, see the [Qwen H16 guide](../../docs/diagnostics/QWEN_H16_ENVIRONMENT.md).

## Image credit

The source frame is from *BridgeData V2: A Dataset for Robot Learning at Scale*, Homer Walke and collaborators, CoRL 2023. `source.jpg` is the reconstructed input frame. The figure displays it in grayscale with DeepVision-VLA diagnostic overlays. [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
