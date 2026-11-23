# BridgeData V2 visual diagnostics

This package runs layer-wise ROI masking and renders source-matched token
Grad-CAM maps for the BridgeData V2 diagnostic. The source index and
annotations identify each episode, sampled frame, instruction and reviewed ROI.
Prepare the same source panel for each model.

- [Verify and redraw the saved results](QUICKSTART.md)
- [Install OpenVLA and official Pi0 base](install/README.md)
- [Rebuild the prepared inputs and run the diagnostic](PUBLIC_CORE.md)
- [Review the measurement and coverage protocol](PROTOCOL.md)
- [Redraw figures and verify source-level statistics](FORMAL_REPORTING.md)
- [Redraw a three-model attribution example](../../examples/bridgev2-attribution/README.md)

The model guides identify the public checkpoints used for the
reproduction. QwenVLA-OFT H16 uses a separately controlled checkpoint. Its
saved evaluation values measure prediction drift and can be verified offline.

[Recompute the three-model Grad-CAM example](ATTRIBUTION.md) · [Bridge action alignment and Qwen first-step metrics](BRIDGE_ACTION_ALIGNMENT.md)
