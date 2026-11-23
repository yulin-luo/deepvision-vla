# BridgeV2 layer-wise diagnosis protocol

This module separates model prediction, visual-token interventions, action references, and attribution. Use a separate environment for each policy. The existing RLBench inference package is independent of these adapters.

Dataset attribution and ROI annotation scope are described in [Data and annotations](DATA.md).

## Policy interfaces

- OpenVLA uses the pinned OpenVLA-7B implementation. A single prediction generates seven action tokens. The diagnostic layers are zero-based decoder blocks 0, 10, 20, and 30. The observed visual layout has 256 image tokens on a 16-by-16 grid, inserted after BOS.
- Pi0 uses a Bridge-adapted checkpoint and its genuine LeRobot implementation. It predicts the complete four-step, seven-channel action chunk from the recorded end-effector state. Use the same full native noise tensor and diffusion settings for every intervention. The selected prefix blocks are 0, 6, 12, and 16. The observed image grid is 16 by 16.
- QwenVLA-OFT uses a Qwen3-VL backbone and a trained parallel action head. The current adapter preserves 16 action queries and the entire 16-by-7 output. Selected decoder blocks are 0, 12, 24, and 34. Derive its image-token layout from the actual processor output: the validated path produces 64 visual tokens on an 8-by-8 grid, despite an explicit earlier resize to 224. Do not infer the grid from the nominal resize alone.

These are adapter contracts for the validated checkpoints, not interchangeable definitions of every model bearing these names. Checkpoint provenance and preprocessing belong to each run configuration.

## Source data and ROI

Keep source image SHA256, original episode identity, frame index, instruction, complete state/action arrays, and annotation revision together. The current panel contains 1,500 distinct episodes, with a middle frame per episode. It retains 1,340 eligible episodes and includes 160 reviewed replacements from the same seeded shard order.

Boxes use original-image xyxy pixel coordinates. Reject an individual box only when its area exceeds 60% of the image; exactly 60% is retained. An enclosing rectangle or total union area is not a substitute for this per-box rule. Keep rejected candidates and the original sample in the request inventory.

Project accepted boxes through the policy's actual crop, resize, padding, grid, and visual sequence offsets. The recovered historical masking rule includes patch centers inside a box. Explicitly select `patch_center`; other membership rules are separate protocols. An empty projected ROI remains unavailable. Never replace it with the whole image or the nearest token.

AI-reviewed visible robot/object rectangles are proxies for visual relevance. They do not establish contact, physical state identity, or a hidden target's extent. Record unresolved object identity and goal ambiguity separately. AI review is not human verification.

## Intervention and controls

The x-axis is the retained-token fraction, not the masked fraction. Keep ratio 1 must reproduce the unmodified prediction. Use a fixed sample/seed permutation, nested masks across ratios, and identical subsets across layers. Only selected visual-token positions are zeroed; cached autoregressive decoding is distinguished from image prefill.

The frozen v55 primary ROI is the union of the reviewed manipulated-object, robot, and applicable interaction-region boxes. The semantic controls exclude that task footprint. Keep ratio 0 removes the selected task-ROI tokens. Task information can also remain elsewhere in the frame and in the instruction.

The full robot/object proxy and equal-token controls are different comparisons. For controls, select fixed, disjoint subsets of the ROI, another visible object, and an explicitly reviewed background region. Exclude task/context tokens from negative pools. Use the same number of distinct tokens for all three regions on the same eligible sources. If any pool is unavailable, retain the entire comparison as blocked at every ratio, including keep ratio 1.

Report nominal ratios and actual integer token counts. Small pools can give identical masks at several ratios: for example, a one-token pool cannot implement eleven distinct retention levels. Successful identical computations may be referenced by multiple output points, but are not additional independent predictions or additional samples. Preserve failed and unknown computations; never replay them to conceal failures.

### Measured Pi state input

Bind the raw state row and anchor to the actual native preprocessing receipt. The input is the recorded six EEF channels, a zero pad at index 6, and the recorded gripper sensor at index 7. Do not clip the sensor, substitute a zero state, or add an unverified axis/origin transform. This demonstrates that the intended recorded values reach preprocessing; it does not establish a physical camera/world-frame equivalence. A camera-style string in a source inventory is not a replacement for this input proof.

## Action metrics

OpenVLA Bridge-orig references use the source-supported neighboring EEF-state difference for the first six channels and the prescribed command-gripper relabeling. Raw recorded TFDS controls are not automatically GT actions.

Bridge Pi0 uses the source-bound four-step dataset-reference MSE. Its six pose components and gripper command were checked against the aligned Bridge conversion at all 1,500 indexed anchors. Qwen H16 reports prediction drift against its own unmasked output in native and normalized action spaces. The study has yet to recover the historical Qwen single-step label producer and verify its frame binding for the current checkpoint and panel.

Native chunk drift and first-step drift are separate fields. Qwen normalized-output drift is separate from native-action drift. Do not pad a near-terminal reference window, truncate a model to another horizon, fit a post-hoc coordinate transform, or pool raw MSE values across different action spaces.

### Episode-bound references

A full-trajectory numerical crosswalk can establish a reference for a particular source episode: verify every state/action row, the source revision, frame offset, gripper transform, action normalization and complete horizon window. Keep that reference as a separate artifact tied to the original prediction and raw-array hashes. Do not rewrite the original run's unknown-reference fields or promote the result to every episode automatically. A numerical crosswalk is also distinct from comparing encoded image pixels or authenticating the model's historical training revision.

Prediction drift and dataset-reference error answer different questions. A large change from the original prediction can reduce error to a verified reference if the original prediction was poor. Retain both metrics and their channel decompositions, including such counterexamples; never choose a target convention by whichever produces a preferred curve.

## Contribution maps

These are gradient contribution maps, not direct attention weights or attention rollout. The selected scalar remains fixed across layers of the same model:

- OpenVLA: teacher-forced cross entropy of its own greedily generated action-token sequence.
- Pi0: mean squared first-six-channel normalized flow velocity at a declared time and fixed full native noise. This is not a full denoising trajectory or physical velocity error.
- Qwen: mean squared complete normalized action-head output.

Observe actual decoder outputs and preserve the unmodified prediction. For Qwen Layer 0, snapshot the decoder output before the caller's subsequent DeepStack addition; its input already contains initial visual tokens. The visual encoder's DeepStack source-layer numbers must not be confused with language-decoder injection indexes. Pi's manual implementation requires a source-proven output seam rather than an unused decoder-module hook.

Keep raw signed maps, ReLU maps, gradient target, hook location, zero/constant flags, and image transforms. The recovered display uses per-layer min-max scaling, bilinear resizing, Jet, and a fixed 0.45 overlay. This reveals relative hotspot positions but hides response-magnitude differences. Preserve a shared-scale rendering when comparing layer strength. Never change the scalar target, swap layer labels, or synthesize hotspots to resemble a reference figure.

## Results and reproduction scope

Keep requested, blocked, actual/unknown attempts, completed independent computations, aliases, errors, and missing records. Report source and geometry matched/unmatched/missing separately. Means must state the eligible source cohort; full-ROI and matched-control means may cover different numbers of sources.

The validated engineering pilot uses 16 fixed sources, all four model-specific layers, and keep ratios 0, 0.5, and 1. The formal run covers all 1,500 requested identities at eleven retention ratios. The final report combines results for the retained episodes with a separate run for the replacements. Semantic controls use their original qualified cohorts. Non-monotonic curves and unexpected background responses remain in the report.
