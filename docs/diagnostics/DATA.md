# BridgeData inputs and ROI annotations

BridgeData V2 is released under **Creative Commons Attribution 4.0**. Its
[official project page](https://rail-berkeley.github.io/bridgedata/) states the
data license and links to the original downloads. The upstream training
[code repository](https://github.com/rail-berkeley/bridge_data_v2) has a separate
[MIT license](https://github.com/rail-berkeley/bridge_data_v2/blob/main/LICENSE).
The code license does not replace the dataset license.

When sharing source frames or derived visualizations, credit BridgeData V2,
link to the source and [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
and identify changes such as resizing, frame selection, ROI annotation, masking
or contribution-map overlays. Retain upstream notices for reused code.

## Diagnostic subset

The diagnostic sample list identifies trajectories, selected frames and source
revisions. Preserve that list when comparing models; do not replace rejected or
failed samples. The diagnostic subset is an independent sample selection, not a
recovered list of the original paper's trajectories.

ROI annotations use original-image pixel coordinates and distinguish manipulated
objects, robot regions and task-dependent goals. AI-reviewed rectangles are
visible-region proxies, not human-verified segmentation or physical-contact
labels. Each annotation keeps its source identity, review status and rejected
candidates. Per-model image transforms determine which visual tokens fall in
each region; token indexes cannot be transferred between models.

Retain source attribution alongside derived annotations. An annotation release
can be distributed as source indexes and annotation records without bundling
the full upstream dataset. Model checkpoints and processed training-data
backups have separate provenance and access terms.

## Reconstructing the fixed panel from local shards

The source index identifies every requested episode by upstream file identity,
episode ID, TFRecord shard, byte offset and record checksum. The image is the
original `image_0` JPEG at frame `(T - 1) // 2`; the exported arrays contain the
complete raw seven-channel state and action sequences. These raw actions are
**not automatically ground truth for any model**. Each model's action conversion
and reference alignment are a separate step.

Use an activated data environment with NumPy 1.26.4 and Pillow. `google-crc32c` is
optional and accelerates TFRecord integrity checks; the reader also has a pure
Python CRC32C implementation. TensorFlow, Torch and a simulator are unnecessary.

Obtain the source index and shard-checksum file for the selected annotation
release, and copy the referenced pinned shards into one local directory. Pass
the two metadata checksums distributed with that release:

```bash
python -m deepvision_vla.diagnostics.reconstruct_bridge \
  --source-index source_index.json \
  --source-index-sha256 "$SOURCE_INDEX_SHA256" \
  --shard-checksums shard_checksums.json \
  --shard-checksums-sha256 "$SHARD_CHECKSUMS_SHA256" \
  --shard-dir /path/to/local/bridge-shards \
  --output /path/to/new/bridge-panel
```

The command only reads local shards. It verifies the shard size/checksum,
TFRecord CRC and record checksum, then checks episode identity, instruction,
anchor and image checksum. The emitted NPZ must also match its original archive
checksum. An archive mismatch retains the emitted files and is reported as a
failure, even if a different NumPy/archive version preserves the numeric values.
It never downloads data, substitutes a failed source,
switches cameras or overwrites an existing output directory. `outcomes.jsonl`
retains every requested source, including failures; `summary.json` reports the
complete requested, reconstructed and failed counts. A failed source returns a
nonzero command status. Interrupted output remains incomplete and is preserved.

The released panel includes 1,340 retained episodes and 160 replacements from
the same seed-42 shard stream. Visual review determined eligibility before the
replacement predictions were run. The 47 checksum-pinned shards reproduce this
independently indexed panel. The upstream episode field is used to reproduce
the sample ID.

Please cite the dataset paper when using these inputs:

```bibtex
@inproceedings{walke2023bridgedata,
  title={BridgeData V2: A Dataset for Robot Learning at Scale},
  author={Walke, Homer and Black, Kevin and Lee, Abraham and Kim, Moo Jin
    and Du, Max and Zheng, Chongyi and Zhao, Tony and Hansen-Estruch, Philippe
    and Vuong, Quan and He, Andre and Myers, Vivek and Fang, Kuan
    and Finn, Chelsea and Levine, Sergey},
  booktitle={Conference on Robot Learning},
  year={2023}
}
```
