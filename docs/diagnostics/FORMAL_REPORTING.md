# Formal diagnostic reporting

The reporter consumes completed diagnostic journals. It does not load a
policy, execute a simulator, change ROI qualification or dispatch another
prediction. The analysis follows the frozen source-paired protocol.

Use a separate CPU-only Python3.12 environment for analysis and plotting:

```bash
python3.12 -m venv .venv/analysis
source .venv/analysis/bin/activate
python -m pip install -r requirements/diagnostics-analysis.txt
python -m pip install -e . --no-deps
python -m pip check
```

The figure tests use NumPy 2.3.5, Matplotlib 3.10.7, and Pillow 13.0.0. The
saved-report redraw was verified on Windows Python 3.12.14 and in a clean Linux
Python 3.12.3 environment. Those runs used a synthetic report. Linux
resource-context checks for analysis were verified separately. Actual
`--analyze` uses the Linux affinity, resource, and signal guards.
Keep this environment separate from all three incompatible native-policy
environments; analysis does not require Torch, Transformers or a simulator.

Preparation is the default and opens only the configuration itself:

```bash
python -m deepvision_vla.diagnostics.formal_reporting \
  --config docs/diagnostics/formal_reporting_config.prepared.json
```

The template deliberately leaves all run paths and output unset. Default
preparation does not open, hash or decode any request/result/reference file.
It returns `prepared_not_authorized` and creates no report directory.

Actual reporting requires an independently approved, exact-byte analysis
permit and all three original job releases:

```bash
python -m deepvision_vla.diagnostics.formal_reporting \
  --config reporting.json --analyze \
  --permit analysis-permit.json --permit-sha256 SHA256
```

To redraw figures from an existing report, keep `report.json` and
`ARTIFACT_MANIFEST.json` in the same directory.

```bash
python -m deepvision_vla.diagnostics.formal_reporting_plots \
  --report path/to/run/report.json --output path/to/new-figures
```

For a single figure with all three models in the paper's four-layer layout:

```bash
python -m deepvision_vla.diagnostics plot redraw \
  --report path/to/run/report.json --output path/to/new-overview --overview
```

The overview reads the saved primary-metric means and confidence intervals. It
shows the verified action error for OpenVLA, source-bound dataset-reference
error for Bridge Pi0, and prediction drift for Qwen H16. Each model has a common
vertical scale across its layers and its own paired episode count. Separate full
redraws include the normalized Qwen drift and all matched-control comparisons.

## Verify published curve values

Keep `report.json`, `ARTIFACT_MANIFEST.json` and
`per_source_metrics.jsonl` in one directory. With the analysis environment
activated, run:

```bash
python -m deepvision_vla.diagnostics plot verify --report path/to/report.json
```

The command checks both file digests, reconstructs the complete episode cohorts
and recomputes the means and pointwise intervals from the episode values. It
also checks the signed difference to baseline, the last-minus-first layer
contrast and any reported semantic region contrasts. Qualification, incomplete
episodes and episode order are checked before resampling. This check needs
the released files and NumPy. Also pass `--source-index "$INPUTS/source_index.json"`
and `--source-index-sha256 "$INDEX_SHA"` to recompute the saved intervals that
resample entire BridgeData shards. The output reports how many panels passed
each check.

Plans prepared with `--include-interaction-in-primary` include reviewed
interaction regions with the object and robot. The semantic-control panels use
the same token budget for task ROI, non-target objects and background. The
report includes paired differences for these regions on the common cohort.

Each model configuration binds `{path,sha256}` descriptors for `parent_state`,
`child_result`, `plan_manifest`, `requests`, `events`, and the ordered
`terminal_journals` list (including any recovery file). It also binds the
original `scope_id`, `permit_id` and complete `run_bindings`. Optional
`references` and `codec` descriptors must match the original child's permitted
reference artifact. `codec_channel_path` explicitly locates the seven labels
inside the bound codec; absent labels remain unavailable rather than guessed.
Any configured codec SHA must equal the admitted child's
`bindings.output_codec_sha256` before the artifact is parsed. Pi0's source-bound
reference evaluation requires that same original codec.

The analysis permit binds the configuration, reporting/statistics source
hashes, analysis plan and all three parent/child artifact hashes. It declares
an available CPU affinity of at most four cores, an original absolute budget
of at most900 seconds and an8GB memory bound. These limits apply to the
report process only; it never acquires a GPU lease or starts a child policy.
An inherited lower soft or hard memory limit is never raised. The resource
context restores its affinity, environment, soft limit and signal handler,
including on setup failure; it rejects an existing real timer before making
any changes.

Before any request, prediction or reference values are read, all three parent
states must declare `complete_requested_inventory_independently_released`,
independent owned-resource release and a qualified SHA-bound child terminal
probe. Failed, partial, active or unknown jobs yield a metadata-only gate
failure; they cannot silently admit partial prediction analysis.

The reporter keeps all1500 requested identities, blocked requests, failures,
unknown entries, missing points, history reuse and aliases. A valid-prefix
journal with a corrupt tail remains incomplete. No missing metric becomes0.
Primary and equal-token random-control panels use separate fixed complete
source cohorts; a missing point removes that source from the entire paired
panel and remains in its exclusion ledger.

Outputs include JSON, CSV, per-source ledgers and PNG/PDF curves for all four
preselected layers and eleven nominal keep fractions. Signed baseline and
last-minus-first-layer differences preserve negative changes. Confidence
intervals use2000 paired source resamples, seed20261114, and are pointwise,
not simultaneous curve bands. Cache aliases are points, not new replicates.

OpenVLA official GT error, Pi0 source-bound full-H4 reference error and Qwen
native/normalized full-H16 drift remain separate metric spaces. Pi0's numeric
reference does not establish physical GT. Qwen Layer0 interventions leave later
native DeepStack reinjections unchanged. Semantic controls absent from the
request manifest are `not_requested`; random non-ROI tokens are not renamed
semantic background.
