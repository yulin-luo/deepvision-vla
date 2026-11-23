# BridgeV2 action alignment

The check uses the fixed 1,500-source BridgeV2 v55 runtime. It selects the source frame and derives labels before reading any model outputs. The original Figure 2 run and the public checkpoint replay have distinct provenance.

| Policy | Reference verified on v55 | Applicable metric |
| --- | --- | --- |
| OpenVLA | Same-frame next-minus-current EEF motion in six channels, with the Bridge-orig gripper relabeling | First-step seven-channel action MSE |
| Archived Bridge-adapted Pi0 route | All four rows at all 1,500 anchors match the archived four-step Bridge reference | Four-step dataset-reference MSE for that archived route |
| Official LeRobot Pi0 base | Pinned configuration declares a 32-channel state, 32-channel output and horizon 50. The released preprocessor does not supply a Bridge-specific feature mapping | Prediction drift on the fixed 150-source experiment |
| QwenVLA-OFT H16 | The first-step target equals the stored checkpoint-specific label at all 1,500 anchors. Its first six channels are reached-state differences and its gripper channel is the recorded command at the same anchor | Supplemental H1 dataset-reference MSE. The main H16 diagnostic reports prediction drift |

## Recheck the references on CPU

Follow [Quickstart](QUICKSTART.md) to download, verify and unpack the v55 runtime. Use the report environment.

```bash
export PYTHONPATH="$DV_CODE/src"
"$DV_REPORT_PYTHON" -m deepvision_vla.diagnostics.action_alignment_audit \
  --runtime "$DV_WORK/runtime" \
  --pi0-base-checkpoint "$PI0_WORK/checkpoint" \
  --output "$DV_WORK/bridge-action-alignment.json"
```

The audit verifies the fixed inventory, original action/state array hashes, anchor and all four action vectors against the archived Bridge reference. The optional `--qwen-stored-evidence` argument accepts the retained per-source Qwen conversion evidence and independently compares the first-step vectors, source hashes and frame indices. Without that ledger, the command still checks the public Bridge arrays and reports the gripper-convention difference.

Add `--qwen-terminals merged_terminals.jsonl --qwen-merge-receipt MERGE_RECEIPT.json` to replay the native H16 predictions on CPU. It verifies the merged ledger digest and the same 1,500 sources in the baseline and every layer/retention cell, then scores step 0 in each prediction. [Verified action audit and H1 replay](../../examples/bridgev2-action-alignment/summary.json)

## Qwen H1 results

The conversion audit linked the 1,500 source anchors by trajectory state and instruction and checked 2,540 stored copies. The first-step targets match across the retained 1,340 and replacement 160 sources. The OpenVLA gripper rule would change 25 rows. Keep the raw Bridge command-gripper channel for this Qwen replay.

On the already saved predictions from the current public checkpoint, unmasked first-step seven-channel MSE was **0.020553**. The six motion channels averaged **0.000714** and the gripper channel alone averaged **0.139587**. At full ROI masking, the layer 0 minus layer 34 H1 MSE gap was **0.004555**, with a source-shard bootstrap 95% interval of **[0.003033, 0.006127]**. These values apply to the current checkpoint and these reconstructed labels. The archived original Qwen evaluation JSON and Figure 2 checkpoint binding have not been recovered.

## Official Pi0 base and physical actions

The published masking run feeds one Bridge camera and a standardized EEF/gripper proxy. A physical-action MSE would require both a verified state adapter and an action encoding, normalization and validity window for Bridge, tied to this exact checkpoint. The pinned base snapshot has no such Bridge contract. The archived four-step Bridge reference remains available for comparing the historical Bridge-adapted route. [Pi0 base reproduction](PI0_BASE.md) · [Official base snapshot](https://huggingface.co/lerobot/pi0_base/tree/6e245399ad403ccf02b8112b73cfdc2633068f38)
