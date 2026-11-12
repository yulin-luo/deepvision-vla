"""Read-only acceptance of a dataset forward; never turn it into rollout success."""
import argparse, hashlib, json, math, pathlib, time
from deepvision_vla.evaluation.validation import require

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-root', required=True)
    p.add_argument('--receipt', required=True)
    a = p.parse_args()
    root = pathlib.Path(a.output_root).resolve(strict=True)
    dest = pathlib.Path(a.receipt).resolve()
    if dest.exists():
        raise FileExistsError('fresh audit receipt required')
    launch = json.loads((root / 'launcher_receipt.json').read_text())
    require(launch['mode'] == 'infer' and launch['returncode'] == 0, "launch['mode']=='infer' and launch['returncode']==0")
    metrics = list((root / 'results').glob('dataset_predict_results/*/*/dataset_metrics.json'))
    require(len(metrics) == 1, 'len(metrics)==1')
    folder = metrics[0].parent
    data = json.loads(metrics[0].read_text())
    preds = [json.loads(l) for l in (folder / 'dataset_predictions.jsonl').read_text().splitlines()]
    require(len(preds) == data['samples'] == 1, "len(preds)==data['samples']==1")
    pred = preds[0]
    require(len(pred['pred_action']) == 7 and all((math.isfinite(v) for v in pred['pred_action'])), "len(pred['pred_action'])==7 and all(math.isfinite(v) for v in pred['pred_action'])")
    require(pred['action_representation'] == 'xyz_delta_euler_absolute_v1' and pred['action_normalization_contract'] == 'quantile_linear_unclipped_v1', "pred['action_representation']=='xyz_delta_euler_absolute_v1' and pred['action_normalization_contract']=='quantile_linear_unclipped_v1'")
    log = (root / 'run.log').read_text()
    finite = []
    strict = []
    device = []
    for line in log.splitlines():
        if line.startswith('STRICT_FINITE_PASS '):
            finite.append(json.loads(line.split(' ', 1)[1]))
        if line.startswith('STRICT_COMPONENT_PASS '):
            strict.append(json.loads(line.split(' ', 1)[1]))
        if line.startswith('SINGLE_GPU_CUDA_READY '):
            device.append(json.loads(line.split(' ', 1)[1]))
    require('STRICT_QWEN_LOADING_PASS' in log, "'STRICT_QWEN_LOADING_PASS' in log")
    require({r['component'] for r in finite} == {'qwen', 'finetuned_dino', 'aligner', 'action_head'}, "{r['component'] for r in finite}=={'qwen','finetuned_dino','aligner','action_head'}")
    require(all((r.get('materialized') is True and r['state_keys'] > 0 for r in finite)), "all(r.get('materialized') is True and r['state_keys']>0 for r in finite)")
    require({r['component'] for r in strict} == {'finetuned_dino', 'aligner', 'action_head'} and len(device) == 1, "{r['component'] for r in strict}=={'finetuned_dino','aligner','action_head'} and len(device)==1")
    result = {'status': 'passed', 'evidence_kind': 'real_dataset_forward_smoke', 'task': launch['task'], 'requested_samples': 1, 'actual_prediction_rows': len(preds), 'simulator_episodes': 0, 'rlbench_success_rate': None, 'not_a_heldout_performance_estimate': True, 'metrics': {k: v for k, v in data.items() if k != 'predictions_path'}, 'gpu': {k: v for k, v in device[0].items() if k != 'pid'}, 'finite_components': finite, 'strict_components': strict, 'input_image_sha256': sha(pred['image_path']), 'episode_id': pred['episode_id'], 'step_id': pred['step_id'], 'prediction_file_sha256': sha(folder / 'dataset_predictions.jsonl'), 'metadata_file_sha256': sha(folder / 'eval_metadata.json'), 'run_log_sha256': sha(root / 'run.log'), 'original_launcher_receipt_sha256': sha(root / 'launcher_receipt.json'), 'legacy_success_rate_text_is_within_threshold_metric': True, 'audit_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
if __name__ == '__main__':
    main()
