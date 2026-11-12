"""Read-only strict acceptance of image inference; no model or simulator import."""
import argparse, hashlib, json, math, pathlib, struct
from deepvision_vla.evaluation.validation import require

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]

def audit(root, reference=None):
    root = pathlib.Path(root).resolve(strict=True)
    launcher = json.loads((root / 'launcher_receipt.json').read_text())
    require(launcher['mode'] == 'image' and launcher['returncode'] == 0, "launcher['mode']=='image' and launcher['returncode']==0")
    require(launcher['requested_samples'] == 1 and launcher['requested_episodes'] == 0, "launcher['requested_samples']==1 and launcher['requested_episodes']==0")
    predictionfile = root / 'results/image_prediction/prediction.json'
    pred = json.loads(predictionfile.read_text())
    require(pred['status'] == 'prediction_completed' and pred['image_size'] == [224, 224], "pred['status']=='prediction_completed' and pred['image_size']==[224,224]")
    require(pred['simulator_episodes'] == 0 and pred['rlbench_success_rate'] is None and (pred['execution_performed'] is False), "pred['simulator_episodes']==0 and pred['rlbench_success_rate'] is None and pred['execution_performed'] is False")
    require(pred['policy_conditioned_on_robot_state'] is False and pred['ground_truth_action_required'] is False, "pred['policy_conditioned_on_robot_state'] is False and pred['ground_truth_action_required'] is False")
    require(pred['action_representation'] == 'xyz_delta_euler_absolute_v1' and pred['action_normalization_contract'] == 'quantile_linear_unclipped_v1', "pred['action_representation']=='xyz_delta_euler_absolute_v1' and pred['action_normalization_contract']=='quantile_linear_unclipped_v1'")
    value = pred['predicted_action_xyz_delta_euler_grip']
    require(len(value) == 7 and all((math.isfinite(v) for v in value)), 'len(value)==7 and all(math.isfinite(v) for v in value)')
    args = launcher['argv']
    requestfile = pathlib.Path(args[args.index('--image-request') + 1])
    request = json.loads(requestfile.read_text())
    require(set(request) <= {'schema_version', 'image', 'instruction', 'current_ee_position_xyz'} and pred['instruction'] == request['instruction'], "set(request)<= {'schema_version','image','instruction','current_ee_position_xyz'} and pred['instruction']==request['instruction']")
    image = pathlib.Path(request['image'])
    image = image if image.is_absolute() else requestfile.parent / image
    require(sha(image) == pred['image_sha256'] and sha(requestfile) == pred['request_sha256'], "sha(image)==pred['image_sha256'] and sha(requestfile)==pred['request_sha256']")
    if pred['execution_xyz_xyzw_grip'] is not None:
        command = pred['execution_xyz_xyzw_grip']
        require(len(command) == 8 and all((math.isfinite(v) for v in command)), 'len(command)==8 and all(math.isfinite(v) for v in command)')
        require(command[:3] == [f32(a + b) for a, b in zip(value[:3], request['current_ee_position_xyz'])], "command[:3]==[f32(a+b) for a,b in zip(value[:3],request['current_ee_position_xyz'])]")
        require(abs(sum((v * v for v in command[3:7])) - 1) < 1e-06 and command[7] == value[6], 'abs(sum(v*v for v in command[3:7])-1)<1e-6 and command[7]==value[6]')
    log = (root / 'run.log').read_text()
    records = {}
    for marker in ['STRICT_FINITE_PASS', 'STRICT_COMPONENT_PASS', 'POLICY_REQUEST', 'MODEL_FORWARD_ENTRY', 'SINGLE_GPU_CUDA_READY', 'IMAGE_DISPATCH_PROTOCOL']:
        records[marker] = [json.loads(line[len(marker) + 1:]) for line in log.splitlines() if line.startswith(marker + ' ')]
    require('STRICT_QWEN_LOADING_PASS' in log and 'X_DISPLAY_READY' not in log, "'STRICT_QWEN_LOADING_PASS' in log and 'X_DISPLAY_READY' not in log")
    require({r['component'] for r in records['STRICT_FINITE_PASS']} == {'qwen', 'finetuned_dino', 'aligner', 'action_head'}, "{r['component'] for r in records['STRICT_FINITE_PASS']}=={'qwen','finetuned_dino','aligner','action_head'}")
    require(all((r['materialized'] is True and r['state_keys'] > 0 for r in records['STRICT_FINITE_PASS'])), "all(r['materialized'] is True and r['state_keys']>0 for r in records['STRICT_FINITE_PASS'])")
    require({r['component'] for r in records['STRICT_COMPONENT_PASS']} == {'finetuned_dino', 'aligner', 'action_head'}, "{r['component'] for r in records['STRICT_COMPONENT_PASS']}=={'finetuned_dino','aligner','action_head'}")
    require(len(records['POLICY_REQUEST']) == len(records['MODEL_FORWARD_ENTRY']) == len(records['SINGLE_GPU_CUDA_READY']) == len(records['IMAGE_DISPATCH_PROTOCOL']) == 1, "len(records['POLICY_REQUEST'])==len(records['MODEL_FORWARD_ENTRY'])==len(records['SINGLE_GPU_CUDA_READY'])==len(records['IMAGE_DISPATCH_PROTOCOL'])==1")
    require(records['POLICY_REQUEST'][0]['instruction'] == request['instruction'], "records['POLICY_REQUEST'][0]['instruction']==request['instruction']")
    comparison = None
    if reference:
        rows = [json.loads(line) for line in pathlib.Path(reference).read_text().splitlines()]
        require(len(rows) == 1, 'len(rows)==1')
        r = rows[0]
        require(sha(r['image_path']) == pred['image_sha256'] and r['task_prompt'] == pred['instruction'], "sha(r['image_path'])==pred['image_sha256'] and r['task_prompt']==pred['instruction']")
        require(r['pred_action'] == value, "r['pred_action']==value")
        comparison = {'same_image': True, 'same_instruction': True, 'predicted_action_exactly_equal': True, 'reference_prediction_file_sha256': sha(reference)}
    return {'status': 'passed', 'evidence_kind': 'real_image_only_forward', 'requested_samples': 1, 'actual_policy_calls': 1, 'actual_model_forward_calls': 1, 'simulator_episodes': 0, 'rlbench_success_rate': None, 'GT_action_input': False, 'robot_state_conditioning': False, 'instruction': pred['instruction'], 'image_sha256': pred['image_sha256'], 'predicted_action': value, 'execution_command': pred['execution_xyz_xyzw_grip'], 'execution_performed': False, 'same_fixture_reference_comparison': comparison, 'strict_components': records['STRICT_COMPONENT_PASS'], 'finite_components': records['STRICT_FINITE_PASS'], 'dispatch_protocol': records['IMAGE_DISPATCH_PROTOCOL'][0], 'gpu': {k: v for k, v in records['SINGLE_GPU_CUDA_READY'][0].items() if k != 'pid'}, 'fixture_scope': 'one training-data image used for runtime regression, not heldout performance', 'evidence_sha256': {'launcher_receipt.json': sha(root / 'launcher_receipt.json'), 'run.log': sha(root / 'run.log'), 'prediction.json': sha(predictionfile)}}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', required=True)
    p.add_argument('--reference-jsonl')
    p.add_argument('--receipt', required=True)
    a = p.parse_args()
    result = audit(a.output_root, a.reference_jsonl)
    with pathlib.Path(a.receipt).open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
