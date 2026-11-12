"""Standalone single-GPU entry around the byte-pinned Full32 evaluator."""
import argparse,hashlib,json,os,pathlib,re,runpy,signal,subprocess,sys,time,uuid
from deepvision_vla.runtime.gpu_lease import gpu_lease
from deepvision_vla.runtime.processes import cleanup_owned_session,session_identity
from deepvision_vla.paths import PACKAGE_ROOT
TASKS=('close_fridge','sweep_to_dustpan','take_umbrella_out_of_umbrella_stand','take_frame_off_hanger','close_laptop_lid','phone_on_base','close_box','place_wine_at_rack_location','water_plants','toilet_seat_down')
EVALUATOR_SHA='ba8d48655afc0d3ec0deb527d1f639aaa243b149b3a97cb4dd871b3d0fa52228'
def pci_key(value):
    domain,bus,device=value.strip().lower().split(':')
    return int(domain,16),int(bus,16),device
def same_owned_group(pid):
    try:
        fields=pathlib.Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return int(fields[2])==os.getpgrp()
    except (OSError,ValueError,IndexError):return False
def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
def resolve(c,key,is_dir=False):
    # Preserve a venv executable's path: resolving its symlink would switch
    # the child to the base environment and lose the editable installation.
    candidate=pathlib.Path(c[key]).expanduser()
    p=candidate.absolute() if key=='python_bin' else candidate.resolve(strict=True)
    if is_dir and not p.is_dir():raise ValueError(f'{key} is not a directory')
    if not is_dir and not p.is_file():raise ValueError(f'{key} is not a file')
    return p
def build(c,mode,task,episodes,out):
    if mode not in ('infer','eval','image'):raise ValueError('unsupported entry mode')
    from deepvision_vla.runtime.checkpoint import validate_checkpoint_binding
    validate_checkpoint_binding(c)
    source=resolve(c,'source_root',True);checkpoint=resolve(c,'checkpoint_root',True)
    script=source/'scripts_rlbench10_ablation/test_rlbench_no_gen_encoder.py'
    if file_sha(script)!=EVALUATOR_SHA:raise ValueError('scientific evaluator differs from accepted origin')
    expected=c['checkpoint_small_file_sha256']
    for rel in ['training_config.json','stats_data.json']:
        if file_sha(checkpoint/rel)!=expected[rel]:raise ValueError(f'checkpoint {rel} hash mismatch')
    cfg=json.loads((checkpoint/'training_config.json').read_text())
    required={'action_chunk':1,'action_dim':7,'action_query_contract':'decoupled_per_dim_v1','action_query_count':7,'action_head_contract':'decoupled_per_dim_trunk_shared_then_per_dim_linear_v1','action_normalization_contract':'quantile_linear_unclipped_v1','action_representation':'xyz_delta_euler_absolute_v1','image_aug_crop_scale':.9}
    for k,v in required.items():
        if cfg.get(k)!=v:raise ValueError(f'checkpoint contract {k}: {cfg.get(k)!r} != {v!r}')
    if task not in TASKS or not 1<=episodes<=50:raise ValueError('task/episode budget outside v0.1 scope')
    profile=resolve(c,'profile_path');dino=resolve(c,'dino_runtime_root',True);python=resolve(c,'python_bin')
    protocol=runpy.run_path(str(source/'utils/rlbench_protocol.py'))
    registry=protocol['load_profile_registry'](profile)
    active=registry['profiles']['hf-legacy-224-v1']
    if protocol['_profile_definition_sha256'](active)!=cfg['rlbench_profile_definition_sha256']:
        raise ValueError('packaged RLBench profile definition differs from checkpoint')
    args=[str(python),'-m','deepvision_vla.runtime.strict_entry','--source-root',str(source),'--checkpoint-root',str(checkpoint),
        '--checkpoint-verification-receipt',str(c['checkpoint_verification_receipt']),
        '--checkpoint-verification-receipt-sha256',c['checkpoint_verification_receipt_sha256'],
        '--checkpoint-inventory-sha256',c['checkpoint_inventory_sha256'],'--',
        '--task-name',task,'--dataset-name','rlbench','--model-path',str(checkpoint/'tfmr'),
        '--replay-or-predict','dataset_predict' if mode in ('infer','image') else 'predict','--cuda','0','--use_robot_state','0','--max-steps','10','--num-episodes',str(episodes),'--eval-seed','42','--load-pointcloud','0',
        '--result-dir',str(out/'results'),'--exp-name','full32_h1','--rlbench-env-profile','hf-legacy-224-v1','--rlbench-profile-path',str(profile),
        '--eval-image-protocol','matched','--action-chunk','1','--action-dim','7','--need-to-sub','0','--allow-random-init','0','--expected-action-representation','xyz_delta_euler_absolute_v1','--dinov3-path',str(dino),'--debug','0']
    if mode=='infer':args+=['--eval-data-path',str(resolve(c,'sample_data_path')),'--dataset-max-samples','1']
    return args
def validate_gpu_selector(value):
    value = str(value)
    if not value.isdigit() and not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value):
        raise ValueError('select one physical GPU index or UUID')
    return value


def task_cli_command(python_bin, mode, configpath, output, gpu, task, episodes=50, wait_for_gpu=0):
    """Build the real public child CLI command in its single supported order."""
    if mode not in ('infer', 'eval', 'image'):
        raise ValueError('unsupported child command')
    return [str(python_bin), '-u', '-m', 'deepvision_vla', mode,
            '--config', str(configpath), '--output', str(output),
            '--gpu', validate_gpu_selector(gpu), '--wait-for-gpu', str(wait_for_gpu),
            '--task', task, '--episodes', str(episodes)]


def observe_gpu(selector):
    observation = subprocess.check_output(
        ['nvidia-smi', '-i', validate_gpu_selector(selector),
         '--query-gpu=index,uuid,pci.bus_id,memory.used', '--format=csv,noheader,nounits'],
        text=True, timeout=10).strip()
    index, gpu_uuid, pci, used = [part.strip() for part in observation.split(',')]
    pci_key(pci)
    return {'index': index, 'uuid': gpu_uuid, 'pci': pci,
            'memory_mib': int(used), 'observation': observation}


def require_idle_gpu(resource):
    compute = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
        text=True, timeout=10)
    graphics = subprocess.check_output(
        ['nvidia-smi', 'pmon', '-i', resource['index'], '-c', '1'],
        text=True, timeout=10)
    occupied = (resource['uuid'] in compute or resource['memory_mib'] > 256
                or any(len(row) > 1 and row[0] == resource['index'] and row[1].isdigit()
                       for row in (line.split() for line in graphics.splitlines())
                       if row and not row[0].startswith('#')))
    if occupied:
        raise RuntimeError('selected physical GPU is occupied; no model was started')


def _execute(c, a):
    out = pathlib.Path(a.output).expanduser().resolve()
    if out.exists():
        raise FileExistsError('fresh output directory required; previous outcomes are preserved')
    request = None
    if a.mode == 'image':
        if not a.request:
            raise ValueError('image mode requires --request')
        from deepvision_vla.inference.image import read_request
        request = read_request(a.request)
    elif a.request:
        raise ValueError('--request is only valid for image mode')
    # Native dependency inspection is CPU-only and precedes any GPU lease,
    # driver query, model construction, output creation or simulator entry.
    native_diagnostics = None
    native_patch = None
    if a.mode == 'eval':
        if c.get('image_only'):
            raise ValueError('image-only configuration cannot run RLBench; regenerate without --image-only and with --deps-root')
        from deepvision_vla.runtime.native import require_native_dependencies, native_environment
        native_diagnostics = require_native_dependencies(c)
        native_patch = native_environment(c)
    args = build(c, a.mode, a.task, 1 if a.mode in ('infer', 'image') else a.episodes, out)
    if request is not None:
        cut = args.index('--')
        args[cut:cut] = ['--image-request', request['request_path']]
    cut = args.index('--')
    args[cut:cut] = ['--wait-gpu-seconds', str(a.wait_for_gpu)]
    launcher = c.get('display_argv')
    if a.mode == 'eval' and not launcher:
        raise ValueError('eval requires a configured display launcher; no guessed offscreen fallback')
    resource = observe_gpu(a.gpu)
    cut = args.index('--')
    args[cut:cut] = ['--expected-gpu-pci', resource['pci']]
    if a.print_command:
        print(json.dumps({'argv': args, 'physical_gpu': a.gpu,
                          'nvml_uuid': resource['uuid'], 'pci': resource['pci'], 'mode': a.mode}, indent=2))
        return

    # Ordinary image/eval entries and controller-launched tasks use the SAME
    # per-task UUID lease. The batch controller never retains a second lease.
    with gpu_lease(resource['uuid'], wait_seconds=a.wait_for_gpu) as lease:
        current = observe_gpu(resource['uuid'])
        if (current['uuid'] != resource['uuid'] or current['pci'] != resource['pci']
                or current['index'] != resource['index']):
            raise RuntimeError('selected physical GPU identity changed before dispatch')
        if a.wait_for_gpu == 0:
            require_idle_gpu(current)
        out.mkdir(parents=True, exist_ok=False)
        cache = out / 'cache'; cache.mkdir()
        temp = out / 'tmp'; temp.mkdir()
        token = uuid.uuid4().hex
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=resource['index'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   DEEPVISION_NVML_GPU_UUID=resource['uuid'], DEEPVISION_RUN_TOKEN=token,
                   PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', WANDB_MODE='offline',
                   HF_HOME=str(cache / 'huggingface'), XDG_CACHE_HOME=str(cache),
                   TORCH_HOME=str(cache / 'torch'), MPLCONFIGDIR=str(cache / 'matplotlib'),
                   TMPDIR=str(temp), TEMP=str(temp), TMP=str(temp), DEEPVISION_RUN_ROOT=str(out))
        for key in ['EVAL_CAM_RES', 'EVAL_CENTER_CROP', 'EVAL_DINO_RAW', 'EVAL_DINO_ABLATE', 'HF10_ALLOW_RUNTIME_DRIFT']:
            env.pop(key, None)
        paths = [str(PACKAGE_ROOT.parent), str(resolve(c, 'source_root', True)),
                 str(resolve(c, 'source_root', True) / 'third_party/lift3d')]
        if a.mode == 'eval':
            # Image/dataset inference neither resolves nor exports native roots.
            paths.extend(str(resolve(c, key, True)) for key in ['rlbench_root', 'pyrep_root'])
            env.update(native_patch)
            args = list(launcher) + args
        env['PYTHONPATH'] = os.pathsep.join(paths)
        receipt = {'status': 'running', 'mode': a.mode, 'task': a.task,
                   'requested_episodes': a.episodes if a.mode == 'eval' else 0,
                   'requested_samples': 0 if a.mode == 'eval' else 1,
                   'argv': args, 'physical_gpu': a.gpu, 'nvml_uuid': resource['uuid'],
                   'expected_cuda_pci': resource['pci'], 'cuda_visible_index': resource['index'],
                   'gpu_prelaunch': current['observation'], 'gpu_lease': lease,
                   'native_dependencies': native_diagnostics,
                   'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'model_result': None}
        receipt_path = out / 'launcher_receipt.json'
        receipt_path.write_text(json.dumps(receipt, indent=2))
        proc = None
        owned = None
        try:
            with (out / 'run.log').open('x') as log:
                try:
                    proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
                    owned = session_identity(proc)
                    receipt.update(child_pid=proc.pid, child_pgid=proc.pid,
                                   child_start_ticks=owned.get('start_ticks'))
                    # Every write after Popen belongs to the cleanup scope.
                    receipt_path.write_text(json.dumps(receipt, indent=2))
                    rc = proc.wait(timeout=1800 if a.mode in ('infer', 'image')
                                   else (900 if a.episodes == 1 else 14400))
                finally:
                    if proc is not None:
                        owned = owned or {'pid': proc.pid, 'pgid': proc.pid, 'start_ticks': None}
                        try:
                            receipt['child_cleanup'] = cleanup_owned_session(proc, owned, token)
                        except BaseException as cleanup_error:
                            receipt['cleanup_unknown'] = True
                            receipt['cleanup_error'] = repr(cleanup_error)
                            raise
            receipt.update(status='process_completed' if rc == 0 else 'process_failed', returncode=rc)
        except BaseException as error:
            receipt.update(status='launcher_failed', error=repr(error))
            raise
        finally:
            receipt['ended_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            receipt_path.write_text(json.dumps(receipt, indent=2))
        # A process return code is not a task-success receipt.
        raise SystemExit(receipt.get('returncode', 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--wait-for-gpu', type=int, default=0)
    parser.add_argument('--print-command', action='store_true')
    parser.add_argument('mode', choices=['infer', 'eval', 'image'])
    parser.add_argument('--task', choices=TASKS, default='close_box')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--request')
    args = parser.parse_args()
    if not 0 <= args.wait_for_gpu <= 600:
        parser.error('--wait-for-gpu must be 0..600 seconds')
    try:
        args.gpu = validate_gpu_selector(args.gpu)
    except ValueError as error:
        parser.error(str(error))
    config = json.loads(pathlib.Path(args.config).read_text())
    def interrupted(signum, frame):
        raise KeyboardInterrupt('launcher interrupted')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return _execute(config, args)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
