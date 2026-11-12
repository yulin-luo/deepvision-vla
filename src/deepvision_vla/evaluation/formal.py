"""Finite single-GPU, no-retry evaluation of the fixed RLBench500 requests."""
import argparse,hashlib,json,os,pathlib,signal,subprocess,sys,time,uuid
from deepvision_vla.evaluation.audit_rollouts import INSTRUCTIONS,aggregate,audit
from deepvision_vla.evaluation.runner import TASKS,build,task_cli_command,validate_gpu_selector,observe_gpu,require_idle_gpu
from deepvision_vla.runtime.processes import cleanup_owned_session,session_identity
from deepvision_vla.paths import ASSETS_ROOT,PACKAGE_ROOT,REPOSITORY_ROOT

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def write_atomic(path,value):
    tmp=path.with_name(path.name+'.pending')
    with tmp.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def identity(pid):
    try:
        fields=pathlib.Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'start_ticks':int(fields[19]),'pgid':int(fields[2])}
    except (OSError,ValueError,IndexError):return {'pid':pid,'start_ticks':None,'pgid':None}

def plan(config,root,gpu):
    gpu=validate_gpu_selector(gpu)
    jobs=[]
    for task in TASKS:
        output=root/'tasks'/task
        # Same actual CLI contract; no GPU/native import or process construction.
        command=build(config,'eval',task,50,output)
        jobs.append({'task':task,'output':str(output),'instruction':INSTRUCTIONS[task],
                     'episodes':[{'episode_id':i,'seed':42+i*1009} for i in range(50)],'scientific_argv':command})
    here=REPOSITORY_ROOT
    seals={str(p.relative_to(here)):sha(p) for p in sorted(PACKAGE_ROOT.rglob('*.py'))}
    seals[str((ASSETS_ROOT/'rlbench_profile.json').relative_to(here))]=sha(ASSETS_ROOT/'rlbench_profile.json')
    return {'schema_version':1,'protocol':'deepvision_v01_rlbench10_strict_single_seed_v1','requested':500,'task_count':10,
        'episodes_per_task':50,'max_policy_steps':10,'base_eval_seed':42,'gpu':str(gpu),'reset_policy':'single_seed_episode_identity_v1',
        'checkpoint_root':config['checkpoint_root'],'source_root':config['source_root'],'helper_sha256':seals,
        'runtime_python':config['python_bin'],'jobs':jobs,'physical_state_identity_claim':False,'automatic_retries':0}

def idle_gpu(gpu):
    resource=observe_gpu(gpu)
    require_idle_gpu(resource)
    return {'index':resource['index'],'nvml_uuid':resource['uuid'],'pci':resource['pci'],'memory_mib':resource['memory_mib'],'observed_utc':stamp()}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu',required=True);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    configpath=pathlib.Path(a.config).resolve(strict=True);config=json.loads(configpath.read_text());root=pathlib.Path(a.output).resolve()
    if root.exists():raise FileExistsError('existing formal root is never resumed/replaced automatically')
    manifest=plan(config,root,a.gpu)
    if a.plan_only:print(json.dumps(manifest,indent=2));return
    resource=idle_gpu(a.gpu)
    # The actual task CLI owns the common per-GPU lease. Keeping a batch
    # lease here would deadlock that child and reserve the card across NFS I/O.
    root.mkdir(parents=True,exist_ok=False);(root/'tasks').mkdir();(root/'intents').mkdir()
    manifest.update(created_utc=stamp(),resource=resource,config_sha256=sha(configpath),controller=identity(os.getpid()),boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),controller_holds_gpu_lease=False)
    with (root/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.flush();os.fsync(f.fileno())
    state={'status':'running','started_utc':stamp(),'controller':manifest['controller'],'jobs':[],'current_task':None,'manifest_sha256':sha(root/'manifest.json')}
    write_atomic(root/'state.json',state);child=None;child_identity=None;child_token=None
    def interrupt(signum,frame):raise KeyboardInterrupt('finite controller interrupted')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        for job in manifest['jobs']:
            for rel,expected in manifest['helper_sha256'].items():
                if sha(REPOSITORY_ROOT/rel)!=expected:raise ValueError('running package changed: '+rel)
            resource=idle_gpu(a.gpu)
            intent={'task':job['task'],'created_utc':stamp(),'resource':resource,'output':job['output'],'status':'intent_before_popen'}
            intentpath=root/'intents'/(job['task']+'.json')
            with intentpath.open('x') as f:json.dump(intent,f,indent=2);f.flush();os.fsync(f.fileno())
            state['current_task']=job['task'];state['jobs'].append(intent);write_atomic(root/'state.json',state)
            argv=task_cli_command(config['python_bin'],'eval',configpath,job['output'],a.gpu,job['task'],50,300)
            child_token=uuid.uuid4().hex
            child_env=dict(os.environ,DEEPVISION_RUN_TOKEN=child_token)
            child_identity=None
            with (root/(job['task']+'.launcher.log')).open('x') as log:
                child=subprocess.Popen(argv,env=child_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                child_identity=session_identity(child)
                intent.update(status='started',child=identity(child.pid),argv=argv);write_atomic(intentpath,intent);write_atomic(root/'state.json',state)
                rc=child.wait(timeout=14500)
                intent['child_cleanup']=cleanup_owned_session(child,child_identity,child_token,terminate_timeout=30)
            intent.update(status='process_exited',returncode=rc,ended_utc=stamp());write_atomic(intentpath,intent);child=None
            result=audit(job['output'],job['task'],range(50));write_atomic(root/(job['task']+'.audit.json'),result)
            write_atomic(root/'results.json',aggregate([audit(j['output'],j['task'],range(50)) for j in manifest['jobs']]))
            write_atomic(root/'state.json',state)
            if rc!=0 or result['status']!='complete':raise RuntimeError('task not fully accepted; no automatic retry or further dispatch: '+job['task'])
        state.update(status='complete',ended_utc=stamp(),current_task=None)
    except BaseException as error:
        state.update(status='stopped',error=repr(error),ended_utc=stamp())
        if child is not None:
            try:
                known=child_identity or {'pid':child.pid,'pgid':child.pid,'start_ticks':None}
                state['child_cleanup']=cleanup_owned_session(child,known,child_token,terminate_timeout=30)
            except BaseException as cleanup_error:
                state.update(cleanup_unknown=True,cleanup_error=repr(cleanup_error))
        raise
    finally:
        write_atomic(root/'results.json',aggregate([audit(j['output'],j['task'],range(50)) for j in manifest['jobs']]))
        state['controller_holds_gpu_lease']=False
        write_atomic(root/'state.json',state)
if __name__=='__main__':main()
