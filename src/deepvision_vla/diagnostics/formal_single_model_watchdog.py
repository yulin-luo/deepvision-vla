"""Reviewable one-shot formal parent. Validation is the default; no Torch.

The parent never acquires a GPU lock. It writes only node-local journals and
isolates child CFS JSON reads in finite helper sessions. Production child argv
is fixed to the existing formal_diagnostic_executor module.
"""
from pathlib import Path,PurePosixPath
from collections import Counter
import argparse
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

MODULE='deepvision_vla.diagnostics.formal_diagnostic_executor'
MODELS={'openvla','pi0','qwen_h16'}
UUID=re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
BINDING_PATHS={'config_sha256':None,'plan_manifest_sha256':'plan_manifest','requests_jsonl_sha256':'requests',
    'inventory_sha256':'inventory','profiles_sha256':'profiles','annotations_sha256':'annotations',
    'geometries_sha256':'geometries','scientific_source_sha256':'scientific_source_seals',
    'runtime_sha256':'runtime_receipt','asset_receipt_sha256':'asset_receipt',
    'output_codec_sha256':'output_codec','history_manifest_sha256':'history_manifest'}
LOCAL_FS={'ext4','xfs','btrfs','tmpfs','overlay','zfs'}

class GuardError(RuntimeError):pass

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def is_sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
def require_posix(value):
    if not isinstance(value,str) or not value.startswith('/') or '\\' in value or '..' in PurePosixPath(value).parts:
        raise GuardError('Explicit absolute POSIX paths required')
    return value

def proc_identity(pid):
    try:
        tail=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        status=(Path('/proc')/str(pid)/'status').read_text();fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in status.splitlines() if ':' in line}
        rss=int(tail[21])*os.sysconf('SC_PAGE_SIZE')
        hwm=int(fields['VmHWM'].split()[0])*1024 if 'VmHWM' in fields else None
        return {'pid':int(pid),'start_ticks':int(tail[19]),'ppid':int(tail[1]),'pgid':int(tail[2]),'sid':int(tail[3]),
            'state':tail[0],'uid':int(fields['Uid'].split()[0]),'rss_bytes':rss,'rss_hwm_bytes':hwm}
    except FileNotFoundError:return None

def identity_key(value):return tuple(value[k] for k in ('pid','start_ticks','pgid','sid','uid'))

def proc_snapshot():
    current={};unreadable=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:
            value=proc_identity(int(p.name))
            if value is not None:current[value['pid']]=value
        except (OSError,ValueError,KeyError,IndexError):unreadable.append(int(p.name))
    return current,unreadable

def enable_subreaper():
    import ctypes
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:raise GuardError('Cannot become private-child subreaper')
    return {'enabled':True,'scope':'this watchdog and its own descendant ancestry only'}

class OwnedFamily:
    def __init__(self,root):
        if root is None or root['pid']!=root['pgid'] or root['pid']!=root['sid']:
            raise GuardError('Private child session PID/PGID/SID identity is required')
        self.root=root;self.known={root['pid']:dict(root)};self.session_trusted=True;self.changes=[]
    def observe(self,snapshot):
        root=snapshot.get(self.root['pid'])
        if root and root['start_ticks']!=self.root['start_ticks']:self.session_trusted=False
        live={pid:v for pid,v in snapshot.items() if pid in self.known and identity_key(v)==identity_key(self.known[pid])}
        for pid,old in self.known.items():
            current=snapshot.get(pid)
            if current and identity_key(current)!=identity_key(old):
                change={'pid':pid,'expected':identity_key(old),'observed':identity_key(current)}
                if change not in self.changes:self.changes.append(change)
        while True:
            added=False
            for pid,value in snapshot.items():
                if pid in self.known:continue
                ancestry=value['ppid'] in live
                private_session=(self.session_trusted and value['sid']==self.root['sid'])
                if value['uid']==self.root['uid'] and (ancestry or private_session):
                    self.known[pid]=dict(value);live[pid]=value;added=True
            if not added:break
        return live

def safe_signal(expected,sig,*,identity_reader=proc_identity,pidfd_open=None,pidfd_signal=None):
    """No PID-number fallback: pidfd prevents PID-reuse check→signal races."""
    pidfd_open=pidfd_open or getattr(os,'pidfd_open',None)
    pidfd_signal=pidfd_signal or getattr(signal,'pidfd_send_signal',None)
    if pidfd_open is None or pidfd_signal is None:return {'sent':False,'reason':'pidfd_unavailable','identity':expected}
    try:fd=pidfd_open(expected['pid'],0)
    except ProcessLookupError:return {'sent':False,'reason':'already_absent','identity':expected}
    try:
        current=identity_reader(expected['pid'])
        if current is None:return {'sent':False,'reason':'already_absent','identity':expected}
        if identity_key(current)!=identity_key(expected):return {'sent':False,'reason':'identity_changed_no_signal','identity':expected,'current':current}
        pidfd_signal(fd,sig,None,0)
        return {'sent':True,'signal':int(sig),'identity':expected}
    except ProcessLookupError:return {'sent':False,'reason':'already_absent','identity':expected}
    finally:os.close(fd)

def kernel_lock_snapshot(path,*,locks_text=None):
    """Read-only stat+/proc/locks; never open/create/flock/unlink the GPU file."""
    try:
        entry=os.stat(path,follow_symlinks=False)
        directory=os.stat(Path(path).parent,follow_symlinks=False)
        if (not stat.S_ISDIR(directory.st_mode) or directory.st_uid!=os.getuid()
                or directory.st_mode & (stat.S_IWGRP|stat.S_IWOTH)):
            return {'status':'unknown','reason':'unsafe_lock_directory'}
        if not stat.S_ISREG(entry.st_mode) or entry.st_nlink!=1 or entry.st_uid!=os.getuid():
            return {'status':'unknown','reason':'unsafe_or_replaced_lock_entry'}
        key=(os.major(entry.st_dev),os.minor(entry.st_dev),entry.st_ino)
        text=Path('/proc/locks').read_text() if locks_text is None else locks_text
        holders=[]
        for line in text.splitlines():
            fields=line.split()
            if '->' in fields:continue  # blocked waiter is not an owner
            if len(fields)<6 or fields[1]!='FLOCK':continue
            try:
                major,minor,inode=fields[5].split(':');candidate=(int(major,16),int(minor,16),int(inode))
                if candidate==key:holders.append(int(fields[4]))
            except (ValueError,IndexError):return {'status':'unknown','reason':'kernel_lock_parse_error'}
        return {'status':'observed','inode':entry.st_ino,'device':entry.st_dev,'holders':holders,'read_only_probe':True}
    except FileNotFoundError:return {'status':'unknown','reason':'lock_file_absent_no_inode_proof','holders':[]}
    except OSError as error:return {'status':'unknown','reason':type(error).__name__}

def parse_nvml_xml(raw,gpu_uuid):
    try:
        root=ET.fromstring(raw);matches=[g for g in root.findall('gpu') if g.findtext('uuid')==gpu_uuid]
        if len(matches)!=1:raise ValueError('Selected UUID absent/duplicate')
        node=matches[0].find('processes')
        if node is None:raise ValueError('Process section missing')
        if node.find('error') is not None or (node.text or '').strip() not in ('','None'):
            raise ValueError('Process section unavailable, not a verified empty list')
        pids=[]
        for process in node.findall('process_info'):
            pid=process.findtext('pid')
            if pid is None or not pid.isdecimal():raise ValueError('Unknown process PID')
            pids.append({'pid':int(pid),'type':process.findtext('type')})
        return {'status':'observed','provider':'nvidia-smi XML NVML-backed','gpu_uuid':gpu_uuid,'processes':pids}
    except (ET.ParseError,ValueError) as error:return {'status':'unknown','reason':str(error),'gpu_uuid':gpu_uuid}

def release_assessment(family,snapshot,unreadable,nvml,lock):
    live=family.observe(snapshot);unknown_pids=[pid for pid in family.known if pid in unreadable]
    family_released=not live and not unknown_pids and not family.changes
    numeric_owned=set(family.known);owned_gpu=[];ambiguous_gpu=[]
    if nvml.get('status')=='observed':
        for entry in nvml['processes']:
            pid=entry['pid']
            if pid not in numeric_owned:continue
            current=snapshot.get(pid)
            if current and identity_key(current)!=identity_key(family.known[pid]):ambiguous_gpu.append(pid)
            else:owned_gpu.append(pid)
        gpu_released=not owned_gpu and not ambiguous_gpu
    else:gpu_released=None
    if lock.get('status')=='observed':
        owned_lock=[pid for pid in lock['holders'] if pid in numeric_owned]
        lock_released=not owned_lock
    else:owned_lock=[];lock_released=None
    return {'observed_owned_family_released':family_released,'live_owned':list(live.values()),'unreadable_owned_pids':unknown_pids,
        'owned_NVML_entries_released':gpu_released,'owned_NVML_pids':owned_gpu,'ambiguous_reused_NVML_pids':ambiguous_gpu,
        'owned_single_UUID_flock_released':lock_released,'owned_lock_holder_pids':owned_lock,
        'selected_UUID_lock_current_state':'unknown' if lock_released is None else 'unlocked' if not lock.get('holders') else 'held_by_foreign' if lock_released else 'held_by_owned',
        'owned_resource_release_verified':family_released and gpu_released is True and lock_released is True,
        'release_scope':'original child, observed descendants and original private session; no foreign process was signalled'}

def child_terminal_complete(value,permit,expected,requested,eligible,permit_file_sha256=None):
    if not isinstance(value,dict):return False
    counts=value.get('counts',{})
    if (not isinstance(counts,dict) or any(type(n) is not int or n<0 for n in counts.values())
            or set(counts)-{'new_complete','history_reuse','within_run_alias','planned_blocked'}
            or sum(counts.values())!=requested or counts.get('planned_blocked',0)!=requested-eligible):return False
    successes=sum(counts.get(k,0) for k in ('new_complete','history_reuse','within_run_alias'))
    return (value.get('status')=='complete_requested_inventory' and value.get('pid')==expected['pid']
        and value.get('start_ticks')==expected['start_ticks'] and value.get('model')==permit['model']
        and value.get('permit_id')==permit['permit_id'] and value.get('scope_id')==permit['scope_id']
        and value.get('bindings')==permit['bindings'] and value.get('evaluation_inventory_complete') is True
        and (permit_file_sha256 is None or value.get('permit_sha256')==permit_file_sha256)
        and value.get('planned',{}).get('requested')==requested and value.get('planned',{}).get('eligible')==eligible
        and value.get('unrecorded_requested')==0 and value.get('terminal_records')==requested
        and value.get('eligible_missing_prediction_records')==0 and successes==eligible
        and value.get('fatal_error') is None and not value.get('cleanup_error')
        and not value.get('terminal_accounting_error') and not value.get('terminal_close_errors')
        and not value.get('terminal_event_error'))

def original_monotonic_deadlines(permit,epoch,monotonic):
    names=('budget_origin_epoch','cpu_deadline_epoch','gpu_deadline_epoch','absolute_deadline_epoch')
    times=[permit.get(k) for k in names]
    if any(type(t) not in (int,float) or not math.isfinite(t) for t in times):raise GuardError('Original absolute permit clocks required')
    origin,cpu,gpu,end=times
    if not origin<=epoch<cpu<=end or not origin<gpu<=end or epoch>=gpu:raise GuardError('Original CPU/GPU/end budget expired; never reset')
    return {name:monotonic+(permit[name]-epoch) for name in names[1:]}

def fixed_child_command(plan):
    return [plan['python_executable'],'-u','-m',MODULE,'--config',plan['config_path'],
        '--permit',plan['permit_path'],'--permit-sha256',plan['permit_sha256']]

def local_filesystem(path,mountinfo=None):
    path=Path(path).resolve();existing=path
    while not existing.exists():existing=existing.parent
    text=Path('/proc/self/mountinfo').read_text() if mountinfo is None else mountinfo
    choices=[]
    for line in text.splitlines():
        left,right=line.split(' - ',1);fields=left.split();mount=Path(fields[4].replace('\\040',' '))
        if existing==mount or mount in existing.parents:choices.append((len(str(mount)),right.split()[0],str(mount)))
    if not choices:raise GuardError('Parent local filesystem cannot be verified')
    _,fstype,mount=max(choices)
    if fstype not in LOCAL_FS:raise GuardError('Parent output/once must be node-local, not '+fstype)
    return {'filesystem':fstype,'mountpoint':mount,'path':str(path)}

def assert_no_parent_gpu_lock_fds():
    found=[]
    for p in Path('/proc/self/fd').iterdir():
        try:target=os.readlink(p)
        except OSError:continue
        if re.search(r'GPU-[0-9a-fA-F-]+\.lock',target) and 'gpu-lock' in target:found.append(target)
    if found:raise GuardError('Parent inherited GPU lease descriptors; refusing launch')
    return True

def preflight(path,expected_sha,*,now=None):
    if os.name!='posix' or not Path('/proc/self/stat').exists():raise GuardError('Actual execution requires Linux proc/pidfds')
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):raise GuardError('Race-safe pidfd signals are required before launch')
    try:
        fd=os.pidfd_open(os.getpid(),0)
        try:signal.pidfd_send_signal(fd,0,None,0)
        finally:os.close(fd)
    except OSError as error:raise GuardError('Kernel pidfd capability is unavailable') from error
    if file_sha(path)!=expected_sha:raise GuardError('Launch-plan bytes differ')
    plan=json.loads(Path(path).read_text())
    if plan.get('schema')!='formal_single_model_watchdog_launch_v1':raise GuardError('Explicit one-model launch plan required')
    for key in ('config_path','permit_path','python_executable','helper_python_executable','nvidia_smi_executable',
                'source_root','source_manifest_path','parent_output','parent_once_file'):
        require_posix(plan.get(key))
    if any(not os.access(plan[key],os.X_OK) for key in ('python_executable','helper_python_executable','nvidia_smi_executable')):
        raise GuardError('Every fixed runtime/helper executable must be executable')
    for key in ('config_sha256','permit_sha256','python_executable_sha256','helper_python_executable_sha256',
                'nvidia_smi_executable_sha256','source_manifest_sha256','watchdog_sha256'):
        if not is_sha(plan.get(key)):raise GuardError('Missing exact launch seal '+key)
    if file_sha(__file__)!=plan['watchdog_sha256'] or file_sha(plan['python_executable'])!=plan['python_executable_sha256']:
        raise GuardError('Watchdog/interpreter bytes differ')
    for key in ('helper_python_executable','nvidia_smi_executable'):
        if file_sha(plan[key])!=plan[key+'_sha256']:raise GuardError('Local helper executable bytes differ')
        local_filesystem(plan[key])
    if Path(plan['nvidia_smi_executable']).name!='nvidia-smi':raise GuardError('Only the fixed NVML-backed nvidia-smi query is allowed')
    if file_sha(plan['config_path'])!=plan['config_sha256'] or file_sha(plan['permit_path'])!=plan['permit_sha256']:
        raise GuardError('Config/permit bytes differ')
    config=json.loads(Path(plan['config_path']).read_text());permit=json.loads(Path(plan['permit_path']).read_text())
    parent_output=PurePosixPath(plan['parent_output']);child_output=PurePosixPath(require_posix(config.get('output')))
    require_posix(config.get('once_file'))
    if (parent_output==child_output or parent_output in child_output.parents or child_output in parent_output.parents
            or plan['parent_once_file']==config['once_file']):
        raise GuardError('Parent/child output and once ownership must be separate')
    if Path(config['output']).exists() or Path(config['once_file']).exists():
        raise GuardError('Existing child output/once is retained; no replay or retry')
    if (plan.get('model') not in MODELS or config.get('model')!=plan['model'] or permit.get('model')!=plan['model']
            or permit.get('schema')!='formal1500_single_model_execution_permit_v1'
            or permit.get('execution_authorized') is not True or permit.get('single_writer_scope') is not True
            or permit.get('reviewed_bank_qualified') is not True or permit.get('history_coverage_review_complete') is not True
            or permit.get('seed')!=42 or not permit.get('permit_id') or not permit.get('scope_id')):
        raise GuardError('A separate exact reviewed single-model permit is required')
    observed={key:file_sha(plan['config_path'] if field is None else config[field]) for key,field in BINDING_PATHS.items()}
    if permit.get('bindings')!=observed:raise GuardError('Permit source/runtime/config byte bindings differ')
    resources=permit.get('resources',{})
    if (not UUID.fullmatch(str(resources.get('gpu_uuid',''))) or type(resources.get('physical_gpu')) is not int
            or not 0<=resources['physical_gpu']<=7 or resources.get('exclusive_gpu') is not True
            or resources.get('cpu_rss_hwm_cap')!=32_000_000_000 or resources.get('outer_watchdog_required') is not True):
        raise GuardError('Exact declared32GB/selected-single-UUID outer guard contract required')
    if file_sha(plan['source_manifest_path'])!=plan['source_manifest_sha256']:raise GuardError('Frozen source manifest differs')
    source_manifest=json.loads(Path(plan['source_manifest_path']).read_text());source=Path(plan['source_root']).resolve()
    files=source_manifest.get('files')
    if not isinstance(files,dict) or not files:raise GuardError('Complete frozen source file map required')
    for name,digest in files.items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or not is_sha(digest):raise GuardError('Invalid source seal entry')
        target=source/name
        if target.is_symlink() or not target.resolve().is_relative_to(source) or file_sha(target)!=digest:raise GuardError('Frozen source bytes differ')
    executor='src/deepvision_vla/diagnostics/formal_diagnostic_executor.py'
    if executor not in files:raise GuardError('The fixed formal executor must be in the frozen source map')
    actual_python={p.relative_to(source).as_posix() for p in (source/'src/deepvision_vla').rglob('*.py')}
    if not actual_python<=set(files):raise GuardError('Unsealed package Python files could shadow the frozen executor')
    scientific_files=json.loads(Path(config['scientific_source_seals']).read_text())
    for item in scientific_files:
        if not is_sha(item.get('sha256')) or file_sha(item['path'])!=item['sha256']:raise GuardError('Original scientific source changed')
    manifest=json.loads(Path(config['plan_manifest']).read_text());work=manifest['prospective_workload'][plan['model']]
    requested,eligible=work.get('requested'),work.get('eligible_proxy_requests')
    if type(requested) is not int or type(eligible) is not int or not 0<=eligible<=requested or requested<=0:
        raise GuardError('Frozen requested/eligible inventory counts required')
    epoch=time.time() if now is None else now;deadlines=original_monotonic_deadlines(permit,epoch,time.monotonic())
    for key,max_value in [('natural_exit_grace_seconds',5),('term_grace_seconds',15),('kill_grace_seconds',5)]:
        if type(plan.get(key)) not in (int,float) or not 0<=plan[key]<=max_value:raise GuardError('Finite bounded exit graces required')
    if plan.get('poll_seconds')!=1 or plan.get('phase_json_probe_timeout_seconds')!=3:raise GuardError('Fixed finite observation cadence required')
    fs_output=local_filesystem(plan['parent_output']);fs_once=local_filesystem(plan['parent_once_file'])
    assert_no_parent_gpu_lock_fds()
    final_seals={plan['config_path']:plan['config_sha256'],plan['permit_path']:plan['permit_sha256'],
        plan['python_executable']:plan['python_executable_sha256'],plan['helper_python_executable']:plan['helper_python_executable_sha256'],
        plan['nvidia_smi_executable']:plan['nvidia_smi_executable_sha256'],plan['source_manifest_path']:plan['source_manifest_sha256'],
        str(Path(__file__).resolve()):plan['watchdog_sha256'],str(path):expected_sha}
    final_seals.update({str(source/name):digest for name,digest in files.items()})
    final_seals.update({item['path']:item['sha256'] for item in scientific_files})
    final_seals.update({config[field]:observed[key] for key,field in BINDING_PATHS.items() if field is not None})
    for name,digest in final_seals.items():
        if file_sha(name)!=digest:raise GuardError('Input mutated before once/Popen')
    deadlines=original_monotonic_deadlines(permit,time.time() if now is None else now,time.monotonic())
    return plan,config,permit,{'bindings':observed,'deadlines_monotonic':deadlines,'requested':requested,'eligible':eligible,
        'source_manifest_sha256':plan['source_manifest_sha256'],'parent_output_filesystem':fs_output,'parent_once_filesystem':fs_once,
        'gpu_lock_path':str(Path('/tmp')/('deepvision-v01-gpu-locks-'+str(os.getuid()))/(resources['gpu_uuid']+'.lock'))}

def run_bounded_helper(command,timeout,*,cwd=None):
    """CPU helper only; preserve a hung helper as unknown, never spawn retries."""
    try:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            start_new_session=True,close_fds=True,cwd=cwd)
    except OSError as error:return {'status':'unknown_helper_launch','error_type':type(error).__name__}
    expected=proc_identity(child.pid)
    try:
        stdout,stderr=child.communicate(timeout=timeout)
        return {'status':'complete' if child.returncode==0 else 'failed','returncode':child.returncode,
            'stdout':stdout.decode(errors='replace'),'stderr':stderr.decode(errors='replace'),'identity':expected}
    except subprocess.TimeoutExpired:
        signals=[]
        if expected is not None:signals.append(safe_signal(expected,signal.SIGKILL))
        try:stdout,stderr=child.communicate(timeout=1)
        except subprocess.TimeoutExpired:return {'status':'unknown_hung_helper','identity':expected,'signals':signals}
        return {'status':'unknown_probe_timeout','identity':expected,'signals':signals}

def isolated_json_read(path,timeout,probe_runtime):
    command=[probe_runtime['python'],'-u',probe_runtime['script'],'_json_probe','--path',str(path)]
    result=run_bounded_helper(command,timeout,cwd=probe_runtime['cwd'])
    if result['status']!='complete':return {'status':'unknown','helper':result}
    try:return json.loads(result['stdout'])
    except ValueError:return {'status':'unknown','reason':'probe_output_not_json'}

def observe_selected_nvml(gpu_uuid,probe_runtime):
    result=run_bounded_helper([probe_runtime['nvidia_smi'],'-q','-x','-i',gpu_uuid],3,cwd=probe_runtime['cwd'])
    if result['status']!='complete':return {'status':'unknown','reason':'NVML_query_failed_or_timed_out','helper':result}
    return parse_nvml_xml(result['stdout'],gpu_uuid)

def phase_deadline_stop(phase,permit,expected,kind):
    if phase.get('status')!='observed':return kind+'_phase_unknown_at_original_deadline'
    value=phase.get('value',{})
    if (value.get('pid')!=expected['pid'] or value.get('start_ticks')!=expected['start_ticks']
            or value.get('permit_id')!=permit['permit_id'] or value.get('scope_id')!=permit['scope_id']):
        return kind+'_phase_identity_unknown_at_original_deadline'
    if kind=='cpu':
        if value.get('model_loads_attempted')==0:return None  # Cached-only; original child refuses future late loads.
        if type(value.get('model_load_count')) is int and value['model_load_count']>=1:return None
        return 'original_cpu_load_deadline'
    if value.get('lease_released') is True:return None
    if value.get('lease_acquired') is True or value.get('cuda_initialized') is True:return 'original_gpu_deadline'
    if value.get('model_loads_attempted')==0:return None
    return 'gpu_phase_unknown_at_original_deadline'

def reap_owned_zombies(family,snapshot):
    for pid,expected in family.known.items():
        value=snapshot.get(pid)
        if (pid!=family.root['pid'] and value is not None and identity_key(value)==identity_key(expected)
                and value['state']=='Z' and value['ppid']==os.getpid()):
            try:os.waitpid(pid,os.WNOHANG)
            except ChildProcessError:pass

def monitor_private_child(child,expected,*,deadlines,cap,natural_grace,term_grace,kill_grace,poll_seconds=1,
        phase_probe=None,permit=None,on_event=lambda value:None,family=None):
    """Monitor an already created private session; production argv is elsewhere."""
    family=family or OwnedFamily(expected);stop=None;signals=[];checked=set();rss_peak=0;hwm_peak=0
    while child.poll() is None:
        snapshot,unreadable=proc_snapshot();live=family.observe(snapshot)
        reap_owned_zombies(family,snapshot)
        rss=sum(v['rss_bytes'] for v in live.values());hwm=max((v['rss_hwm_bytes'] or 0 for v in live.values()),default=0)
        rss_peak=max(rss_peak,rss);hwm_peak=max(hwm_peak,hwm)
        on_event({'event':'owned_resource_observation','owned':list(live.values()),'rss_sum_bytes':rss,'maximum_process_VmHWM_bytes':hwm,
            'unreadable_owned_pids':[pid for pid in family.known if pid in unreadable]})
        now=time.monotonic()
        if any(pid in unreadable for pid in family.known):stop='owned_resource_observation_unavailable';break
        if rss>cap or hwm>cap:stop='declared_owned_CPU_RSS_or_HWM_cap_exceeded';break
        if family.changes:stop='owned_identity_or_group_changed';break
        if now>=deadlines['absolute_deadline_epoch']:stop='original_absolute_end_deadline';break
        if phase_probe is not None:
            for kind in ('cpu','gpu'):
                key=kind+'_deadline_epoch'
                if key not in checked and now>=deadlines[key]:
                    checked.add(key);reason=phase_deadline_stop(phase_probe(kind),permit,expected,kind)
                    on_event({'event':'original_phase_deadline_observed','phase':kind,'stop_reason':reason})
                    if reason:stop=reason;break
            if stop:break
        time.sleep(poll_seconds)
    natural_return=child.poll()
    # No broad killpg: each current identity is pidfd-bound and individually reverified.
    initial_deadline=time.monotonic()+natural_grace
    while time.monotonic()<initial_deadline:
        snapshot,unreadable=proc_snapshot();live=family.observe(snapshot)
        reap_owned_zombies(family,snapshot)
        child.poll()
        if not live:break
        time.sleep(min(.1,max(0,initial_deadline-time.monotonic())))
    snapshot,unreadable=proc_snapshot();live=family.observe(snapshot)
    if live:
        if stop is None:stop='owned_family_left_after_natural_child_exit'
        for sig,grace in ((signal.SIGTERM,term_grace),(signal.SIGKILL,kill_grace)):
            snapshot,unreadable=proc_snapshot();live=family.observe(snapshot)
            for value in sorted(live.values(),key=lambda v:v['pid']==expected['pid']):
                result=safe_signal(family.known[value['pid']],sig);signals.append(result);on_event({'event':'owned_signal',**result})
            until=time.monotonic()+grace
            while time.monotonic()<until:
                child.poll();snapshot,unreadable=proc_snapshot();live=family.observe(snapshot)
                # Reap only exactly known zombies that became direct children.
                reap_owned_zombies(family,snapshot)
                if not live:break
                time.sleep(min(.1,max(0,until-time.monotonic())))
            if not live:break
    child.poll();snapshot,unreadable=proc_snapshot();family.observe(snapshot);reap_owned_zombies(family,snapshot)
    snapshot,unreadable=proc_snapshot();live=family.observe(snapshot)
    return family,{'returncode':child.returncode,'natural_returncode_before_cleanup':natural_return,
        'stop_reason':stop,'cleanup_signals':signals,'observed_owned_processes':list(family.known.values()),
        'live_owned_after_cleanup':list(live.values()),'identity_changes':family.changes,
        'observed_owned_RSS_sum_peak_bytes':rss_peak,'observed_max_process_VmHWM_peak_bytes':hwm_peak,
        'memory_scope':'actual observed owned RSS sum plus per-process VmHWM; shared pages may be conservatively double-counted'}

def final_parent_status(returncode,terminal_complete,release,stop_reason,probe_unknown=False):
    if stop_reason:return 'failed_parent_guard',2
    if returncode!=0:return 'failed_child_exit',2
    if not terminal_complete:return 'partial_or_unknown_child_inventory',2
    if probe_unknown or release.get('owned_resource_release_verified') is not True:return 'unknown_owned_resource_release',3
    return 'complete_requested_inventory_independently_released',0

def execute(path,expected_sha):
    plan,config,permit,evidence=preflight(path,expected_sha)
    if plan.get('launch_authorized') is not True:raise GuardError('Root launch plan has not authorized execution')
    out=Path(plan['parent_output']);out.mkdir(parents=True,exist_ok=False)
    helper_source=Path(__file__).read_bytes()
    if hashlib.sha256(helper_source).hexdigest()!=plan['watchdog_sha256']:raise GuardError('Parent helper source changed before launch')
    helper_script=out/'watchdog_frozen.py'
    with helper_script.open('xb') as f:f.write(helper_source)
    helper_script.chmod(0o444)
    probe_runtime={'python':plan['helper_python_executable'],'script':str(helper_script),
        'nvidia_smi':plan['nvidia_smi_executable'],'cwd':str(out)}
    once=Path(plan['parent_once_file']);once.parent.mkdir(parents=True,exist_ok=True)
    parent=proc_identity(os.getpid());state={'schema':'formal_single_model_watchdog_result_v1','status':'starting',
        'parent':parent,'launch_plan_sha256':expected_sha,'model':plan['model'],'evidence':evidence,
        'parent_holds_GPU_lease':False,'GPU_lease_owner_policy':'only original child NativeRuntime',
        'command':fixed_child_command(plan),'child_attempts':0,'model_imports_in_parent':0,'scope_clock_epoch':time.time(),
        'node_local_helper_runtime':probe_runtime,'subreaper':enable_subreaper()}
    with once.open('x') as f:json.dump({'launch_plan_sha256':expected_sha,'parent':parent,'permit_sha256':plan['permit_sha256']},f);f.flush();os.fsync(f.fileno())
    events=(out/'parent_events.jsonl').open('x')
    def event(value):events.write(json.dumps({'epoch':time.time(),**value},separators=(',',':'))+'\n');events.flush()
    def save():
        p=out/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(out/'state.json')
    child=None;family=None;expected=None;monitor=None
    def interrupted(_s,_f):raise InterruptedError('Root parent watchdog interrupted; no retry')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        # No CFS reads after this point in the parent. Only bounded helper sessions
        # read progress/result JSON; source/runtime/config were sealed above.
        env=dict(os.environ,PYTHONPATH=str(Path(plan['source_root'])/'src'),PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        (out/'tmp').mkdir();env.update(TMPDIR=str(out/'tmp'),TEMP=str(out/'tmp'),TMP=str(out/'tmp'))
        with (out/'child_stdout.log').open('x') as log:
            child=subprocess.Popen(fixed_child_command(plan),env=env,cwd=out,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            expected=proc_identity(child.pid);state.update(status='running',child=expected,child_attempts=1);save()
            if expected is None:raise GuardError('Child identity disappeared before it could be proven')
            family=OwnedFamily(expected)
            family,monitor=monitor_private_child(child,expected,deadlines=evidence['deadlines_monotonic'],
                cap=permit['resources']['cpu_rss_hwm_cap'],natural_grace=plan['natural_exit_grace_seconds'],
                term_grace=plan['term_grace_seconds'],kill_grace=plan['kill_grace_seconds'],on_event=event,
                phase_probe=lambda kind:isolated_json_read(str(PurePosixPath(config['output'])/'progress.json'),3,probe_runtime),permit=permit,family=family)
            state['monitor']=monitor
    except BaseException as error:
        state.update(parent_error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        if child is not None and expected is not None and monitor is None:
            family,monitor=monitor_private_child(child,expected,deadlines={'absolute_deadline_epoch':time.monotonic()-1},
                cap=permit['resources']['cpu_rss_hwm_cap'],natural_grace=0,term_grace=plan['term_grace_seconds'],kill_grace=plan['kill_grace_seconds'],on_event=event,family=family)
            state['monitor']=monitor
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if family is not None:
            nvml=observe_selected_nvml(permit['resources']['gpu_uuid'],probe_runtime)
            lock=kernel_lock_snapshot(evidence['gpu_lock_path']);snapshot,unreadable=proc_snapshot()
            release=release_assessment(family,snapshot,unreadable,nvml,lock)
            state.update(NVML_release_probe=nvml,single_UUID_kernel_flock_probe=lock,independent_release=release)
        else:release={'owned_resource_release_verified':False};state['independent_release']=release
        terminal=isolated_json_read(str(PurePosixPath(config['output'])/'result.json'),3,probe_runtime) if child is not None else {'status':'unknown','reason':'no_child'}
        valid=(terminal.get('status')=='observed' and expected is not None and
            child_terminal_complete(terminal['value'],permit,expected,evidence['requested'],evidence['eligible'],plan['permit_sha256']))
        state.update(child_terminal_probe=terminal,child_complete_inventory_qualified=valid)
        stop=(monitor or {}).get('stop_reason') or ('parent_exception' if state.get('parent_error') else None)
        status,exitcode=final_parent_status(child.returncode if child else None,valid,release,stop,
            probe_unknown=terminal.get('status')!='observed')
        state.update(status=status,parent_exitcode=exitcode,finished_epoch=time.time())
        assert_no_parent_gpu_lock_fds();save();events.close()
        with (out/'result.json').open('x') as f:json.dump(state,f,indent=2)
    return state

def json_probe(path):
    try:
        with Path(path).open('rb') as f:raw=f.read(8_000_001)
        if len(raw)>8_000_000:raise ValueError('Scalar child state exceeds8MB cap')
        value=json.loads(raw)
        return {'status':'observed','value':value,'sha256':hashlib.sha256(raw).hexdigest()}
    except (OSError,ValueError) as error:return {'status':'unknown','reason':type(error).__name__,'message':str(error)}

def main():
    if len(sys.argv)>1 and sys.argv[1]=='_json_probe':
        parser=argparse.ArgumentParser();parser.add_argument('--path',required=True)
        args=parser.parse_args(sys.argv[2:]);print(json.dumps(json_probe(args.path)));return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-plan',required=True);parser.add_argument('--launch-plan-sha256',required=True)
    parser.add_argument('--execute',action='store_true',help='Root-only one-shot execution; absent means validation only')
    args=parser.parse_args()
    if args.execute:
        result=execute(args.launch_plan,args.launch_plan_sha256);print(json.dumps(result));raise SystemExit(result['parent_exitcode'])
    plan,config,permit,evidence=preflight(args.launch_plan,args.launch_plan_sha256)
    print(json.dumps({'status':'validated_not_executed','command':fixed_child_command(plan),'evidence':evidence,'once_consumed':False,'child_Popen_calls':0}))

if __name__=='__main__':main()
