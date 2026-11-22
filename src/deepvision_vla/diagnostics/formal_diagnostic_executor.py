"""Explicitly permitted single-model fixed1500 executor; never a dispatcher.

The default remains data-only: this entry requires a separate root-issued permit
and immutable reviewed evidence. It imports real backends only for an actually
new computation; all cache-only work avoids another model load/GPU lease.
"""
import argparse
from collections import Counter
from contextlib import ExitStack,contextmanager
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys
import time
import traceback

from . import formal_execution_core as core
from .formal_plan import canonical_sha,pi_measured_input_verified
from .geometry import VisualLayout
from deepvision_vla.diagnostics.input_inventory import validate_linked_inventory, safe_payload_path
from deepvision_vla.diagnostics.predictors import ModelOnce, chunk_drift, chunk_bits, HORIZON
from deepvision_vla.diagnostics.factory_imports import configure_factory_imports
from deepvision_vla.diagnostics.resources import require_available
from deepvision_vla.diagnostics.resources import require_own_memory
from deepvision_vla.diagnostics.runtime_observation import actual_memory
from .resources import verify_cuda_pci,pci_identity
from .random_controls import validate_bindings,EXCLUSION
from ..runtime.gpu_lease import gpu_lease


def load_rows(path):
    raw=Path(path).read_bytes()
    return json.loads(raw)if raw.lstrip().startswith(b'[')else[json.loads(line)for line in raw.splitlines()if line.strip()]


def validate_runtime_data(config,manifest,profile):
    inventory=load_rows(config['inventory'])
    if core.file_sha(config['inventory'])!=manifest['inventory_sha256']:
        raise ValueError('Formal plan original inventory byte seal differs')
    root=Path(config['data_root']);staged=json.loads((root/'inventory.json').read_text(encoding='utf-8'))
    original=[json.loads(x)for x in (root/'manifest.jsonl').read_bytes().splitlines()if x.strip()]
    files=json.loads((root/'files.json').read_text(encoding='utf-8'));receipt=json.loads((root/'transfer_receipt.json').read_text(encoding='utf-8'))
    linked=validate_linked_inventory(staged,original,files,receipt,core.file_sha(root/'inventory.json'),core.file_sha(root/'manifest.jsonl'))
    if inventory!=staged:raise ValueError('Execution inventory must preserve exact original staged rows')
    geometries={}
    for row in load_rows(config['geometries']):
        if row['model']!=config['model']:continue
        key=row['sample_id']
        if key in geometries:raise ValueError('Duplicate same-model geometry')
        geometries[key]=row
    for source in inventory:
        geometry=geometries.get(source['sample_id'])
        if geometry is None:continue  # Request plan retains qualification blocking, not a replacement source.
        if (geometry.get('model_identity_sha256')!=canonical_sha(profile['identity'])
                or geometry.get('source_row_sha256')!=canonical_sha(source)):
            raise ValueError('Geometry/source/model identity mismatch')
    return inventory,geometries,linked


def random_control_stream_kwargs(config,manifest):
    """Bind explicit v2 metadata before staged-data checks, once or backends."""
    fields=('random_control_metadata','random_control_metadata_sha256')
    if manifest.get('include_random_controls') is not True:
        if any(name in config for name in fields):
            raise ValueError('v1 config cannot supply undeclared random control metadata')
        return {}
    if manifest.get('schema')!='formal1500_data_only_request_plan_v2':
        raise ValueError('Random controls require an explicit v2 manifest')
    bindings=validate_bindings(manifest.get('random_control_evidence'),manifest.get('models',[]))
    path=config.get('random_control_metadata');declared=config.get('random_control_metadata_sha256')
    if not isinstance(path,str) or not path.strip() or not core.is_sha(declared):
        raise ValueError('v2 config requires explicit random metadata path and SHA256')
    if declared!=bindings['metadata_sha256'] or core.file_sha(path)!=declared:
        raise ValueError('Config/plan/current random metadata byte seals differ')
    return {'random_control_metadata_path':path}


def validate_random_control_runtime_evidence(config,manifest,inventory,geometries,profile,observed):
    """Join sealed metadata to bank/layout receipts without recomputing masks."""
    binding=manifest['random_control_evidence'];model=config['model']
    if binding['source_bank_sha256']!=observed['annotations_sha256']:
        raise ValueError('Random controls refer to another source-bank byte revision')
    original_ids={source['sample_id'] for source in inventory};metadata={}
    for row in load_rows(config['random_control_metadata']):
        key=(row.get('sample_id'),row.get('model'))
        if key in metadata or key[0]not in original_ids or key[1]not in core.LAYERS:
            raise ValueError('Random metadata has duplicate/foreign source/model identity')
        metadata[key]=row
    annotations={}
    for row in load_rows(config['annotations']):
        key=row.get('sample_id')
        if key in annotations or key not in original_ids:raise ValueError('Source-bank identity differs')
        annotations[key]=row
    for source in inventory:
        sid=source['sample_id'];row=metadata.get((sid,model));annotation=annotations.get(sid);geometry=geometries.get(sid)
        if row is None or annotation is None or geometry is None:
            raise ValueError('Complete selected-model random source/bank/geometry evidence required before once')
        roles=annotation.get('roles')
        if not isinstance(roles,list):raise ValueError('Bound original source roles are missing')
        roles_sha=canonical_sha(roles)
        if (annotation.get('roles_sha256',roles_sha)!=roles_sha
                or annotation.get('review',{}).get('roles_sha256')!=roles_sha
                or row.get('relevant_roles_sha256')!=roles_sha):
            raise ValueError('Random metadata original source role dictionary differs')
        if (geometry.get('actual_geometry_verified')is not True or geometry.get('geometry_source')!='actual_processor'
                or not isinstance(geometry.get('audit_link'),str) or not geometry['audit_link'].strip()
                or not core.is_sha(geometry.get('audit_sha256'))
                or geometry.get('source_row_sha256')!=canonical_sha(source)
                or geometry.get('model_identity_sha256')!=canonical_sha(profile['identity'])
                or geometry.get('processor_sha256')!=profile['identity']['processor_sha256']):
            raise ValueError('Actual random-control geometry/model/source receipt is unbound')
        layout=core.restore_visual_layout(geometry['layout'])
        if layout.image_sha256!=source['image_sha256'] or list(layout.source_size)!=source['image_size']:
            raise ValueError('Actual random-control layout belongs to another source image')
        expected={'ordinal':source['ordinal'],'sample_id':sid,'model':model,
            'image_sha256':source['image_sha256'],'source_inventory_row_sha256':canonical_sha(source),
            'source_bank_sha256':binding['source_bank_sha256'],
            'model_identity_sha256':canonical_sha(profile['identity']),
            'processor_sha256':profile['identity']['processor_sha256'],
            'actual_geometry_file_sha256':binding['geometry_file_sha256_by_model'][model],
            'actual_geometry_audit_sha256':geometry['audit_sha256'],'actual_layout':geometry['layout'],
            'control_kind':'random_nonROI_visual','primary_membership':'patch_center','random_exclusion_rule':EXCLUSION,
            'logical_layers_same_schedule':list(core.LAYERS[model]),'semantic_background_claim':False,
            'human_verified':False,'policy_results_used':False,'NN_dispatch_authorized':False,
            'runtime_GT_and_probe_target_gates_not_checked':True,'new_processor_calls':0}
        if any(canonical_sha(row.get(key))!=canonical_sha(value) for key,value in expected.items()):
            raise ValueError('Random metadata source/roles/actual geometry binding mismatch')
    if core.file_sha(config['random_control_metadata'])!=config['random_control_metadata_sha256']:
        raise ValueError('Random metadata changed during evidence validation')


def read_source(source,geometry,config,file_map):
    root=Path(config['data_root']);payload={}
    for field,key in (('image_path','image_sha256'),('arrays_path','arrays_sha256')):
        relative=source['original_source'][field];path=safe_payload_path(root,relative);raw=path.read_bytes()
        if len(raw)!=file_map[relative]['bytes']or hashlib.sha256(raw).hexdigest()!=source[key]:
            raise ValueError('Current original source bytes differ: '+field)
        payload[field]=raw
    from PIL import Image
    import numpy as np
    with Image.open(io.BytesIO(payload['image_path']))as original:
        original.load()
        if list(original.size)!=source['image_size']:raise ValueError('Original image dimensions differ')
        image=original.convert('RGB')
    with np.load(io.BytesIO(payload['arrays_path']),allow_pickle=False)as arrays:
        values={key:arrays[key].copy()for key in ('states','actions')}
    steps=source['original_source'].get('steps')
    for value in values.values():
        if value.ndim!=2 or value.shape[1]!=7 or not np.isfinite(value).all()or(steps is not None and len(value)!=steps):
            raise ValueError('Actual original NPZ shape/finite/episode length differs')
    anchor=source['anchor']
    if type(anchor)is not int or not 0<=anchor<len(values['states']):raise ValueError('Actual original anchor missing')
    if (not geometry or geometry.get('actual_geometry_verified')is not True or geometry.get('geometry_source')!='actual_processor'
            or geometry.get('image_sha256')!=source['image_sha256']or geometry.get('arrays_sha256')!=source['arrays_sha256']):
        raise ValueError('Reviewed actual same-input processor geometry required')
    if config['model']=='pi0':
        audit=geometry.get('state_input_audit')
        if not pi_measured_input_verified(source,geometry):raise ValueError('Actual measured Pi state input proof is missing')
        observed=np.asarray(audit['measured_state7'],np.float32)
        if observed.shape!=(7,)or observed.tobytes()!=values['states'][anchor].astype(np.float32).tobytes():
            raise ValueError('Current measured Pi anchor state differs from accepted native preprocessing proof')
    layout=core.restore_visual_layout(geometry['layout'])
    return {'image':image,**values},layout


def calculate_metrics(prediction,baseline,source,data,row,profile):
    """Fresh reference metrics never become part of the NN-success identity."""
    result={'native_prediction_drift':chunk_drift(prediction['native'],baseline['native']),
        'gt_mse':None,'gt_mapping_verified':False,'source_bound_dataset_reference_mse':None,
        'source_bound_dataset_reference_first_step_mse':None,'dataset_reference_status':'no_sealed_dataset_reference_artifact'}
    if profile['identity']['model']=='qwen_h16':
        result['normalized_prediction_drift']=chunk_drift(prediction['normalized'],baseline['normalized'])
    if row.get('gt_mse_allowed'):
        if profile['identity']['model']!='openvla'or profile.get('action_mapping',{}).get('gt_mapping_verified')is not True:
            raise ValueError('This protocol permits only its explicitly verified official OpenVLA GT mapping')
        from .adapters.openvla_runtime import reference_action,ACTION_SPACE
        from .contracts import action_mse
        try:
            target=reference_action(data['states'].tolist(),data['actions'].tolist(),source['anchor'])
            result.update(gt_mse=action_mse(prediction['native'][0],target,ACTION_SPACE,ACTION_SPACE),
                gt_mapping_verified=True,reference_action=target,gt_mse_status='verified_official_reference')
        except ValueError as error:
            result.update(gt_mse=None,gt_mapping_verified=False,gt_mse_status='official_reference_unavailable_no_padding',
                gt_reference_reason=str(error))
    if row['keep_ratio']==1:result['keep1_baseline_bit_exact']=validate_keep1(prediction,baseline)
    return result


def validate_keep1(prediction,baseline):
    exact=chunk_bits(prediction['native'])==chunk_bits(baseline['native'])
    if 'normalized'in prediction:exact=exact and chunk_bits(prediction['normalized'])==chunk_bits(baseline['normalized'])
    if not exact:raise RuntimeError('Keep1 intervention changed the deterministic baseline bits')
    return True


def evaluate_qualified_metrics(prediction,baseline,source,data,row,profile,references=None):
    """Keep1 is a protocol gate, never caught as a metric-only failure."""
    if row['keep_ratio']==1:validate_keep1(prediction,baseline)
    try:
        metrics=calculate_metrics(prediction,baseline,source,data,row,profile)
        if references is not None:metrics.update(references.evaluate(prediction,source,profile['identity']['model']))
        return metrics,None
    except Exception as error:
        return {'metric_status':'metric_evaluation_error_raw_prediction_preserved','metric_error_type':type(error).__name__,
            'metric_error':str(error),'gt_mse':None,'source_bound_dataset_reference_mse':None},error


@contextmanager
def observe_return(owner,name,capture):
    """Delegate the genuine method and return the SAME object, restoring attrs."""
    original=getattr(owner,name);had=name in owner.__dict__;value=owner.__dict__.get(name)
    def observed(*args,**kwargs):
        capture['calls']=capture.get('calls',0)+1
        if capture['calls']!=1:
            capture['blocked_duplicate_calls']=capture.get('blocked_duplicate_calls',0)+1
            raise RuntimeError('Unexpected duplicate native prediction entry was blocked')
        capture['native_calls_executed']=capture.get('native_calls_executed',0)+1
        result=original(*args,**kwargs)
        capture.update(returned=True,returned_calls=capture.get('returned_calls',0)+1,
            returned_object=result,returned_object_id=id(result),method=name)
        return result
    setattr(owner,name,observed)
    try:yield
    finally:
        if had:setattr(owner,name,value)
        else:delattr(owner,name)


def native_observer_target(engine,model):
    if model=='openvla':return engine.adapter.model,'predict_action'
    if model=='pi0':return engine.adapter.policy,'select_action'
    return engine.adapter,'predict_normalized'


def _json_numeric_values(value):
    if isinstance(value,list):return [_json_numeric_values(x)for x in value]
    if type(value)in(int,float):
        if math.isfinite(value):return value
        return {'nonfinite':'nan'if math.isnan(value)else'+inf'if value>0 else'-inf'}
    raise ValueError('Observed native output is not numeric')


def take_unqualified_return(capture,model,engine):
    """No NN/tokenizer replay. Only actual returned values/shared normal decode."""
    value=capture.pop('returned_object',None)
    if not capture.get('returned')or value is None:return None
    shape=list(value.shape)if hasattr(value,'shape')else None
    dtype=str(value.dtype)if hasattr(value,'dtype')else'python_native_values'
    if hasattr(value,'detach'):plain=value.detach().cpu().tolist()
    elif hasattr(value,'tolist'):plain=value.tolist()
    else:plain=value
    artifact={'qualification':'RAW_UNQUALIFIED','native_method':capture['method'],
        'native_method_calls':capture['native_calls_executed'],'native_method_attempted_entries':capture['calls'],
        'blocked_duplicate_calls':capture.get('blocked_duplicate_calls',0),'returned_calls':capture['returned_calls'],
        'returned_object_id':capture['returned_object_id'],'observed_shape':shape,'observed_dtype':dtype,
        'raw_values':_json_numeric_values(plain),'raw_value_encoding':'finite_json_numbers_with_explicit_nonfinite_tags',
        'verified_cache_success':False,'prediction':None}
    try:
        if model=='openvla':prediction={'native':[plain]}
        elif model=='pi0':prediction={'native':plain[0]}
        else:
            from .adapters.qwen_h16 import decode_h16_chunk
            prediction={'normalized':plain,'native':decode_h16_chunk(plain,engine.adapter.statistics)}
            artifact['native_value_provenance']='unchanged shared decode_h16_chunk of the actual returned normalized values; no NN replay'
        artifact['prediction']=core.prediction_payload(prediction,model)
    except (ValueError,TypeError,IndexError,KeyError)as error:artifact['native_shape_or_decode_error']=type(error).__name__
    return artifact


def verify_factory_assets(config):
    """Bind actual native weight paths to the separately sealed asset receipt."""
    model=config['model'];factory=config['factory_kwargs']
    if model in('openvla','pi0'):
        if Path(factory['asset_receipt']).resolve()!=Path(config['asset_receipt']).resolve():
            raise ValueError('Actual factory uses a different asset receipt')
        root=Path(factory['checkpoint'])
    else:root=Path(factory['checkpoint_root'])
    prefix=config.get('asset_receipt_prefix','').rstrip('/')
    receipt=json.loads(Path(config['asset_receipt']).read_text(encoding='utf-8'));files=receipt['files'];count=0
    for item in files:
        logical=item['path']
        if prefix:
            if not logical.startswith(prefix+'/'):continue
            logical=logical[len(prefix)+1:]
        if not logical.endswith(('.safetensors','.bin','.pt','.pth','.ckpt')):continue
        path=safe_payload_path(root,logical)
        if item.get('verified')is not True or path.stat().st_size!=item['size']or core.file_sha(path)!=item['sha256']:
            raise ValueError('Actual native factory weight bytes differ from its pinned acquisition')
        count+=1
    if count!={'openvla':3,'pi0':1,'qwen_h16':2}[model]:
        raise ValueError('Selected native weight components are absent/ambiguous; no prefix guessing')
    return count


class NativeRuntime:
    """Lazy genuine CPUfactory→single exclusive GPUlease; no pool or retries."""
    def __init__(self,config,permit,state):
        self.config,self.permit,self.state=config,permit,state;self.engine=None;self.stack=ExitStack()
        self.persist=lambda:None
        self.monotonic_limits={name:state['scope_clock_monotonic']+(permit[name]-state['scope_clock_epoch'])
            for name in('cpu_deadline_epoch','gpu_deadline_epoch','absolute_deadline_epoch')}
    def begin(self):
        if self.engine is not None:return
        p=self.permit;r=p['resources'];physical=r['physical_gpu']
        if time.time()>=p['cpu_deadline_epoch']or time.monotonic()>=self.monotonic_limits['cpu_deadline_epoch']:
            raise TimeoutError('Original absolute CPU load budget expired')
        if time.time()>=p['gpu_deadline_epoch']or time.monotonic()>=self.monotonic_limits['gpu_deadline_epoch']:
            raise TimeoutError('Original GPU deadline already expired; do not materialize another model')
        self.state['memory_before_cpu_load']=actual_memory(True)
        selected=require_available(physical,r['gpu_uuid'],sharing=False,minimum_free_bytes=r['load_free_gpu_bytes_min'])
        if pci_identity(selected['pci'])!=pci_identity(r['gpu_pci']):raise RuntimeError('Permit PCI differs from selected physicalGPU')
        os.environ['CUDA_VISIBLE_DEVICES']=str(physical);os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        if hasattr(os,'sched_getaffinity'):
            allowed=sorted(os.sched_getaffinity(0))[:8]
            if not allowed:raise RuntimeError('No allowed CPU affinity')
            os.sched_setaffinity(0,allowed)
            self.state['actual_cpu_affinity']=sorted(os.sched_getaffinity(0))
        configure_factory_imports(self.config['model'],self.config['factory_kwargs'])
        self.state['phase']='verifying_native_asset_bytes';self.persist()
        self.state['actual_weight_components_rehashed']=verify_factory_assets(self.config)
        if time.time()>=p['cpu_deadline_epoch']or time.monotonic()>=self.monotonic_limits['cpu_deadline_epoch']:
            raise TimeoutError('Original CPU deadline expired during asset verification; no model materialization')
        self.state['phase']='cpu_loading';self.state['model_loads_attempted']+=1;self.persist()
        self.engine=ModelOnce(self.config['model'],self.config['factory_kwargs'])
        if self.engine.torch.get_num_threads()!=8:raise RuntimeError('The unchanged real factory must truthfully use8TorchCPUthreads')
        self.state['model_load_count']+=1;self.state['memory_after_cpu_load']=actual_memory()
        self.persist()
        if time.time()>=p['cpu_deadline_epoch']or time.monotonic()>=self.monotonic_limits['cpu_deadline_epoch']:
            raise TimeoutError('Original CPU load deadline expired during materialization')
        runtime=json.loads(Path(self.config['runtime_receipt']).read_text(encoding='utf-8'))
        versions=runtime.get('versions',{})
        if not {'torch','transformers'}.issubset(versions):raise ValueError('Actual native Torch/Transformers runtime evidence is required')
        for name,expected in versions.items():
            if name not in('torch','transformers','timm','tokenizers'):
                raise ValueError('Unsupported runtime version field must not be silently ignored: '+name)
            module=sys.modules.get(name)
            actual=module.__version__ if module is not None else importlib.metadata.version(name)
            if actual!=expected:raise RuntimeError('Native runtime version differs: '+name)
        if self.config['model']=='pi0':
            noise=hashlib.sha256(self.engine.noise.numpy().tobytes()).hexdigest()
        else:noise=canonical_sha({'kind':'deterministic_no_external_noise','seed':42})
        if noise!=p['noise_identity_sha256']:raise RuntimeError('Actual seed/noise bytes differ from the permitted/history identity')
        require_available(physical,r['gpu_uuid'],sharing=False,minimum_free_bytes=r['load_free_gpu_bytes_min'])
        if time.time()>=p['gpu_deadline_epoch']or time.monotonic()>=self.monotonic_limits['gpu_deadline_epoch']:
            raise TimeoutError('Original GPU phase deadline expired before lease')
        self.state['lease']=self.stack.enter_context(gpu_lease(r['gpu_uuid']));self.state['lease_acquired']=True
        require_available(physical,r['gpu_uuid'],sharing=False,minimum_free_bytes=r['load_free_gpu_bytes_min'])
        self.state['cuda_binding']=verify_cuda_pci(r['gpu_pci']);torch=self.engine.torch;torch.cuda.init()
        if torch.cuda.device_count()!=1:raise RuntimeError('Expected exactly the selected physicalGPU')
        torch.cuda.set_per_process_memory_fraction(r['allocator_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
        self.engine.to_gpu();self.state['phase']='evaluating';self.budget()
        self.persist()
    def budget(self):
        if (time.time()>=min(self.permit['gpu_deadline_epoch'],self.permit['absolute_deadline_epoch'])
                or time.monotonic()>=min(self.monotonic_limits['gpu_deadline_epoch'],self.monotonic_limits['absolute_deadline_epoch'])):
            raise TimeoutError('Original absolute GPU/execution deadline expired')
        r=self.permit['resources'];info=require_available(r['physical_gpu'],r['gpu_uuid'],sharing=False,
            minimum_free_bytes=r['free_gpu_bytes_min'],maximum_own_bytes=r['observed_gpu_bytes_cap'])
        self.state['last_observed_own_gpu_bytes']=require_own_memory(info,os.getpid())
        self.state['last_cpu_memory_observation']=actual_memory()
    def close(self):
        try:
            if self.engine is not None:self.engine.close()
            self.engine=None;gc.collect()
            torch=sys.modules.get('torch')
            if torch is not None and torch.cuda.is_initialized():torch.cuda.empty_cache()
        finally:
            self.stack.close()
            if self.state.get('lease_acquired'):self.state['lease_released']=True
            torch=sys.modules.get('torch');initialized=torch.cuda.is_initialized()if torch is not None else False
            self.state['cuda_initialized']=initialized
            self.state['cuda_context_release_requires_parent_observed_process_exit']=initialized


def observed_bindings(config,config_path,manifest_path):
    paths={'config_sha256':config_path,'plan_manifest_sha256':manifest_path,
        'requests_jsonl_sha256':config['requests'],'inventory_sha256':config['inventory'],
        'profiles_sha256':config['profiles'],'annotations_sha256':config['annotations'],
        'geometries_sha256':config['geometries'],'scientific_source_sha256':config['scientific_source_seals'],
        'runtime_sha256':config['runtime_receipt'],'asset_receipt_sha256':config['asset_receipt'],
        'output_codec_sha256':config['output_codec'],'history_manifest_sha256':config['history_manifest']}
    return {key:core.file_sha(path)for key,path in paths.items()}


def claim_once(path,permit,permit_file_sha256=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    value={'permit_id':permit['permit_id'],'scope_id':permit['scope_id'],'pid':os.getpid(),
        'epoch':time.time(),'permit_semantic_sha256':canonical_sha(permit),'permit_file_sha256':permit_file_sha256}
    with path.open('x')as stream:json.dump(value,stream);stream.flush();os.fsync(stream.fileno())
    core.sync_directory(path.parent)
    return value


def run(config_path,permit_path,permit_sha):
    clock_epoch,clock_mono=time.time(),time.monotonic()
    config=json.loads(Path(config_path).read_text(encoding='utf-8'));permit=core.sealed_json(permit_path,permit_sha)
    observed=observed_bindings(config,config_path,config['plan_manifest'])
    core.validate_permit(permit,observed)
    if config['model']!=permit['model']:raise ValueError('Separate permit allows one selected model only')
    manifest=json.loads(Path(config['plan_manifest']).read_text(encoding='utf-8'));profiles=json.loads(Path(config['profiles']).read_text(encoding='utf-8'))
    profile=next(p for p in profiles if p['identity']['model']==config['model'])
    if canonical_sha(profile.get('action_mapping',{}))!=profile['identity'].get('action_mapping_sha256'):
        raise ValueError('Actual metric/action mapping differs from its sealed profile identity')
    evidence=manifest.get('evidence_input_sha256',{})
    if any(evidence.get(name)!=observed[name+'_sha256']for name in('profiles','annotations','geometries')):
        raise ValueError('Plan was not compiled from these exact reviewed profile/bank/geometry bytes')
    random_kwargs=random_control_stream_kwargs(config,manifest)
    inventory,geometries,file_map=validate_runtime_data(config,manifest,profile)
    if canonical_sha([s['sample_id']for s in inventory])!=manifest.get('original_id_order_sha256'):
        raise ValueError('Actual original1500 source order differs from reviewed manifest')
    if random_kwargs:validate_random_control_runtime_evidence(config,manifest,inventory,geometries,profile,observed)
    counts=core.validate_stream(config['requests'],manifest,inventory,profile,**random_kwargs)
    for item in json.loads(Path(config['scientific_source_seals']).read_text(encoding='utf-8')):
        if core.file_sha(item['path'])!=item['sha256']:raise ValueError('Scientific source bytes differ before any backend import')
    # Runtime ABI/codec receipts are immutable root-reviewed metadata. Their
    # native factories retain their own strict version/source/statistics gates.
    history=core.VerifiedHistoryIndex(config['history_manifest'],observed['history_manifest_sha256'])
    references=None
    if config.get('dataset_references'):
        from .dataset_reference_metrics import DatasetReferences
        if not permit.get('metric_reference_sha256'):raise ValueError('Optional source-bound reference artifact needs a separate permit SHA')
        references=DatasetReferences(config['dataset_references'],permit['metric_reference_sha256'],
            config['output_codec'],observed['output_codec_sha256'])
    elif permit.get('metric_reference_sha256')is not None:raise ValueError('Permitted metric reference artifact is missing')
    if random_kwargs and core.file_sha(config['random_control_metadata'])!=config['random_control_metadata_sha256']:
        raise ValueError('Random metadata changed before once consumption')
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=False)
    once=claim_once(config['once_file'],permit,permit_sha)
    registry=core.AttemptRegistry(config['attempt_registry'],{'scope_id':permit['scope_id'],
        'model':permit['model'],'budget_origin_epoch':permit['budget_origin_epoch'],
        'cpu_deadline_epoch':permit['cpu_deadline_epoch'],'gpu_deadline_epoch':permit['gpu_deadline_epoch'],
        'absolute_deadline_epoch':permit['absolute_deadline_epoch'],'max_total_distinct_attempts':permit['max_total_distinct_attempts']})
    events=core.DurableJSONL(out/'events.jsonl');terminals=core.DurableJSONL(out/'results.jsonl');successes=core.NewSuccessIndex(out/'new_successes.jsonl')
    raw_outputs=core.DurableJSONL(out/'raw_native_outputs.jsonl')
    state={'schema':'formal1500_single_model_execution_v1','status':'starting','model':config['model'],
        'permit_id':permit['permit_id'],'scope_id':permit['scope_id'],'permit_sha256':permit_sha,'bindings':observed,
        'pid':os.getpid(),'start_ticks':int(Path('/proc/self/stat').read_text(encoding='utf-8').rsplit(')',1)[1].split()[19]),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text(encoding='utf-8').strip(),'host':os.uname().nodename,
        'scope_clock_epoch':clock_epoch,'scope_clock_monotonic':clock_mono,
        'started_epoch':time.time(),'budget_origin_epoch':permit['budget_origin_epoch'],
        'absolute_deadline_epoch':permit['absolute_deadline_epoch'],'planned':counts,'terminal_records':0,
        'counts':{},'model_loads_attempted':0,'model_load_count':0,'new_native_computations':0,'new_computation_dispatches':0,'new_actual_model_attempts':0,
        'new_attempts_with_unknown_entry':0,'history_reuse_records':0,'within_run_alias_records':0,
        'lease_acquired':False,'lease_released':False,'model_predictions_authorized_by_separate_permit':True,
        'plan_remains_data_only_launch_false':True,'optimizer_steps':0,'paper_reproduction':False,
        'source_bound_metric_reference_sha256':permit.get('metric_reference_sha256'),
        'human_verified':False,'physical_contact_claim':False,'once':once}
    status_counts=Counter();runtime=NativeRuntime(config,permit,state);source_id=None;data=layout=None;baseline=None
    stop_reason=None;fatal=None
    def save():
        state['updated_epoch']=time.time();state['counts']=dict(status_counts)
        temp=out/'progress.tmp';temp.write_text(json.dumps(state,indent=2),encoding='utf-8');temp.replace(out/'progress.json')
    def finish(row,status,**extra):
        terminals.append('request_terminal',request_id=row['request_id'],ordinal=row['ordinal'],sample_id=row['sample_id'],
            model=row['model'],region=row['region'],layer=row['layer'],keep_ratio=row['keep_ratio'],status=status,**extra)
        status_counts[status]+=1;state['terminal_records']+=1
        if extra.get('actual_attempted')is True:
            state['new_actual_model_attempts']+=1;state['new_native_computations']+=1
        elif extra.get('actual_attempted')is None:state['new_attempts_with_unknown_entry']+=1
        if status=='history_reuse':state['history_reuse_records']+=1
        elif status=='within_run_alias':state['within_run_alias_records']+=1
        if state['terminal_records']%100==0:save()
    def expired(_signal,_frame):raise TimeoutError('Original execution scope absolute deadline expired')
    runtime.persist=save
    signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(permit['absolute_deadline_epoch']-time.time())))
    events.append('authorized_run_intent',permit_id=permit['permit_id'],scope_id=permit['scope_id'],bindings=observed)
    save()
    try:
        for row in core.iter_selected_requests(config['requests'],config['model']):
            events.append('request_received',request_id=row['request_id'],planned_status=row['status'])
            if row['status']=='blocked':
                finish(row,'planned_blocked',actual_attempted=False,reasons=row['blocked_reasons']);continue
            if stop_reason:
                finish(row,'unattempted_after_stop',actual_attempted=False,reasons=[stop_reason]);continue
            if (time.time()>=permit['absolute_deadline_epoch']
                    or time.monotonic()>=runtime.monotonic_limits['absolute_deadline_epoch']):
                stop_reason='original_absolute_scope_expired';fatal={'type':'TimeoutError','message':stop_reason};signal.alarm(0)
                finish(row,'unattempted_after_stop',actual_attempted=False,reasons=[stop_reason]);continue
            source=inventory[row['ordinal']-1]
            if source_id!=source['sample_id']:
                source_id=source['sample_id'];data=layout=None;baseline=None
                try:data,layout=read_source(source,geometries.get(source_id),config,file_map)
                except Exception as error:
                    stop_reason='source_pre_entry_error';fatal={'type':type(error).__name__,'message':str(error)}
                    finish(row,'source_error',actual_attempted=False,error=fatal);continue
            if row['region']!='baseline'and baseline is None:
                finish(row,'dependency_blocked',actual_attempted=False,reasons=['baseline_prediction_unavailable']);continue
            identity=core.computation_identity(source,profile,row,permit);key=canonical_sha(identity)
            prediction=None;origin=None;attempt_path=None;entries_before=None;outcome_written=False;raw_artifact=None;raw_capture={}
            try:
                if not set(row['positions']).issubset(layout.token_positions):raise ValueError('Plan mask contains nonvisual/wrong-geometry positions')
                prior=history.get(key,identity)
                if prior is not None:prediction=prior;origin='history_reuse'
                else:
                    cached=successes.get(key,identity)
                    if cached is not None:
                        prediction,_=cached;origin='within_run_alias'
                if prediction is None:
                    excluded=history.blocked(key,identity)
                    if excluded is not None:
                        finish(row,'historical_unknown_or_failure_not_retried',actual_attempted=False,
                            computation_key=key,prior_status=excluded['status'],origin_artifact_sha256=excluded['origin_artifact_sha256'])
                        continue
                    existing=registry.inspect(key)
                    if existing is not None:raise core.PriorAttemptExists(existing['status'],existing['path'])
                    if registry.existing_count>=permit['max_total_distinct_attempts']:
                        raise TimeoutError('Original cumulative distinct-attempt budget exhausted')
                    runtime.begin();runtime.budget()
                    actual=runtime.engine.layout(source,data)
                    if canonical_sha(asdict(actual))!=canonical_sha(asdict(layout)):
                        raise ValueError('Actual same-source model layout differs from accepted processor geometry')
                    attempt_path=registry.reserve(key,identity,{'permit_id':permit['permit_id'],'pid':os.getpid(),
                        'scope_id':permit['scope_id'],'start_ticks':state['start_ticks'],'request_id':row['request_id']})
                    events.append('durable_attempt_intent',request_id=row['request_id'],computation_key=key,
                        registry_path=str(attempt_path),actual_attempted=None)
                    entries_before=runtime.engine.native_entries();state['new_computation_dispatches']+=1;save()
                    owner,method=native_observer_target(runtime.engine,config['model'])
                    with observe_return(owner,method,raw_capture):
                        computed=runtime.engine.predict(source,data,row,actual)
                    if raw_capture.get('calls')!=1 or raw_capture.get('returned')is not True:
                        raise RuntimeError('Native return observation did not cover exactly one real prediction')
                    prediction=core.prediction_payload(computed,config['model'])
                    raw_capture.pop('returned_object',None)
                    offset,length=raw_outputs.append('raw_native_return',request_id=row['request_id'],computation_key=key,
                        identity=identity,prediction=prediction,qualification='pending_resource_and_protocol_checks')
                    raw_artifact={'path':str(raw_outputs.path),'offset':offset,'bytes':length}
                    runtime.engine.torch.cuda.synchronize();runtime.budget()
                    actual_entry=runtime.engine.native_entries()>entries_before
                    if not actual_entry:raise RuntimeError('Native prediction returned without an observed actual model entry')
                    if row['keep_ratio']==1:validate_keep1(prediction,baseline)
                    registry.outcome(attempt_path,status='complete',actual_attempted=True,prediction=prediction)
                    outcome_written=True
                    successes.add(key,identity,prediction,row['request_id']);origin='new_complete'
                    state['actual_native_counters']=runtime.engine.counters_snapshot()
                metrics,metric_error=evaluate_qualified_metrics(prediction,prediction if row['region']=='baseline'else baseline,
                    source,data,row,profile,references)
                if metric_error is not None:
                    # Qualified native success survives a metric-only failure.
                    stop_reason='metric_metadata_error_no_further_dispatch';fatal={'type':type(metric_error).__name__,'message':str(metric_error)}
                if row['region']=='baseline':baseline=prediction
                finish(row,origin,actual_attempted=origin=='new_complete',computation_key=key,prediction=prediction,metrics=metrics,
                    raw_prediction_artifact=raw_artifact)
            except core.PriorAttemptExists as error:
                # No retry. A root-reviewed historical success export may
                # resolve this in a separately permitted future invocation.
                finish(row,'prior_attempt_not_retried',actual_attempted=False,computation_key=key,
                    prior_attempt_state=error.state,prior_attempt_path=error.path,
                    prior_actual_entry_unknown=error.state.startswith('unknown'))
            except Exception as error:
                actual_entry=None
                if raw_artifact is None and raw_capture.get('returned'):
                    try:
                        recovered=take_unqualified_return(raw_capture,config['model'],runtime.engine)
                        if recovered is not None:
                            offset,length=raw_outputs.append('RAW_UNQUALIFIED',request_id=row['request_id'],computation_key=key,identity=identity,**recovered)
                            raw_artifact={'path':str(raw_outputs.path),'offset':offset,'bytes':length,'qualification':'RAW_UNQUALIFIED'}
                            prediction=recovered['prediction']
                    except Exception as raw_error:
                        state.setdefault('raw_preservation_errors',[]).append({'request_id':row['request_id'],'type':type(raw_error).__name__})
                    finally:raw_capture.pop('returned_object',None)
                if attempt_path is not None and entries_before is not None:
                    actual_entry=runtime.engine.native_entries()>entries_before
                    if not outcome_written:
                        registry.outcome(attempt_path,status='error',actual_attempted=actual_entry,
                            prediction=prediction,error={'type':type(error).__name__,'message':str(error),'raw_prediction_artifact':raw_artifact})
                finish(row,'execution_error',actual_attempted=actual_entry if attempt_path is not None else False,
                    computation_key=key,error={'type':type(error).__name__,'message':str(error)},
                    raw_unqualified_prediction=prediction,raw_prediction_artifact=raw_artifact,verified_cache_success=False)
                stop_reason='resource_or_runtime_error_no_further_dispatch';fatal={'type':type(error).__name__,'message':str(error)}
                signal.alarm(0)  # Cleanup/remaining-denominator emission only; no new model/source calls.
        successful=sum(status_counts[k]for k in('new_complete','history_reuse','within_run_alias'))
        state['status']='complete_requested_inventory'if not fatal and successful==counts['eligible']and state['terminal_records']==counts['requested']else'partial'
    except BaseException as error:
        fatal={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()};state['status']='partial'
        signal.alarm(0)
        try:
            if 'raw_capture'in locals()and raw_capture.get('returned')and raw_capture.get('returned_object')is not None:
                recovered=take_unqualified_return(raw_capture,config['model'],runtime.engine)
                if recovered is not None:raw_outputs.append('RAW_UNQUALIFIED_INTERRUPT',request_id=row['request_id'],**recovered)
        except BaseException as raw_error:state['interrupted_raw_preservation_error']=type(raw_error).__name__
        finally:
            if 'raw_capture'in locals():raw_capture.pop('returned_object',None)
        # Catchable interruptions still retain an explicit terminal row for
        # every remaining request, using metadata/ledger only (no model/source
        # decode or retries). A hard SIGKILL/power loss cannot run this code;
        # its immutable plan + durable intents remain authoritative for parent
        # readback and a separately permitted continuation.
        try:
            try:terminals.close()
            except BaseException as close_error:state['original_terminal_close_error']=type(close_error).__name__
            recovered=core.scan_terminal_prefix(out/'results.jsonl')
            original_ids=recovered.pop('request_ids');state['terminal_recovery_prefix']=recovered
            state['original_terminal_jsonl_sha256']=core.file_sha(out/'results.jsonl')
            state['terminal_records']=recovered['valid_records'];status_counts=Counter(recovered['counts'])
            state['new_actual_model_attempts']=recovered['actual_attempts'];state['new_native_computations']=recovered['actual_attempts']
            state['new_attempts_with_unknown_entry']=recovered['unknown_entries'];state['history_reuse_records']=recovered['history_reuse_records']
            state['within_run_alias_records']=recovered['within_run_alias_records']
            terminals=core.DurableJSONL(out/'recovery_results.jsonl')
            state['terminal_files']=['results.jsonl','recovery_results.jsonl']
            for pending in core.iter_selected_requests(config['requests'],config['model']):
                if pending['request_id']in original_ids:continue
                if pending['status']=='blocked':finish(pending,'planned_blocked',actual_attempted=False,reasons=pending['blocked_reasons']);continue
                source=inventory[pending['ordinal']-1]
                identity=core.computation_identity(source,profile,pending,permit);key=canonical_sha(identity)
                prior=registry.inspect(key)
                if prior is None:finish(pending,'unattempted_after_interrupt',actual_attempted=False,reasons=['worker_interrupted_no_new_dispatch'])
                elif prior.get('outcome')is not None:
                    outcome=prior['outcome'];owner=prior['intent'].get('owner',{})
                    finish(pending,'known_attempt_terminal_incomplete',actual_attempted=outcome.get('actual_attempted')if owner.get('permit_id')==permit['permit_id']else False,
                        computation_key=key,prior_status=prior['status'],prediction=outcome.get('prediction'),metrics_missing=True)
                else:
                    own=prior.get('intent',{}).get('owner',{}).get('permit_id')==permit['permit_id']
                    finish(pending,'unknown_attempt_after_interrupt',actual_attempted=None if own else False,
                        historical_entry_unknown=not own,computation_key=key,prior_state=prior['status'])
        except BaseException as accounting_error:state['terminal_accounting_error']={'type':type(accounting_error).__name__,'message':str(accounting_error)}
    finally:
        signal.alarm(0)
        try:runtime.close()
        except BaseException as cleanup_error:
            state['status']='partial';state['cleanup_error']={'type':type(cleanup_error).__name__,'message':str(cleanup_error)}
        finally:
            state.update(finished_epoch=time.time(),fatal_error=fatal,
                unrecorded_requested=counts['requested']-state['terminal_records'],
                evaluation_inventory_complete=state['terminal_records']==counts['requested'],
                missing_prediction_records=counts['requested']-sum(status_counts[k]for k in ('new_complete','history_reuse','within_run_alias')),
                eligible_missing_prediction_records=counts['eligible']-sum(status_counts[k]for k in ('new_complete','history_reuse','within_run_alias')),
                terminal_jsonl_sha256=core.file_sha(out/'results.jsonl'),new_successes_jsonl_sha256=core.file_sha(successes.journal.path),
                raw_native_outputs_jsonl_sha256=core.file_sha(raw_outputs.path))
            state['terminal_files']=state.get('terminal_files',['results.jsonl'])
            state['terminal_file_sha256']={name:core.file_sha(out/name)for name in state['terminal_files']}
            save()
            try:events.append('run_terminal',status=state['status'],terminal_records=state['terminal_records'],
                unrecorded_requested=state['unrecorded_requested'],counts=dict(status_counts))
            except BaseException as event_error:
                state['status']='partial';state['terminal_event_error']={'type':type(event_error).__name__,'message':str(event_error)}
            for journal in (events,terminals,successes,raw_outputs):
                try:journal.close()
                except BaseException as close_error:
                    state['status']='partial';state.setdefault('terminal_close_errors',[]).append(type(close_error).__name__)
            save()
            (out/'result.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path);parser.add_argument('--permit',required=True,type=Path)
    parser.add_argument('--permit-sha256',required=True)
    args=parser.parse_args();state=run(args.config,args.permit,args.permit_sha256)
    print(json.dumps({'status':state['status'],'counts':state['counts'],'requested':state['planned']['requested'],
        'terminal_records':state['terminal_records'],'new_actual_model_attempts':state['new_actual_model_attempts']}))
    if state['status']=='partial':raise SystemExit(2)


if __name__=='__main__':main()
