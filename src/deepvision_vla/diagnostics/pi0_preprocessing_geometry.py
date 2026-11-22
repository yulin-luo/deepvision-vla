"""Fixed1500 genuine Pi0 CPU preprocessing, one strict policy materialization.

Normalize/prepare methods run on the actual loaded policy. Its learned model
modules and sampling/prediction entry points are guarded and never executed.
Visual offset0/prefix304 remain source-derived and bound to prior real NN proof;
this run observes processed images/state/language, not vision embeddings.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import signal
import sys
import time
import traceback

from .adapters import openpi_bridge
from .adapters.openpi_bridge import MODEL_ID,MODEL_REVISION,bridge_eef_state8,load_bridge_pi0
from deepvision_vla.diagnostics.adapters.pi0_gradients import prepare_native_inputs, validate_cpu_receipt, FORWARD_SHA
from .formal_plan import canonical_sha,is_sha
from deepvision_vla.diagnostics.input_inventory import validate_linked_inventory, safe_payload_path, failure_status
from deepvision_vla.diagnostics.preprocessing_limits import _cpu_limits
from deepvision_vla.diagnostics.factory_imports import configure_factory_imports
from deepvision_vla.diagnostics.runtime_observation import MIN_AVAILABLE, MAX_RSS, parse_kb, memory_observation, actual_memory

SOURCES=1500
ADAPTER_SOURCE_FILES=('adapters/openpi_bridge.py','adapters/openpi_bridge_config.py',
    'adapters/openpi_bridge_helper_compat.py','adapters/openpi_bridge_unused_head.py',
    'adapters/pi0_gradients.py','factory_imports.py','input_payloads.py','input_inventory.py',
    'runtime_observation.py','preprocessing_limits.py')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while chunk:=stream.read(1024*1024):digest.update(chunk)
    return digest.hexdigest()


class PreprocessingBoundaryViolation(RuntimeError):pass








def verify_source_metadata(checkpoint,source,receipt):
    if receipt.get('status')!='complete' or receipt.get('model')!=MODEL_ID or receipt.get('revision')!=MODEL_REVISION:
        raise ValueError('Completed pinned author Pi0 asset receipt is required')
    files=receipt['files'];by_path={r['path']:r for r in files}
    if len(files)!=len(by_path)or'model.safetensors'not in by_path:raise ValueError('Actual unique weight/source asset inventory is required')
    metadata={}
    for name,item in by_path.items():
        logical=PurePosixPath(name)
        if logical.is_absolute()or'..'in logical.parts or'\\'in name:raise ValueError('Unsafe asset path')
        path=Path(checkpoint).joinpath(*logical.parts)
        if item.get('verified')is not True or path.stat().st_size!=item['size']:raise ValueError('Incomplete pinned asset: '+name)
        if name=='model.safetensors':
            if item.get('sha256')!=openpi_bridge.WEIGHT_SHA256:raise ValueError('Wrong author checkpoint weight seal')
            continue  # Actual weight bytes are read once by the genuine strict CPU factory, not this metadata pass.
        if path.suffix not in ('.json','.py','.md','.model','.txt')and name!='.gitattributes':raise ValueError('Unexpected nonweight preprocessing asset')
        if sha(path)!=item['sha256']:raise ValueError('Pinned preprocessing asset bytes differ: '+name)
        metadata[name]=item['sha256']
    if not {'config.json','configuration_pi0.py','modeling_pi0.py','tokenizer_config.json'}.issubset(metadata):
        raise ValueError('Native config/source/tokenizer assets missing')
    sources={}
    for relative,expected in openpi_bridge._DEPS.items():
        if sha(Path(source)/relative)!=expected:raise ValueError('Pinned LeRobot preprocessing dependency differs: '+relative)
        sources['dependency/'+relative]=expected
    for name in ('configuration_pi0.py','modeling_pi0.py'):sources['checkpoint/'+name]=metadata[name]
    for relative in ADAPTER_SOURCE_FILES:sources['diagnostics/'+relative]=sha(Path(__file__).parent/relative)
    return sources,metadata


def identity_hashes(sources,metadata):
    return {'source_sha256':canonical_sha(sources),'processor_sha256':canonical_sha({
        'checkpoint_nonweight_assets':metadata,'normalize_source_sha256':openpi_bridge._DEPS['lerobot/common/policies/normalize.py'],
        'native_prepare_inputs_module_sha256':sources['diagnostics/adapters/pi0_gradients.py']})}


def validate_prior_proof(proof,cpu,cpu_sha,asset_sha):
    validate_cpu_receipt(cpu)
    audit=proof.get('native_input_audit',{})
    if (proof.get('status')!='complete'or proof.get('lease_released')is not True
            or proof.get('model')!=MODEL_ID or proof.get('revision')!=MODEL_REVISION
            or proof.get('cpu_receipt_sha256')!=cpu_sha or proof.get('asset_receipt_sha256')!=asset_sha
            or proof.get('native_calls')!={'embed_prefix':5,'denoise_step':5}
            or proof.get('backward_calls')!=4 or proof.get('probes_completed')!=4
            or audit.get('image_shapes')!=[[1,3,224,224]] or audit.get('language_shape')!=[1,48]
            or audit.get('prepared_state_shape')!=[1,32]
            or proof.get('output_alias_source_proof',{}).get('helper_sha256')!=openpi_bridge._DEPS['lerobot/common/policies/pi0/paligemma_with_expert.py']
            or proof.get('output_alias_source_proof',{}).get('forward_source_sha256')!=FORWARD_SHA):
        raise ValueError('Actual completed same-source Pi prefix NN proof and CPU receipt are required')
    layout=proof.get('layout',{})
    if layout.get('grid_shape')!=[16,16]or layout.get('token_positions')!=list(range(256)):
        raise ValueError('Prior actual Pi visual prefix proof differs')


@contextmanager
def forbid_cuda(torch,counters):
    old={name:getattr(torch.cuda,name)for name in ('init','_lazy_init')}
    def refused(*_args,**_kwargs):
        counters['blocked_cuda_attempts']+=1;raise PreprocessingBoundaryViolation('CPU native preprocessing cannot initialize CUDA')
    for name in old:setattr(torch.cuda,name,refused)
    try:yield
    finally:
        for name,value in old.items():setattr(torch.cuda,name,value)


@contextmanager
def observe_strict_materialization(pretrained_class,checkpoint,audit):
    """Delegate the genuine pinned classmethod; no synthetic load results."""
    descriptor=pretrained_class.__dict__['_load_as_safetensor'];original=descriptor.__func__
    def counted(cls,model,model_file,map_location,strict):
        row={'class':cls.__name__,'file':str(model_file),'map_location':map_location,'outer_strict':strict,'status':'started'}
        audit.append(row)
        if len(audit)!=1 or strict is not True or map_location!='cpu' or Path(model_file).resolve()!=(Path(checkpoint)/'model.safetensors').resolve():
            raise PreprocessingBoundaryViolation('One actual strictCPU materialization from the pinned file is required')
        result=original(cls,model,model_file,map_location,strict)
        row.update(status='returned_successfully',returned_same_policy=result is model,
            actual_state_key_count=len(result.state_dict()),meta_tensors=sum(p.is_meta for p in result.parameters()),
            native_strict_loader_success=True,
            missing_unexpected_evidence='native strict=True loader success + actual active-header audit; no invented missing/unexpected arrays')
        return result
    setattr(pretrained_class,'_load_as_safetensor',classmethod(counted))
    try:yield
    finally:setattr(pretrained_class,'_load_as_safetensor',descriptor)


@contextmanager
def forbid_learned_forward_during_load(torch,counters):
    def refused(_module,_args):
        counters['blocked_network_forward_attempts']+=1
        raise PreprocessingBoundaryViolation('Policy construction must not execute a network forward')
    handle=torch.nn.modules.module.register_module_forward_pre_hook(refused)
    try:yield
    finally:handle.remove()


@contextmanager
def protect_policy_network(policy,counters):
    """Actual normalization outside model remains allowed; no fake outputs."""
    handles=[];old=[]
    def refused(_module,_args):
        counters['blocked_network_forward_attempts']+=1
        raise PreprocessingBoundaryViolation('A learned policy.model module was called during preprocessing')
    def deny_method(*_args,**_kwargs):
        counters['blocked_prediction_method_attempts']+=1
        raise PreprocessingBoundaryViolation('Sampling/prediction/embedding/denoising is forbidden')
    def replace(obj,name):
        if hasattr(obj,name):
            had=name in obj.__dict__;value=obj.__dict__.get(name);old.append((obj,name,had,value));setattr(obj,name,deny_method)
    try:
        for module in policy.model.modules():handles.append(module.register_forward_pre_hook(refused))
        # Manual Pi helper can bypass Module.__call__; deny its direct forward too.
        for obj,names in ((policy,('forward','select_action','predict_action')),
                          (policy.model,('forward','sample_actions','embed_prefix','denoise_step')),
                          (policy.model.paligemma_with_expert,('forward',))):
            for name in names:replace(obj,name)
        yield {'learned_network_module_guards':len(handles),'normalization_modules_outside_model_allowed':True}
    finally:
        for handle in handles:handle.remove()
        for obj,name,had,value in reversed(old):
            if had:setattr(obj,name,value)
            else:delattr(obj,name)


def state_input_audit(state7,native_audit,anchor,array_sha):
    expected=bridge_eef_state8(state7)
    if native_audit.get('native_state8_before_normalization')!=expected:
        raise ValueError('Genuine native input differs from measured state7 + zero + raw sensor gripper')
    return {'measured_state7':list(state7),'native_state8_before_normalization':expected,
        'source_array_sha256':array_sha,'original_anchor':anchor,'zero_pad_index':6,'raw_gripper_destination_index':7,
        'raw_gripper_preserved':expected[7]==float(state7[6]),'gripper_clipped_or_binarized':False,
        'zero_state_substitute_used':False,'extra_axis_or_origin_transform_applied':False,
        'coordinate_semantics':'original dataset state channels passed through; no external frame/origin equivalence asserted',
        'physical_frame_origin_verified_by_preprocessing':False,'formal_planner_state_frame_contract_update_pending':True}


def layout_from_prepared(row,image_shapes,state_shape,language_shape,vision):
    if (image_shapes!=[[1,3,224,224]]or list(state_shape)!=[1,32]or list(language_shape)!=[1,48]
            or vision.get('image_size')!=224 or vision.get('patch_size')!=14):
        raise ValueError('Actual native prepared tensors or materialized SigLIP config differ')
    from .geometry import VisualLayout
    w,h=row['image_size'];content=openpi_bridge.resize_geometry((w,h))
    return VisualLayout(row['image_sha256'],(w,h),(0,0,w,h),(224,224),content,(16,16),tuple(range(256)))


def prepare_original_inputs(row,image_raw,arrays_raw):
    """Read the frozen image and measured anchor state; no policy substitution."""
    from PIL import Image
    import numpy as np
    with Image.open(io.BytesIO(image_raw))as image:
        image.load()
        if list(image.size)!=row['image_size']:raise ValueError('Original image dimensions differ')
        rgb=image.convert('RGB')
    if not isinstance(row['instruction'],str)or not row['instruction'].strip():raise ValueError('Original instruction is empty')
    with np.load(io.BytesIO(arrays_raw),allow_pickle=False)as arrays:states=arrays['states']
    anchor=row['anchor']
    if (states.ndim!=2 or states.shape[1]!=7 or type(anchor)is not int or not 0<=anchor<len(states)
            or not np.isfinite(states[anchor]).all()):
        raise ValueError('Original measured state7 is absent/nonfinite at its declared anchor')
    steps=row.get('original_source',{}).get('steps')
    if steps is not None and steps!=len(states):raise ValueError('Original raw episode state length differs')
    return rgb,states[anchor].tolist()


def _load(args,receipt,cpu,proof,identity,state):
    sources,metadata=verify_source_metadata(args.checkpoint,args.lerobot_source,receipt)
    validate_prior_proof(proof,cpu,sha(args.cpu_receipt),sha(args.asset_receipt))
    expected=identity_hashes(sources,metadata)
    if (identity.get('model')!='pi0'or identity.get('checkpoint_revision')!=MODEL_REVISION
            or any(identity.get(k)!=v for k,v in expected.items())
            or any(not is_sha(identity.get(k))for k in ('checkpoint_sha256','action_mapping_sha256'))):
        raise ValueError('Root Pi model identity is not bound to exact native preprocessing sources')
    state['dependency_import_path']=configure_factory_imports('pi0',{'lerobot_source_root':str(args.lerobot_source)})
    state['memory_before_load']=actual_memory(True)
    import torch
    if torch.cuda.is_initialized():raise PreprocessingBoundaryViolation('CPU preprocessing inherited a CUDA context')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    from lerobot.common.policies.pretrained import PreTrainedPolicy
    state['model_loads_attempted']=1
    with forbid_cuda(torch,state['guards']),forbid_learned_forward_during_load(torch,state['guards']),observe_strict_materialization(PreTrainedPolicy,args.checkpoint,state['strict_materializations']):
        adapter=load_bridge_pi0(args.checkpoint,lerobot_source_root=args.lerobot_source,asset_receipt=args.asset_receipt)
    policy=adapter.policy
    if any(p.is_meta or p.device.type!='cpu'for p in policy.parameters())or torch.cuda.is_initialized():
        raise PreprocessingBoundaryViolation('Policy must be genuinely materialized on CPU only')
    if (adapter.active_state_header_audit!=cpu['active_state_header_audit']
            or adapter.unused_expert_head_audit!=cpu['unused_expert_head_audit']
            or len(state['strict_materializations'])!=1
            or state['strict_materializations'][0].get('actual_state_key_count')!=783):
        raise ValueError('Actual strict policy load differs from original valid CPU source/header binding')
    config=policy.model.paligemma_with_expert.paligemma.config.vision_config
    vision={'image_size':config.image_size,'patch_size':config.patch_size,'source':'actual materialized policy config; vision forward not executed'}
    if vision['image_size']!=224 or vision['patch_size']!=14 or policy.config.tokenizer_max_length!=48:
        raise ValueError('Actual policy vision/token contract differs')
    state.update(model_load_count=1,memory_after_load=actual_memory(),active_state_header_audit=adapter.active_state_header_audit,
        unused_expert_head_audit=adapter.unused_expert_head_audit,config_compatibility_audit=adapter.config_compatibility_audit,
        constructor_compatibility_audit=adapter.constructor_compatibility_audit,source_hashes=sources,
        processor_metadata_hashes=metadata,actual_vision_config=vision,
        normalize_source_sha256=sha(Path(inspect_module_path(policy.normalize_inputs.__class__))),
        parameter_count=sum(p.numel()for p in policy.parameters()),prior_acceptance_suite_repeated=False)
    if state['normalize_source_sha256']!=openpi_bridge._DEPS['lerobot/common/policies/normalize.py']:
        raise ValueError('Actual Normalize class source differs from the sealed dependency')
    return adapter,torch,sources,vision


def inspect_module_path(cls):
    import inspect
    return inspect.getfile(cls)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','checkpoint','lerobot-source','asset-receipt','cpu-receipt','model-proof','model-identity','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    for name in ('expected-inventory-sha256','expected-original-manifest-sha256','expected-model-proof-sha256'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args();limits=_cpu_limits();root=args.data_root
    inventory_raw=(root/'inventory.json').read_bytes();original_raw=(root/'manifest.jsonl').read_bytes()
    inventory_sha=hashlib.sha256(inventory_raw).hexdigest();original_sha=hashlib.sha256(original_raw).hexdigest()
    if inventory_sha!=args.expected_inventory_sha256 or original_sha!=args.expected_original_manifest_sha256 or sha(args.model_proof)!=args.expected_model_proof_sha256:
        raise ValueError('Fixed1500/source NN proof byte seal differs')
    rows=json.loads(inventory_raw);original=[json.loads(x)for x in original_raw.splitlines()if x.strip()]
    files=json.loads((root/'files.json').read_text(encoding='utf-8'));transfer=json.loads((root/'transfer_receipt.json').read_text(encoding='utf-8'))
    file_map=validate_linked_inventory(rows,original,files,transfer,inventory_sha,original_sha)
    receipt=json.loads(args.asset_receipt.read_text(encoding='utf-8'));cpu=json.loads(args.cpu_receipt.read_text(encoding='utf-8'))
    proof=json.loads(args.model_proof.read_text(encoding='utf-8'));identity=json.loads(args.model_identity.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=False);start=time.time();deadline=time.monotonic()+600
    state={'schema':'pi0_cpu_native_preprocessing_geometry_batch_v1','status':'starting','pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text(encoding='utf-8').rsplit(')',1)[1].split()[19]),'started_epoch':start,'limits':limits,
        'requested':1500,'recorded':0,'counts':{},'model_loads_attempted':0,'model_load_count':0,'strict_materializations':[],
        'model_forward_calls':0,'learned_network_forward_calls':0,'prediction_calls':0,'gpu_calls':0,'optimizer_steps':0,
        'preprocessing_calls_attempted':0,'preprocessing_calls_completed':0,'payload_files_rehashed':0,
        'guards':{'blocked_cuda_attempts':0,'blocked_network_forward_attempts':0,'blocked_prediction_method_attempts':0},
        'inventory_sha256':inventory_sha,'original_manifest_sha256':original_sha,'asset_receipt_sha256':sha(args.asset_receipt),
        'cpu_receipt_sha256':sha(args.cpu_receipt),'prior_actual_prefix_nn_proof_sha256':sha(args.model_proof),
        'model_identity_sha256':canonical_sha(identity),'physical_frame_origin_verified_by_preprocessing':False,
        'global_gt_gate_changed':False,'memory_policy':{'min_host_available_bytes_before_load':MIN_AVAILABLE,'max_rss_hwm_bytes':MAX_RSS,'outer_memory_watchdog_required':True}}
    def save():
        state['updated_epoch']=time.time();tmp=args.output/'progress.tmp';tmp.write_text(json.dumps(state,indent=2),encoding='utf-8');tmp.replace(args.output/'progress.json')
    def expired(_signum,_frame):raise TimeoutError('Original600second CPU native preprocessing deadline expired')
    save();signal.signal(signal.SIGALRM,expired);signal.alarm(600)
    adapter=torch=None;sources={};vision={};global_error=None;counts=Counter();digest=hashlib.sha256()
    try:
        try:adapter,torch,sources,vision=_load(args,receipt,cpu,proof,identity,state)
        except Exception as error:
            global_error=(type(error).__name__,str(error));state['initialization_error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
        if adapter is not None:guard=protect_policy_network(adapter.policy,state['guards'])
        else:
            @contextmanager
            def unavailable_guard():yield {}
            guard=unavailable_guard()
        with guard as network_guard,(args.output/'geometry.jsonl').open('xb')as stream:
            state['network_guard']=network_guard
            for row in rows:
                record={'schema':'pi0_cpu_native_preprocessing_geometry_v1','ordinal':row['ordinal'],'sample_id':row['sample_id'],'model':'pi0',
                    'source_row_sha256':canonical_sha(row),'original_source_row_sha256':canonical_sha(row['original_source']),
                    'original_source':row['original_source'],'image_sha256':row['image_sha256'],'arrays_sha256':row['arrays_sha256'],
                    'image_size':row['image_size'],'anchor':row['anchor'],'instruction':row['instruction'],
                    'model_identity_sha256':canonical_sha(identity),'processor_sha256':identity.get('processor_sha256'),
                    'source_hashes':sources,'geometry_source':'actual_processor','actual_geometry_verified':False,'layout':None,
                    'prior_actual_prefix_nn_proof_sha256':args.expected_model_proof_sha256,
                    'preprocessing_call_attempted':False,'preprocessing_call_completed':False,'model_forward_calls':0,'gpu_calls':0,
                    'human_verified':False,'semantic_roi_review_performed':False,'global_gt_gate_changed':False,'paper_reproduction':False}
                try:
                    if global_error:
                        if global_error[0]=='TimeoutError':raise TimeoutError(global_error[1])
                        raise RuntimeError('Actual native policy preprocessing unavailable: '+':'.join(global_error))
                    if time.monotonic()>=deadline:raise TimeoutError('No new native preparation after original deadline')
                    state['last_memory_observation']=actual_memory()
                    payload={}
                    for field,key in (('image_path','image_sha256'),('arrays_path','arrays_sha256')):
                        relative=row['original_source'][field];raw=safe_payload_path(root,relative).read_bytes()
                        if len(raw)!=file_map[relative]['bytes']or hashlib.sha256(raw).hexdigest()!=row[key]:raise ValueError('Current original image/array bytes differ')
                        payload[field]=raw;state['payload_files_rehashed']+=1
                    anchor=row['anchor']
                    rgb,state7=prepare_original_inputs(row,payload['image_path'],payload['arrays_path'])
                    record['preprocessing_call_attempted']=True;state['preprocessing_calls_attempted']+=1
                    with forbid_cuda(torch,state['guards']),torch.no_grad():prepared,audit=prepare_native_inputs(adapter.policy,rgb,state7,row['instruction'])
                    images,image_masks,tokens,language_masks,prepared_state=prepared
                    tensors=list(images)+list(image_masks)+[tokens,language_masks,prepared_state]
                    if any(t.device.type!='cpu'for t in tensors)or torch.cuda.is_initialized()or any(not bool(torch.isfinite(t).all())for t in images+[prepared_state]):
                        raise PreprocessingBoundaryViolation('Actual native prepared tensors must be finite CPU values')
                    if len(image_masks)!=1 or tuple(image_masks[0].shape)!=(1,)or not bool(image_masks[0].all())or tuple(language_masks.shape)!=(1,48):
                        raise ValueError('Native camera/language masks differ from one-camera contract')
                    layout=layout_from_prepared(row,audit['image_shapes'],prepared_state.shape,tokens.shape,vision)
                    record.update(status='complete',actual_geometry_verified=True,layout=asdict(layout),native_preparation_audit=audit,
                        state_input_audit=state_input_audit(state7,audit,anchor,row['arrays_sha256']),
                        native_language_token_ids=tokens.cpu().tolist(),native_language_mask=language_masks.cpu().tolist(),
                        image_and_arrays_current_sha_verified=True,preprocessing_call_completed=True,
                        visual_sequence_position_binding='source-derived visual prefix0..255 + previously actual304-token NN prefix proof',
                        prefix_embedding_forward_performed=False,prefix_shape_this_run_observed=False,source_derived_prefix_length=304)
                    state['preprocessing_calls_completed']+=1;state['last_memory_observation']=actual_memory()
                except Exception as error:
                    record.update(status=failure_status(error),error_type=type(error).__name__,error=str(error))
                    if isinstance(error,(PreprocessingBoundaryViolation,TimeoutError,MemoryError)):
                        global_error=(type(error).__name__,str(error));signal.alarm(0)
                record['audit_hash_scope']='canonical_record_before_audit_link_and_sha';record['audit_sha256']=canonical_sha(record)
                record['audit_link']=str((args.output/'geometry.jsonl').resolve())+'#ordinal='+str(row['ordinal'])
                raw=(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n').encode();stream.write(raw);digest.update(raw)
                counts[record['status']]+=1;state['recorded']+=1;state['counts']=dict(counts)
                if state['recorded']%25==0:stream.flush();save()
        if adapter is not None:
            if any(p.grad is not None for p in adapter.policy.parameters()):raise RuntimeError('Native preprocessing accumulated parameter gradients')
            state['final_memory_observation']=actual_memory()
        cuda=torch.cuda.is_initialized()if torch is not None else False
        state.update(status='complete'if counts['complete']==1500 and state['model_load_count']==1 and not cuda and not any(state['guards'].values())else'partial',
            cuda_initialized=cuda,geometry_jsonl_sha256=digest.hexdigest(),finished_epoch=time.time(),launch_allowed=False,
            semantic_roi_review_performed=False,model_predictions_performed=False)
        save();(args.output/'result.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        print(json.dumps({'status':state['status'],'counts':dict(counts),'model_load_count':state['model_load_count'],'learned_forward_calls':0,'gpu_calls':0}))
        if state['status']!='complete':raise SystemExit(2)
    except BaseException as error:
        # Normal row/initialization failures already retain all1500 rows.
        # An exceptional finalizer/emitter failure must still publish its
        # recorded/missing denominator rather than masquerading as success.
        if not (args.output/'result.json').exists():
            state.update(status='partial',terminal_error_type=type(error).__name__,terminal_error=str(error),
                missing_geometry_rows=SOURCES-state['recorded'],finished_epoch=time.time())
            save();(args.output/'result.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        raise
    finally:
        signal.alarm(0);adapter=None;gc.collect()


if __name__=='__main__':main()
