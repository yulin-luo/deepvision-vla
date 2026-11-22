"""Fixed1500 actual CPU OpenVLA processor observations, without model weights.

Pixel/text outputs are observed for every source. Visual sequence positions
are a separate source-derived BOS insertion bound to an earlier real model
receipt, not image-token IDs emitted by this processor. Learned models are
forbidden; parameterless torchvision transforms remain native and allowed.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import inspect
import io
import json
import os
from pathlib import Path,PurePosixPath
import signal
import sys
import time
import traceback

from .adapters.openvla_runtime import MODEL_ID,MODEL_REVISION,prepare_prompt_tokens
from .geometry import VisualLayout
from .formal_plan import canonical_sha,is_sha
from deepvision_vla.diagnostics.input_inventory import validate_linked_inventory, safe_payload_path, failure_status
from deepvision_vla.diagnostics.preprocessing_limits import _cpu_limits

SOURCES=1500
METADATA_FILES=frozenset({'added_tokens.json','config.json','configuration_prismatic.py','modeling_prismatic.py',
    'preprocessor_config.json','processing_prismatic.py','processor_config.json','special_tokens_map.json',
    'tokenizer.json','tokenizer.model','tokenizer_config.json'})
SOURCE_FILES=frozenset({'configuration_prismatic.py','modeling_prismatic.py','processing_prismatic.py'})
_AUDIT={'installed':False,'depth':0,'counters':None}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while chunk:=stream.read(1024*1024):digest.update(chunk)
    return digest.hexdigest()


class LearnedModelWorkForbidden(RuntimeError):pass


def weight_path(path):
    try:name=os.fsdecode(path).lower()
    except TypeError:return False
    return name.endswith(('.safetensors','.bin','.pt','.pth','.ckpt'))


def _weight_audit(event,args):
    if _AUDIT['depth'] and event=='open' and args and weight_path(args[0]):
        _AUDIT['counters']['blocked_weight_file_opens']+=1
        raise LearnedModelWorkForbidden('Processor geometry cannot open a model weight file')


@contextmanager
def forbid_learned_work(torch,counters):
    """Guard real learned work while allowing native parameterless transforms.

    These are process-local protective checks, not an OS sandbox. No native
    processor/predict function is replaced. All temporary methods are restored.
    One inactive-after-exit audit callback avoids1500 accumulated active hooks.
    """
    if not _AUDIT['installed']:sys.addaudithook(_weight_audit);_AUDIT['installed']=True
    old_counters=_AUDIT['counters'];_AUDIT['counters']=counters;_AUDIT['depth']+=1
    parameter=torch.nn.Parameter;module=torch.nn.Module
    original_new=parameter.__dict__['__new__'];original_call=module._call_impl
    changes=[]
    def replace(obj,name,new):
        original=vars(obj).get(name);present=name in vars(obj)
        changes.append((obj,name,original,present));setattr(obj,name,new)
    def denied(kind):
        def refused(*_args,**_kwargs):
            counters[kind]+=1;raise LearnedModelWorkForbidden('Learned model/weight/CUDA work is forbidden: '+kind)
        return refused
    def module_call(self,*args,**kwargs):
        if any(True for _ in self.parameters()):
            counters['blocked_learned_forward_attempts']+=1
            raise LearnedModelWorkForbidden('A learned module forward is forbidden')
        counters['parameterless_module_calls']+=1
        return original_call(self,*args,**kwargs)
    parameter.__new__=staticmethod(denied('blocked_parameter_constructions'))
    module._call_impl=module_call
    replace(torch,'load',denied('blocked_weight_loader_calls'))
    replace(torch.cuda,'init',denied('blocked_cuda_initializations'))
    modeling=sys.modules.get('transformers.modeling_utils')
    if modeling is not None and hasattr(modeling,'PreTrainedModel'):
        replace(modeling.PreTrainedModel,'from_pretrained',classmethod(denied('blocked_model_from_pretrained')))
    timm=sys.modules.get('timm')
    if timm is not None:replace(timm,'create_model',denied('blocked_model_constructors'))
    safetensors=sys.modules.get('safetensors')
    if safetensors is not None:replace(safetensors,'safe_open',denied('blocked_weight_loader_calls'))
    try:yield
    finally:
        parameter.__new__=original_new;module._call_impl=original_call
        for obj,name,original,present in reversed(changes):
            if present:setattr(obj,name,original)
            else:delattr(obj,name)
        _AUDIT['depth']-=1;_AUDIT['counters']=old_counters


def guard_counters():
    return {k:0 for k in ('blocked_parameter_constructions','blocked_learned_forward_attempts','blocked_weight_file_opens',
        'blocked_weight_loader_calls','blocked_cuda_initializations','blocked_model_from_pretrained',
        'blocked_model_constructors','parameterless_module_calls')}


def verify_metadata(checkpoint,receipt,inventory):
    """Read only named source/config/tokenizer bytes, never weight shards/index."""
    if (receipt.get('status')!='complete' or receipt.get('model')!=MODEL_ID or receipt.get('revision')!=MODEL_REVISION
            or inventory.get('model')!=MODEL_ID or inventory.get('revision')!=MODEL_REVISION):
        raise ValueError('Completed pinned official OpenVLA acquisition metadata is required')
    expected={r['path']:r for r in inventory['files']};received={r['path']:r for r in receipt['files']}
    if (len(expected)!=len(inventory['files']) or len(received)!=len(receipt['files'])
            or len(expected)!=17 or set(expected)!=set(received)
            or not METADATA_FILES.issubset(expected.keys()&received.keys())):
        raise ValueError('Required processor metadata/source inventory is absent or duplicated')
    hashes={}
    for name in sorted(METADATA_FILES):
        path=Path(checkpoint)/name;item=expected[name];actual=received[name]
        if path.stat().st_size!=item['size'] or actual.get('size')!=item['size'] or actual.get('verified')is not True:
            raise ValueError('Processor source/tokenizer size or acquisition status differs: '+name)
        content=path.read_bytes();digest=hashlib.sha256(content).hexdigest()
        if digest!=actual.get('sha256'):raise ValueError('Acquired processor metadata SHA differs: '+name)
        if item.get('sha256'):
            if digest!=item['sha256']:raise ValueError('Official metadata SHA differs: '+name)
        elif hashlib.sha1(('blob '+str(len(content))+'\0').encode()+content).hexdigest()!=item.get('git_blob'):
            raise ValueError('Official source Git blob differs: '+name)
        hashes[name]=digest
    return hashes


def identity_hashes(hashes):
    return {'source_sha256':canonical_sha({k:hashes[k]for k in sorted(SOURCE_FILES)}),
            'processor_sha256':canonical_sha({k:hashes[k]for k in sorted(METADATA_FILES)})}


def validate_config(config,preprocessor):
    if (config.get('model_type')!='openvla' or config.get('image_sizes')!=[224,224]
            or config.get('image_resize_strategy')!='resize-naive' or config.get('use_fused_vision_backbone')is not True
            or config.get('timm_model_ids')!=['vit_large_patch14_reg4_dinov2.lvd142m','vit_so400m_patch14_siglip_224']
            or preprocessor.get('image_resize_strategy')!='resize-naive'
            or preprocessor.get('input_sizes')!=[[3,224,224],[3,224,224]]):
        raise ValueError('Expected sealed native fused224/patch14 resize-naive OpenVLA configuration')


def validate_native_transforms(processor):
    """Inspect real processor transform parameters, not a reimplemented image path."""
    if (processor.image_resize_strategy!='resize-naive' or processor.tvf_do_letterbox is not False
            or not processor.use_fused_vision_backbone
            or [list(x)for x in processor.input_sizes]!=[[3,224,224],[3,224,224]]
            or len(processor.tvf_resize_params)!=2 or len(processor.tvf_crop_params)!=2
            or any(list(x['size'])!=[224,224]for x in processor.tvf_resize_params)
            or any(list(x['output_size'])!=[224,224]for x in processor.tvf_crop_params)):
        raise ValueError('Actual full-image resize-naive/no-op224crop transform differs')
    return json.loads(json.dumps({'image_resize_strategy':processor.image_resize_strategy,
        'source_content_policy':'entire original RGB image; no external pre-resize/crop',
        'use_fused_vision_backbone':processor.use_fused_vision_backbone,'input_sizes':processor.input_sizes,
        'resize':processor.tvf_resize_params,'center_crop':processor.tvf_crop_params,
        'normalize':processor.tvf_normalize_params,'letterbox':processor.tvf_do_letterbox,
        'center_crop_effect':'no-op after native224x224resize'}))


def validate_prior_model_proof(proof,hashes,receipt_sha,inventory_sha):
    if (proof.get('status')!='complete' or proof.get('lease_released')is not True
            or proof.get('model')!=MODEL_ID or proof.get('revision')!=MODEL_REVISION
            or proof.get('asset_receipt_sha256')!=receipt_sha or proof.get('official_inventory_sha256')!=inventory_sha
            or proof.get('all_decoder_layer_count_required')!=32
            or proof.get('generation_model_forward_calls')!=7 or proof.get('generation_projector_calls')!=1
            or proof.get('teacher_forward_calls')!=4 or proof.get('backward_calls')!=4
            or proof.get('generation',{}).get('projected_shapes')!=[[1,256,4096]]):
        raise ValueError('Same-source actual completed OpenVLA projector/model proof is required')
    layout=proof.get('layout',{})
    if (layout.get('grid_shape')!=[16,16] or layout.get('canvas_size')!=[224,224]
            or layout.get('token_positions')!=list(range(1,257))
            or any(proof.get('source_hashes',{}).get(k)!=v for k,v in hashes.items())):
        raise ValueError('Actual prior model visual sequence/source proof differs')


def prepare_rgb(raw,size,instruction):
    from PIL import Image
    if not isinstance(instruction,str)or not instruction.strip():raise ValueError('Original instruction is empty')
    with Image.open(io.BytesIO(raw))as image:
        image.load()
        if list(image.size)!=size:raise ValueError('Actual source image dimensions differ')
        return image.convert('RGB')  # Native processor performs full-image224resize; no Qwen pre-resize/crop.


def prompt_text(instruction):
    if not isinstance(instruction,str)or not instruction.strip():raise ValueError('Original instruction is empty')
    return 'In: What action should the robot take to '+instruction.strip().lower()+'?\nOut:'


def layout_from_observation(row,ids,mask,pixel_shape,bos_token_id):
    if (len(ids)!=1 or len(mask)!=1 or not ids[0] or len(ids[0])!=len(mask[0])
            or type(bos_token_id)is not int or ids[0][0]!=bos_token_id or bos_token_id!=1
            or any(x!=1 for x in mask[0]) or list(pixel_shape)!=[1,6,224,224]):
        raise ValueError('Actual native BOS/mask/fused pixel observation differs')
    prepared,prepared_mask,appended=prepare_prompt_tokens(ids[0],mask[0])
    width,height=row['image_size']
    layout=VisualLayout(row['image_sha256'],(width,height),(0,0,width,height),(224,224),(0,0,224,224),
        (224//14,224//14),tuple(range(1,257)))
    return layout,{'processor_input_ids':ids[0],'runtime_input_ids_after_empty_token':prepared,
        'runtime_attention_mask_after_empty_token':prepared_mask,'empty_token_appended':appended,
        'pixel_values_shape':list(pixel_shape),'bos_token_id_observed':bos_token_id,
        'processor_emits_visual_token_ids':False,'actual_processor_observations':['pixel_values','text_input_ids','attention_mask'],
        'visual_sequence_positions_source':'sealed native BOS insertion + reused actual256projector proof, not processor image token IDs',
        'source_documented_patch_size':14,'prior_model_grid_reused':[16,16],
        'runtime_prefill_sequence_length_source_derived':len(prepared)+256}


def _load_processor(args,receipt,inventory,proof,identity,loader_state):
    hashes=verify_metadata(args.checkpoint,receipt,inventory)
    validate_prior_model_proof(proof,hashes,sha(args.asset_receipt),sha(args.official_inventory))
    expected=identity_hashes(hashes)
    if (identity.get('model')!='openvla' or identity.get('checkpoint_revision')!=MODEL_REVISION
            or any(identity.get(k)!=v for k,v in expected.items())
            or any(not is_sha(identity.get(k))for k in ('checkpoint_sha256','action_mapping_sha256'))):
        raise ValueError('Root model identity does not bind exact source/processor acquisition metadata')
    config=json.loads((args.checkpoint/'config.json').read_text(encoding='utf-8'));preprocessor=json.loads((args.checkpoint/'preprocessor_config.json').read_text(encoding='utf-8'))
    validate_config(config,preprocessor);loader_state['source_verified_before_processor_import']=True
    import torch
    if torch.cuda.is_initialized():raise LearnedModelWorkForbidden('CPU worker inherited CUDA initialization')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    counters=loader_state['guards']
    with forbid_learned_work(torch,counters):
        import transformers
        import timm
        from transformers import AutoProcessor,PreTrainedModel
    if transformers.__version__!='4.40.1' or timm.__version__!='0.9.10':
        raise RuntimeError('Use genuine validated OpenVLA Transformers4.40.1/timm0.9.10 environment')
    with forbid_learned_work(torch,counters):
        processor=AutoProcessor.from_pretrained(args.checkpoint,local_files_only=True,trust_remote_code=True)
    if (processor.image_processor.image_resize_strategy!='resize-naive'
            or [list(x)for x in processor.image_processor.input_sizes]!=[[3,224,224],[3,224,224]]
            or processor.tokenizer.bos_token_id!=1):raise ValueError('Actual native processor config/BOS differs')
    transforms=validate_native_transforms(processor.image_processor)
    classes={name:inspect.getfile(value.__class__)for name,value in (
        ('processor',processor),('image_processor',processor.image_processor),('tokenizer',processor.tokenizer))}
    for name in ('processor','image_processor'):
        if sha(classes[name])!=hashes['processing_prismatic.py']:
            raise RuntimeError('Actual trusted processor class source differs from sealed official bytes')
    loader_state.update(processor_loaded=True,actual_class_sources=classes,
        actual_class_source_sha256={k:sha(v)for k,v in classes.items()},
        versions={'torch':torch.__version__,'transformers':transformers.__version__,'timm':timm.__version__},
        metadata_hashes=hashes,actual_native_image_transforms=transforms,weight_files_read=0,new_model_validation_performed=False)
    return processor,torch,hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','checkpoint','asset-receipt','official-inventory','model-proof','model-identity','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    for name in ('expected-inventory-sha256','expected-original-manifest-sha256','expected-model-proof-sha256'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args();limits=_cpu_limits()
    os.environ['HF_MODULES_CACHE']=str((args.output/'hf-modules').resolve())
    root=args.data_root;inventory_raw=(root/'inventory.json').read_bytes();original_raw=(root/'manifest.jsonl').read_bytes()
    inventory_sha=hashlib.sha256(inventory_raw).hexdigest();original_sha=hashlib.sha256(original_raw).hexdigest()
    if inventory_sha!=args.expected_inventory_sha256 or original_sha!=args.expected_original_manifest_sha256 or sha(args.model_proof)!=args.expected_model_proof_sha256:
        raise ValueError('Fixed data/model-proof byte seals differ')
    rows=json.loads(inventory_raw);original=[json.loads(line)for line in original_raw.splitlines()if line.strip()]
    files=json.loads((root/'files.json').read_text(encoding='utf-8'));staging=json.loads((root/'transfer_receipt.json').read_text(encoding='utf-8'))
    file_map=validate_linked_inventory(rows,original,files,staging,inventory_sha,original_sha)
    receipt=json.loads(args.asset_receipt.read_text(encoding='utf-8'));official=json.loads(args.official_inventory.read_text(encoding='utf-8'))
    proof=json.loads(args.model_proof.read_text(encoding='utf-8'));identity=json.loads(args.model_identity.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=False);start=time.time();deadline=time.monotonic()+600
    state={'schema':'openvla_cpu_processor_geometry_batch_v1','status':'starting','pid':os.getpid(),
        'start_ticks':int(Path('/proc/self/stat').read_text(encoding='utf-8').rsplit(')',1)[1].split()[19]),'started_epoch':start,'limits':limits,
        'requested':1500,'recorded':0,'counts':{},'processor_calls_attempted':0,'processor_calls_completed':0,
        'payload_files_rehashed':0,'model_calls':0,'learned_model_forward_calls':0,'gpu_calls':0,'weight_files_read':0,
        'optimizer_steps':0,'inventory_sha256':inventory_sha,'original_manifest_sha256':original_sha,
        'asset_receipt_sha256':sha(args.asset_receipt),'official_inventory_sha256':sha(args.official_inventory),
        'prior_actual_model_proof_sha256':sha(args.model_proof),'model_identity_sha256':canonical_sha(identity),
        'processor_loader':{'processor_loaded':False,'guards':guard_counters()},'new_model_validation_performed':False}
    def save():
        state['updated_epoch']=time.time();temp=args.output/'progress.tmp';temp.write_text(json.dumps(state,indent=2),encoding='utf-8');temp.replace(args.output/'progress.json')
    def expired(_signum,_frame):raise TimeoutError('Original600second CPU processor budget expired')
    save();signal.signal(signal.SIGALRM,expired);signal.alarm(600)
    processor=torch=None;hashes={};global_error=None;counts=Counter();digest=hashlib.sha256()
    try:
        try:processor,torch,hashes=_load_processor(args,receipt,official,proof,identity,state['processor_loader'])
        except Exception as error:
            global_error=(type(error).__name__,str(error));state['initialization_error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
        with (args.output/'geometry.jsonl').open('xb')as stream:
            for row in rows:
                record={'schema':'openvla_cpu_processor_geometry_v1','ordinal':row['ordinal'],'sample_id':row['sample_id'],'model':'openvla',
                    'source_row_sha256':canonical_sha(row),'original_source_row_sha256':canonical_sha(row['original_source']),
                    'original_source':row['original_source'],'image_sha256':row['image_sha256'],'arrays_sha256':row['arrays_sha256'],
                    'image_size':row['image_size'],'instruction':row['instruction'],'anchor':row['anchor'],
                    'model_identity_sha256':canonical_sha(identity),'processor_sha256':identity.get('processor_sha256'),
                    'source_hashes':hashes,'geometry_source':'actual_processor','actual_geometry_verified':False,'layout':None,
                    'prior_actual_model_proof_sha256':args.expected_model_proof_sha256,'processor_call_attempted':False,'processor_call_completed':False,
                    'model_calls':0,'learned_model_forward_calls':0,'gpu_calls':0,'weight_files_read':0,'human_verified':False,
                    'semantic_roi_review_performed':False,'new_model_validation_performed':False,'paper_reproduction':False}
                try:
                    if global_error:
                        if global_error[0]=='TimeoutError':raise TimeoutError(global_error[1])
                        raise RuntimeError('Processor unavailable: '+':'.join(global_error))
                    if time.monotonic()>=deadline:raise TimeoutError('No new processor calls after original deadline')
                    image_bytes=None
                    for field,key in (('image_path','image_sha256'),('arrays_path','arrays_sha256')):
                        relative=row['original_source'][field];path=safe_payload_path(root,relative);raw=path.read_bytes()
                        if len(raw)!=file_map[relative]['bytes'] or hashlib.sha256(raw).hexdigest()!=row[key]:
                            raise ValueError('Actual staged image/array bytes differ: '+field)
                        state['payload_files_rehashed']+=1
                        if field=='image_path':image_bytes=raw
                    image=prepare_rgb(image_bytes,row['image_size'],row['instruction'])
                    record['processor_call_attempted']=True;state['processor_calls_attempted']+=1
                    with forbid_learned_work(torch,state['processor_loader']['guards']):
                        inputs=processor(prompt_text(row['instruction']),image,return_tensors='pt')
                    if set(inputs)!={'input_ids','attention_mask','pixel_values'} or any(v.device.type!='cpu'for v in inputs.values()if isinstance(v,torch.Tensor)) or torch.cuda.is_initialized():
                        raise LearnedModelWorkForbidden('Actual native processor keys/device differ from CPU-only contract')
                    if not bool(torch.isfinite(inputs.pixel_values).all()):raise ValueError('Native normalized pixels are nonfinite')
                    layout,observed=layout_from_observation(row,inputs.input_ids.tolist(),inputs.attention_mask.tolist(),inputs.pixel_values.shape,processor.tokenizer.bos_token_id)
                    observed.update(pixel_values_dtype_observed=str(inputs.pixel_values.dtype),
                        runtime_pixel_cast_not_executed='OpenVLARuntime.predict later casts pixels toBF16 on model device; CPU processor verification stops before that cast')
                    record.update(status='complete',actual_geometry_verified=True,layout=asdict(layout),observed_native_processor=observed,
                        image_and_arrays_current_sha_verified=True,processor_call_completed=True,
                        visual_sequence_position_binding='source-derived offset1 + earlier actual256patch projector proof')
                    state['processor_calls_completed']+=1
                except Exception as error:
                    record.update(status=failure_status(error),error_type=type(error).__name__,error=str(error))
                    if isinstance(error,(LearnedModelWorkForbidden,TimeoutError)):
                        global_error=(type(error).__name__,str(error));signal.alarm(0)
                record['audit_hash_scope']='canonical_record_before_audit_link_and_sha';record['audit_sha256']=canonical_sha(record)
                record['audit_link']=str((args.output/'geometry.jsonl').resolve())+'#ordinal='+str(row['ordinal'])
                raw=(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n').encode();stream.write(raw);digest.update(raw)
                counts[record['status']]+=1;state['recorded']+=1;state['counts']=dict(counts)
                if state['recorded']%25==0:stream.flush();save()
        loaded=sys.modules.get('torch');cuda=loaded.cuda.is_initialized()if loaded is not None else False
        guards=state['processor_loader']['guards'];blocked=sum(v for k,v in guards.items()if k.startswith('blocked_'))
        if hashes and verify_metadata(args.checkpoint,receipt,official)!=hashes:raise RuntimeError('Metadata changed during processor run')
        state.update(status='complete'if counts['complete']==1500 and not cuda and not blocked else'partial',
            geometry_jsonl_sha256=digest.hexdigest(),finished_epoch=time.time(),cuda_initialized=cuda,
            gpu_calls=None if cuda else 0,launch_allowed=False,semantic_roi_review_performed=False)
        save();(args.output/'result.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        print(json.dumps({'status':state['status'],'counts':dict(counts),'model_calls':0,'gpu_calls':state['gpu_calls']}))
        if state['status']!='complete':raise SystemExit(2)
    finally:signal.alarm(0)


if __name__=='__main__':main()
