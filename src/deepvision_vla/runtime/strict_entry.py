"""Require strict checkpoint keys and finite saved tensors before prediction."""
import argparse,hashlib,json,pathlib,runpy,sys,os,subprocess,time,ctypes,functools,types
from deepvision_vla.evaluation.runner import pci_key,same_owned_group
from deepvision_vla.runtime.tensors import assert_finite,state_materialization
from deepvision_vla.runtime.checkpoint import component_watch_paths,validate_checkpoint_binding
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--checkpoint-root',required=True);p.add_argument('--wait-gpu-seconds',type=int,default=0);p.add_argument('--expected-gpu-pci',required=True);p.add_argument('--image-request')
p.add_argument('--checkpoint-verification-receipt',required=True)
p.add_argument('--checkpoint-verification-receipt-sha256',required=True)
p.add_argument('--checkpoint-inventory-sha256',required=True)
a,forward=p.parse_known_args()
if forward and forward[0]=='--':forward=forward[1:]
root=pathlib.Path(a.source_root).resolve(strict=True);cp=pathlib.Path(a.checkpoint_root).resolve(strict=True)
checkpoint_binding_config={'checkpoint_root':str(cp),
    'checkpoint_verification_receipt':a.checkpoint_verification_receipt,
    'checkpoint_verification_receipt_sha256':a.checkpoint_verification_receipt_sha256,
    'checkpoint_inventory_sha256':a.checkpoint_inventory_sha256}
forward_parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
forward_parser.add_argument('--dinov3-path',required=True)
dino_forward,_=forward_parser.parse_known_args(forward)
checkpoint_binding_config['dino_runtime_root']=dino_forward.dinov3_path
validate_checkpoint_binding(checkpoint_binding_config)
entry=root/'scripts_rlbench10_ablation/test_rlbench_no_gen_encoder.py'
if sha(entry)!='ba8d48655afc0d3ec0deb527d1f639aaa243b149b3a97cb4dd871b3d0fa52228':raise ValueError('scientific source SHA mismatch')
sys.path.insert(0,str(root));sys.dont_write_bytecode=True
import torch
import transformers
from transformers import Qwen3VLForConditionalGeneration
if transformers.__version__!='4.57.0' or not pathlib.Path(transformers.__file__).resolve().is_relative_to(root):raise ValueError('wrong Transformers implementation')
# Complete slow NFS imports before testing for an idle device. Never change
# CUDA_VISIBLE_DEVICES after imports, never choose another card, never stop a
# foreign process. This wait is before model construction and GPU allocation.
selected=os.environ.get('DEEPVISION_NVML_GPU_UUID','')
if not selected or ',' in os.environ.get('CUDA_VISIBLE_DEVICES',''):raise ValueError('one explicitly selected physical GPU required')
deadline=time.monotonic()+a.wait_gpu_seconds
wait_logged=False
while True:
    row=subprocess.check_output(['nvidia-smi','-i',selected,'--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    index,uuid,used=[v.strip() for v in row.split(',')]
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
    foreign_compute=any(line.split(',')[0].strip()==uuid and not same_owned_group(int(line.split(',')[1].strip())) for line in compute.splitlines() if ',' in line)
    graphics=subprocess.check_output(['nvidia-smi','pmon','-i',index,'-c','1'],text=True,timeout=10)
    foreign_graphics=any(len(v)>1 and v[0]==index and v[1].isdigit() and not same_owned_group(int(v[1])) for v in [line.split() for line in graphics.splitlines()] if v and not v[0].startswith('#'))
    if not foreign_compute and not foreign_graphics and int(used)<=256:break
    if time.monotonic()>=deadline:raise RuntimeError('selected GPU remained occupied; model construction not entered')
    if not wait_logged:print('WAITING_FOR_SELECTED_GPU '+json.dumps({'uuid':uuid,'index':index,'seconds':a.wait_gpu_seconds,'model_loads':0}),flush=True);wait_logged=True
    time.sleep(10)
print('CHECKPOINT_BINDING_PASS '+json.dumps(validate_checkpoint_binding(checkpoint_binding_config)),flush=True)
driver=ctypes.CDLL('libcuda.so.1')
if driver.cuInit(0)!=0:raise RuntimeError('CUDA driver initialization failed before model load')
count=ctypes.c_int()
if driver.cuDeviceGetCount(ctypes.byref(count))!=0 or count.value!=1:raise ValueError('driver must see exactly one selected device')
dev=ctypes.c_int();bus=ctypes.create_string_buffer(32)
if driver.cuDeviceGet(ctypes.byref(dev),0)!=0 or driver.cuDeviceGetPCIBusId(bus,32,dev)!=0:raise RuntimeError('CUDA PCI identity unavailable')
actual_pci=bus.value.decode()
if pci_key(actual_pci)!=pci_key(a.expected_gpu_pci):raise ValueError('CUDA device differs from the NVML-selected physical PCI device')
torch.cuda.init()
if torch.cuda.device_count()!=1:raise ValueError('expected exactly one visible CUDA device')
health=torch.ones((32,32),device='cuda');torch.cuda.synchronize()
if not bool((health@health==32).all()):raise ValueError('CUDA health mismatch')
del health
print('SINGLE_GPU_CUDA_READY '+json.dumps({'nvml_uuid':uuid,'physical_pci':actual_pci,'local_index':0,'pid':os.getpid(),'device':torch.cuda.get_device_name(0)}),flush=True)
original=Qwen3VLForConditionalGeneration.from_pretrained
request_context={}
def trace_policy_calls(namespace,model):
    predict=namespace.get('model_predict')
    if not callable(predict) or getattr(predict,'_release_observed',False):return
    @functools.wraps(predict)
    def observed(*args,**kwargs):
        caller=sys._getframe(1);episode=caller.f_locals.get('episode_id')
        if episode is None and isinstance(caller.f_locals.get('row'),dict):episode=caller.f_locals['row'].get('episode_id')
        task=getattr(args[0],'task_name',None);text=kwargs.get('task_description',args[6] if len(args)>6 else None);step=kwargs.get('step',args[11] if len(args)>11 else None)
        request_context.clear();request_context.update(task=task,episode_id=episode,step=step,instruction=text)
        print('POLICY_REQUEST '+json.dumps(request_context),flush=True)
        return predict(*args,**kwargs)
    observed._release_observed=True;namespace['model_predict']=observed
    def forward_entry(module,inputs):print('MODEL_FORWARD_ENTRY '+json.dumps(request_context),flush=True)
    model.model.register_forward_pre_hook(forward_entry)
def finite(state,component):
    print('STRICT_FINITE_PASS '+json.dumps(assert_finite(state,component,torch)),flush=True)
def strict_qwen(cls,*args,**kwargs):
    caller_namespace=sys._getframe(1).f_globals
    validate_checkpoint_binding(checkpoint_binding_config)
    kwargs['output_loading_info']=True
    model,info=original(*args,**kwargs)
    validate_checkpoint_binding(checkpoint_binding_config)
    if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError(f'Qwen strict loading failed: {info}')
    finite(model.state_dict(),'qwen');print('STRICT_QWEN_LOADING_PASS',flush=True);trace_policy_calls(caller_namespace,model);return model
Qwen3VLForConditionalGeneration.from_pretrained=classmethod(strict_qwen)
watched=component_watch_paths(cp)
loaded={};torch_load=torch.load;state_load=torch.nn.Module.load_state_dict
def observe_torch_load(file,*args,**kwargs):
    key=str(pathlib.Path(file).resolve()) if isinstance(file,(str,pathlib.Path)) else None
    if key in watched:validate_checkpoint_binding(checkpoint_binding_config)
    state=torch_load(file,*args,**kwargs)
    if key in watched:
        validate_checkpoint_binding(checkpoint_binding_config)
        name=watched[key]
        if state_materialization(state,torch)=='metadata_only':
            print('META_STATE_INSPECTION_NOT_FINITE_PROOF '+json.dumps({'component':name,'keys':len(state)}),flush=True)
        else:finite(state,name);loaded[id(state)]=name
    return state
def strict_component(module,state,*args,**kwargs):
    name=loaded.get(id(state));strict=kwargs.get('strict',args[0] if args else True)
    if name and strict is not True:raise ValueError(f'{name} requires strict=True')
    result=state_load(module,state,*args,**kwargs)
    if name:
        if result.missing_keys or result.unexpected_keys:raise ValueError(f'{name} state mismatch')
        print('STRICT_COMPONENT_PASS '+json.dumps({'component':name,'keys':len(state)}),flush=True)
    return result
torch.load=observe_torch_load;torch.nn.Module.load_state_dict=strict_component
sys.argv=[str(entry),*forward]
mode=forward[forward.index('--replay-or-predict')+1] if '--replay-or-predict' in forward else 'predict'
if a.image_request:
    if mode!='dataset_predict':raise ValueError('image dispatch is only supported in the non-simulator entry')
    from deepvision_vla.inference.image import image_dispatch_tree
    tree=image_dispatch_tree(entry.read_text(),a.image_request)
    print('IMAGE_DISPATCH_PROTOCOL '+json.dumps({'source_sha256':sha(entry),'execution_ast_sha256':hashlib.sha256(__import__('ast').dump(tree,include_attributes=False).encode()).hexdigest(),'gt_action_input':False,'simulator_episodes':0}),flush=True)
elif mode=='predict':
    from deepvision_vla.runtime import reset as reset_protocol
    from deepvision_vla.runtime.reset import POLICY,transform_evaluator,install_reset_adapter,execution_fingerprint
    install_reset_adapter();tree=transform_evaluator(entry.read_text())
    print('RESET_PROTOCOL '+json.dumps({'policy':POLICY,'outer_attempts':1,'wrapper_attempts':1,'seed_formula':'eval_seed + episode_id * 1009','execution_ast_sha256':execution_fingerprint(tree),'source_sha256':sha(entry),'adapter_sha256':sha(reset_protocol.__file__)}),flush=True)
if a.image_request or mode=='predict':
    previous_main=sys.modules.get('__main__');module=types.ModuleType('__main__')
    module.__dict__.update(__file__=str(entry),__package__=None,__cached__=None,__spec__=None);sys.modules['__main__']=module
    try:exec(compile(tree,str(entry),'exec'),module.__dict__)
    finally:sys.modules['__main__']=previous_main
else:runpy.run_path(str(entry),run_name='__main__')
