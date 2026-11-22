"""Pinned third-party LeRobot pi0 Bridge adapter, distinct from official pi05.

The trained policy and normalizers are loaded unchanged. Bridge EEF state has
one documented dummy slot inserted; this does not convert EEF pose to joints.
GT action relabeling/time alignment is deliberately not implemented here.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys

from ..geometry import VisualLayout

MODEL_ID = "HaomingSong/lerobot-pi0-bridge"
MODEL_REVISION = "93c2840a2c68c337da0d472a2f52d1fd8d62132a"
LEROBOT_COMMIT = "5d24ce31606eb3e48a0f644530f928c5f119a724"
WEIGHT_SHA256 = "7078a1ddf199fc4964340e7f4622f3f98e79b2acfb9e83ac538fe5dcade2a90d"
_SOURCE_BLOBS = {"configuration_pi0.py":"5dc6d0cfcbc26b8d3200f845d67efe001f2707ff",
                 "modeling_pi0.py":"ab0f9fb2a91ed6339d6ff4d791da618ca0ff9a94"}
_DEPS = {
 "lerobot/common/policies/pi0/paligemma_with_expert.py":"1aba460337aacf005fd1da9396a320febb22491b383e3ad76f3636f1f33ce8ae",
 "lerobot/common/policies/pi0/flex_attention.py":"879be3dd236ff12734668f6abf01fed6c0a68dfa08c8b2b78cafdb45f8725de8",
 "lerobot/common/policies/pretrained.py":"efd5cb098eee803d9732f596d7c1dd9487bf621596af79c3ae06bff1bf0a13ed",
 "lerobot/common/policies/normalize.py":"b266de275633ae8b636b59cb563b55b36a50d48ad8637e686a82fed72cffed56"}


def bridge_eef_state8(state7, *, frame="bridge_dataset_topdown"):
    """Preserve measured Bridge EEF state, inserting only its documented pad.

    The sensor gripper channel is continuous raw proprioception, not a binary
    action label or a guaranteed [0,1] value. Its trained MEAN_STD normalizer
    receives the original finite reading, including encoder overshoot.
    """
    if frame != "bridge_dataset_topdown":
        raise ValueError("Supply dataset-frame EEF pose; simulator base rotation must be converted separately")
    values = [float(x) for x in state7]
    if len(values)!=7 or not all(math.isfinite(x) for x in values):
        raise ValueError("Expected finite Bridge xyz/rpy/gripper state7")
    return values[:6]+[0.0,values[6]]


def resize_geometry(source_size, target_size=(224,224)):
    """Match checkpoint resize_with_pad: floor resize, then LEFT/TOP padding."""
    width,height=source_size;target_width,target_height=target_size
    if any(type(x) is not int or x<=0 for x in (*source_size,*target_size)):
        raise ValueError("Image dimensions must be positive integers")
    ratio=max(width/target_width,height/target_height)
    resized_width,resized_height=int(width/ratio),int(height/ratio)
    if min(resized_width,resized_height)<1:raise ValueError("Image aspect ratio is unsupported")
    left,top=target_width-resized_width,target_height-resized_height
    return (left,top,target_width,target_height)


def validate_native_noise_shape(shape, horizon=4):
    if tuple(shape)!=(1,horizon,32):
        raise ValueError("pi0 fixed noise must preserve full [1,horizon,32] native shape")


@contextmanager
def zero_prefix_hidden_inference(norm_layer, positions, *, prefix_length):
    """Zero hidden storage before LN so the manual loop residual sees it too.

    Inference only. Returning a cloned LN input would leave the native loop's
    original residual untouched and implement a different intervention.
    """
    import torch
    positions=tuple(positions)
    if len(set(positions))!=len(positions) or any(type(x) is not int or x<0 or x>=256 for x in positions):
        raise ValueError("Only this checkpoint's 256 visual prefix positions may be dropped")
    observed={"calls":0,"operator":"inference_inplace_prefix_hidden_zero","gradient_supported":False}
    def hook(module,args,kwargs):
        if torch.is_grad_enabled():raise RuntimeError("This inference intervention does not support Grad-CAM")
        hidden=args[0] if args else kwargs.get('hidden_states')
        if hidden is None or hidden.ndim!=3 or tuple(hidden.shape[:2])!=(1,prefix_length):
            raise RuntimeError("Hook was not reached on the declared native image/text prefix")
        observed['calls']+=1
        if positions:hidden[:,list(positions),:]=0
        return None
    handle=norm_layer.register_forward_pre_hook(hook,with_kwargs=True)
    try:yield observed
    finally:handle.remove()


def _import_local(name,path):
    if name in sys.modules:
        module=sys.modules[name]
        if Path(module.__file__).resolve()!=path.resolve():raise RuntimeError("Model module alias collision: "+name)
        return module
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None);raise
    return module


def load_bridge_pi0(checkpoint, *, lerobot_source_root, asset_receipt):
    """Load real author checkpoint + pinned real HF LeRobot dependency, strictly.

    No GPU movement here. Caller owns GPU selection, PCI check, lease/budget.
    The full dependency checkout must be the stated commit; critical imported
    files are also checked below, including its historically non-strict loader.
    """
    checkpoint=Path(checkpoint).resolve();source=Path(lerobot_source_root).resolve()
    receipt=json.loads(Path(asset_receipt).read_text(encoding='utf-8'))
    if receipt.get('status')!='complete' or receipt.get('model')!=MODEL_ID or receipt.get('revision')!=MODEL_REVISION:
        raise RuntimeError("Complete pinned pi0 Bridge asset receipt is required")
    snapshot={}
    for item in receipt['files']:
        path=checkpoint/item['path'];s=path.stat()
        if not item.get('verified') or s.st_size!=item['size']:raise RuntimeError("Incomplete asset receipt")
        if s.st_size<1024*1024 and hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError("Checkpoint metadata changed: "+item['path'])
        snapshot[path]=(s.st_size,s.st_mtime_ns,s.st_ino)
    weight=next((x for x in receipt['files'] if x['path']=='model.safetensors'),None)
    if weight is None or weight.get('sha256')!=WEIGHT_SHA256:raise RuntimeError("Wrong pi0 Bridge weights")
    for relative,expected in _DEPS.items():
        if hashlib.sha256((source/relative).read_bytes()).hexdigest()!=expected:
            raise RuntimeError("Wrong historical LeRobot dependency: "+relative)
    for name,expected in _SOURCE_BLOBS.items():
        b=(checkpoint/name).read_bytes()
        if hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest()!=expected:
            raise RuntimeError("Wrong author model source: "+name)
    import transformers
    if transformers.__version__!='4.48.1':raise RuntimeError("Use isolated pi0 Bridge Transformers4.48.1 environment")
    from lerobot.common.policies.pretrained import PreTrainedPolicy
    from lerobot.common.policies import normalize
    from lerobot.common.policies.pi0 import paligemma_with_expert
    for module,relative in [(inspect.getmodule(PreTrainedPolicy),'lerobot/common/policies/pretrained.py'),
                            (normalize,'lerobot/common/policies/normalize.py'),
                            (paligemma_with_expert,'lerobot/common/policies/pi0/paligemma_with_expert.py')]:
        if Path(module.__file__).resolve()!=(source/relative).resolve():raise RuntimeError("Imported dependency differs from pinned source")
    previous=sys.modules.get('paligemma_with_expert')
    if previous is not None and previous is not paligemma_with_expert:raise RuntimeError("Conflicting PaliGemma helper")
    # The publisher omitted this file from the bundle. This aliases an actual
    # pinned upstream module, not a dependency stub or reimplemented model.
    sys.modules['paligemma_with_expert']=paligemma_with_expert
    config_module=_import_local('configuration_pi0',checkpoint/'configuration_pi0.py')
    module=_import_local('deepvision_author_bridge_pi0',checkpoint/'modeling_pi0.py')
    from .openpi_bridge_config import parse_bridge_pi0_config
    from .openpi_bridge_helper_compat import provided_subconfig_constructor_compat
    from .openpi_bridge_unused_head import absent_unused_expert_head_compat, audit_active_state_header
    config,config_audit=parse_bridge_pi0_config(checkpoint,config_module.PI0Config)
    with provided_subconfig_constructor_compat(paligemma_with_expert) as constructor_audit:
        with absent_unused_expert_head_compat(paligemma_with_expert,checkpoint) as unused_head_audit:
            policy=module.PI0Policy.from_pretrained(checkpoint,config=config,map_location='cpu',strict=True,
                                                   local_files_only=True,token=False)
    if unused_head_audit['constructor_calls']!=1:
        raise RuntimeError("Expected one proven expert-head structural compatibility operation")
    active_state_audit=audit_active_state_header(policy,checkpoint)
    for path,identity in snapshot.items():
        s=path.stat()
        if (s.st_size,s.st_mtime_ns,s.st_ino)!=identity:raise RuntimeError("Checkpoint changed during loading")
    import torch
    normalizers={k:v for k,v in policy.state_dict().items() if 'normalize_' in k or 'unnormalize_' in k}
    if not normalizers or not all(bool(torch.isfinite(v).all()) for v in normalizers.values()):
        raise RuntimeError("Missing or nonfinite trained normalizer buffers")
    adapter=Pi0BridgeAdapter(policy)
    adapter.config_compatibility_audit=config_audit
    adapter.constructor_compatibility_audit=constructor_audit
    adapter.unused_expert_head_audit=unused_head_audit
    adapter.active_state_header_audit=active_state_audit
    return adapter


class Pi0BridgeAdapter:
    """Author-native full action chunks; no GT target or arbitrary denormalizer."""
    def __init__(self,policy):
        config=policy.config
        if tuple(config.input_features['observation.state'].shape)!=(8,) or tuple(config.output_features['action'].shape)!=(7,):
            raise ValueError("Checkpoint input/output specification changed")
        if config.chunk_size!=4 or config.max_action_dim!=32 or config.tokenizer_max_length!=48 or config.empty_cameras!=0:
            raise ValueError("Unsupported checkpoint horizon/token/camera configuration")
        if config.adapt_to_pi_aloha or config.use_delta_joint_actions_aloha:
            raise ValueError("Aloha transforms must not be applied to Bridge EEF state")
        self.policy=policy.eval();self.config=config

    def visual_layout(self,image_sha256,source_size):
        # This pinned checkpoint has one SigLIP224/14 camera and 256 prefix
        # patches before its 48 text tokens. Runtime hook length checks bind it.
        return VisualLayout(image_sha256,tuple(source_size),(0,0,*source_size),
                            (224,224),resize_geometry(source_size),(16,16),tuple(range(256)))

    def predict(self,image_rgb,state7,instruction,*,fixed_noise,layer=None,dropped_positions=()):
        import numpy as np
        import torch
        array=np.asarray(image_rgb)
        if array.dtype!=np.uint8 or array.ndim!=3 or array.shape[-1]!=3:
            raise ValueError("Supply original uint8 RGB image")
        if not isinstance(instruction,str) or not instruction.strip():raise ValueError("Instruction is required")
        validate_native_noise_shape(fixed_noise.shape,self.config.chunk_size)
        if not isinstance(fixed_noise,torch.Tensor) or not bool(torch.isfinite(fixed_noise).all()):
            raise ValueError("Fixed noise must be a finite real Torch tensor")
        device=next(self.policy.parameters()).device
        batch={'observation.state':torch.tensor([bridge_eef_state8(state7)],dtype=torch.float32,device=device),
               'observation.images.image_0':torch.from_numpy(array.copy()).permute(2,0,1).float()[None,...].to(device)/255,
               'task':[instruction]}
        # Author's sampler mutates x_t/noise in place. A pristine clone per
        # ratio is mandatory; reset queue/state and rebuild the observation.
        noise=fixed_noise.detach().clone().to(device=device,dtype=torch.float32)
        self.policy.reset()
        observed=None
        with torch.no_grad():
            if layer is None:
                if dropped_positions:raise ValueError("A layer is required for masking")
                result=self.policy.select_action(batch,noise=noise)
            else:
                layers=self.policy.model.paligemma_with_expert.paligemma.language_model.model.layers
                if type(layer) is not int or not 0<=layer<len(layers):raise ValueError("Layer outside real prefix model")
                with zero_prefix_hidden_inference(layers[layer].input_layernorm,dropped_positions,
                                                  prefix_length=256+self.config.tokenizer_max_length) as observed:
                    result=self.policy.select_action(batch,noise=noise)
                if observed['calls']!=1:raise RuntimeError("Expected one actual prefix hidden-state intervention")
        if tuple(result.shape)!=(1,4,7) or not bool(torch.isfinite(result).all()):
            raise RuntimeError("Unexpected native pi0 Bridge prediction")
        return {'actions':result.detach().cpu().numpy()[0],'hook':observed,
                'gt_mapping_verified':False,'action_contract':'author_bridge_native_mean_std_denormalized'}
