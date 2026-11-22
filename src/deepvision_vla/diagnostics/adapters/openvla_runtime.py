"""Reusable official OpenVLA runtime for model-once diagnostic batches.

The caller owns process, device selection, resource checks and GPU leases.
This adapter preserves the original canary's native prediction science.
"""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import threading

from ..bridge import bridge_orig_targets
from ..contracts import ActionSpace
from ..geometry import VisualLayout, project_roi_boxes
from ..interventions import zero_visual_tokens


MODEL_ID = "openvla/openvla-7b"
MODEL_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
ACTION_SPACE = ActionSpace(("dx","dy","dz","droll","dpitch","dyaw","open_gripper"),
    "reached_state_delta", "bridge_orig/observation/state", "bridge_orig_xyz_units",
    "bridge_orig_rpy_units", "physical", "openvla_bridge_orig_relabel_v1", True)
ESSENTIAL_FILES = {"config.json", "configuration_prismatic.py", "modeling_prismatic.py",
    "processing_prismatic.py", "preprocessor_config.json", "processor_config.json",
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "model.safetensors.index.json"}


def _read_json(value):
    return json.loads(Path(value).read_text(encoding="utf8")) if isinstance(value,(str,Path)) else value


def _identity(path):
    link=path.lstat();target=path.resolve(strict=True);stat=target.stat()
    return (str(target),stat.st_size,stat.st_mtime_ns,stat.st_ino,stat.st_dev,
            link.st_size,link.st_mtime_ns,link.st_ino)


def validate_assets(checkpoint, asset_receipt, official_inventory):
    """Verify all pinned snapshot bytes once; allow genuine HF leaf symlinks."""
    checkpoint=Path(checkpoint)
    receipt,inventory=map(_read_json,(asset_receipt,official_inventory))
    if (receipt.get("status")!="complete" or receipt.get("model")!=MODEL_ID
            or receipt.get("revision")!=MODEL_REVISION
            or inventory.get("model")!=MODEL_ID or inventory.get("revision")!=MODEL_REVISION):
        raise ValueError("Official completed OpenVLA snapshot identity differs")
    expected={row["path"]:row for row in inventory["files"]}
    received={row["path"]:row for row in receipt["files"]}
    if (len(expected)!=len(inventory["files"]) or len(received)!=len(receipt["files"])
            or set(expected)!=set(received) or not ESSENTIAL_FILES.issubset(expected)):
        raise ValueError("Checkpoint inventory is incomplete or duplicated")
    identities={};hashes={}
    for name,item in expected.items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Unsafe logical checkpoint path")
        path=checkpoint.joinpath(*relative.parts)
        before=_identity(path)
        found=received[name]
        if not found.get("verified") or before[1]!=item["size"] or found.get("size")!=item["size"]:
            raise ValueError("Checkpoint file was not acquired completely: "+name)
        sha=hashlib.sha256();git=hashlib.sha1(("blob "+str(item["size"])+"\0").encode())
        with path.open("rb") as stream:
            while chunk:=stream.read(8*1024*1024):sha.update(chunk);git.update(chunk)
        digest=sha.hexdigest()
        if digest!=found.get("sha256") or (item.get("sha256") and digest!=item["sha256"]):
            raise ValueError("Pinned checkpoint checksum differs: "+name)
        if not item.get("sha256") and item.get("git_blob")!=git.hexdigest():
            raise ValueError("Pinned source Git blob differs: "+name)
        if before!=_identity(path):raise RuntimeError("Checkpoint changed while being verified")
        identities[name]=before;hashes[name]=digest
    index=_read_json(checkpoint/"model.safetensors.index.json")
    shards=set(index["weight_map"].values())
    if not shards or not shards.issubset(expected):raise ValueError("Weight shard index is incomplete")
    return identities,hashes


def strict_loading_info(info):
    if not isinstance(info,dict):raise ValueError("Actual pretrained loading information is required")
    for key in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs"):
        if key not in info or info[key]:raise ValueError("Strict pretrained load failed: "+key)


def prepare_prompt_tokens(input_ids,attention_mask=None):
    """Match native empty-action-token insertion and keep the mask aligned."""
    ids=list(input_ids);mask=None if attention_mask is None else list(attention_mask)
    if not ids or any(type(x)is not int or x<0 for x in ids):raise ValueError("Expected nonempty integer token IDs")
    if mask is not None and len(mask)!=len(ids):raise ValueError("Prompt mask length differs from token length")
    appended=ids[-1]!=29871
    if appended:
        ids.append(29871)
        if mask is not None:mask.append(1)
    return ids,mask,appended


def build_visual_layout(image_sha256,source_size,*,strategy,input_sizes,grids):
    if strategy!="resize-naive":raise ValueError("This official snapshot requires its native resize-naive processor")
    if not grids or len(set(tuple(x) for x in grids))!=1:raise ValueError("Fused visual grids differ")
    if not input_sizes or len(set(tuple(x[-2:]) for x in input_sizes))!=1:
        raise ValueError("Fused processor image sizes differ")
    rows,cols=map(int,grids[0]);height,width=map(int,input_sizes[0][-2:])
    sw,sh=source_size
    # Reviewed pinned native code inserts projected patches after the BOS.
    return VisualLayout(image_sha256,tuple(source_size),(0,0,sw,sh),
        (width,height),(0,0,width,height),(rows,cols),tuple(range(1,1+rows*cols)))


def validate_intervention(layer,positions,layout,layer_count):
    positions=tuple(positions)
    if layer is None and positions:raise ValueError("Dropped positions require an explicit layer")
    if layer is not None and (type(layer)is not int or not 0<=layer<layer_count):
        raise ValueError("Requested OpenVLA layer does not exist")
    if len(set(positions))!=len(positions) or not set(positions).issubset(layout.token_positions):
        raise ValueError("Intervention includes duplicate, nonvisual or wrong-grid positions")
    return positions


def checked_action(values):
    if len(values)!=7:raise ValueError("Native OpenVLA prediction must remain seven-dimensional")
    result=[float(value) for value in values]
    if not all(math.isfinite(x) for x in result):raise ValueError("Non-finite native action")
    return result


def reference_action(states,actions,anchor):
    if type(anchor)is not int:raise ValueError("Original frame index must be an integer")
    for row in bridge_orig_targets(states,actions):
        if row["original_frame_index"]==anchor:return row["action"]
    raise ValueError("Frame is not an eligible Bridge-orig transition")


class OpenVLARuntime:
    action_space=ACTION_SPACE
    reference_action=staticmethod(reference_action)

    def __init__(self,model,processor,torch,checkpoint,identities,hashes):
        self.model,self.processor,self.torch=model.eval(),processor,torch
        self.model_id,self.model_revision=MODEL_ID,MODEL_REVISION
        self.checkpoint,self.identities,self.source_hashes=Path(checkpoint),identities,hashes
        self.layer_count=len(model.language_model.model.layers)
        if self.layer_count<=0:raise ValueError("No language layers")
        if "bridge_orig" not in model.norm_stats or model.get_action_dim("bridge_orig")!=7:
            raise ValueError("The explicit Bridge-orig seven-dimensional decoder is required")
        self._lock=threading.Lock();self.last_call=None
        self.counters=dict(request_calls=0,prediction_calls_attempted=0,prediction_calls_completed=0,
                           model_forward_calls=0,projector_calls=0,prefill_calls=0,cached_decode_calls=0)

    def assert_assets_unchanged(self):
        for name,identity in self.identities.items():
            if _identity(self.checkpoint/name)!=identity:raise RuntimeError("Checkpoint source identity changed: "+name)

    def to(self,device):
        with self._lock:self.model.to(device=device,dtype=self.torch.bfloat16).eval()
        return self

    def visual_layout(self,image_sha256,source_size):
        ip=self.processor.image_processor
        grids=[self.model.vision_backbone.featurizer.patch_embed.grid_size]
        if self.model.vision_backbone.use_fused_vision_backbone:
            grids.append(self.model.vision_backbone.fused_featurizer.patch_embed.grid_size)
        return build_visual_layout(image_sha256,source_size,strategy=ip.image_resize_strategy,
                                   input_sizes=ip.input_sizes,grids=grids)

    def roi_positions(self,boxes,*,image_sha256,source_size,membership):
        layout=self.visual_layout(image_sha256,source_size)
        return project_roi_boxes(boxes,image_sha256=image_sha256,layout=layout,membership=membership)

    def predict(self,image,instruction,*,image_sha256,layer=None,dropped_positions=(),seed=42):
        if not self._lock.acquire(blocking=False):raise RuntimeError("One runtime cannot predict concurrent samples")
        call={"status":"preparing","layer":layer,"seed":seed,"model_forward_calls":0,
              "projector_calls":0,"projected_shapes":[],"model":MODEL_ID,"revision":MODEL_REVISION}
        self.last_call=call;self.counters["request_calls"]+=1
        handles=[]
        try:
            if self.model.training:raise RuntimeError("The diagnostic model must remain in eval mode")
            if not isinstance(instruction,str) or not instruction.strip():raise ValueError("Instruction is empty")
            if type(seed)is not int:raise ValueError("Seed must be an integer")
            image=image.convert("RGB")
            layout=self.visual_layout(image_sha256,image.size)
            positions=validate_intervention(layer,dropped_positions,layout,self.layer_count)
            prompt='In: What action should the robot take to '+instruction.strip().lower()+'?\nOut:'
            inputs=self.processor(prompt,image,return_tensors='pt')
            if tuple(inputs.input_ids.shape[:1])!=(1,):raise ValueError("Only one image/prompt at a time is supported")
            ids,mask,appended=prepare_prompt_tokens(inputs.input_ids[0].tolist(),
                inputs.attention_mask[0].tolist() if 'attention_mask' in inputs else None)
            if appended:
                inputs['input_ids']=self.torch.tensor([ids],dtype=inputs.input_ids.dtype)
                if mask is not None:inputs['attention_mask']=self.torch.tensor([mask],dtype=inputs.attention_mask.dtype)
            device=next(self.model.parameters()).device
            inputs={key:value.to(device,dtype=self.torch.bfloat16) if key=='pixel_values' else value.to(device)
                    for key,value in inputs.items()}
            def count_forward(module,args):call['model_forward_calls']+=1
            def observe_projector(module,args,output):
                call['projector_calls']+=1;call['projected_shapes'].append(list(output.shape))
                if tuple(output.shape[:2])!=(1,len(layout.token_positions)):
                    raise RuntimeError("Observed projected patches differ from actual visual grid")
            handles.append(self.model.register_forward_pre_hook(count_forward))
            handles.append(self.model.projector.register_forward_hook(observe_projector))
            observation_layer=0 if layer is None else layer
            self.torch.manual_seed(seed)
            call['status']='predicting';self.counters['prediction_calls_attempted']+=1
            with zero_visual_tokens(self.model.language_model.model.layers[observation_layer],positions) as observed:
                call['hook']=observed  # retain actual coverage even if prediction fails
                with self.torch.inference_mode():
                    actions=self.model.predict_action(**inputs,unnorm_key='bridge_orig',do_sample=False)
            result=checked_action(actions)
            if (observed['prefill_calls']!=1 or call['projector_calls']!=1
                    or call['model_forward_calls']!=1+observed['cached_decode_calls']):
                raise RuntimeError("Actual native prefill/decode/projector accounting differs")
            call.update(status='complete',actions=result,hook=dict(observed),layout=asdict(layout),
                norm_stats_key='bridge_orig',empty_token_appended=appended,
                prompt_tokens=len(ids),native_generation_reached_seven_steps=call['model_forward_calls']==7)
            self.counters['prediction_calls_completed']+=1
            return call
        except BaseException as error:
            call.update(status='failed',error_type=type(error).__name__,error=str(error))
            raise
        finally:
            for handle in handles:handle.remove()
            for name in ('model_forward_calls','projector_calls'):
                self.counters[name]+=call[name]
            if 'hook' in call:
                self.counters['prefill_calls']+=call['hook']['prefill_calls']
                self.counters['cached_decode_calls']+=call['hook']['cached_decode_calls']
            self._lock.release()


def load_openvla_runtime(checkpoint,asset_receipt,official_inventory):
    """Load the genuine CPU model once. Device transfer remains caller-owned."""
    checkpoint=Path(checkpoint)
    identities,hashes=validate_assets(checkpoint,asset_receipt,official_inventory)
    import torch
    import timm
    import transformers
    from transformers import AutoModelForVision2Seq,AutoProcessor
    if transformers.__version__!='4.40.1' or timm.__version__!='0.9.10':
        raise RuntimeError("Use the validated OpenVLA environment, not the Qwen/OpenPI environment")
    processor=AutoProcessor.from_pretrained(checkpoint,trust_remote_code=True,local_files_only=True)
    model,info=AutoModelForVision2Seq.from_pretrained(checkpoint,trust_remote_code=True,
        local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager',
        low_cpu_mem_usage=True,output_loading_info=True)
    strict_loading_info(info)
    if any(parameter.is_meta for parameter in model.parameters()):raise RuntimeError("Model weights remain unmaterialized")
    if any(parameter.device.type!='cpu' for parameter in model.parameters()):
        raise RuntimeError("The reusable factory must first load the model on CPU")
    runtime=OpenVLARuntime(model,processor,torch,checkpoint,identities,hashes)
    runtime.assert_assets_unchanged()
    runtime.loading_info=info
    runtime.versions={'torch':torch.__version__,'transformers':transformers.__version__,'timm':timm.__version__}
    return runtime
