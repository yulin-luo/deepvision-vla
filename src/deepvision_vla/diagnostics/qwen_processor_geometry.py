"""Fixed1500 CPU processor geometry only: no model, weights or NN forward."""
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

from .adapters import qwen_h16
from deepvision_vla.diagnostics.model_geometry import actual_visual_layout
from deepvision_vla.diagnostics.adapters.qwen_h16_contracts import validate_cpu_receipt, REVISION
from .formal_plan import canonical_sha,is_sha
from deepvision_vla.diagnostics.input_inventory import SOURCES, validate_linked_inventory, safe_payload_path, failure_status
from deepvision_vla.diagnostics.preprocessing_limits import CPU_THREADS, _cpu_limits

BUDGET_SECONDS = 600
PROCESSOR_SOURCE_PATHS = {
    "transformers/__init__.py","transformers/models/auto/processing_auto.py","transformers/processing_utils.py",
    "transformers/image_processing_utils.py","transformers/image_processing_utils_fast.py",
    "transformers/image_utils.py","transformers/image_transforms.py","transformers/tokenization_utils_base.py",
    "transformers/tokenization_utils_fast.py","transformers/models/qwen2/tokenization_qwen2_fast.py",
    "transformers/models/qwen3_vl/processing_qwen3_vl.py","transformers/models/qwen2_vl/image_processing_qwen2_vl.py",
    "transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py",
    "transformers/models/qwen3_vl/video_processing_qwen3_vl.py","transformers/video_processing_utils.py",
}


class NeuralExecutionForbidden(RuntimeError):pass


@contextmanager
def forbid_neural_modules(module_class,counters):
    """Guard actual Torch Module construction/calls, without a fake backend."""
    original_init,original_call = module_class.__init__,module_class._call_impl
    def refused_init(_self,*_args,**_kwargs):
        counters["blocked_nn_constructor_attempts"] += 1
        raise NeuralExecutionForbidden("CPU geometry must not construct an nn.Module")
    def refused_call(_self,*_args,**_kwargs):
        counters["blocked_nn_forward_attempts"] += 1
        raise NeuralExecutionForbidden("CPU geometry must not run an nn.Module forward")
    module_class.__init__,module_class._call_impl = refused_init,refused_call
    try:yield
    finally:module_class.__init__,module_class._call_impl = original_init,original_call


def verify_file_seals(root,seals,required):
    root = Path(root).resolve()
    if not required.issubset(seals):raise ValueError("Required pre-import source seals are absent")
    for relative,expected in seals.items():
        logical = PurePosixPath(relative)
        if logical.is_absolute() or ".." in logical.parts or "\\" in relative or not is_sha(expected):
            raise ValueError("Unsafe or unsealed source path")
        path = root.joinpath(*logical.parts).resolve(strict=True)
        if not path.is_relative_to(root) or qwen_h16.file_sha256(path) != expected:
            raise ValueError("Pre-import file bytes differ: "+relative)


def reject_conflicting_transformers(source_root,modules=None):
    root = Path(source_root).resolve();modules = sys.modules if modules is None else modules
    for name,module in list(modules.items()):
        if name == "transformers" or name.startswith("transformers."):
            path = getattr(module,"__file__",None)
            if not path or not Path(path).resolve().is_relative_to(root):
                raise RuntimeError("Conflicting Transformers namespace; no hot replacement: "+name)


def validate_config(config):
    vision = config.get("vision_config",{})
    if (config.get("model_type") != "qwen3_vl" or vision.get("patch_size") != 16
            or vision.get("spatial_merge_size") != 2 or vision.get("temporal_patch_size") != 2
            or vision.get("in_channels") != 3 or type(config.get("image_token_id")) is not int):
        raise ValueError("Expected the sealed native Qwen patch16/merge2 still-image configuration")








def prepare_rgb(raw,size,instruction):
    from PIL import Image
    if not isinstance(instruction,str) or not instruction.strip():raise ValueError("Original instruction is empty")
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        if list(image.size) != size:raise ValueError("Actual original image size differs")
        return image.convert("RGB").resize((224,224),resample=Image.Resampling.BICUBIC)


def layout_from_observation(row,config,input_ids,grid,pixel_shape,placeholder):
    """Validate native processor observations, never substitute a fixed grid."""
    validate_config(config)
    if (len(input_ids) != 1 or not input_ids[0] or any(type(x) is not int for x in input_ids[0])
            or type(placeholder) is not int or placeholder == config["image_token_id"]):
        raise ValueError("Expected one native prompt and a non-image pad/eos query ID")
    if len(grid) != 1 or len(grid[0]) != 3 or grid[0][0] != 1:raise ValueError("Expected one actual still-image grid")
    t,h,w = grid[0]
    if any(type(x) is not int or x <= 0 for x in (t,h,w)):raise ValueError("Invalid actual processor grid")
    expected_width = 3*2*16*16
    if list(pixel_shape) != [t*h*w,expected_width]:raise ValueError("Actual image patch tensor does not match the declared grid")
    full_ids = input_ids[0]+[placeholder]*16
    observed = {"image_grid_thw":grid,"visual_positions":[i for i,value in enumerate(full_ids) if value == config["image_token_id"]],
        "action_query_count":16,"original_image_size":row["image_size"],"prepared_image_size":[224,224],
        "image_transform":"full_image_bicubic_resize_224","prompt_length":len(input_ids[0]),"patch_size":16,"spatial_merge_size":2}
    layout = actual_visual_layout(row,observed)
    return layout,observed,{"full_sequence_length":len(full_ids),"pixel_values_shape":list(pixel_shape),
        "action_query_accounting":"sixteen_native_pad_or_eos_IDs_only; no_embeddings_constructed"}




def _load_processor(args,cpu,reference,seals,identity,loader_state):
    verify_file_seals(args.source_root,cpu["source_hashes"],qwen_h16.CRITICAL_SOURCE_FILES)
    verify_file_seals(args.source_root,seals,PROCESSOR_SOURCE_PATHS)
    reference_hashes = {row["path"]:row["sha256"] for row in reference["files"]}
    verify_file_seals(args.config_dir,reference_hashes,{"config.json","tokenizer.json","tokenizer_config.json","preprocessor_config.json"})
    if any(path.name.endswith((".safetensors",".bin",".pt",".pth",".ckpt")) for path in args.config_dir.iterdir()):
        raise ValueError("Processor reference must not contain model weights")
    if (identity.get("model") != "qwen_h16" or identity.get("checkpoint_revision") != REVISION
            or identity.get("source_sha256") != canonical_sha(cpu["source_hashes"])
            or identity.get("processor_sha256") != qwen_h16.file_sha256(args.config_dir/"reference_source_manifest.json")
            or any(not is_sha(identity.get(k)) for k in ("checkpoint_sha256","action_mapping_sha256"))):
        raise ValueError("Supplied model metadata is not bound to the CPU source/processor proof")
    config = json.loads((args.config_dir/"config.json").read_text(encoding="utf8"));validate_config(config)
    reject_conflicting_transformers(args.source_root)
    loader_state["source_seals_verified_before_import"] = True
    sys.path.insert(0,str(args.source_root.resolve()))
    import transformers
    import torch
    import tokenizers
    if transformers.__version__ != "4.57.0" or not Path(transformers.__file__).resolve().is_relative_to(args.source_root.resolve()):
        raise RuntimeError("The actual pinned Transformers4.57.0 processor source is required")
    if torch.cuda.is_initialized():raise RuntimeError("CPU-only geometry inherited a CUDA context")
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    from transformers import AutoProcessor
    counters = loader_state["neural_execution_guards"]
    with forbid_neural_modules(torch.nn.Module,counters):
        processor = AutoProcessor.from_pretrained(args.config_dir,local_files_only=True,trust_remote_code=False)
    if (processor.image_processor.patch_size != 16 or processor.image_processor.merge_size != 2
            or processor.image_token_id != config["image_token_id"]):raise RuntimeError("Actual processor/config geometry contract differs")
    classes = {name:inspect.getfile(value.__class__) for name,value in (
        ("processor",processor),("image_processor",processor.image_processor),("tokenizer",processor.tokenizer))}
    for path in classes.values():
        actual = Path(path).resolve()
        if not actual.is_relative_to(args.source_root.resolve()) or actual.relative_to(args.source_root.resolve()).as_posix() not in seals:
            raise RuntimeError("Actual processor class source was not sealed before import")
    loader_state.update(actual_class_source_paths=classes,versions={"transformers":transformers.__version__,"torch":torch.__version__,"tokenizers":tokenizers.__version__},
                        processor_loaded=True,cuda_initialized=torch.cuda.is_initialized())
    return processor,config,torch


def _base_record(row,identity,processor_sha,source_hashes):
    return {"schema":"qwen_cpu_processor_geometry_v1","ordinal":row["ordinal"],"sample_id":row["sample_id"],"model":"qwen_h16",
        "source_row_sha256":canonical_sha(row),"original_source_row_sha256":canonical_sha(row["original_source"]),
        "original_source":row["original_source"],"image_sha256":row["image_sha256"],"arrays_sha256":row["arrays_sha256"],
        "image_size":row["image_size"],"anchor":row["anchor"],"instruction":row["instruction"],
        "model_identity_sha256":canonical_sha(identity),"processor_sha256":processor_sha,"source_hashes":source_hashes,
        "geometry_source":"actual_processor","actual_geometry_verified":False,"layout":None,
        "processor_call_attempted":False,"processor_call_completed":False,
        "model_calls":0,"nn_forward_calls":0,"gpu_calls":0,"optimizer_steps":0,
        "human_verified":False,"semantic_roi_review_performed":False,"paper_reproduction":False,
        "validation_scope":"actual CPU processor/token/layout only; no new model/weight validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root","source-root","config-dir","cpu-receipt","processor-source-seals","model-identity","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    for name in ("expected-inventory-sha256","expected-original-manifest-sha256","expected-reference-manifest-sha256"):
        parser.add_argument("--"+name,required=True)
    args = parser.parse_args();limits = _cpu_limits()
    root = args.data_root;inventory_raw = (root/"inventory.json").read_bytes();original_raw = (root/"manifest.jsonl").read_bytes()
    inventory_sha = hashlib.sha256(inventory_raw).hexdigest();original_sha = hashlib.sha256(original_raw).hexdigest()
    if inventory_sha != args.expected_inventory_sha256 or original_sha != args.expected_original_manifest_sha256:
        raise ValueError("Immutable staged/original inventory byte seal differs")
    inventory = json.loads(inventory_raw);original = [json.loads(line) for line in original_raw.splitlines() if line.strip()]
    receipt = json.loads((root/"transfer_receipt.json").read_text(encoding='utf-8'));files = json.loads((root/"files.json").read_text(encoding='utf-8'))
    file_map = validate_linked_inventory(inventory,original,files,receipt,inventory_sha,original_sha)
    reference_path = args.config_dir/"reference_source_manifest.json"
    processor_sha = qwen_h16.file_sha256(reference_path)
    if processor_sha != args.expected_reference_manifest_sha256:raise ValueError("Reference processor manifest differs")
    reference = json.loads(reference_path.read_text(encoding='utf-8'));cpu = json.loads(args.cpu_receipt.read_text(encoding='utf-8'))
    validate_cpu_receipt(cpu,qwen_h16.file_sha256(Path(qwen_h16.__file__)))
    seals = json.loads(args.processor_source_seals.read_text(encoding='utf-8'));identity = json.loads(args.model_identity.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=False);started = time.time();deadline = time.monotonic()+600
    state = {"schema":"qwen_cpu_processor_geometry_batch_v1","status":"starting","pid":os.getpid(),
        "start_ticks":int(Path('/proc/self/stat').read_text(encoding='utf-8').rsplit(')',1)[1].split()[19]),"started_epoch":started,
        "limits":limits,"requested":1500,"recorded":0,"counts":{},"processor_calls_attempted":0,"processor_calls_completed":0,
        "payload_files_rehashed":0,"model_calls":0,"nn_forward_calls":0,"gpu_calls":0,"optimizer_steps":0,
        "inventory_sha256":inventory_sha,"original_manifest_sha256":original_sha,"staging_receipt_sha256":qwen_h16.file_sha256(root/'transfer_receipt.json'),
        "processor_reference_manifest_sha256":processor_sha,"cpu_receipt_sha256":qwen_h16.file_sha256(args.cpu_receipt),
        "source_seals_sha256":qwen_h16.file_sha256(args.processor_source_seals),"model_identity_sha256":canonical_sha(identity),
        "processor_loader":{"processor_loaded":False,"neural_execution_guards":{"blocked_nn_constructor_attempts":0,"blocked_nn_forward_attempts":0}}}
    def save():
        state["updated_epoch"] = time.time();temp = args.output/"progress.tmp"
        temp.write_text(json.dumps(state,indent=2),encoding='utf-8');temp.replace(args.output/"progress.json")
    def expired(_signum,_frame):raise TimeoutError("Original600second CPU processor budget exceeded")
    save();signal.signal(signal.SIGALRM,expired);signal.alarm(600)
    global_error = None;processor = config = torch = None;counts = Counter();records_sha = hashlib.sha256()
    try:
        try:processor,config,torch = _load_processor(args,cpu,reference,seals,identity,state["processor_loader"])
        except Exception as error:global_error = (type(error).__name__,str(error));state["initialization_error"] = {"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
        with (args.output/"geometry.jsonl").open("xb") as stream:
            for row in inventory:
                record = _base_record(row,identity,processor_sha,cpu["source_hashes"])
                try:
                    if global_error:
                        if global_error[0] == "TimeoutError":raise TimeoutError(global_error[1])
                        raise RuntimeError("Processor unavailable: "+global_error[0]+": "+global_error[1])
                    if time.monotonic() >= deadline:raise TimeoutError("No new processor calls after the original deadline")
                    source = row["original_source"];payload = {}
                    for field,hash_key in (("image_path","image_sha256"),("arrays_path","arrays_sha256")):
                        path = safe_payload_path(root,source[field]);raw = path.read_bytes()
                        if len(raw) != file_map[source[field]]["bytes"] or hashlib.sha256(raw).hexdigest() != row[hash_key]:
                            raise ValueError("Current original payload bytes differ: "+field)
                        state["payload_files_rehashed"] += 1
                        if field == "image_path":payload["image"] = raw
                    image = prepare_rgb(payload["image"],row["image_size"],row["instruction"])
                    messages = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":row["instruction"]}]}]
                    state["processor_calls_attempted"] += 1
                    record["processor_call_attempted"] = True
                    with forbid_neural_modules(torch.nn.Module,state["processor_loader"]["neural_execution_guards"]):
                        inputs = processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt")
                    if tuple(inputs.attention_mask.shape) != tuple(inputs.input_ids.shape):
                        raise ValueError("Actual prompt mask/token shapes differ")
                    if any(value.device.type != "cpu" for value in inputs.values() if isinstance(value,torch.Tensor)) or torch.cuda.is_initialized():
                        raise NeuralExecutionForbidden("Processor left CPU-only execution")
                    token = processor.tokenizer.pad_token_id
                    if token is None:token = processor.tokenizer.eos_token_id
                    layout,observed,extra = layout_from_observation(row,config,inputs.input_ids.tolist(),inputs.image_grid_thw.tolist(),inputs.pixel_values.shape,token)
                    state["processor_calls_completed"] += 1
                    record.update(status="complete",actual_geometry_verified=True,layout=asdict(layout),observed_native_layout=observed,
                                  processor_only_metadata=extra,image_and_arrays_current_sha_verified=True,processor_call_completed=True)
                except Exception as error:
                    kind = failure_status(error)
                    record.update(status=kind,error_type=type(error).__name__,error=str(error))
                    if isinstance(error,(NeuralExecutionForbidden,TimeoutError)):global_error = (type(error).__name__,str(error));signal.alarm(0)
                record["audit_hash_scope"] = "canonical_record_before_audit_link_and_sha"
                record["audit_sha256"] = canonical_sha(record)
                record["audit_link"] = str((args.output/"geometry.jsonl").resolve())+"#ordinal="+str(row["ordinal"])
                raw = (json.dumps(record,separators=(",",":"),allow_nan=False)+"\n").encode();stream.write(raw);records_sha.update(raw)
                counts[record["status"]] += 1;state["recorded"] += 1;state["counts"] = dict(counts)
                if state["recorded"] % 25 == 0:stream.flush();save()
        loaded_torch = sys.modules.get("torch")
        cuda_initialized = loaded_torch.cuda.is_initialized() if loaded_torch is not None else False
        state.update(status="complete" if counts["complete"] == 1500 and not cuda_initialized else "partial",finished_epoch=time.time(),
            geometry_jsonl_sha256=records_sha.hexdigest(),cuda_initialized=cuda_initialized,gpu_calls=None if cuda_initialized else 0,
            launch_allowed=False,semantic_roi_review_performed=False,model_validation_performed=False)
        save();(args.output/"result.json").write_text(json.dumps(state,indent=2),encoding='utf-8');print(json.dumps({"status":state["status"],"counts":state["counts"],"model_calls":0,"gpu_calls":state["gpu_calls"]}))
        if state["status"] != "complete":raise SystemExit(2)
    finally:signal.alarm(0)


if __name__ == "__main__":main()
