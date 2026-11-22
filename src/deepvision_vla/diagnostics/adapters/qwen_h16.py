"""Qwen-only LAST0 H16 components, without H1/query conversion.

Heavy imports occur only inside the real loader/forward methods. This module
does not define a Bridge GT mapping: stored LAST0 actions and Bridge reached
state labels are distinct contracts until their semantics are established.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


STATS_SHA256 = "0e30b5c17c91bdd3f3b3c9ea1814af189dd8192ad5ea37179acf3597efa694c7"
HORIZON = 16
DIMENSIONS = 7
CRITICAL_SOURCE_FILES = {
    "transformers/models/qwen3_vl/modeling_qwen3_vl.py",
    "transformers/models/qwen3_vl/processing_qwen3_vl.py",
    "deepvision/models/action_head.py",
}


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def validate_h16_training(config):
    expected = {
        "action_chunk": 16,
        "action_dim": 7,
        "action_query_contract": "per_step_shared_v1",
        "action_query_count": 16,
        "action_head_contract": "shared_per_step_mlp_resnet_to_7d_v1",
        "action_query_placeholder_contract": "active_pad_or_eos_token_ids_then_zero_embedding_v1",
        "qwen_position_contract": "token_ids_mrope",
        "vla_attention_mask_contract": "prompt_causal_action_bidirectional",
        "attention_collection_mask_contract": "corrected_causal",
        "action_task_attention_contract": "action_only",
        "lerobot_action_contract": "stored_7d_action_no_second_difference",
        "use_dinov3": 0,
        "input_image_num": 1,
        "image_aug": 0,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"Unsupported Qwen H16 field {name}: expected {value!r}")
    if int(config.get("use_robot_state", 0)) != 0:
        raise ValueError("This adapter is image/instruction-only; robot-state conditioning is unsupported")
    if set(config.get("data_input_image_resolution_counts", {})) != {"224x224"}:
        raise ValueError("Training image resolution is not the verified 224x224 contract")
    return expected


def read_h16_statistics(path):
    if file_sha256(path) != STATS_SHA256:
        raise ValueError("Qwen LAST0 statistics do not match the producer hash")
    action = json.loads(Path(path).read_text(encoding="utf8"))["last0_lerobot_v3"]["action"]
    for name in ("mask", "q01", "q99"):
        if len(action[name]) != HORIZON * DIMENSIONS:
            raise ValueError("H16 statistics must have 112 row-major channels")
    for index, (mask, low, high) in enumerate(zip(action["mask"], action["q01"], action["q99"])):
        if bool(mask) != (index % 7 != 6):
            raise ValueError("Unexpected gripper/continuous normalization mask")
        if not math.isfinite(low) or not math.isfinite(high) or (mask and high <= low):
            raise ValueError("Invalid action quantiles")
    return action


def decode_h16_chunk(normalized, action_statistics):
    """Decode the complete H16 chunk in producer row-major ordering.

    This mirrors the reference decoder: gripper is thresholded at .5, continuous
    channels use q01/q99 without clipping. It is not a Bridge coordinate map.
    """
    if len(normalized) != HORIZON or any(len(row) != DIMENSIONS for row in normalized):
        raise ValueError("Expected the full [16,7] output, not an H1/7-query conversion")
    if any(len(action_statistics[k]) != 112 for k in ("mask", "q01", "q99")):
        raise ValueError("Expected 112 statistics values")
    result = []
    for step, row in enumerate(normalized):
        decoded = []
        for channel, raw in enumerate(row):
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("Non-finite Qwen action")
            index = step * DIMENSIONS + channel
            if channel == 6:
                value = float(value >= 0.5)
            if action_statistics["mask"][index]:
                low, high = action_statistics["q01"][index], action_statistics["q99"][index]
                value = 0.5 * (value + 1) * (high - low) + low
            decoded.append(value)
        result.append(decoded)
    return result


def _check_files(root, hashes, required):
    if not required.issubset(hashes):
        raise ValueError("Required source/processor hashes were not supplied")
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha256(path) != expected:
            raise ValueError("Source/processor bytes differ: " + name)


class QwenH16Adapter:
    def __init__(self, model, processor, head, config, statistics, torch):
        self.model, self.processor, self.head = model, processor, head
        self.config, self.statistics, self.torch = config, statistics, torch
        self.gt_action_mapping_verified = False
        self.last_input_layout = None

    def to(self, device, dtype=None):
        dtype = dtype or self.torch.bfloat16
        self.model.to(device=device, dtype=dtype).eval()
        self.head.to(device=device, dtype=dtype).eval()
        return self

    def predict_normalized(self, image, instruction):
        torch = self.torch
        from PIL import Image
        if not instruction.strip():
            raise ValueError("Instruction must be nonempty")
        # Training supplied224x224 frames; preserve this explicit pre-transform.
        original_size = image.size
        image = image.convert("RGB").resize((224, 224), resample=Image.Resampling.BICUBIC)
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                     {"type": "text", "text": instruction}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        base = inputs.input_ids.to(device)
        attention = inputs.attention_mask.to(device)
        token = self.processor.tokenizer.pad_token_id
        if token is None:
            token = self.processor.tokenizer.eos_token_id
        if token is None:
            raise ValueError("The trained placeholder requires an actual pad/eos token")
        queries = torch.full((1,16), token, dtype=base.dtype, device=device)
        full_ids = torch.cat((base,queries),dim=1)
        embeddings = self.model.get_input_embeddings()(full_ids)
        embeddings[:, -16:, :] = 0
        attention = torch.cat((attention,torch.ones((1,16),dtype=attention.dtype,device=device)),dim=1)
        positions = (full_ids[0] == self.model.config.image_token_id).nonzero().flatten()
        grid = inputs.image_grid_thw.to(device)
        self.last_input_layout = {"image_grid_thw":grid.detach().cpu().tolist(),
            "visual_positions":positions.detach().cpu().tolist(), "action_query_count":16,
            "original_image_size":list(original_size),"prepared_image_size":[224,224],
            "image_transform":"full_image_bicubic_resize_224","prompt_length":base.shape[1],
            "patch_size":self.model.config.vision_config.patch_size,
            "spatial_merge_size":self.model.config.vision_config.spatial_merge_size}
        with torch.inference_mode():
            outputs = self.model.model(inputs_embeds=embeddings,
                pixel_values=inputs.pixel_values.to(device=device,dtype=torch.bfloat16),
                attention_mask=attention, image_grid_thw=grid, return_dict=True,
                action_length=16, position_input_ids=full_ids,
                visual_pos_masks=full_ids == self.model.config.image_token_id,
                qwen_position_contract="token_ids_mrope", action_task_attention_contract="action_only",
                attention_collection_mask_contract="corrected_causal",
                vla_attention_mask_contract="prompt_causal_action_bidirectional", use_cache=False)
            predictions = self.head.predict_action(outputs.last_hidden_state[:,-16:,:])
        values = predictions.float().cpu().tolist()
        if len(values) != 1:
            raise ValueError("Diagnosis requires a single sample")
        decode_h16_chunk(values[0],self.statistics)  # shape/finite check, no GT access
        return values[0]


def load_qwen_h16(*, source_root, config_dir, checkpoint_root, statistics_path,
                  source_sha256, processor_sha256):
    """Construct the real CPU model and strict-load both saved components.

    Call in an independent Transformers4.57 process. No module replacement,
    hidden GPU transfer, foundation/random head fallback or H1 conversion.
    """
    source_root, config_dir, checkpoint_root = map(Path,(source_root,config_dir,checkpoint_root))
    _check_files(source_root,source_sha256,CRITICAL_SOURCE_FILES)
    _check_files(config_dir,processor_sha256,{"config.json","tokenizer.json","tokenizer_config.json","preprocessor_config.json"})
    config = json.loads((checkpoint_root/"training_config.json").read_text(encoding="utf8"))
    validate_h16_training(config)
    statistics = read_h16_statistics(statistics_path)
    exports = checkpoint_root/"portable"
    backbone = json.loads((exports/"qwen_backbone/manifest.json").read_text(encoding="utf8"))
    head_manifest = json.loads((exports/"shared_action_head/manifest.json").read_text(encoding="utf8"))
    for manifest in (backbone,head_manifest):
        if manifest.get("action_query_contract") != "per_step_shared_v1" or manifest.get("action_query_count") != 16 or manifest.get("source_action_chunk") != 16 or manifest.get("source_action_dim") != 7:
            raise ValueError("Portable component is not the H16 per-step policy")
        if manifest["source_model_config_sha256"] != file_sha256(config_dir/"config.json"):
            raise ValueError("Model architecture config differs from the component producer")
    for folder,manifest in [("qwen_backbone",backbone),("shared_action_head",head_manifest)]:
        if file_sha256(exports/folder/manifest["weights_file"]) != manifest["weights_sha256"]:
            raise ValueError("Portable component checksum mismatch")
    if "transformers" in sys.modules:
        if not Path(sys.modules["transformers"].__file__).resolve().is_relative_to(source_root.resolve()):
            raise RuntimeError("Start a separate process; another Transformers is already imported")
    sys.path.insert(0,str(source_root.resolve()))
    import torch
    import transformers
    from transformers import AutoConfig,AutoProcessor,Qwen3VLForConditionalGeneration
    from safetensors.torch import load_file
    if transformers.__version__ != "4.57.0":
        raise RuntimeError("The saved source requires modified Transformers4.57.0")
    actual = Path(transformers.__file__).resolve()
    if not actual.is_relative_to(source_root.resolve()):
        raise RuntimeError("Wrong model implementation")
    hf_config = AutoConfig.from_pretrained(config_dir,local_files_only=True)
    model = Qwen3VLForConditionalGeneration._from_config(hf_config,torch_dtype=torch.bfloat16)
    state = load_file(str(exports/"qwen_backbone"/backbone["weights_file"]),device="cpu")
    if not all(name.startswith("model.") for name in state):
        raise ValueError("Unexpected backbone export prefixes")
    model.model.load_state_dict({name[len("model."):]:value for name,value in state.items()},strict=True,assign=True)
    model.tie_weights()
    if model.lm_head.weight is not model.get_input_embeddings().weight:
        raise ValueError("Producer requires tied input/LM weights")
    head_path = source_root/"deepvision/models/action_head.py"
    spec = importlib.util.spec_from_file_location("_diagnostic_qwen_head",head_path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    shape = head_manifest["tensor_shapes"]["model.fc1.weight"]
    head = module.PerStepSharedActionHead(input_dim=shape[1],hidden_dim=shape[0],action_dim=7,action_chunk=16)
    head.load_state_dict(load_file(str(exports/"shared_action_head"/head_manifest["weights_file"]),device="cpu"),strict=True,assign=True)
    processor = AutoProcessor.from_pretrained(config_dir,local_files_only=True)
    return QwenH16Adapter(model.eval(),processor,head.eval(),config,statistics,torch)
