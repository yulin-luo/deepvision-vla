"""Reusable diagnostics extracted from validated native implementations."""
from deepvision_vla.diagnostics.model_geometry import actual_visual_layout
import math
import struct

class ModelOnce:
    """Thin caller-owned wrapper around the three immutable real factories."""

    def __init__(self, model, factory):
        import torch
        torch.set_num_threads(8)
        if torch.cuda.is_initialized():
            raise RuntimeError('Model-once process must begin without a CUDA context')
        self.kind = model
        self.counters = {}
        self.handles = []
        self.original_methods = []
        if model == 'openvla':
            from deepvision_vla.diagnostics.adapters.openvla_runtime import load_openvla_runtime
            self.adapter = load_openvla_runtime(**factory)
            self.model = self.adapter.model
            self.layers = self.model.language_model.model.layers
        elif model == 'pi0':
            from deepvision_vla.diagnostics.adapters.openpi_bridge import load_bridge_pi0
            self.adapter = load_bridge_pi0(**factory)
            self.model = self.adapter.policy
            self.layers = self.model.model.paligemma_with_expert.paligemma.language_model.model.layers
        elif model == 'qwen_h16':
            from deepvision_vla.diagnostics.adapters.qwen_h16 import load_qwen_h16
            self.adapter = load_qwen_h16(**factory)
            self.model = self.adapter.model
            self.layers = self.model.model.language_model.layers
        else:
            raise ValueError('Unknown model')
        if len(self.layers) != DEPTH[model] or any((p.device.type != 'cpu' for p in self.model.parameters())):
            raise RuntimeError('Factory must produce the expected genuine CPU decoder before any lease')
        self.torch = torch
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU factory initialized CUDA before the GPU lease')
        self.noise = torch.randn((1, 4, 32), generator=torch.Generator().manual_seed(42)) if model == 'pi0' else None

    def layout(self, case, data):
        if self.kind != 'qwen_h16':
            return self.adapter.visual_layout(case['image_sha256'], data['image'].size)
        from PIL import Image
        image = data['image'].resize((224, 224), resample=Image.Resampling.BICUBIC)
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': case['instruction']}]}]
        inputs = self.adapter.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
        actual = {'image_grid_thw': inputs.image_grid_thw.tolist(), 'visual_positions': (inputs.input_ids[0] == self.model.config.image_token_id).nonzero().flatten().tolist(), 'action_query_count': 16, 'original_image_size': case['image_size'], 'prepared_image_size': [224, 224], 'image_transform': 'full_image_bicubic_resize_224', 'prompt_length': inputs.input_ids.shape[1], 'patch_size': self.model.config.vision_config.patch_size, 'spatial_merge_size': self.model.config.vision_config.spatial_merge_size}
        return actual_visual_layout(case, actual)

    def to_gpu(self):
        if self.kind == 'pi0':
            self.adapter.policy.to('cuda:0').eval()
        else:
            self.adapter.to('cuda:0')

        def count(name):
            self.counters[name] = self.counters.get(name, 0) + 1
        if self.kind == 'qwen_h16':
            self.handles += [self.model.model.register_forward_pre_hook(lambda *_: count('model_forward_calls')), self.adapter.head.model.register_forward_pre_hook(lambda *_: count('head_forward_calls'))]
        elif self.kind == 'pi0':
            native = self.model.model
            for name in ('sample_actions', 'embed_prefix', 'denoise_step'):
                original = getattr(native, name)
                self.original_methods.append((native, name, original))

                def wrapped(*a, _name=name, _original=original, **kw):
                    count(_name)
                    return _original(*a, **kw)
                setattr(native, name, wrapped)

    def predict(self, case, data, row, layout):
        torch = self.torch
        torch.manual_seed(42)
        layer, positions = (row['layer'], tuple(row['positions']))
        if self.kind == 'openvla':
            result = self.adapter.predict(data['image'], case['instruction'], image_sha256=case['image_sha256'], layer=layer, dropped_positions=positions, seed=42)
            self.counters = dict(self.adapter.counters)
            if result['native_generation_reached_seven_steps'] is not True:
                raise RuntimeError('Native action generation ended before seven actual decoding steps')
            return {'native': [result['actions']], 'hook': result['hook']}
        if self.kind == 'pi0':
            result = self.adapter.predict(data['image'], data['states'][case['anchor']].tolist(), case['instruction'], fixed_noise=self.noise, layer=layer, dropped_positions=positions)
            return {'native': result['actions'].tolist(), 'hook': result['hook']}
        from deepvision_vla.diagnostics.interventions import zero_visual_tokens
        from deepvision_vla.diagnostics.adapters.qwen_h16 import decode_h16_chunk
        target = self.layers[0 if layer is None else layer]
        with zero_visual_tokens(target, positions) as hook:
            normalized = self.adapter.predict_normalized(data['image'], case['instruction'])
        if hook['prefill_calls'] != 1 or hook['cached_decode_calls'] != 0:
            raise RuntimeError('Qwen must produce one actual parallel-H16 forward')
        if actual_visual_layout(case, self.adapter.last_input_layout) != layout:
            raise RuntimeError('Actual Qwen forward layout differs from the CPU processor derivation')
        return {'normalized': normalized, 'native': decode_h16_chunk(normalized, self.adapter.statistics), 'hook': dict(hook)}

    def counters_snapshot(self):
        return dict(self.adapter.counters) if self.kind == 'openvla' else dict(self.counters)

    def native_entries(self):
        key = 'sample_actions' if self.kind == 'pi0' else 'model_forward_calls'
        return self.counters_snapshot().get(key, 0)

    def close(self):
        for handle in self.handles:
            handle.remove()
        for target, name, original in self.original_methods:
            setattr(target, name, original)
        self.handles.clear()
        self.original_methods.clear()

DEPTH = {'openvla': 32, 'pi0': 18, 'qwen_h16': 36}

HORIZON = {'openvla': 1, 'pi0': 4, 'qwen_h16': 16}

LAYERS = {'openvla': (0, 10, 20, 30), 'pi0': (0, 6, 12, 16), 'qwen_h16': (0, 12, 24, 34)}

REVISIONS = {'openvla': '47a0ec7fc4ec123775a391911046cf33cf9ed83f', 'pi0': '93c2840a2c68c337da0d472a2f52d1fd8d62132a', 'qwen_h16': 'b131353d58b77daf2b6308e5ea4ea0aa4a417a79'}

def finite_chunk(values, horizon):
    if len(values) != horizon or any((len(row) != 7 for row in values)):
        raise ValueError('Full native Hx7 chunk is required; no padding/truncation')
    if any((type(x) not in (int, float) or not math.isfinite(x) for row in values for x in row)):
        raise ValueError('Nonfinite native chunk')
    return values

def chunk_drift(actual, baseline):
    finite_chunk(actual, len(baseline))
    finite_chunk(baseline, len(baseline))
    errors = [(a - b) ** 2 for row, ref in zip(actual, baseline) for a, b in zip(row, ref)]
    return {'full_chunk_mse': sum(errors) / len(errors), 'first_step_mse': sum(errors[:7]) / 7}

def chunk_bits(values):
    return b''.join((struct.pack('!d', float(x)) for row in values for x in row))

