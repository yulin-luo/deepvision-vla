"""Reusable diagnostics extracted from validated native implementations."""
from deepvision_vla.diagnostics.roi_filter import filter_candidate_annotation
from deepvision_vla.diagnostics.adapters.openpi_bridge import bridge_eef_state8

def prepare_image(sample, path):
    """Validate the actual image and instruction before resource/model work."""
    from PIL import Image
    instruction = sample.get('instruction')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('A nonempty task instruction is required')
    size = sample.get('image_size')
    if not isinstance(size, list) or len(size) != 2 or any((type(x) is not int or x <= 0 for x in size)):
        raise ValueError('Original image dimensions must be explicit positive integers')
    with Image.open(path) as original:
        original.load()
        if original.size != tuple(size):
            raise ValueError('Source image size changed')
        return original.convert('RGB')

def screened_roi(sample):
    return filter_candidate_annotation({'sample_id': sample['sample_id'], 'image_sha256': sample['image_sha256'], 'image_size': sample['image_size'], 'coordinate_space': 'source_image_pixels', 'candidate_boxes': [{'xyxy_px': box} for box in sample['roi_boxes']], 'human_verified': False})

def prepare_sample_inputs(sample, image_path, arrays_path):
    """Validate cheap native inputs on CPU before any GPU lease/model load."""
    import numpy as np
    from PIL import Image
    instruction = sample['instruction']
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('A nonempty task instruction is required')
    anchor = sample['anchor']
    if type(anchor) is not int or anchor < 0:
        raise ValueError('The sample anchor must be a nonnegative integer')
    with Image.open(image_path) as original:
        if original.size != tuple(sample['image_size']):
            raise ValueError('Original image dimensions changed')
        image = original.convert('RGB')
    with np.load(arrays_path, allow_pickle=False) as arrays:
        states = arrays['states']
        if states.ndim != 2 or states.shape[1] != 7 or anchor >= len(states):
            raise ValueError('Expected measured Bridge state7 at the declared anchor')
        state7 = states[anchor].tolist()
    state8 = bridge_eef_state8(state7)
    return (image, state7, {'state_frame': 'unverified_dataset_frame', 'physical_frame_origin_verified': False, 'extra_frame_transform_applied': False, 'anchor': anchor, 'native_state8': state8, 'gripper_input': 'raw_measured_passthrough', 'gripper_clipped_or_binarized': False})

