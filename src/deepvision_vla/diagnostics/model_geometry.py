"""Reusable diagnostics extracted from validated native implementations."""
from deepvision_vla.diagnostics.geometry import VisualLayout

def actual_visual_layout(sample, observed):
    grids = observed['image_grid_thw']
    if len(grids) != 1 or len(grids[0]) != 3 or grids[0][0] != 1:
        raise ValueError('The canary requires one still-image temporal grid')
    _, height, width = grids[0]
    merge, patch = (observed['spatial_merge_size'], observed['patch_size'])
    if type(merge) is not int or type(patch) is not int or merge <= 0 or (patch <= 0) or height % merge or width % merge:
        raise ValueError('Image grid cannot be mapped to actual merged tokens')
    positions = tuple(observed['visual_positions'])
    if len(positions) != height // merge * (width // merge) or not positions or positions != tuple(range(positions[0], positions[0] + len(positions))):
        raise ValueError('Visual sequence is not the actual contiguous single-image grid')
    if observed['action_query_count'] != 16 or observed['original_image_size'] != sample['image_size'] or observed['prepared_image_size'] != [224, 224] or (observed['image_transform'] != 'full_image_bicubic_resize_224'):
        raise ValueError('H16 query or image preprocessing contract changed')
    source_width, source_height = sample['image_size']
    return VisualLayout(sample['image_sha256'], tuple(sample['image_size']), (0, 0, source_width, source_height), (width * patch, height * patch), (0, 0, width * patch, height * patch), (height // merge, width // merge), positions)

