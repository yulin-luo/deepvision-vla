"""Separate shape-only meta inspection from actual saved-weight finite proof."""
def state_materialization(state,torch):
    tensors=[v for v in state.values() if isinstance(v,torch.Tensor)]
    if not tensors:raise ValueError('checkpoint contains no tensors')
    meta=sum(v.is_meta for v in tensors)
    if meta==len(tensors):return 'metadata_only'
    if meta:raise ValueError('mixed materialized and meta checkpoint tensors')
    return 'materialized'
def assert_finite(state,component,torch):
    if state_materialization(state,torch)!='materialized':raise ValueError(f'{component} has not been materialized; no finite proof')
    bad=[k for k,v in state.items() if isinstance(v,torch.Tensor) and (v.is_floating_point() or v.is_complex()) and not bool(torch.isfinite(v).all())]
    if bad:raise ValueError(f'{component} has nonfinite saved tensors: {bad[:8]}')
    return {'component':component,'state_keys':len(state),'materialized':True}
