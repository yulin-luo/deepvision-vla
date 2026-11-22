"""Genuine post-decoder OUTPUT attribution, separate from INPUT probes.

The output hook observes the actual Tensor (or tuple[0]) and returns no
replacement. A preparatory input prehook enables a real input leaf's gradient
when the parameters are frozen. Consequently native caller-side operations,
including Qwen DeepStack inplace additions, retain their real autograd path.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class ResidualOutputCapture:
    target_name: str
    hook_location: str
    expected_shape: tuple[int,int,int]
    input_calls: int = 0
    forward_calls: int = 0
    backward_calls: int = 0
    activation: Any = None
    activation_values: Any = None
    gradient: Any = None


def _grad_context(torch):
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise RuntimeError("Decoder OUTPUT capture requires gradients outside inference_mode")


@contextmanager
def capture_residual_output(module, *, expected_shape, target_name,
                            hook_location="decoder_residual_output",
                            hidden_keyword="hidden_states",expected_prefix_calls=1):
    """Capture one actual output and scalar backward, without replacing it.

    The input hook only enables an eligible real leaf to build the native
    decoder graph; attribution activation/gradient are taken at OUTPUT. It
    never changes input values or returns a clone to a residual branch.
    Original input/output .grad, input requires_grad, and hooks are restored.
    """
    import torch
    _grad_context(torch)
    shape = tuple(expected_shape)
    if len(shape) != 3 or shape[0] != 1 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("Declare exact positive [1,prefix_tokens,width] dimensions")
    if not isinstance(target_name,str) or not target_name.strip():
        raise ValueError("An explicit scalar gradient target is required")
    if hook_location != "decoder_residual_output":
        raise ValueError("This hook captures genuine decoder OUTPUT, not INPUT")
    if type(expected_prefix_calls) is not int or expected_prefix_calls != 1:
        raise ValueError("Declare exactly one prefix call")
    if not isinstance(hidden_keyword,str) or not hidden_keyword:
        raise ValueError("Declare the actual hidden-state keyword")
    capture = ResidualOutputCapture(target_name,hook_location,shape)
    module_handles,tensor_handles,remembered = [],[],{}

    def remember(tensor):
        if id(tensor) not in remembered:
            restore_grad = tensor.is_leaf or tensor.retains_grad
            old_grad = tensor.grad if restore_grad else None
            remembered[id(tensor)] = (tensor,tensor.requires_grad,restore_grad,old_grad,
                                     old_grad.detach().clone() if old_grad is not None else None)

    def input_hook(_module,args,kwargs):
        _grad_context(torch);capture.input_calls += 1
        if capture.input_calls != 1:raise RuntimeError("Decoder received multiple prefix calls")
        if args and hidden_keyword in kwargs:raise ValueError("Ambiguous hidden input")
        hidden = args[0] if args else kwargs.get(hidden_keyword)
        if (not isinstance(hidden,torch.Tensor) or tuple(hidden.shape) != shape
                or not hidden.is_floating_point() or hidden.is_inference()):
            raise ValueError("Actual decoder input differs from the declared floating prefix")
        remember(hidden)
        if not hidden.requires_grad:
            if not hidden.is_leaf:raise RuntimeError("Cannot enable gradient on a non-leaf input")
            hidden.requires_grad_(True)
        return None

    def output_hook(_module,_args,_kwargs,output):
        _grad_context(torch);capture.forward_calls += 1
        if capture.forward_calls != 1:raise RuntimeError("Decoder produced multiple prefix outputs")
        hidden = output[0] if isinstance(output,tuple) and output else output
        if (not isinstance(hidden,torch.Tensor) or tuple(hidden.shape) != shape
                or not hidden.is_floating_point() or hidden.is_inference()):
            raise ValueError("Decoder output must be the declared Tensor or tuple[0]")
        if not hidden.requires_grad:
            raise RuntimeError("The real decoder detached its output; do not invent an output gradient leaf")
        remember(hidden)
        capture.activation = hidden
        capture.activation_values = hidden.detach().clone()

        def gradient_hook(gradient):
            capture.backward_calls += 1
            if capture.backward_calls != 1:raise RuntimeError("Use one scalar backward per output capture")
            if tuple(gradient.shape) != shape:raise RuntimeError("Output gradient shape changed")
            capture.gradient = gradient.detach().clone()
            return None

        tensor_handles.append(hidden.register_hook(gradient_hook))
        return None  # Preserve the actual output Tensor/container and alias.

    normal_exit = False
    try:
        module_handles.append(module.register_forward_pre_hook(input_hook,with_kwargs=True))
        module_handles.append(module.register_forward_hook(output_hook,with_kwargs=True))
        yield capture
        normal_exit = True
    finally:
        for handle in module_handles+tensor_handles:handle.remove()
        for tensor,enabled,restore_grad,old_grad,old_values in remembered.values():
            if restore_grad:
                if old_grad is not None:
                    with torch.no_grad():old_grad.copy_(old_values)
                tensor.grad = old_grad
            if not enabled:tensor.requires_grad_(False)
        if normal_exit and (capture.input_calls != 1 or capture.forward_calls != 1 or capture.backward_calls != 1):
            raise RuntimeError("Expected one real decoder input/output and scalar backward")


def token_gradcam(capture,visual_positions,*,target_name):
    """Raw OUTPUT Grad-CAM; no input-map relabeling or display normalization."""
    import torch
    if target_name != capture.target_name:raise ValueError("Scalar target differs from the output capture")
    if capture.input_calls != 1 or capture.forward_calls != 1 or capture.backward_calls != 1 or capture.gradient is None:
        raise RuntimeError("A genuine one-prefix decoder-output gradient is required")
    positions = tuple(visual_positions)
    if (not positions or len(set(positions)) != len(positions)
            or any(type(x) is not int or x < 0 or x >= capture.expected_shape[1] for x in positions)):
        raise ValueError("Declare actual unique visual sequence positions")
    acts = capture.activation_values[0,list(positions),:]
    grads = capture.gradient[0,list(positions),:]
    if not bool(torch.isfinite(acts).all()) or not bool(torch.isfinite(grads).all()):
        raise ValueError("Non-finite output activations/gradients cannot produce a map")
    dtype = torch.float64 if acts.dtype == torch.float64 or grads.dtype == torch.float64 else torch.float32
    acts,grads = acts.to(dtype),grads.to(dtype)
    alpha = grads.mean(dim=0)
    signed = (acts*alpha.unsqueeze(0)).sum(dim=-1)
    relu = signed.clamp_min(0)
    if not bool(torch.isfinite(signed).all()):raise ValueError("Output contribution arithmetic overflowed")
    return {"target_name":target_name,"hook_location":"decoder_residual_output",
            "activation_semantics":"residual_stream_output_not_input",
            "activation_snapshot":"decoder_output_before_caller_postprocessing",
            "visual_positions":list(positions),"raw_signed":signed.detach().cpu(),
            "raw_relu":relu.detach().cpu(),"channel_weights":alpha.detach().cpu(),
            "constant_signed":bool((signed == signed[0]).all()),"constant_relu":bool((relu == relu[0]).all()),
            "zero_gradient":bool((grads == 0).all()),"display_normalization_applied":False}
