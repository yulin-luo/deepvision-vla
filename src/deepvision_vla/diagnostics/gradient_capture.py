"""Gradient-safe attribution at the residual-stream INPUT, not layer output.

Use a decoder prehook for OpenVLA/Qwen, or the selected Pali input-layernorm
prehook for Pi's manual layer loop. No activation replacement or masking is
performed. A Tensor hook observes the gradient of the original shared input,
including both its normalization and residual branches.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class ResidualInputCapture:
    target_name: str
    hook_location: str
    expected_shape: tuple[int, int, int]
    expected_prefix_calls: int = 1
    forward_calls: int = 0
    backward_calls: int = 0
    activation: Any = None
    activation_values: Any = None
    gradient: Any = None


def _valid_grad_context(torch):
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise RuntimeError("Residual-input capture requires enabled gradients outside inference_mode")


@contextmanager
def capture_residual_input(module, *, expected_shape, target_name,
                           hook_location, hidden_keyword="hidden_states",
                           expected_prefix_calls=1):
    """Capture one real prefix and its backward pass without changing values.

    Backpropagate the declared scalar inside this context. A missing/duplicate
    forward or backward is an error. Only a floating non-inference leaf may
    have requires_grad enabled here; original flags and input .grad are restored
    on exit. Parameter flags, model mode and weights are never modified.
    """
    import torch
    _valid_grad_context(torch)
    shape = tuple(expected_shape)
    if len(shape) != 3 or shape[0] != 1 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("Declare an exact positive [1,prefix_tokens,width] input shape")
    if not isinstance(target_name, str) or not target_name.strip():
        raise ValueError("An explicit scalar gradient target name is required")
    if hook_location not in {"decoder_residual_input", "pi_prefix_layernorm_residual_input"}:
        raise ValueError("Declare the actual residual INPUT hook location")
    if expected_prefix_calls != 1 or type(expected_prefix_calls) is not int:
        raise ValueError("This capture supports exactly one declared prefix call")
    if not isinstance(hidden_keyword, str) or not hidden_keyword:
        raise ValueError("Declare the hidden-state keyword")
    capture = ResidualInputCapture(target_name, hook_location, shape)
    tensor_handles = []
    original_requires_grad = None
    original_grad = None
    original_grad_values = None
    restore_input_grad = False

    def input_hook(_module, args, kwargs):
        nonlocal original_requires_grad, original_grad, original_grad_values, restore_input_grad
        _valid_grad_context(torch)
        capture.forward_calls += 1
        if capture.forward_calls != 1:
            raise RuntimeError("Selected module was called more than the declared single prefix")
        if args and hidden_keyword in kwargs:
            raise ValueError("Ambiguous positional and keyword hidden input")
        hidden = args[0] if args else kwargs.get(hidden_keyword)
        if (not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != shape
                or not hidden.is_floating_point() or hidden.is_inference()):
            raise ValueError("Actual floating residual input differs from the declared prefix")
        capture.activation = hidden  # Actual alias; never return a clone to the module.
        capture.activation_values = hidden.detach().clone()
        original_requires_grad = hidden.requires_grad
        restore_input_grad = hidden.is_leaf or hidden.retains_grad
        if restore_input_grad:
            original_grad = hidden.grad
            if original_grad is not None:
                original_grad_values = original_grad.detach().clone()
        if not hidden.requires_grad:
            if not hidden.is_leaf:
                raise RuntimeError("Cannot enable gradients on a non-leaf residual input")
            hidden.requires_grad_(True)

        def gradient_hook(gradient):
            capture.backward_calls += 1
            if capture.backward_calls != 1:
                raise RuntimeError("Use one declared scalar backward per prefix capture")
            if tuple(gradient.shape) != shape:
                raise RuntimeError("Residual gradient shape changed")
            capture.gradient = gradient.detach().clone()
            return None  # Preserve the original aggregated gradient.

        tensor_handles.append(hidden.register_hook(gradient_hook))
        return None  # Preserve the original arguments, Tensor storage and values.

    module_handle = module.register_forward_pre_hook(input_hook, with_kwargs=True)
    normal_exit = False
    try:
        yield capture
        normal_exit = True
    finally:
        module_handle.remove()
        for handle in tensor_handles:
            handle.remove()
        hidden = capture.activation
        if hidden is not None and original_requires_grad is not None:
            if restore_input_grad:
                if original_grad is not None:
                    with torch.no_grad():
                        original_grad.copy_(original_grad_values)
                hidden.grad = original_grad
            if not original_requires_grad:
                hidden.requires_grad_(False)
        if normal_exit and (capture.forward_calls != 1 or capture.backward_calls != 1):
            raise RuntimeError("Expected one real prefix and one scalar backward inside capture context")


def token_gradcam(capture, visual_positions, *, target_name):
    """Return unnormalized signed and ReLU token Grad-CAM, with explicit target.

    Channel weights are the mean gradient over the declared visual positions.
    Constant/zero maps remain unchanged and are flagged; no hot-map fallback or
    per-layer display normalization is applied. Inputs/gradients must be finite.
    """
    import torch
    if target_name != capture.target_name:
        raise ValueError("Gradient target name differs from the capture")
    if capture.forward_calls != 1 or capture.backward_calls != 1 or capture.gradient is None:
        raise RuntimeError("An actual single-prefix scalar gradient is required")
    positions = tuple(visual_positions)
    if (not positions or len(set(positions)) != len(positions)
            or any(type(x) is not int or x < 0 or x >= capture.expected_shape[1] for x in positions)):
        raise ValueError("Declare distinct actual visual sequence positions")
    acts = capture.activation_values[0, list(positions), :]
    grads = capture.gradient[0, list(positions), :]
    if not bool(torch.isfinite(acts).all()) or not bool(torch.isfinite(grads).all()):
        raise ValueError("Non-finite activations or gradients cannot produce a contribution map")
    dtype = torch.float64 if acts.dtype == torch.float64 or grads.dtype == torch.float64 else torch.float32
    acts, grads = acts.to(dtype), grads.to(dtype)
    alpha = grads.mean(dim=0)
    signed = (acts * alpha.unsqueeze(0)).sum(dim=-1)
    relu = signed.clamp_min(0)
    if not bool(torch.isfinite(signed).all()):
        raise ValueError("Contribution arithmetic overflowed")
    return {"target_name": target_name, "hook_location": capture.hook_location,
            "activation_semantics": "residual_stream_input_not_legacy_layer_output",
            "visual_positions": list(positions), "raw_signed": signed.detach().cpu(),
            "raw_relu": relu.detach().cpu(), "channel_weights": alpha.detach().cpu(),
            "constant_signed": bool((signed == signed[0]).all()),
            "constant_relu": bool((relu == relu[0]).all()),
            "zero_gradient": bool((grads == 0).all()), "display_normalization_applied": False}
