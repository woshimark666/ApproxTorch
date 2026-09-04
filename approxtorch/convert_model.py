"""Model conversion utilities for integer and 16-bit floating-point layers."""

from typing import Literal

import torch
import torch.nn as nn

from .nn import (
    Conv2d_bf16,
    Conv2d_fp16,
    Conv2d_int8,
    Conv2d_uint8,
    Linear_bf16,
    Linear_fp16,
)


IntQType = Literal["int8", "uint8"]
FloatQType = Literal["fp16", "bf16"]
GradType = Literal["ste", "lre", "custom"]

__all__ = ["convert_int_model", "convert_float_model"]


def _move_qparams_to_device(module: nn.Module, device: torch.device) -> None:
    """Keep generated quantization state beside the source layer's weights.

    ``lut``, ``dx`` and ``dw`` deliberately stay on the device supplied by the
    caller. This preserves the common workflow where a CPU model is converted
    with CUDA LUTs and the whole model is moved to CUDA afterwards.
    """
    external_buffers = {"lut", "dx", "dw"}
    for name, buffer in module._buffers.items():
        if buffer is not None and name not in external_buffers:
            module._buffers[name] = buffer.to(device=device)


def _copy_conv_state(
    source: nn.Conv2d,
    target: nn.Module,
    qtype: IntQType | FloatQType,
) -> None:
    """Copy parameters and initialize integer qparams when required."""
    device = source.weight.device
    _move_qparams_to_device(target, device)
    target_dtype = target.weight.dtype

    target.weight = nn.Parameter(
        source.weight.detach().to(dtype=target_dtype).clone(
            memory_format=torch.preserve_format
        ),
        requires_grad=source.weight.requires_grad,
    )
    if source.bias is not None:
        target.bias = nn.Parameter(
            source.bias.detach().to(dtype=target_dtype).clone(),
            requires_grad=source.bias.requires_grad,
        )

    if qtype == "int8":
        target._reset_scale_w_from_weight()
    elif qtype == "uint8":
        with torch.no_grad():
            reduce_dims = tuple(range(1, target.weight.dim()))
            target.w_min.copy_(
                target.weight.detach().amin(dim=reduce_dims).to(target.w_min.dtype)
            )
            target.w_max.copy_(
                target.weight.detach().amax(dim=reduce_dims).to(target.w_max.dtype)
            )
        target._reset_qparams_from_stats()

    # Replacing a child after model.eval() would otherwise insert it in training
    # mode, which would unexpectedly resume EMA updates.
    target.train(source.training)


def _copy_linear_state(source: nn.Linear, target: nn.Module) -> None:
    """Copy a Linear layer into a fixed-dtype approximate Linear module."""
    device = source.weight.device
    _move_qparams_to_device(target, device)
    target_dtype = target.weight.dtype

    target.weight = nn.Parameter(
        source.weight.detach().to(dtype=target_dtype).clone(
            memory_format=torch.preserve_format
        ),
        requires_grad=source.weight.requires_grad,
    )
    if source.bias is not None:
        target.bias = nn.Parameter(
            source.bias.detach().to(dtype=target_dtype).clone(),
            requires_grad=source.bias.requires_grad,
        )
    target.train(source.training)


def _module_graph(
    model: nn.Module,
) -> tuple[list[nn.Module], list[tuple[nn.Module, str, nn.Module]]]:
    """Return unique modules plus every parent reference, including aliases."""
    modules = [model]
    references: list[tuple[nn.Module, str, nn.Module]] = []
    visited = {id(model)}

    def visit(parent: nn.Module) -> None:
        for name, child in parent._modules.items():
            if child is None:
                continue
            references.append((parent, name, child))
            if id(child) in visited:
                continue
            visited.add(id(child))
            modules.append(child)
            visit(child)

    visit(model)
    return modules, references


def _apply_replacements(
    model: nn.Module,
    references: list[tuple[nn.Module, str, nn.Module]],
    replacements: dict[int, nn.Module],
) -> nn.Module:
    """Replace every reference while preserving shared-module aliases."""
    for parent, name, child in references:
        if id(parent) in replacements:
            continue
        replacement = replacements.get(id(child))
        if replacement is not None:
            setattr(parent, name, replacement)
    return replacements.get(id(model), model)


def _make_int_conv(
    module: nn.Conv2d,
    lut: torch.Tensor,
    qtype: IntQType,
    grad: GradType,
    dx: torch.Tensor | None,
    dw: torch.Tensor | None,
    scale_momentum: float,
    update_scale: bool,
    weight_bits: int,
) -> nn.Module:
    if isinstance(module.padding, str):
        raise NotImplementedError(
            "convert_int_model does not support Conv2d string padding "
            f"{module.padding!r}"
        )
    if module.padding_mode != "zeros":
        raise NotImplementedError(
            "convert_int_model only supports Conv2d padding_mode='zeros', "
            f"got {module.padding_mode!r}"
        )

    geometry = dict(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        lut=lut,
        bias=module.bias is not None,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
    )

    if qtype == "int8":
        new_module = Conv2d_int8(
            **geometry,
            grad=grad,
            dx=dx,
            dw=dw,
            update_scale=update_scale,
            scale_momentum=scale_momentum,
            weight_bits=weight_bits,
        )
    elif qtype == "uint8":
        new_module = Conv2d_uint8(
            **geometry,
            grad=grad,
            dx=dx,
            dw=dw,
            update_scale=update_scale,
            scale_momentum=scale_momentum,
        )

    _copy_conv_state(module, new_module, qtype)
    return new_module


def convert_int_model(
    model: nn.Module,
    lut: torch.Tensor,
    qtype: IntQType = "int8",
    grad: GradType = "ste",
    dx: torch.Tensor | None = None,
    dw: torch.Tensor | None = None,
    ignore_first_conv: bool = True,
    scale_momentum: float = 0.05,
    update_scale: bool = True,
    weight_bits: int = 8,
) -> nn.Module:
    """Replace selected nn.Conv2d layers with integer approximate layers.

    int8 uses symmetric activations and per-channel symmetric weights.
    uint8 uses asymmetric activations and per-channel asymmetric weights.
    The model is modified in place and returned, except when model itself is
    an nn.Conv2d; in that case the replacement module is returned.

    Convert the model before constructing its optimizer because replacements
    create new Parameter objects.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not isinstance(lut, torch.Tensor):
        raise TypeError(f"lut must be a torch.Tensor, got {type(lut).__name__}")
    if qtype not in ("int8", "uint8"):
        raise ValueError(f"qtype must be 'int8' or 'uint8', got {qtype!r}")
    if lut.dtype != torch.float32:
        raise TypeError(
            f"integer lut must have dtype torch.float32, got {lut.dtype}"
        )
    if lut.numel() != 256 * 256:
        raise ValueError(f"lut must have 65536 elements, got {lut.numel()}")
    if not lut.is_contiguous():
        raise ValueError("lut must be contiguous")
    if grad not in ("ste", "lre", "custom"):
        raise ValueError(
            f"grad must be 'ste', 'lre' or 'custom', got {grad!r}"
        )
    if not isinstance(ignore_first_conv, bool):
        raise TypeError("ignore_first_conv must be a bool")
    if not isinstance(update_scale, bool):
        raise TypeError("update_scale must be a bool")
    if not 0.0 <= scale_momentum <= 1.0:
        raise ValueError(
            "scale_momentum must be between 0 and 1, "
            f"got {scale_momentum}"
        )
    if isinstance(weight_bits, bool) or not isinstance(weight_bits, int):
        raise TypeError("weight_bits must be an int")
    if qtype == "int8" and not 3 <= weight_bits <= 8:
        raise ValueError(
            f"weight_bits must be between 3 and 8, got {weight_bits}"
        )
    if qtype == "uint8" and weight_bits != 8:
        raise ValueError("uint8 quantization uses a fixed weight_bits=8")

    modules, references = _module_graph(model)
    convs = [module for module in modules if isinstance(module, nn.Conv2d)]
    if ignore_first_conv:
        convs = convs[1:]

    # Build every replacement before mutating the model. Unsupported geometry
    # therefore leaves the caller's original model intact.
    replacements = {
        id(module): _make_int_conv(
            module,
            lut,
            qtype,
            grad,
            dx,
            dw,
            scale_momentum,
            update_scale,
            weight_bits,
        )
        for module in convs
    }
    return _apply_replacements(model, references, replacements)


def _validate_float_lut(lut: torch.Tensor, qtype: FloatQType) -> None:
    if not isinstance(lut, torch.Tensor):
        raise TypeError(f"lut must be a torch.Tensor, got {type(lut).__name__}")
    if lut.dtype != torch.uint32:
        raise TypeError(
            f"{qtype} lut must have dtype torch.uint32, got {lut.dtype}"
        )
    side = 1024 if qtype == "fp16" else 128
    if tuple(lut.shape) != (side, side):
        raise ValueError(
            f"{qtype} lut must have shape ({side}, {side}), "
            f"got {tuple(lut.shape)}"
        )
    if not lut.is_contiguous():
        raise ValueError("lut must be contiguous in row-major order")


def _make_float_module(
    module: nn.Conv2d | nn.Linear,
    lut: torch.Tensor,
    qtype: FloatQType,
    optimized: bool,
) -> nn.Module:
    if isinstance(module, nn.Conv2d):
        if isinstance(module.padding, str):
            raise NotImplementedError(
                "convert_float_model does not support Conv2d string padding "
                f"{module.padding!r}"
            )
        if module.padding_mode != "zeros":
            raise NotImplementedError(
                "convert_float_model only supports Conv2d "
                "padding_mode='zeros', "
                f"got {module.padding_mode!r}"
            )
        layer_type = Conv2d_fp16 if qtype == "fp16" else Conv2d_bf16
        target = layer_type(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            lut,
            bias=module.bias is not None,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            optimized=optimized,
        )
        _copy_conv_state(module, target, qtype)
        return target

    layer_type = Linear_fp16 if qtype == "fp16" else Linear_bf16
    target = layer_type(
        module.in_features,
        module.out_features,
        lut,
        bias=module.bias is not None,
        optimized=optimized,
    )
    _copy_linear_state(module, target)
    return target


def convert_float_model(
    model: nn.Module,
    lut: torch.Tensor,
    qtype: FloatQType = "fp16",
    ignore_first_conv: bool = True,
    optimized: bool = True,
) -> nn.Module:
    """Replace nn.Conv2d and nn.Linear with FP16/BF16 LUT layers.

    Every selected layer receives parameters in the dtype named by qtype.
    ignore_first_conv affects only the first unique convolution; all Linear
    layers are converted. The function preserves shared-module aliases and
    builds every replacement before mutating model.

    The surrounding model is not cast automatically. Its inputs and remaining
    floating-point layers must use a compatible dtype at execution time.
    Convert before constructing an optimizer because replacements create new
    Parameter objects.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if qtype not in ("fp16", "bf16"):
        raise ValueError(f"qtype must be 'fp16' or 'bf16', got {qtype!r}")
    _validate_float_lut(lut, qtype)
    if not isinstance(ignore_first_conv, bool):
        raise TypeError("ignore_first_conv must be a bool")
    if not isinstance(optimized, bool):
        raise TypeError("optimized must be a bool")

    modules, references = _module_graph(model)
    first_conv = next(
        (module for module in modules if isinstance(module, nn.Conv2d)),
        None,
    )
    selected = [
        module
        for module in modules
        if isinstance(module, (nn.Conv2d, nn.Linear))
        and not (
            ignore_first_conv
            and first_conv is not None
            and module is first_conv
        )
    ]

    replacements = {
        id(module): _make_float_module(module, lut, qtype, optimized)
        for module in selected
    }
    return _apply_replacements(model, references, replacements)
