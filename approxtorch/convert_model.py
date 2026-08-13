"""Utilities for converting ``torch.nn.Conv2d`` layers to ApproxTorch."""

from typing import Literal

import torch
import torch.nn as nn

from .nn import Conv2d_int8, Conv2d_uint8


QType = Literal["int8", "uint8"]
GradType = Literal["ste", "lre", "custom"]


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


def _copy_conv_state(source: nn.Conv2d, target: nn.Module, qtype: QType) -> None:
    """Copy parameters and initialize qparams from the copied weights."""
    device = source.weight.device
    _move_qparams_to_device(target, device)

    target.weight = nn.Parameter(
        source.weight.detach().clone(memory_format=torch.preserve_format),
        requires_grad=source.weight.requires_grad,
    )
    if source.bias is not None:
        target.bias = nn.Parameter(
            source.bias.detach().clone(),
            requires_grad=source.bias.requires_grad,
        )

    if qtype == "int8":
        target._reset_scale_w_from_weight()
    else:
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


def _make_approx_conv(
    module: nn.Conv2d,
    lut: torch.Tensor,
    qtype: QType,
    grad: GradType,
    dx: torch.Tensor | None,
    dw: torch.Tensor | None,
    scale_momentum: float,
    update_scale: bool,
    weight_bits: int,
) -> nn.Module:
    if isinstance(module.padding, str):
        raise NotImplementedError(
            "convert_model does not support Conv2d string padding "
            f"{module.padding!r}"
        )
    if module.padding_mode != "zeros":
        raise NotImplementedError(
            "convert_model only supports Conv2d padding_mode='zeros', "
            f"got {module.padding_mode!r}"
        )

    common_args = dict(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        lut=lut,
        grad=grad,
        dx=dx,
        dw=dw,
        bias=module.bias is not None,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        update_scale=update_scale,
        scale_momentum=scale_momentum,
    )

    if qtype == "int8":
        new_module = Conv2d_int8(**common_args, weight_bits=weight_bits)
    else:
        new_module = Conv2d_uint8(**common_args)

    _copy_conv_state(module, new_module, qtype)
    return new_module


def convert_model(
    model: nn.Module,
    lut: torch.Tensor,
    qtype: QType = "int8",
    grad: GradType = "ste",
    dx: torch.Tensor | None = None,
    dw: torch.Tensor | None = None,
    ignore_first_conv: bool = True,
    scale_momentum: float = 0.05,
    update_scale: bool = True,
    weight_bits: int = 8,
) -> nn.Module:
    """Replace selected ``nn.Conv2d`` layers with approximate convolutions.

    Quantization is selected by ``qtype`` and follows the fixed strategies of
    the maintained layers:

    - ``int8``: per-tensor symmetric activations and per-channel symmetric
      weights;
    - ``uint8``: per-tensor asymmetric activations and per-channel asymmetric
      weights.

    Consequently, quantizer configuration is intentionally not part of this
    API. ``weight_bits`` applies only to ``int8``; unsigned quantization is
    fixed at 8 bits. The model is modified in place and returned, except when
    ``model`` itself is an ``nn.Conv2d``, in which case the replacement module
    is returned. Convert a model before constructing its optimizer because
    replacement creates new ``Parameter`` objects.

    Args:
        model: Model whose exact convolution layers should be replaced.
        lut: Flattened or 2-D 256 x 256 approximate-multiplier LUT.
        qtype: Signed ``"int8"`` or unsigned ``"uint8"`` convolution.
        grad: Backward estimator: ``"ste"``, ``"lre"`` or ``"custom"``.
        dx: Input-operand gradient LUT required by ``lre`` and ``custom``.
        dw: Weight-operand gradient LUT required by ``lre`` and ``custom``.
        ignore_first_conv: Keep the first ``nn.Conv2d`` exact when true.
        scale_momentum: EMA momentum for activation and weight statistics.
        update_scale: Update quantization statistics during training.
        weight_bits: Signed weight precision from 3 to 8 bits (int8 only).

    Returns:
        The converted model.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not isinstance(lut, torch.Tensor):
        raise TypeError(f"lut must be a torch.Tensor, got {type(lut).__name__}")
    if lut.numel() != 256 * 256:
        raise ValueError(f"lut must have 65536 elements, got {lut.numel()}")
    if qtype not in ("int8", "uint8"):
        raise ValueError(f"qtype must be 'int8' or 'uint8', got {qtype!r}")
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

    named_modules = dict(model.named_modules())
    convs = [
        (name, module)
        for name, module in named_modules.items()
        if isinstance(module, nn.Conv2d)
    ]
    if ignore_first_conv:
        convs = convs[1:]

    # Build every replacement before mutating the model. If an unsupported
    # convolution is encountered, the caller keeps the original model intact.
    replacements = [
        (
            name,
            _make_approx_conv(
                module,
                lut,
                qtype,
                grad,
                dx,
                dw,
                scale_momentum,
                update_scale,
                weight_bits,
            ),
        )
        for name, module in convs
    ]

    converted_model = model
    for name, new_module in replacements:
        if not name:
            converted_model = new_module
            continue
        parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent_module = named_modules[parent_name] if parent_name else model
        setattr(parent_module, attr_name, new_module)

    return converted_model
