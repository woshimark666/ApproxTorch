from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .float_lut import FloatKind, validate_mantissa_lut


IntQType = Literal["int8", "int4", "uint8"]
FloatQType = Literal["fp16", "float16", "bf16", "bfloat16"]
QType = IntQType | FloatQType

__all__ = [
    "load_lut",
    "load_float_lut",
    "load_lre_grad_lut",
    "load_half_custom_grad_lut",
    "load_custom_grad_lut",
]

_FLOAT_KINDS: dict[str, FloatKind] = {
    "fp16": "fp16",
    "float16": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
}
_FLOAT_SIDES: dict[FloatKind, int] = {"fp16": 1024, "bf16": 128}
_INTEGER_SIDES: dict[str, int] = {"int8": 256, "uint8": 256, "int4": 16}


def _check_element_count(count: int, side: int, qtype: str) -> None:
    expected = side * side
    if count != expected:
        raise ValueError(
            f"{qtype} LUT must contain {expected} elements, got {count}"
        )


def _load_float_binary(path: Path, side: int, qtype: str) -> torch.Tensor:
    data = path.read_bytes()
    expected_bytes = side * side * 4
    if len(data) != expected_bytes:
        raise ValueError(
            f"{qtype} binary LUT must contain {expected_bytes} bytes, "
            f"got {len(data)}"
        )
    # The raw float LUT ABI is always little-endian uint32.
    values = np.frombuffer(data, dtype="<u4").astype(np.uint32, copy=True)
    return torch.from_numpy(values).reshape(side, side)


def _load_float_pytorch(path: Path) -> torch.Tensor:
    lut = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(lut, torch.Tensor):
        raise TypeError(
            f"PyTorch LUT file must contain a tensor, got {type(lut).__name__}"
        )
    return lut


def _load_float_text(path: Path, side: int, qtype: str) -> torch.Tensor:
    values = np.loadtxt(path, dtype=np.int64)
    _check_element_count(int(values.size), side, qtype)
    uint32_max = np.iinfo(np.uint32).max
    if np.any(values < 0) or np.any(values > uint32_max):
        raise ValueError(
            f"{qtype} text LUT entries must fit in unsigned 32-bit storage"
        )
    values = values.astype(np.uint32, copy=False).reshape(side, side)
    return torch.from_numpy(values).contiguous()


def load_float_lut(
    file_path: str | Path,
    qtype: FloatQType = "fp16",
) -> torch.Tensor:
    """Load an FP16/BF16 mantissa LUT as a CPU ``torch.uint32`` tensor.

    ``.bin`` files are interpreted as raw little-endian uint32 entries;
    ``.pt`` and ``.pth`` files must contain a uint32 tensor; every other
    suffix is parsed as a whitespace-delimited integer text file. FP16 uses
    shape ``[1024, 1024]`` and BF16 uses ``[128, 128]``.

    For generated exact LUTs that include a manifest and SHA-256 metadata,
    prefer :func:`approxtorch.float_lut.load_exact_lut`.
    """
    try:
        kind = _FLOAT_KINDS[qtype]
    except KeyError as error:
        raise ValueError(
            "qtype must be 'fp16', 'float16', 'bf16', or 'bfloat16', "
            f"got {qtype!r}"
        ) from error

    path = Path(file_path).expanduser()
    side = _FLOAT_SIDES[kind]
    suffix = path.suffix.lower()
    if suffix == ".bin":
        lut = _load_float_binary(path, side, qtype)
    elif suffix in (".pt", ".pth"):
        lut = _load_float_pytorch(path)
    else:
        lut = _load_float_text(path, side, qtype)

    return validate_mantissa_lut(lut, kind, require_cuda=False)


def load_lut(
    file_path: str | Path,
    qtype: QType = "int8",
) -> torch.Tensor:
    """Load an integer or FP16/BF16 approximate-multiplier LUT.

    Integer LUTs preserve the historical flattened ``torch.int32`` return
    format. Floating-point LUTs use the two-dimensional ``torch.uint32``
    contract required by the FP16/BF16 CUDA kernels.
    """
    if qtype in _FLOAT_KINDS:
        return load_float_lut(file_path, qtype)
    try:
        side = _INTEGER_SIDES[qtype]
    except KeyError as error:
        raise ValueError(
            "qtype must be 'int8', 'int4', 'uint8', 'fp16', 'float16', "
            f"'bf16', or 'bfloat16', got {qtype!r}"
        ) from error

    lut_array = np.loadtxt(file_path, dtype=np.int32)
    _check_element_count(int(lut_array.size), side, qtype)
    return torch.from_numpy(lut_array.reshape(-1)).to(dtype=torch.int32)


def load_lre_grad_lut(
    da_file_path: str | Path,
    db_file_path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    da_lut = torch.from_numpy(
        np.loadtxt(da_file_path, dtype=np.float32)
    ).reshape(-1)
    db_lut = torch.from_numpy(
        np.loadtxt(db_file_path, dtype=np.float32)
    ).reshape(-1)
    return da_lut, db_lut


def load_half_custom_grad_lut(file_path: str | Path) -> torch.Tensor:
    """Load the weight-gradient LUT used with an STE input gradient."""
    return torch.from_numpy(
        np.loadtxt(file_path, dtype=np.float32)
    ).reshape(-1)


def load_custom_grad_lut(
    dx_file_path: str | Path,
    dw_file_path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load pair-wise custom gradients indexed as ``[x + 128, w + 128]``."""
    dx_lut = torch.from_numpy(
        np.loadtxt(dx_file_path, dtype=np.float32)
    ).reshape(-1)
    dw_lut = torch.from_numpy(
        np.loadtxt(dw_file_path, dtype=np.float32)
    ).reshape(-1)
    return dx_lut, dw_lut
