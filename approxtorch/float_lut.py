"""Validation and loading helpers for FP16/BF16 mantissa LUTs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Literal

import torch


FloatKind = Literal["fp16", "bf16"]
LutSource = Literal["binary", "pytorch"]

_SPECS = {
    "fp16": (1024, "fp16_exact_mantissa_lut"),
    "bf16": (128, "bf16_exact_mantissa_lut"),
}
_LUT_CACHE: dict[tuple[str, str, str, str], torch.Tensor] = {}


def _spec(kind: str) -> tuple[int, str]:
    try:
        return _SPECS[kind]
    except KeyError as error:
        raise ValueError(f"kind must be 'fp16' or 'bf16', got {kind!r}") from error


def validate_mantissa_lut(
    lut: torch.Tensor,
    kind: FloatKind,
    *,
    require_cuda: bool = True,
) -> torch.Tensor:
    """Validate the shape, dtype, layout, and optional device of a LUT."""
    side, _ = _spec(kind)
    if not isinstance(lut, torch.Tensor):
        raise TypeError(f"lut must be a torch.Tensor, got {type(lut).__name__}")
    if lut.dtype != torch.uint32:
        raise TypeError(f"{kind} lut must have dtype torch.uint32, got {lut.dtype}")
    if tuple(lut.shape) != (side, side):
        raise ValueError(
            f"{kind} lut must have shape ({side}, {side}), got {tuple(lut.shape)}"
        )
    if not lut.is_contiguous():
        raise ValueError("lut must be contiguous in row-major order")
    if require_cuda and not lut.is_cuda:
        raise ValueError("lut must be a CUDA tensor")
    return lut


def clear_lut_cache() -> None:
    """Drop cached device LUT tensors."""
    _LUT_CACHE.clear()


def _load_manifest(directory: Path, stem: str, kind: str, side: int) -> dict:
    manifest_path = directory / f"{stem}.manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"LUT manifest not found: {manifest_path}") from error
    expected_size = side * side * 4
    if manifest.get("schema_version") != 2:
        raise ValueError("LUT manifest schema_version must be 2")
    if manifest.get("version") != "rtl-mantissa-result-v2":
        raise ValueError("LUT manifest version must be rtl-mantissa-result-v2")
    if manifest.get("kind") != kind:
        raise ValueError(f"LUT manifest kind does not match {kind!r}")
    if manifest.get("shape") != [side, side]:
        raise ValueError(f"LUT manifest shape does not match [{side}, {side}]")
    if manifest.get("logical_dtype") != "uint32":
        raise TypeError("LUT manifest dtype must be uint32")
    if manifest.get("element_size_bytes") != 4:
        raise ValueError("LUT manifest element_size_bytes must be 4")
    if manifest.get("data_size_bytes") != expected_size:
        raise ValueError(
            f"LUT manifest data_size_bytes must be {expected_size}"
        )
    if manifest.get("binary_byte_order") != "little":
        raise ValueError("LUT manifest binary_byte_order must be little")
    return manifest


def _check_file_hash(data: bytes, expected: str | None, label: str) -> None:
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"{label} failed SHA-256 validation")


def _load_binary(
    directory: Path,
    stem: str,
    side: int,
    manifest: dict,
) -> torch.Tensor:
    path = directory / f"{stem}.bin"
    data = path.read_bytes()
    expected_size = side * side * 4
    if len(data) != expected_size:
        raise ValueError(
            f"LUT binary has {len(data)} bytes, expected {expected_size}"
        )
    _check_file_hash(data, manifest.get("data_sha256"), "LUT binary")
    if sys.byteorder == "little":
        return torch.frombuffer(bytearray(data), dtype=torch.uint32).reshape(
            side, side
        )
    values = [
        int.from_bytes(data[index : index + 4], byteorder="little")
        for index in range(0, len(data), 4)
    ]
    return torch.tensor(values, dtype=torch.uint32).reshape(side, side)


def _load_pytorch(
    directory: Path,
    stem: str,
    manifest: dict,
) -> torch.Tensor:
    path = directory / f"{stem}.pt"
    data = path.read_bytes()
    file_entry = manifest.get("files", {}).get("pytorch", {})
    _check_file_hash(data, file_entry.get("sha256"), "PyTorch LUT file")
    lut = torch.load(path, map_location="cpu", weights_only=True)
    return lut


def load_exact_lut(
    kind: FloatKind,
    device: torch.device | str,
    *,
    directory: Path | str,
    source: LutSource = "binary",
) -> torch.Tensor:
    """Load an exact mantissa LUT once per source directory and CUDA device."""
    side, stem = _spec(kind)
    if source not in ("binary", "pytorch"):
        raise ValueError(
            f"source must be 'binary' or 'pytorch', got {source!r}"
        )
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError(f"device must be CUDA, got {target}")
    if target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    directory = Path(directory).expanduser().resolve()
    key = (kind, str(target), str(directory), source)
    cached = _LUT_CACHE.get(key)
    if cached is not None:
        return cached

    manifest = _load_manifest(directory, stem, kind, side)
    if source == "binary":
        lut = _load_binary(directory, stem, side, manifest)
    else:
        lut = _load_pytorch(directory, stem, manifest)
    validate_mantissa_lut(lut, kind, require_cuda=False)
    lut = lut.to(device=target, non_blocking=False)
    validate_mantissa_lut(lut, kind)
    _LUT_CACHE[key] = lut
    return lut
