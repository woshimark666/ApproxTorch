from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from approxtorch.float_lut import clear_lut_cache, load_exact_lut, validate_mantissa_lut
from tools import generate_float_mantissa_lut as generator


@pytest.mark.parametrize("kind", ("fp16", "bf16"))
def test_exact_lut_contents_manifest_and_determinism(tmp_path: Path, kind: str) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    spec = generator.SPECS[kind]
    generator.write_lut(spec, first, ("binary", "pytorch"))
    generator.write_lut(spec, second, ("binary", "pytorch"))

    names = (
        f"{spec.stem}.bin",
        f"{spec.stem}.pt",
        f"{spec.stem}.manifest.json",
    )
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes()

    manifest = json.loads((first / names[2]).read_text(encoding="utf-8"))
    raw = (first / names[0]).read_bytes()
    assert len(raw) == spec.data_size_bytes
    assert manifest["shape"] == [spec.side, spec.side]
    assert manifest["logical_dtype"] == spec.logical_dtype
    assert manifest["q_format"] == spec.q_format
    assert manifest["lut_order"] == "LUT[A][B]"
    assert manifest["element_count"] == spec.element_count
    assert manifest["data_size_bytes"] == spec.data_size_bytes
    assert manifest["data_sha256"] == hashlib.sha256(raw).hexdigest()

    tensor = torch.load(first / names[1], map_location="cpu", weights_only=True)
    expected_dtype = torch.uint32
    assert tensor.dtype == expected_dtype
    assert tensor.shape == (spec.side, spec.side)
    rows = torch.arange(spec.side, dtype=torch.int64)[:, None] + spec.offset
    cols = torch.arange(spec.side, dtype=torch.int64)[None, :] + spec.offset
    product = rows * cols
    fraction_bits = spec.side.bit_length() - 1
    expected = torch.where(
        (product & (1 << (2 * fraction_bits + 1))) != 0,
        spec.side | ((product >> (fraction_bits + 1)) & (spec.side - 1)),
        (product >> fraction_bits) & (spec.side - 1),
    )
    # Exhaustive contents plus row-major LUT[A][B] flattening.
    assert torch.equal(tensor.to(torch.int64), expected)
    flat = tensor.reshape(-1).to(torch.int64)
    for row, col in (
        (0, 0),
        (0, spec.side - 1),
        (1, 0),
        (spec.side - 1, spec.side - 1),
    ):
        assert flat[row * spec.side + col].item() == expected[row, col].item()


def test_validate_approximate_lut_contract() -> None:
    fp16 = torch.zeros((1024, 1024), dtype=torch.uint32)
    assert validate_mantissa_lut(fp16, "fp16", require_cuda=False) is fp16
    with pytest.raises(TypeError, match="dtype"):
        validate_mantissa_lut(fp16.to(torch.int64), "fp16", require_cuda=False)
    with pytest.raises(ValueError, match="shape"):
        validate_mantissa_lut(fp16.reshape(-1), "fp16", require_cuda=False)
    with pytest.raises(ValueError, match="contiguous"):
        validate_mantissa_lut(fp16.t(), "fp16", require_cuda=False)


def test_loader_rejects_corrupted_binary(tmp_path: Path) -> None:
    spec = generator.SPECS["bf16"]
    generator.write_lut(spec, tmp_path, ("binary",))
    path = tmp_path / f"{spec.stem}.bin"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="SHA-256"):
        load_exact_lut("bf16", "cuda:0", directory=tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kind", ("fp16", "bf16"))
def test_loader_gpu_cache_and_source_equivalence(tmp_path: Path, kind: str) -> None:
    spec = generator.SPECS[kind]
    generator.write_lut(spec, tmp_path, ("binary", "pytorch"))
    clear_lut_cache()
    binary = load_exact_lut(kind, "cuda:0", directory=tmp_path, source="binary")
    again = load_exact_lut(kind, "cuda:0", directory=tmp_path, source="binary")
    pytorch = load_exact_lut(kind, "cuda:0", directory=tmp_path, source="pytorch")
    expected_dtype = torch.uint32
    assert binary.data_ptr() == again.data_ptr()
    assert binary.is_cuda and binary.is_contiguous()
    assert binary.dtype == expected_dtype
    assert torch.equal(binary.cpu(), pytorch.cpu())

    if torch.cuda.device_count() > 1:
        other = load_exact_lut(kind, "cuda:1", directory=tmp_path, source="binary")
        assert other.device.index == 1
        assert other.data_ptr() != binary.data_ptr()
