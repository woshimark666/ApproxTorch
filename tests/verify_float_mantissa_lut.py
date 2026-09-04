#!/usr/bin/env python3
"""Standalone exhaustive and determinism checks for float mantissa LUTs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch

from approxtorch.float_lut import clear_lut_cache, load_exact_lut


ROOT = Path(__file__).parents[1]
GENERATOR = ROOT / "tools" / "generate_float_mantissa_lut.py"


def generate(output_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--kind", "all",
            "--format", "all",
            "--output-dir", str(output_dir),
        ],
        check=True,
    )


def verify_kind(first: Path, second: Path, kind: str) -> None:
    if kind == "fp16":
        side, offset, dtype, expected_size = 1024, 1024, torch.uint32, 4_194_304
    else:
        side, offset, dtype, expected_size = 128, 128, torch.uint32, 65_536
    stem = f"{kind}_exact_mantissa_lut"
    names = (f"{stem}.bin", f"{stem}.pt", f"{stem}.manifest.json")
    for name in names:
        if (first / name).read_bytes() != (second / name).read_bytes():
            raise AssertionError(f"non-deterministic output: {name}")

    raw = (first / names[0]).read_bytes()
    manifest = json.loads((first / names[2]).read_text(encoding="utf-8"))
    if len(raw) != expected_size:
        raise AssertionError(f"{kind} binary size mismatch")
    if hashlib.sha256(raw).hexdigest() != manifest["data_sha256"]:
        raise AssertionError(f"{kind} data hash mismatch")
    expected_fields = {
        "schema_version": 2,
        "version": "rtl-mantissa-result-v2",
        "shape": [side, side],
        "logical_dtype": str(dtype).removeprefix("torch."),
        "q_format": (
            "bit10 normalization + bits9:0 fraction"
            if kind == "fp16"
            else "bit7 normalization + bits6:0 fraction"
        ),
        "layout": "row-major",
        "lut_order": "LUT[A][B]",
        "element_count": side * side,
        "data_size_bytes": expected_size,
        "binary_byte_order": "little",
    }
    for key, value in expected_fields.items():
        if manifest.get(key) != value:
            raise AssertionError(f"{kind} manifest field mismatch: {key}")

    saved = torch.load(first / names[1], map_location="cpu", weights_only=True)
    rows = torch.arange(side, dtype=torch.int64)[:, None] + offset
    cols = torch.arange(side, dtype=torch.int64)[None, :] + offset
    if saved.dtype != dtype or saved.shape != (side, side):
        raise AssertionError(f"{kind} PyTorch dtype/shape mismatch")
    product = rows * cols
    fraction_bits = side.bit_length() - 1
    expected = torch.where(
        (product & (1 << (2 * fraction_bits + 1))) != 0,
        side | ((product >> (fraction_bits + 1)) & (side - 1)),
        (product >> fraction_bits) & (side - 1),
    )
    if not torch.equal(saved.to(torch.int64), expected):
        raise AssertionError(f"{kind} exhaustive entry mismatch")
    flat = saved.reshape(-1).to(torch.int64)
    for row, col in ((0, 0), (0, side - 1), (1, 0), (side - 1, side - 1)):
        if flat[row * side + col].item() != expected[row, col].item():
            raise AssertionError(f"{kind} flatten index mismatch")

    if torch.cuda.is_available():
        clear_lut_cache()
        binary = load_exact_lut(kind, "cuda:0", directory=first, source="binary")
        again = load_exact_lut(kind, "cuda:0", directory=first, source="binary")
        pytorch = load_exact_lut(kind, "cuda:0", directory=first, source="pytorch")
        if binary.data_ptr() != again.data_ptr():
            raise AssertionError(f"{kind} same-device cache miss")
        if not torch.equal(binary.cpu(), pytorch.cpu()):
            raise AssertionError(f"{kind} binary/.pt logical mismatch")
        if torch.cuda.device_count() > 1:
            other = load_exact_lut(kind, "cuda:1", directory=first, source="binary")
            if other.device.index != 1 or other.data_ptr() == binary.data_ptr():
                raise AssertionError(f"{kind} multi-GPU cache pointer reuse")
    print(
        f"PASS {kind}: {side * side} entries, {expected_size} bytes, "
        f"sha256={manifest['data_sha256']}"
    )


def verify_corruption_rejected(source: Path, work: Path) -> None:
    shutil.copytree(source, work)
    binary = work / "bf16_exact_mantissa_lut.bin"
    data = bytearray(binary.read_bytes())
    data[-1] ^= 1
    binary.write_bytes(data)
    clear_lut_cache()
    try:
        load_exact_lut("bf16", "cuda:0", directory=work, source="binary")
    except ValueError as error:
        if "SHA-256" not in str(error):
            raise
    else:
        raise AssertionError("corrupted binary LUT was accepted")
    print("PASS corrupted binary SHA-256 rejection")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="approxtorch-lut-test-") as temporary:
        root = Path(temporary)
        first, second = root / "first", root / "second"
        generate(first)
        generate(second)
        for kind in ("fp16", "bf16"):
            verify_kind(first, second, kind)
        if torch.cuda.is_available():
            verify_corruption_rejected(first, root / "corrupt")
    print("All float mantissa LUT checks passed")


if __name__ == "__main__":
    main()
