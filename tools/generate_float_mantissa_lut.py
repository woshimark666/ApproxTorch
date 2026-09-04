#!/usr/bin/env python3
"""Generate deterministic RTL-compatible mantissa-result LUTs.

Every LUT entry is produced with Python integer arithmetic. CUDA is never
initialized, and PyTorch is imported only when a .pt output is requested.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable


@dataclass(frozen=True)
class LutSpec:
    kind: str
    side: int
    offset: int
    typecode: str
    logical_dtype: str
    q_format: str

    @property
    def element_size_bytes(self) -> int:
        return 4 if self.typecode == "I" else 2

    @property
    def element_count(self) -> int:
        return self.side * self.side

    @property
    def data_size_bytes(self) -> int:
        return self.element_count * self.element_size_bytes

    @property
    def stem(self) -> str:
        return f"{self.kind}_exact_mantissa_lut"


SPECS = {
    "fp16": LutSpec(
        "fp16", 1024, 1024, "I", "uint32",
        "bit10 normalization + bits9:0 fraction",
    ),
    "bf16": LutSpec(
        "bf16", 128, 128, "I", "uint32",
        "bit7 normalization + bits6:0 fraction",
    ),
}


def generate_values(spec: LutSpec) -> array.array:
    """Return native-endian storage in row-major LUT[A][B] order."""
    values = array.array(spec.typecode)
    if values.itemsize != spec.element_size_bytes:
        raise RuntimeError(
            f"array typecode {spec.typecode!r} has unexpected item size "
            f"{values.itemsize}"
        )
    fraction_bits = spec.side.bit_length() - 1
    fraction_mask = spec.side - 1
    normalization_product_bit = 2 * fraction_bits + 1
    for row in range(spec.side):
        lhs = spec.offset + row
        for col in range(spec.side):
            product = lhs * (spec.offset + col)
            if product & (1 << normalization_product_bit):
                entry = spec.side | (
                    (product >> (fraction_bits + 1)) & fraction_mask
                )
            else:
                entry = (product >> fraction_bits) & fraction_mask
            values.append(entry)
    if len(values) != spec.element_count:
        raise AssertionError("internal LUT element-count error")
    return values


def little_endian_bytes(values: array.array) -> bytes:
    encoded = array.array(values.typecode, values)
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pytorch_bytes(values: array.array, spec: LutSpec) -> bytes:
    import torch

    dtype = torch.uint32
    tensor = torch.frombuffer(values, dtype=dtype).clone().reshape(
        spec.side, spec.side
    )
    buffer = io.BytesIO()
    # BytesIO gives the zip archive a stable root name, making the bytes
    # independent of the destination directory and filename.
    torch.save(tensor, buffer)
    return buffer.getvalue()


def write_lut(spec: LutSpec, output_dir: Path, formats: Iterable[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = frozenset(formats)
    unknown = selected.difference(("binary", "pytorch"))
    if not selected or unknown:
        raise ValueError(f"invalid output formats: {sorted(selected)!r}")

    values = generate_values(spec)
    raw = little_endian_bytes(values)
    if len(raw) != spec.data_size_bytes:
        raise AssertionError("internal LUT byte-size error")

    files: dict[str, dict[str, object]] = {}
    if "binary" in selected:
        binary_name = f"{spec.stem}.bin"
        (output_dir / binary_name).write_bytes(raw)
        files["binary"] = {
            "name": binary_name,
            "file_size_bytes": len(raw),
            "sha256": sha256_bytes(raw),
        }

    if "pytorch" in selected:
        pytorch_name = f"{spec.stem}.pt"
        serialized = pytorch_bytes(values, spec)
        (output_dir / pytorch_name).write_bytes(serialized)
        files["pytorch"] = {
            "name": pytorch_name,
            "file_size_bytes": len(serialized),
            "sha256": sha256_bytes(serialized),
        }

    manifest = {
        "schema_version": 2,
        "version": "rtl-mantissa-result-v2",
        "kind": spec.kind,
        "shape": [spec.side, spec.side],
        "logical_dtype": spec.logical_dtype,
        "q_format": spec.q_format,
        "layout": "row-major",
        "lut_order": "LUT[A][B]",
        "formula": "entry(i,j)=truncate_normalize((offset+i)*(offset+j))",
        "element_count": spec.element_count,
        "element_size_bytes": spec.element_size_bytes,
        "data_size_bytes": spec.data_size_bytes,
        "binary_byte_order": "little",
        "data_sha256": sha256_bytes(raw),
        "files": files,
    }
    manifest_path = output_dir / f"{spec.stem}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=("fp16", "bf16", "all"), default="all"
    )
    parser.add_argument(
        "--format",
        choices=("binary", "pytorch", "all"),
        default="all",
        dest="output_format",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kinds = SPECS if args.kind == "all" else (args.kind,)
    formats = (
        ("binary", "pytorch")
        if args.output_format == "all"
        else (args.output_format,)
    )
    for kind in kinds:
        print(write_lut(SPECS[kind], args.output_dir, formats))


if __name__ == "__main__":
    main()
