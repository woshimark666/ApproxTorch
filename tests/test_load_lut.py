from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from approxtorch.load_lut import load_float_lut, load_lut


class LoadLutTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_bfloat16_text_and_aliases(self) -> None:
        values = (
            np.arange(128 * 128, dtype=np.uint32).reshape(128, 128) & 0xFF
        )
        path = self.directory / "bf16.txt"
        np.savetxt(path, values, fmt="%u")

        long_name = load_lut(path, qtype="bfloat16")
        short_name = load_float_lut(path, qtype="bf16")

        self.assertEqual(long_name.dtype, torch.uint32)
        self.assertEqual(tuple(long_name.shape), (128, 128))
        self.assertTrue(long_name.is_contiguous())
        self.assertFalse(long_name.is_cuda)
        torch.testing.assert_close(long_name, short_name)
        torch.testing.assert_close(
            long_name.to(torch.int64), torch.from_numpy(values).to(torch.int64)
        )

    def test_fp16_raw_little_endian_binary(self) -> None:
        values = (
            np.arange(1024 * 1024, dtype=np.uint32) & 0x7FF
        ).astype("<u4")
        path = self.directory / "fp16.bin"
        path.write_bytes(values.tobytes())

        lut = load_lut(path, qtype="fp16")

        self.assertEqual(lut.dtype, torch.uint32)
        self.assertEqual(tuple(lut.shape), (1024, 1024))
        self.assertTrue(lut.is_contiguous())
        np.testing.assert_array_equal(lut.numpy().reshape(-1), values)

    def test_float16_pytorch_tensor(self) -> None:
        expected = torch.zeros(1024, 1024, dtype=torch.uint32)
        expected[3, 7] = 1234
        path = self.directory / "fp16.pt"
        torch.save(expected, path)

        actual = load_lut(path, qtype="float16")

        self.assertEqual(actual.dtype, torch.uint32)
        torch.testing.assert_close(actual, expected)

    def test_invalid_float_artifacts_are_rejected(self) -> None:
        wrong_dtype = self.directory / "wrong.pt"
        torch.save(torch.zeros(128, 128, dtype=torch.uint16), wrong_dtype)
        with self.assertRaisesRegex(TypeError, "torch.uint32"):
            load_float_lut(wrong_dtype, qtype="bf16")

        wrong_size = self.directory / "wrong.bin"
        wrong_size.write_bytes(bytes(16))
        with self.assertRaisesRegex(ValueError, "65536 bytes"):
            load_float_lut(wrong_size, qtype="bfloat16")

        negative = self.directory / "negative.txt"
        values = np.zeros((128, 128), dtype=np.int64)
        values[0, 0] = -1
        np.savetxt(negative, values, fmt="%d")
        with self.assertRaisesRegex(ValueError, "unsigned 32-bit"):
            load_float_lut(negative, qtype="bf16")

        with self.assertRaisesRegex(ValueError, "qtype"):
            load_float_lut(negative, qtype="float32")

    def test_integer_loader_contract_is_preserved(self) -> None:
        values = np.arange(16 * 16, dtype=np.int32).reshape(16, 16)
        path = self.directory / "int4.txt"
        np.savetxt(path, values, fmt="%d")

        lut = load_lut(path, qtype="int4")

        self.assertEqual(lut.dtype, torch.int32)
        self.assertEqual(tuple(lut.shape), (16 * 16,))
        torch.testing.assert_close(lut, torch.from_numpy(values).reshape(-1))


if __name__ == "__main__":
    unittest.main()
