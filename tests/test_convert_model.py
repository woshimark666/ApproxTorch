import inspect
import unittest

import torch
import torch.nn as nn

import approxtorch
from approxtorch.convert_model import convert_float_model, convert_int_model
from approxtorch.nn import (
    Conv2d_bf16,
    Conv2d_fp16,
    Conv2d_int8,
    Conv2d_uint8,
    Linear_bf16,
    Linear_fp16,
    conv2d_bf16,
    conv2d_fp16,
)
from approxtorch.nn.Conv2d_uint8 import uint8_qparams


class ConvertModelTest(unittest.TestCase):
    def setUp(self):
        self.lut = torch.zeros(256, 256, dtype=torch.float32)

    def test_public_api_is_split_by_number_domain(self):
        int_parameters = inspect.signature(convert_int_model).parameters
        float_parameters = inspect.signature(convert_float_model).parameters

        self.assertIs(approxtorch.convert_int_model, convert_int_model)
        self.assertIs(approxtorch.convert_float_model, convert_float_model)
        self.assertFalse(callable(approxtorch.convert_model))
        self.assertIn("grad", int_parameters)
        self.assertIn("weight_bits", int_parameters)
        self.assertNotIn("optimized", int_parameters)
        self.assertIn("optimized", float_parameters)
        self.assertNotIn("grad", float_parameters)
        self.assertTrue(callable(conv2d_fp16))
        self.assertTrue(callable(conv2d_bf16))

    @staticmethod
    def _float_lut(kind):
        side = 1024 if kind == "fp16" else 128
        dtype = torch.uint32
        values = torch.arange(side, dtype=torch.int64)
        return ((side + values[:, None]) * (side + values[None, :])).to(dtype)

    def test_int8_conversion_preserves_state_and_skips_first_conv(self):
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.Sequential(nn.ReLU(), nn.Conv2d(4, 5, 1)),
        )
        source = model[1][1]
        source.weight.requires_grad_(False)
        expected_weight = source.weight.detach().clone()
        expected_bias = source.bias.detach().clone()
        model.eval()

        result = convert_int_model(model, self.lut, qtype="int8", weight_bits=5)

        self.assertIs(result, model)
        self.assertIsInstance(model[0], nn.Conv2d)
        self.assertIsInstance(model[1][1], Conv2d_int8)
        converted = model[1][1]
        torch.testing.assert_close(converted.weight, expected_weight)
        torch.testing.assert_close(converted.bias, expected_bias)
        self.assertFalse(converted.weight.requires_grad)
        self.assertFalse(converted.training)
        self.assertEqual(converted.weight_bits, 5)
        expected_scale = expected_weight.abs().amax(dim=(1, 2, 3)) / 15
        torch.testing.assert_close(converted.scale_w, expected_scale)

    def test_uint8_conversion_initializes_weight_qparams(self):
        model = nn.Sequential(nn.Conv2d(2, 3, 3, bias=False))
        source_weight = model[0].weight.detach().clone()

        convert_int_model(
            model,
            self.lut,
            qtype="uint8",
            ignore_first_conv=False,
            update_scale=False,
        )

        self.assertIsInstance(model[0], Conv2d_uint8)
        converted = model[0]
        torch.testing.assert_close(converted.weight, source_weight)
        self.assertIsNone(converted.bias)
        self.assertFalse(converted.update_scale)

        expected_min = source_weight.amin(dim=(1, 2, 3))
        expected_max = source_weight.amax(dim=(1, 2, 3))
        expected_scale, expected_zero = uint8_qparams(expected_min, expected_max)
        torch.testing.assert_close(converted.w_min, expected_min)
        torch.testing.assert_close(converted.w_max, expected_max)
        torch.testing.assert_close(converted.scale_w, expected_scale)
        torch.testing.assert_close(converted.zero_w, expected_zero)

    def test_root_conv_is_returned_as_replacement(self):
        source = nn.Conv2d(3, 4, 1)
        expected_weight = source.weight.detach().clone()

        converted = convert_int_model(
            source,
            self.lut,
            qtype="int8",
            ignore_first_conv=False,
        )

        self.assertIsInstance(converted, Conv2d_int8)
        torch.testing.assert_close(converted.weight, expected_weight)

    def test_float_conversion_casts_state_and_preserves_geometry(self):
        for kind, cls, dtype in (
            ("fp16", Conv2d_fp16, torch.float16),
            ("bf16", Conv2d_bf16, torch.bfloat16),
        ):
            with self.subTest(kind=kind):
                model = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(
                        4,
                        6,
                        (3, 2),
                        stride=(2, 1),
                        padding=(2, 1),
                        dilation=(2, 1),
                    ),
                )
                source = model[1]
                source.weight.requires_grad_(False)
                expected_weight = source.weight.detach().to(dtype)
                expected_bias = source.bias.detach().to(dtype)
                model.eval()

                result = convert_float_model(
                    model,
                    self._float_lut(kind),
                    qtype=kind,
                    ignore_first_conv=False,
                    optimized=False,
                )

                self.assertIs(result, model)
                converted = model[1]
                self.assertIsInstance(converted, cls)
                self.assertEqual(converted.weight.dtype, dtype)
                self.assertEqual(converted.bias.dtype, dtype)
                self.assertEqual(converted.lut.dtype, torch.uint32)
                self.assertIn(f"dtype={dtype}", repr(converted))
                torch.testing.assert_close(converted.weight, expected_weight)
                torch.testing.assert_close(converted.bias, expected_bias)
                self.assertFalse(converted.weight.requires_grad)
                self.assertFalse(converted.training)
                self.assertFalse(converted.optimized)
                self.assertEqual(converted.stride, (2, 1))
                self.assertEqual(converted.padding, (2, 1))
                self.assertEqual(converted.dilation, (2, 1))
                self.assertEqual(converted.groups, 1)

    def test_root_bf16_conv_is_returned_and_float_contract_is_checked(self):
        source = nn.Conv2d(2, 4, 1, bias=False)
        expected = source.weight.detach().to(torch.bfloat16)
        lut = self._float_lut("bf16")

        converted = convert_float_model(
            source,
            lut,
            qtype="bf16",
            ignore_first_conv=False,
        )

        self.assertIsInstance(converted, Conv2d_bf16)
        self.assertIsNone(converted.bias)
        torch.testing.assert_close(converted.weight, expected)
        self.assertIs(
            convert_float_model(converted, lut, qtype="bf16"), converted
        )

        with self.assertRaisesRegex(TypeError, "torch.uint32"):
            convert_float_model(source, lut.to(torch.int32), qtype="bf16")

    def test_float_conversion_also_converts_linear(self):
        for kind, layer_type, dtype in (
            ("fp16", Linear_fp16, torch.float16),
            ("bf16", Linear_bf16, torch.bfloat16),
        ):
            with self.subTest(kind=kind):
                model = nn.Sequential(nn.Conv2d(2, 2, 1), nn.Linear(4, 3))
                source = model[1]
                source.bias.requires_grad_(False)
                expected_weight = source.weight.detach().to(dtype)
                expected_bias = source.bias.detach().to(dtype)
                model.eval()

                convert_float_model(model, self._float_lut(kind), qtype=kind)

                self.assertIs(type(model[0]), nn.Conv2d)
                converted = model[1]
                self.assertIsInstance(converted, layer_type)
                self.assertEqual(converted.weight.dtype, dtype)
                self.assertEqual(converted.bias.dtype, dtype)
                self.assertEqual(converted.lut.dtype, torch.uint32)
                self.assertIn(f"dtype={dtype}", repr(converted))
                torch.testing.assert_close(converted.weight, expected_weight)
                torch.testing.assert_close(converted.bias, expected_bias)
                self.assertFalse(converted.bias.requires_grad)
                self.assertFalse(converted.training)

    def test_root_float_linear_is_returned(self):
        source = nn.Linear(4, 3, bias=False)
        expected = source.weight.detach().to(torch.float16)

        converted = convert_float_model(
            source, self._float_lut("fp16"), qtype="fp16"
        )

        self.assertIsInstance(converted, Linear_fp16)
        self.assertIsNone(converted.bias)
        torch.testing.assert_close(converted.weight, expected)

    def test_shared_module_aliases_remain_shared(self):
        int_model = nn.Module()
        shared_conv = nn.Conv2d(2, 3, 1)
        int_model.first = shared_conv
        int_model.second = shared_conv
        convert_int_model(
            int_model,
            self.lut,
            qtype="int8",
            ignore_first_conv=False,
        )
        self.assertIsInstance(int_model.first, Conv2d_int8)
        self.assertIs(int_model.first, int_model.second)

        float_model = nn.Module()
        shared_linear = nn.Linear(4, 3)
        float_model.first = shared_linear
        float_model.second = shared_linear
        convert_float_model(
            float_model, self._float_lut("bf16"), qtype="bf16"
        )
        self.assertIsInstance(float_model.first, Linear_bf16)
        self.assertIs(float_model.first, float_model.second)

    def test_float_grouped_conversion_is_rejected_without_partial_mutation(self):
        model = nn.Sequential(
            nn.Conv2d(4, 4, 1),
            nn.Conv2d(4, 4, 3, padding=1, groups=2),
        )

        with self.assertRaisesRegex(NotImplementedError, "only groups=1"):
            convert_float_model(
                model,
                self._float_lut("bf16"),
                qtype="bf16",
                ignore_first_conv=False,
            )

        self.assertIsInstance(model[0], nn.Conv2d)
        self.assertIsInstance(model[1], nn.Conv2d)

    def test_unsupported_uint8_group_does_not_partially_convert(self):
        model = nn.Sequential(
            nn.Conv2d(4, 4, 1),
            nn.Conv2d(4, 4, 3, padding=1, groups=2),
        )

        with self.assertRaises(NotImplementedError):
            convert_int_model(
                model,
                self.lut,
                qtype="uint8",
                ignore_first_conv=False,
            )

        self.assertIsInstance(model[0], nn.Conv2d)
        self.assertIsInstance(model[1], nn.Conv2d)

    def test_invalid_qtype_and_lut_fail_early(self):
        model = nn.Sequential(nn.Conv2d(1, 1, 1))

        with self.assertRaisesRegex(ValueError, "qtype"):
            convert_int_model(model, self.lut, qtype="int4")
        with self.assertRaisesRegex(ValueError, "65536"):
            convert_int_model(model, torch.zeros(10))
        with self.assertRaisesRegex(TypeError, "torch.float32"):
            convert_int_model(
                model, torch.zeros(256, 256, dtype=torch.int32)
            )
        with self.assertRaisesRegex(ValueError, "qtype"):
            convert_float_model(model, self._float_lut("fp16"), qtype="int8")
        with self.assertRaisesRegex(TypeError, "torch.uint32"):
            convert_float_model(
                model,
                torch.zeros(128, 128, dtype=torch.uint16),
                qtype="bf16",
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            convert_float_model(
                model,
                torch.zeros(10, 10, dtype=torch.uint32),
                qtype="fp16",
            )


if __name__ == "__main__":
    unittest.main()
