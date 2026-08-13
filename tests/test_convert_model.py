import inspect
import unittest

import torch
import torch.nn as nn

import approxtorch
from approxtorch.convert_model import convert_model
from approxtorch.nn import Conv2d_int8, Conv2d_uint8
from approxtorch.nn.Conv2d_uint8 import uint8_qparams


class ConvertModelTest(unittest.TestCase):
    def setUp(self):
        self.lut = torch.zeros(256, 256, dtype=torch.float32)

    def test_public_api_has_only_unified_entry(self):
        parameters = inspect.signature(convert_model).parameters

        self.assertNotIn("x_quantizer", parameters)
        self.assertNotIn("w_quantizer", parameters)
        self.assertFalse(hasattr(approxtorch, "to_qat_int8"))
        self.assertNotIn("x_quantizer", inspect.signature(Conv2d_int8).parameters)
        self.assertNotIn("w_quantizer", inspect.signature(Conv2d_int8).parameters)

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

        result = convert_model(model, self.lut, qtype="int8", weight_bits=5)

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

        convert_model(
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

        converted = convert_model(
            source,
            self.lut,
            qtype="int8",
            ignore_first_conv=False,
        )

        self.assertIsInstance(converted, Conv2d_int8)
        torch.testing.assert_close(converted.weight, expected_weight)

    def test_unsupported_uint8_group_does_not_partially_convert(self):
        model = nn.Sequential(
            nn.Conv2d(4, 4, 1),
            nn.Conv2d(4, 4, 3, padding=1, groups=2),
        )

        with self.assertRaises(NotImplementedError):
            convert_model(
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
            convert_model(model, self.lut, qtype="int4")
        with self.assertRaisesRegex(ValueError, "65536"):
            convert_model(model, torch.zeros(10))


if __name__ == "__main__":
    unittest.main()
