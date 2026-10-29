"""True STE contracts, with a CPU LUT oracle and optional real CUDA execution.

Run from the repository root with:
    python -m unittest discover -s approxtorch/backend/csrc/claude -p test_conv_true_ste.py -v
CPU tests replace only the three CUDA forward ops; quantization, the module,
autograd boundary and convolution backward are the real implementation.
"""

import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch
import torch.nn.functional as F

import approxtorch as at
from approxtorch.nn import Conv2d_int8, quantization


def _cpu_im2col(x, kernel_size, stride, padding, dilation):
    return (F.unfold(x.float(), kernel_size, dilation=dilation,
                     padding=padding, stride=stride) + 128).to(torch.uint8)


def _cpu_bgemm_save(x, w, lut):
    xi = x.long() if x.dtype == torch.uint8 else x.round().clamp(-128, 127).long() + 128
    wi = w.round().clamp(-128, 127).long() + 128
    products = lut.reshape(256, 256)[xi[:, None, :, :], wi[None, :, :, None]]
    y = products.long().sum(dim=2).to(torch.int32).float()
    return y, xi.to(torch.uint8), wi.to(torch.uint8)


def _cpu_depthwise(x, w, lut, kernel_size, stride, padding, dilation):
    batch, channels, height, width = x.shape
    kh, kw = kernel_size
    xi = _cpu_im2col(x, kernel_size, stride, padding, dilation)
    xi = xi.long().reshape(batch, channels, kh * kw, -1)
    wi = w.round().clamp(-128, 127).long() + 128
    products = lut.reshape(256, 256)[xi, wi[None, :, :, None]]
    oh = (height + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    ow = (width + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    y = products.long().sum(dim=2).to(torch.int32).float()
    return y.reshape(batch, channels, oh, ow), wi.to(torch.uint8)


class _TrueSTEContracts:
    device_type = "cpu"

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        # The reference Conv2d uses the same FP32 precision as the new backward.
        tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cudnn, "allow_tf32", tf32)
        torch.manual_seed(17)
        self.device = torch.device(self.device_type)
        if self.device_type == "cpu":
            for name, function in (
                ("im2col_u8", _cpu_im2col),
                ("bgemm_fake_int8_claude_save", _cpu_bgemm_save),
                ("dwconv_fake_int8_claude", _cpu_depthwise),
            ):
                stack.enter_context(patch.object(at.backend.ops, name, function))

        signed = torch.arange(-128, 128, dtype=torch.float32, device=self.device)
        a, b = signed[:, None], signed[None, :]
        # Asymmetric, with nonzero products at x=0, to exercise actual LUT and
        # padding semantics rather than just an exact-multiply forward.
        self.lut = (a * b + a - 2 * b + 7).contiguous()

    def make_module(self, **kwargs):
        groups = kwargs.get("groups", 1)
        module = Conv2d_int8(
            2, 2 if groups == 2 else 3, kwargs.pop("kernel_size", 3),
            self.lut, grad="true-ste", bias=kwargs.pop("bias", True),
            **kwargs,
        ).to(self.device).eval()
        with torch.no_grad():
            module.weight.copy_(torch.randn_like(module.weight) * 0.4 + 0.031)
            module.scale_x.fill_(0.025)
            module.scale_w.copy_(torch.linspace(
                0.015, 0.035, module.out_channels, device=self.device))
        return module

    def reference(self, module, x):
        xr = x.detach().clone().requires_grad_(x.requires_grad)
        wr = module.weight.detach().clone().requires_grad_(module.weight.requires_grad)
        br = (module.bias.detach().clone().requires_grad_(module.bias.requires_grad)
              if module.bias is not None else None)
        y = F.conv2d(xr, wr, br, module.stride, module.padding,
                     module.dilation, module.groups)
        return y, (xr, wr, br)

    def assert_gradients(self, module, x, reference):
        for actual, expected in zip((x, module.weight, module.bias), reference):
            if actual is None:
                self.assertIsNone(expected)
            elif not actual.requires_grad:
                self.assertIsNone(actual.grad)
            else:
                torch.testing.assert_close(actual.grad, expected.grad,
                                           rtol=2e-5, atol=3e-5)

    def test_forward_parity_and_float_backward(self):
        for geometry in (
            dict(kernel_size=1),
            dict(kernel_size=3, padding=1, weight_bits=3),
            dict(kernel_size=3, padding=2, stride=2, dilation=2, bias=False),
            dict(kernel_size=(2, 3), padding=(1, 2), stride=(2, 1), dilation=(1, 2)),
            dict(kernel_size=3, groups=2, padding=1, stride=2),
            dict(kernel_size=3, groups=2, padding=2, dilation=2, weight_bits=4, bias=False),
        ):
            with self.subTest(**geometry):
                module = self.make_module(**geometry)
                old_ste = copy.deepcopy(module)
                old_ste.grad = "ste"
                # Both input and upstream gradient deliberately have strides.
                x = torch.randn(2, 2, 9, 7, device=self.device).transpose(2, 3)
                x.requires_grad_()
                actual = module(x)
                with torch.no_grad():
                    torch.testing.assert_close(actual, old_ste(x), rtol=0, atol=0)
                expected, operands = self.reference(module, x)
                self.assertFalse(torch.equal(actual, expected))
                go = torch.randn_like(actual.transpose(2, 3)).transpose(2, 3)
                actual.backward(go)
                expected.backward(go)
                self.assert_gradients(module, x, operands)

    def test_saturated_values_and_scales_do_not_change_backward(self):
        module = Conv2d_int8(1, 1, 1, self.lut, grad="true-ste", bias=True)
        module = module.to(self.device).eval()
        with torch.no_grad():
            module.weight.fill_(2.25)
        outputs = []
        go = torch.tensor([[[[0.5, -1.5]]]], device=self.device)
        for sx, sw in ((0.001, 0.001), (0.2, 0.03)):
            module.scale_x.fill_(sx)
            module.scale_w.fill_(sw)
            module.zero_grad(set_to_none=True)
            x = torch.tensor([[[[1.25, -2.5]]]], device=self.device, requires_grad=True)
            y = module(x)
            expected, operands = self.reference(module, x)
            y.backward(go)
            expected.backward(go)
            self.assert_gradients(module, x, operands)
            self.assertTrue(torch.all(x.grad != 0))
            self.assertTrue(torch.all(module.weight.grad != 0))
            outputs.append(y.detach())
        self.assertFalse(torch.equal(*outputs))

    def test_frozen_parameters_and_bias_only_backward(self):
        for need_x, need_w, need_bias in ((True, False, False), (False, True, False),
                                          (False, False, True)):
            with self.subTest(x=need_x, weight=need_w, bias=need_bias):
                module = self.make_module(padding=1)
                module.weight.requires_grad_(need_w)
                module.bias.requires_grad_(need_bias)
                x = torch.randn(2, 2, 5, 7, device=self.device, requires_grad=need_x)
                y = module(x)
                expected, operands = self.reference(module, x)
                go = torch.randn_like(y)
                y.backward(go)
                expected.backward(go)
                self.assert_gradients(module, x, operands)

    def test_existing_ste_keeps_quantized_masked_gradients(self):
        module = self.make_module(padding=1, weight_bits=3)
        module.grad = "ste"
        module.scale_x.fill_(0.005)
        x = torch.randn(2, 2, 5, 7, device=self.device, requires_grad=True)
        y = module(x)
        go = torch.randn_like(y)

        sx = module.scale_x.detach()
        sw = module.scale_w.detach().view(-1, 1, 1, 1)
        xv, wv = x.detach() / sx, module.weight.detach() / sw
        xmask = (xv >= -128) & (xv <= 127)
        wmask = (wv >= -4) & (wv <= 3)
        self.assertTrue(torch.any(~xmask) and torch.any(~wmask))
        xq = xv.round().clamp(-128, 127).requires_grad_()
        wq = wv.round().clamp(-4, 3).requires_grad_()
        raw = F.conv2d(xq, wq, padding=1)
        reference = raw * (sx * module.scale_w.detach()).view(1, -1, 1, 1)
        reference.backward(go)
        y.backward(go)
        torch.testing.assert_close(x.grad, xq.grad * xmask / sx,
                                   rtol=2e-5, atol=3e-5)
        torch.testing.assert_close(module.weight.grad, wq.grad * wmask / sw,
                                   rtol=2e-5, atol=3e-5)
        torch.testing.assert_close(module.bias.grad, go.sum(dim=(0, 2, 3)),
                                   rtol=2e-5, atol=3e-5)

    def test_signed_quantization_boundaries(self):
        values = torch.tensor(
            [-129, -128, -127.75, -127.5, -0.5, 0.5, 126.5, 127, 127.25, 128],
            device=self.device)
        expected = values.new_tensor([-128, -128, -128, -128, 0, 0, 126, 127, 127, 127])
        mask = values.new_tensor([0, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        scale = values.new_tensor(0.5)
        x = (values * scale).requires_grad_()
        q = quantization.static_quantize_int8(x, scale)
        torch.testing.assert_close(q, expected, rtol=0, atol=0)
        q.sum().backward()
        torch.testing.assert_close(x.grad, mask / scale, rtol=0, atol=0)

        scales = values.new_tensor([0.25, 0.5])
        w = (values[None, :] * scales[:, None]).requires_grad_()
        q = quantization.static_quantize_int8(w, scales, ch_axis=0)
        torch.testing.assert_close(q, expected.expand_as(w), rtol=0, atol=0)
        q.sum().backward()
        torch.testing.assert_close(w.grad, mask[None, :] / scales[:, None], rtol=0, atol=0)

    def test_all_modes_use_full_signed_endpoints(self):
        signed = torch.arange(-128, 128, dtype=torch.float32, device=self.device)
        grad_tables = {
            "ste": {}, "true-ste": {},
            "lre": dict(dx=signed, dw=signed),
            "custom": dict(dx=signed[None, :].expand(256, 256).contiguous(),
                           dw=signed[:, None].expand(256, 256).contiguous()),
        }
        for bits in range(3, 9):
            for mode, tables in grad_tables.items():
                for groups in ((1,) if mode == "custom" else (1, 2)):
                    with self.subTest(bits=bits, mode=mode, groups=groups):
                        module = Conv2d_int8(
                            2, 2, 1, self.lut, grad=mode, groups=groups,
                            weight_bits=bits, bias=False, **tables,
                        ).to(self.device).eval()
                        qmin_w, qmax_w = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
                        scales = signed.new_tensor([0.5, 0.25])
                        with torch.no_grad():
                            module.scale_x.fill_(1.0)
                            module.scale_w.copy_(scales)
                            values = signed.new_tensor([qmin_w - 1, qmax_w + 1]) * scales
                            module.weight.copy_(values.view(2, 1, 1, 1).expand_as(module.weight))
                            x = signed.new_tensor([-129, -128, 128]).view(1, 1, 1, 3)
                            y = module(x.expand(1, 2, 1, 3))
                        self.assertEqual((module.qmin, module.qmax), (-128, 127))
                        xi = torch.tensor([0, 0, 255], device=self.device)
                        wi = torch.tensor([qmin_w + 128, qmax_w + 128], device=self.device)
                        expected = self.lut[xi[None, :], wi[:, None]] * (2 // groups)
                        expected = (expected * scales[:, None]).view(1, 2, 1, 3)
                        torch.testing.assert_close(y, expected, rtol=0, atol=0)

    def test_scale_convention_remains_absmax_over_qmax(self):
        for bits in range(3, 9):
            with self.subTest(bits=bits):
                module = self.make_module(weight_bits=bits, scale_momentum=1.0)
                module._update_scale(torch.tensor([-1.0, 2.0], device=self.device))
                torch.testing.assert_close(module.scale_x, module.scale_x.new_tensor(2.0 / 127),
                                           rtol=0, atol=0)
                module._reset_scale_w_from_weight()
                qmax_w = 2 ** (bits - 1) - 1
                expected = module.weight.detach().abs().amax(dim=(1, 2, 3)) / qmax_w
                torch.testing.assert_close(module.scale_w, expected, rtol=0, atol=0)
                with torch.no_grad():
                    module.weight.mul_(0.75)
                module._update_scale_w()
                expected = module.weight.detach().abs().amax(dim=(1, 2, 3)) / qmax_w
                torch.testing.assert_close(module.scale_w, expected, rtol=0, atol=0)

    def test_ema_and_deferred_backward(self):
        module = self.make_module(padding=1, scale_momentum=0.2).train()
        old_ste = copy.deepcopy(module)
        old_ste.grad = "ste"
        inputs, outputs, references, gradients = [], [], [], []
        for magnitude in (0.5, 3.0):
            x = (torch.randn(2, 2, 5, 7, device=self.device) * magnitude).requires_grad_()
            y = module(x)
            with torch.no_grad():
                torch.testing.assert_close(y, old_ste(x), rtol=0, atol=0)
            torch.testing.assert_close(module.scale_x, old_ste.scale_x, rtol=0, atol=0)
            torch.testing.assert_close(module.scale_w, old_ste.scale_w, rtol=0, atol=0)
            expected, operands = self.reference(module, x)
            go = torch.randn_like(y)
            expected.backward(go)
            inputs.append(x)
            outputs.append(y)
            references.append(operands)
            gradients.append(go)

        # Buffers can change again without affecting either pending backward.
        module.scale_x.fill_(0.123)
        module.scale_w.fill_(0.456)
        torch.autograd.backward(outputs, gradients)
        for x, (xr, _, _) in zip(inputs, references):
            torch.testing.assert_close(x.grad, xr.grad, rtol=2e-5, atol=3e-5)
        torch.testing.assert_close(module.weight.grad,
                                   sum(operands[1].grad for operands in references),
                                   rtol=2e-5, atol=3e-5)
        torch.testing.assert_close(module.bias.grad,
                                   sum(operands[2].grad for operands in references),
                                   rtol=2e-5, atol=3e-5)

        module.freeze_scale()
        sx, sw = module.scale_x.clone(), module.scale_w.clone()
        with torch.no_grad():
            y = module(inputs[0])
        self.assertFalse(y.requires_grad)
        torch.testing.assert_close(module.scale_x, sx, rtol=0, atol=0)
        torch.testing.assert_close(module.scale_w, sw, rtol=0, atol=0)

    def test_inplace_relu_after_layer(self):
        module = self.make_module(padding=1)
        x = torch.randn(2, 2, 5, 7, device=self.device, requires_grad=True)
        y = module(x)
        go = (y.detach() > 0).to(y.dtype)
        expected, operands = self.reference(module, x)
        F.relu(y, inplace=True).sum().backward()
        expected.backward(go)
        self.assert_gradients(module, x, operands)

    def test_conversion_and_validation(self):
        source = torch.nn.Conv2d(2, 2, 3, padding=1, groups=2).to(self.device).eval()
        converted = at.convert_int_model(
            source, self.lut, grad="true-ste", ignore_first_conv=False)
        self.assertIsInstance(converted, Conv2d_int8)
        self.assertEqual(converted.grad, "true-ste")
        self.assertFalse(converted.training)
        torch.testing.assert_close(converted.weight, source.weight, rtol=0, atol=0)
        torch.testing.assert_close(converted.bias, source.bias, rtol=0, atol=0)
        x = torch.randn(1, 2, 5, 7, device=self.device, requires_grad=True)
        y = converted(x)
        expected, operands = self.reference(converted, x)
        go = torch.randn_like(y)
        y.backward(go)
        expected.backward(go)
        self.assert_gradients(converted, x, operands)
        with self.assertRaisesRegex(ValueError, "only for qtype='int8'"):
            at.convert_int_model(source, self.lut, qtype="uint8", grad="true-ste")
        with self.assertRaises(ValueError):
            Conv2d_int8(2, 3, 3, self.lut, grad="unknown")


class TestTrueSTECPU(_TrueSTEContracts, unittest.TestCase):
    """Real Python/autograd path with only the CUDA forward ops substituted."""


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class TestTrueSTECUDA(_TrueSTEContracts, unittest.TestCase):
    """The same contracts against the actual CUDA forward kernels."""

    device_type = "cuda"


if __name__ == "__main__":
    unittest.main()
