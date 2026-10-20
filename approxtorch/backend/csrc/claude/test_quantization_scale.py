"""CUDA regressions for scale updates while earlier forwards await backward."""

import copy

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Fused quantization requires CUDA"
)


@pytest.fixture(params=range(torch.cuda.device_count()))
def device(request):
    return torch.device("cuda", request.param)


@pytest.mark.parametrize("asymmetric", [False, True], ids=["int8", "uint8"])
def test_quantization_preserves_each_forward_scale(device, asymmetric):
    from approxtorch.nn import quantization

    x = torch.tensor(
        [-80.0, -63.5, -32.0, 0.0, 1.0, 31.75, 63.5, 80.0],
        device=device, requires_grad=True,
    )
    scale = torch.tensor(0.5, device=device)
    zero_point = torch.tensor(128.0, device=device)
    qmin, qmax = (0, 255) if asymmetric else (-127, 127)
    outputs, expected_outputs, upstream = [], [], []
    expected_grad = torch.zeros_like(x)

    for index, value in enumerate((0.5, 0.25)):
        scale.fill_(value)
        if asymmetric:
            output = quantization.static_quantize_uint8(x, scale, zero_point)
        else:
            output = quantization.static_quantize_int8(x, scale)
        outputs.append(output)

        # Compute the oracle from immutable numeric qparams, not the live buffer.
        v = x.detach() / value + (128.0 if asymmetric else 0.0)
        expected_outputs.append(v.round().clamp(qmin, qmax))
        go = torch.arange(1, x.numel() + 1, device=device).float() * (index + 1)
        upstream.append(go)
        expected_grad += go * ((v >= qmin) & (v <= qmax)) / value

    # Neither pending backward may depend on these new buffer contents.
    scale.fill_(0.125)
    zero_point.fill_(0.0)
    torch.autograd.backward(outputs, upstream)

    for actual, expected in zip(outputs, expected_outputs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("grad", ["ste", "lre", "custom"])
@pytest.mark.parametrize("kernel", [1, 3])
def test_conv_deferred_backward_matches_immediate_backward(device, grad, kernel):
    from approxtorch.nn import Conv2d_int8

    torch.manual_seed(17)
    signed = torch.arange(-128, 128, dtype=torch.float32, device=device)
    lut = (signed[:, None] * signed[None, :]).contiguous()
    kwargs = {}
    if grad == "lre":
        kwargs = dict(dx=signed.clone(), dw=signed.clone())
    elif grad == "custom":
        kwargs = dict(
            dx=signed[None, :].expand(256, 256).contiguous(),
            dw=signed[:, None].expand(256, 256).contiguous(),
        )
    deferred = Conv2d_int8(
        2, 3, kernel, lut, grad=grad, bias=True, padding=kernel // 2,
        scale_momentum=0.2, **kwargs,
    ).to(device).train()
    deferred.scale_x.fill_(0.01)
    immediate = copy.deepcopy(deferred)
    inputs = [
        (torch.randn(2, 2, 5, 7, device=device) * magnitude).requires_grad_()
        for magnitude in (0.5, 2.0)
    ]
    reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    outputs = [deferred(x) for x in inputs]
    upstream = [torch.randn_like(y) for y in outputs]
    torch.autograd.backward(outputs, upstream)

    for x, expected_x, output, go in zip(
        inputs, reference_inputs, outputs, upstream
    ):
        expected_output = immediate(expected_x)
        expected_output.backward(go)
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, expected_x.grad, rtol=2e-5, atol=2e-5)

    for actual, expected in zip(deferred.parameters(), immediate.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(deferred.scale_x, immediate.scale_x, rtol=0, atol=0)
    torch.testing.assert_close(deferred.scale_w, immediate.scale_w, rtol=0, atol=0)
