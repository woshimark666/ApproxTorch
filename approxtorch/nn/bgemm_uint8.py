# uint8 LUT-BGEMM（uint8 × uint8 近似乘法器，非对称量化）：
#
#   y[n, o, l] = sum_k lut[ x[n,k,l] * 256 + w[o,k] ]
#
#   x   [N, K, L]  uint8 量化值 [0, 255]（fp32 整数值；仅推理时也可直接传
#                  uint8 dtype，值本身就是 LUT 行索引，kernel 跳过 prepass）
#   w   [O, K]     uint8 量化值 [0, 255]（fp32 整数值）
#   lut [65536]    近似乘法器真值表，布局 lut[x_u8][w_u8]
#                  （两个操作数都用原始值作索引，x 高字节、w 低字节）
#
# 注意：输出是「原始 LUT 和」Σ_k q_x·q_w 的近似，不含零点修正。
# 非对称量化的交叉项（-z_w·Σq_x - z_x·Σq_w + K·z_x·z_w）由调用方
# （Conv2d_uint8）用可微 torch op 在外面补，见其 forward。
import torch
from torch.autograd import Function
import approxtorch as at


class _bgemm_uint8_base(Function):

    @staticmethod
    def forward(ctx, x, w, lut):
        ctx.save_for_backward(x, w)
        return at.backend.ops.bgemm_fake_uint8_claude(x, w, lut)


class _bgemm_uint8_ste(_bgemm_uint8_base):
    # STE：把近似乘法器当精确乘法。forward 输出是未居中的 Σ_k q_x·q_w，
    # 所以这里的梯度也是对原始乘积的：d/dq_x = q_w，d/dq_w = q_x。
    # 零点修正项在模块 forward 里是可微的 torch op，autograd 会自动
    # 补上 -z 部分，合成后恰好是期望的
    #   dy/dq_x = s·(q_w - z_w)，dy/dq_w = s·(q_x - z_x)
    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x = torch.einsum("nol,ok->nkl", grad_output, w)
        grad_w = torch.einsum("nol,nkl->ok", grad_output, x)
        return grad_x, grad_w, None


def bgemm_uint8_ste(x, w, lut):
    return _bgemm_uint8_ste.apply(x, w, lut)


class _bgemm_uint8(_bgemm_uint8_base):
    # 仅前向（推理/测试用）：误反传时给出明确报错而不是静默错梯度
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "bgemm_uint8 is forward-only; use bgemm_uint8_ste for training")


def bgemm_uint8(x, w, lut):
    return _bgemm_uint8.apply(x, w, lut)
