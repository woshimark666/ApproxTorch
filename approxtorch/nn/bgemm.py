import torch
import torch.nn as nn
from torch.autograd import Function
import approxtorch as at

class _bgemm_int8_base(Function):

    @staticmethod
    def forward(ctx, x, w, lut, dx, dw, coeff):
        ctx.save_for_backward(x, w)
        ctx.dx = dx
        ctx.dw = dw
        ctx.coeff = coeff

        return at.backend.ops.bgemm_fake_int8_claude(x, w, lut)



class _bgemm_int8_ste(_bgemm_int8_base):
    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x = torch.einsum("nol,ok->nkl", grad_output, w)
        grad_w = torch.einsum("nol,nkl->ok", grad_output, x)
        return grad_x, grad_w, None, None, None, None


def bgemm_int8_ste(x, w, lut):
    return _bgemm_int8_ste.apply(x, w, lut, None, None, None)



class _bgemm_int8_lre(_bgemm_int8_base):
    # gemm 级 LRE：x 是已展开的 [N,K,L]。backward 载荷只存 forward 内部现成的
    # uint8 量化映像（比 fp32 省 4 倍），梯度与 fp32 保存逐位一致。
    # 卷积请用下面的 conv2d_int8_lre（连 unfold 都不物化）。

    @staticmethod
    def forward(ctx, x, w, lut, dx, dw):
        y, xq, wq = at.backend.ops.bgemm_fake_int8_claude_save(x, w, lut)
        ctx.save_for_backward(xq, wq)
        ctx.dx = dx
        ctx.dw = dw
        return y

    @staticmethod
    def backward(ctx, grad_outputs):
        x, w = ctx.saved_tensors
        w = w.transpose(0, 1).contiguous()
        grad_x, grad_w = at.backend.ops.bgemm_lre_backward_claude(
            grad_output=grad_outputs, x=x, w=w, dx=ctx.dx, dw=ctx.dw)
        grad_w = grad_w.transpose(0, 1).contiguous()
        return grad_x, grad_w, None, None, None

def bgemm_int8_lre(x, w, lut, dx, dw):
    return _bgemm_int8_lre.apply(x, w, lut, dx, dw)


class _conv2d_int8_lre(Function):
    # conv 级 LRE Function：输入是 fake-quant 后的图像 x [B,C,H,W]（精确整数）
    # 和摊平权重 w [O,K]，返回 y [B,O,L]。
    #
    # forward: x 先转 int8 小图，再用 im2col_u8 直接展开成 u8 LUT 索引喂
    # bgemm kernel —— fp32 unfold 张量和 kernel 内的 fp32->u8 prepass 都消失
    # （原来这一段每元素 9 字节流量只为得到 1 字节索引）。
    #
    # backward: grad_x = fold(W'^T @ go) 和 grad_w = go @ X'^T 恰好就是普通
    # 卷积的 backward-data / backward-weight，把 1D LUT 映射(与 unfold 可交换)
    # 提前作用到算子上即可整体交给 cuDNN 隐式 GEMM（不再物化 X'、不再 fold，
    # 实测 k=3 比 cuBLAS 路径快 2.2-2.8x）。唯一例外：unfold 的 0 padding 映射
    # 后是 dw[128] != 0 而 cuDNN 只会补 0，所以 weight pass 用 lut_map_pad 把
    # dw[128] 预先填进边界、以 padding=0 调用；data pass 与 padding 无关。
    # 1x1/stride1/无 padding 时 cuDNN 反而更慢（实测 0.82x），保留 cuBLAS 路径。

    @staticmethod
    def forward(ctx, x, w, lut, dx, dw, geom):
        kernel_size, stride, padding, dilation = geom
        xq8 = x.detach().to(torch.int8)
        xu8 = at.backend.ops.im2col_u8(xq8, kernel_size, stride, padding, dilation)
        y, _, wq = at.backend.ops.bgemm_fake_int8_claude_save(xu8, w, lut)
        ctx.save_for_backward(xq8, wq)
        ctx.geom = geom
        ctx.dx = dx
        ctx.dw = dw
        return y

    @staticmethod
    def backward(ctx, grad_outputs):
        xq8, wq = ctx.saved_tensors
        (kh, kw), (sh, sw), (ph, pw), (dilh, dilw) = ctx.geom
        dx, dw = ctx.dx, ctx.dw
        B, C, H, W = xq8.shape
        O = wq.shape[0]

        if (kh, kw, sh, sw, ph, pw, dilh, dilw) == (1, 1, 1, 1, 0, 0, 1, 1):
            # 1x1：图像本身就是展开矩阵
            grad_x, grad_w = at.backend.ops.bgemm_lre_backward_claude(
                grad_output=grad_outputs, x=xq8.flatten(2),
                w=wq.transpose(0, 1).contiguous(), dx=dx, dw=dw)
            return (grad_x.view(B, C, H, W),
                    grad_w.transpose(0, 1).contiguous(),
                    None, None, None, None)

        OH = (H + 2 * ph - dilh * (kh - 1) - 1) // sh + 1
        OW = (W + 2 * pw - dilw * (kw - 1) - 1) // sw + 1
        go = grad_outputs.reshape(B, O, OH, OW)
        wp = at.backend.ops.lut_map(wq, dx).view(O, C, kh, kw)
        xp = at.backend.ops.lut_map_pad(xq8, dw, (ph, pw))

        # 与 cuBLAS 路径同精度（torch 对 conv 默认开 TF32，会悄悄降梯度精度）；
        # 训练中形状固定，benchmark 让 cuDNN 选最优算法
        prev_tf32 = torch.backends.cudnn.allow_tf32
        prev_bench = torch.backends.cudnn.benchmark
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
        try:
            if ph == 0 and pw == 0:
                grad_x, grad_w, _ = torch.ops.aten.convolution_backward(
                    go, xp, wp, None, [sh, sw], [0, 0], [dilh, dilw],
                    False, [0, 0], 1, [True, True, False])
            else:
                # data pass 用原 padding 几何（input 只取形状，零存储 expand）
                dummy_x = go.new_empty(1).expand(B, C, H, W)
                grad_x, _, _ = torch.ops.aten.convolution_backward(
                    go, dummy_x, wp, None, [sh, sw], [ph, pw], [dilh, dilw],
                    False, [0, 0], 1, [True, False, False])
                dummy_w = go.new_empty(1).expand(O, C, kh, kw)
                _, grad_w, _ = torch.ops.aten.convolution_backward(
                    go, xp, dummy_w, None, [sh, sw], [0, 0], [dilh, dilw],
                    False, [0, 0], 1, [False, True, False])
        finally:
            torch.backends.cudnn.allow_tf32 = prev_tf32
            torch.backends.cudnn.benchmark = prev_bench

        return grad_x, grad_w.reshape(O, -1), None, None, None, None

def conv2d_int8_lre(x, w, lut, dx, dw, geom):
    return _conv2d_int8_lre.apply(x, w, lut, dx, dw, geom)


def conv2d_int8_ste(x, w, lut, id_lut, geom):
    # STE 就是恒等 LUT 的 LRE：DX[q(w)] = q(w)、DW[q(x)] = q(x) 时上面的
    # backward 恰好退化成 grad_x = W^T·go、grad_w = go·X^T（即原 einsum 对）。
    # 顺带 padding 也自动正确：恒等 LUT 下 lut[128] = 0，与 cuDNN 的 0 padding
    # 一致。激活保存从 fp32 unfold 矩阵变成 int8 小图 + uint8 权重。
    # id_lut: arange(256) - 128 的 fp32 CUDA 张量（模块持有，免每步重建）。
    return _conv2d_int8_lre.apply(x, w, lut, id_lut, id_lut, geom)


class _bgemm_int8_bqsg64(_bgemm_int8_base):

    @staticmethod
    def backward(ctx, grad_outputs):
        x, w = ctx.saved_tensors
        coeff = ctx.coeff

        grad_x, grad_w = at.backend.ops.bgemm_bqsg64_backward(grad_outputs, x, w, coeff)

        return grad_x, grad_w, None, None, None, None

def bgemm_int8_bqsg64(x, w, lut, coeff):
    return _bgemm_int8_bqsg64.apply(x, w, lut, None, None, coeff)