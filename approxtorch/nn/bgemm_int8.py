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
    #
    # groups: 支持 1（普通卷积，forward 走 im2col_u8 + bgemm）和 depthwise
    # （groups == C == O，forward 走专用 dwconv kernel）。backward 两种情况
    # 同一套代码：cuDNN 的 convolution_backward 原生支持 groups，LRE 的
    # 因式分解按通道独立成立。

    @staticmethod
    def forward(ctx, x, w, lut, dx, dw, geom):
        kernel_size, stride, padding, dilation, groups = geom
        xq8 = x.detach().to(torch.int8)
        if groups == 1:
            xu8 = at.backend.ops.im2col_u8(xq8, kernel_size, stride, padding, dilation)
            y, _, wq = at.backend.ops.bgemm_fake_int8_claude_save(xu8, w, lut)
        else:
            # depthwise：y 与逐通道跑 LUT-BGEMM 逐位一致（同 tap 顺序）
            y, wq = at.backend.ops.dwconv_fake_int8_claude(
                xq8, w, lut, kernel_size, stride, padding, dilation)
        ctx.save_for_backward(xq8, wq)
        ctx.geom = geom
        ctx.dx = dx
        ctx.dw = dw
        return y

    @staticmethod
    def backward(ctx, grad_outputs):
        xq8, wq = ctx.saved_tensors
        (kh, kw), (sh, sw), (ph, pw), (dilh, dilw), g = ctx.geom
        dx, dw = ctx.dx, ctx.dw
        B, C, H, W = xq8.shape
        O = wq.shape[0]

        if g == 1 and (kh, kw, sh, sw, ph, pw, dilh, dilw) == (1, 1, 1, 1, 0, 0, 1, 1):
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
        wp = at.backend.ops.lut_map(wq, dx).view(O, C // g, kh, kw)
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
                    False, [0, 0], g, [True, True, False])
            else:
                # data pass 用原 padding 几何（input 只取形状，零存储 expand）
                dummy_x = go.new_empty(1).expand(B, C, H, W)
                grad_x, _, _ = torch.ops.aten.convolution_backward(
                    go, dummy_x, wp, None, [sh, sw], [ph, pw], [dilh, dilw],
                    False, [0, 0], g, [True, False, False])
                dummy_w = go.new_empty(1).expand(O, C // g, kh, kw)
                _, grad_w, _ = torch.ops.aten.convolution_backward(
                    go, xp, dummy_w, None, [sh, sw], [0, 0], [dilh, dilw],
                    False, [0, 0], g, [False, True, False])
        finally:
            torch.backends.cudnn.allow_tf32 = prev_tf32
            torch.backends.cudnn.benchmark = prev_bench

        return grad_x, grad_w.reshape(O, -1), None, None, None, None

def conv2d_int8_lre(x, w, lut, dx, dw, geom):
    return _conv2d_int8_lre.apply(x, w, lut, dx, dw, geom)


# 1x1 ste backward 用的 cuBLAS op 需要一张 [256] 解码表把保存的 uint8 索引
# 还原成量化值（v = idx - 128）；按 device 缓存，免每步重建
_u8_decode_tables = {}
def _u8_decode_table(device):
    t = _u8_decode_tables.get(device)
    if t is None:
        t = torch.arange(256, device=device, dtype=torch.float32) - 128
        _u8_decode_tables[device] = t
    return t


class _conv2d_int8_ste(Function):
    # conv 级 STE Function。STE 把近似乘法器当精确乘法看待，所以反传就是
    # 普通矩阵乘的梯度（原 einsum 对）：
    #     grad_x = W^T·go（unfold 空间，再 fold 回图像）
    #     grad_w = go·X^T
    # 而 "einsum + fold" 在数学上恰好就是普通卷积的 backward-data /
    # backward-weight，所以整体交给 cuDNN 的 convolution_backward —— 梯度
    # 数值与 einsum 相同（仅求和顺序不同，fp32 round-off 级差异）。
    #
    # forward 与 lre 共用 u8 im2col 路径（y 逐位一致）；backward 载荷从
    # fp32 unfold 矩阵 (+ fp32 w) 变成 int8 图像 + uint8 权重索引，
    # 反传时解码回量化值（精确整数，cast 无损）。
    # STE 下量化 0 就是真 0，padding 无需任何特殊处理。

    @staticmethod
    def forward(ctx, x, w, lut, geom):
        kernel_size, stride, padding, dilation, groups = geom
        xq8 = x.detach().to(torch.int8)
        if groups == 1:
            xu8 = at.backend.ops.im2col_u8(xq8, kernel_size, stride, padding, dilation)
            y, _, wq = at.backend.ops.bgemm_fake_int8_claude_save(xu8, w, lut)
        else:
            # depthwise（groups == C == O）：专用 LUT kernel
            y, wq = at.backend.ops.dwconv_fake_int8_claude(
                xq8, w, lut, kernel_size, stride, padding, dilation)
        ctx.save_for_backward(xq8, wq)
        ctx.geom = geom
        return y

    @staticmethod
    def backward(ctx, grad_outputs):
        xq8, wq = ctx.saved_tensors
        (kh, kw), (sh, sw), (ph, pw), (dilh, dilw), g = ctx.geom
        B, C, H, W = xq8.shape
        O = wq.shape[0]

        if g == 1 and (kh, kw, sh, sw, ph, pw, dilh, dilw) == (1, 1, 1, 1, 0, 0, 1, 1):
            # 1x1：图像本身就是展开矩阵，cuBLAS op 比 cuDNN 快（实测 0.82x）。
            # op 内部先按解码表还原数值、再做的两个 GEMM 就是上面的 einsum 对
            dec = _u8_decode_table(xq8.device)
            grad_x, grad_w = at.backend.ops.bgemm_lre_backward_claude(
                grad_output=grad_outputs, x=xq8.flatten(2),
                w=wq.transpose(0, 1).contiguous(), dx=dec, dw=dec)
            return (grad_x.view(B, C, H, W),
                    grad_w.transpose(0, 1).contiguous(), None, None)

        OH = (H + 2 * ph - dilh * (kh - 1) - 1) // sh + 1
        OW = (W + 2 * pw - dilw * (kw - 1) - 1) // sw + 1
        go = grad_outputs.reshape(B, O, OH, OW)
        xf = xq8.float()                                    # 量化值，精确整数
        wf = (wq.float() - 128.0).view(O, C // g, kh, kw)   # u8 索引 -> 量化值

        # 与 cuBLAS 同精度（torch 对 conv 默认开 TF32，会悄悄降梯度精度）；
        # 训练中形状固定，benchmark 让 cuDNN 选最优算法
        prev_tf32 = torch.backends.cudnn.allow_tf32
        prev_bench = torch.backends.cudnn.benchmark
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
        try:
            grad_x, grad_w, _ = torch.ops.aten.convolution_backward(
                go, xf, wf, None, [sh, sw], [ph, pw], [dilh, dilw],
                False, [0, 0], g, [True, True, False])
        finally:
            torch.backends.cudnn.allow_tf32 = prev_tf32
            torch.backends.cudnn.benchmark = prev_bench

        return grad_x, grad_w.reshape(O, -1), None, None

def conv2d_int8_ste(x, w, lut, geom):
    return _conv2d_int8_ste.apply(x, w, lut, geom)


class _bgemm_int8_bqsg64(_bgemm_int8_base):

    @staticmethod
    def backward(ctx, grad_outputs):
        x, w = ctx.saved_tensors
        coeff = ctx.coeff

        grad_x, grad_w = at.backend.ops.bgemm_bqsg64_backward(grad_outputs, x, w, coeff)

        return grad_x, grad_w, None, None, None, None

def bgemm_int8_bqsg64(x, w, lut, coeff):
    return _bgemm_int8_bqsg64.apply(x, w, lut, None, None, coeff)