import torch
import approxtorch as at
from torch.autograd import Function



# 我们要把量化也包在这里, 这里输出就是反量化的结果
class _bgemm_uint8_ste(Function):
    @staticmethod
    def forward(ctx, q_x, q_w, lut, 
                x_quantizer, w_quantizer,
                scale_x, zero_x, scale_w, zero_w):
        # feature (N, CKK, L)
        # weight (O, CKK)
        
        
        ctx.save_for_backward(q_x, q_w)
        q_output = at.backend.ops.bgemm_uint8(q_x, q_w, lut)
        q_output = q_output.to(torch.float)
        # output is (N, O, L)
        q_x = q_x.to(torch.float)
        q_w = q_w.to(torch.float)
        
        ctx.save_for_backward(q_x, q_w, scale_x, zero_x, scale_w, zero_w)
        ctx.x_quantizer = x_quantizer
        ctx.w_quantizer = w_quantizer
        
        K = q_w.shape[1]

        # q_w @ 1 → (O,)  每行求和
        qw_sum = q_w.sum(dim=1)  # (O,)

        # 1 @ q_x → (N, 1, L) 或 (N, CKK, L) 按列求和 → (N, 1, L)
        qx_sum = q_x.sum(dim=1, keepdim=True)  # (N, 1, L)
        
        # dequantiation
        match w_quantizer[2]:
            case 'tensor':
                output = scale_x * scale_w * (
                q_output
                - zero_x * qw_sum.view(1, -1, 1)      # (1, O, 1)
                - zero_w * qx_sum                       # (N, 1, L)
                + zero_w * zero_x * K
            )
            case 'channel':
                s_w = scale_w.view(1, -1, 1)   # (1, O, 1)
                z_w = zero_w.view(1, -1, 1)    # (1, O, 1)

                output = scale_x * s_w * (
                    q_output
                    - zero_x * qw_sum.view(1, -1, 1)
                    - z_w * qx_sum
                    + z_w * zero_x * K
                )
                
        return output
    
    # @staticmethod
    # def backward(ctx, grad_output):
    #     q_x, q_w, scale_x, zero_x, scale_w, zero_w = ctx.saved_tensors
    #     x_quantizer = ctx.x_quantizer
    #     w_quantizer = ctx.w_quantizer
        
        
    #     if w_quantizer[2] == 'channel':
    #         s_w = scale_w.view(1, -1, 1) # (1, O, 1)
    #         z_w = zero_w.view(1, -1, 1)  # (1, O, 1)
    #     else:
    #         s_w = scale_w
    #         z_w = zero_w
            
    #     grad_scaled = grad_output * scale_x * s_w
        
    #     # -----------------------------------------------------------
    #     # 3. 计算 q_x 的梯度 (Shape: N, K, L)
    #     # -----------------------------------------------------------
    #     grad_q_x = None
    #     if ctx.needs_input_grad[0]: # 检查 q_x 是否需要梯度
    #         # 第一部分：MatMul 的反向传播
    #         # Einsum: grad(N, O, L) @ q_w(O, K) -> (N, K, L)
    #         # 相当于 grad_scaled 乘以权重转置
    #         grad_mm_x = torch.einsum('nol,ok->nkl', grad_scaled, q_w)

    #         # 第二部分：Zero Point 修正项 (- z_w * qx_sum) 的反向传播
    #         # Forward: - z_w * sum(q_x, dim=1)
    #         # Backward: 将 grad_scaled 沿着 O 维度求和 (因为 qx_sum 在 O 维度被广播了)
    #         # 乘以 -z_w
    #         # sum_grad: (N, 1, L)
    #         grad_zp_x = -z_w * grad_scaled
    #         grad_zp_x = grad_zp_x.sum(dim=1, keepdim=True) 
            
    #         grad_q_x = grad_mm_x + grad_zp_x

    #     # -----------------------------------------------------------
    #     # 4. 计算 q_w 的梯度 (Shape: O, K)
    #     # -----------------------------------------------------------
    #     grad_q_w = None
    #     if ctx.needs_input_grad[1]: # 检查 q_w 是否需要梯度
    #         # 第一部分：MatMul 的反向传播
    #         # Einsum: grad(N, O, L) @ q_x(N, K, L) -> (O, K)
    #         grad_mm_w = torch.einsum('nol,nkl->ok', grad_scaled, q_x)

    #         # 第二部分：Zero Point 修正项 (- zero_x * qw_sum) 的反向传播
    #         # Forward: - zero_x * sum(q_w, dim=1)
    #         # Backward: 将 grad_scaled 沿着 N, L 维度求和 (因为 qw_sum 在 N, L 被广播了)
    #         # 乘以 -zero_x
    #         grad_zp_w = -zero_x * grad_scaled
    #         grad_zp_w = grad_zp_w.sum(dim=(0, 2)).view(-1, 1) # (O, 1)
            
    #         grad_q_w = grad_mm_w + grad_zp_w
        
    #     return grad_q_x, grad_q_w, None, None, None, None, None, None, None
    
def bgemm_uint8_ste(qfeature, qweight, lut, x_quantizer, w_quantizer, scale_x, zero_x, scale_w, zero_w):
    return _bgemm_uint8_ste.apply(qfeature, qweight, lut, x_quantizer, w_quantizer, scale_x, zero_x, scale_w, zero_w)