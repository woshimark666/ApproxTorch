import torch
import approxtorch as at

torch.manual_seed(41)
torch.cuda.manual_seed_all(41)

def test_gradient_fetch():
    accuracte_grad = torch.load('accurate_grad.pth', weights_only=True).cuda()
    exact_lut = at.load_lut('exact.txt').cuda()

    B = 32
    C = 3
    H = 32
    W = 32
    K = 3
    O = 6

    feature = torch.ones((B, C, H, W), dtype=torch.float32, device='cuda', requires_grad=True)
    weight = torch.randn((O, C, K, K), dtype=torch.float32, device='cuda', requires_grad=True)
    bias = torch.randn((O), dtype=torch.float32, device='cuda', requires_grad=True)

    feature1 = feature.clone().detach().requires_grad_(True)
    weight1 = weight.clone().detach().requires_grad_(True)
    bias1 = bias.clone().detach().requires_grad_(True)

    stride = 1
    padding = 1
    dilation = 1
    criterion = torch.nn.L1Loss()


    output = at.nn.conv2d_int8(feature, weight, exact_lut, accuracte_grad,
                                bias, stride, padding, dilation)
    radom_target = torch.randn(output.shape, dtype=torch.float32, device='cuda', requires_grad=False) 
    radom_target1 = radom_target.clone().detach().requires_grad_(False)

    loss_my = criterion(output, radom_target)
    loss_my.backward()
    grad_feature_my = feature.grad
    grad_weight_my = weight.grad



    output_official = torch.nn.functional.conv2d(feature1, weight1, bias1, stride, padding, dilation)
    loss_official = criterion(output_official, radom_target1)
    loss_official.backward()
    grad_feature_official = feature1.grad
    grad_weight_official = weight1.grad

    print('grad_feature_my', grad_feature_my)
    print('grad_weight_my', grad_weight_my)
    print('grad_feature_official', grad_feature_official)
    print('grad_weight_official', grad_weight_official)
    print('grad_feature_my - grad_feature_official', grad_feature_my - grad_feature_official)
    print('grad_weight_my - grad_weight_official', grad_weight_my - grad_weight_official)

    print((grad_feature_my - grad_feature_official).sum())
    print((grad_weight_my - grad_weight_official).sum())
    
    
    
def test_cuda_gradient():
    M = 128
    N = 128
    K = 128
    
    # upstream_grad = torch.randint(-10, 10, (M, N), dtype=torch.float32, device='cuda')
    upstream_grad = torch.ones((M, N), dtype=torch.float32, device='cuda')
    
    A = torch.randint(-127, 127, (M, K), dtype=torch.int8, device='cuda')
    B = torch.randint(-127, 127, (K, N), dtype=torch.int8, device='cuda')
    a = A.clone().detach().to(torch.float32).requires_grad_(True)
    b = B.clone().detach().to(torch.float32).requires_grad_(True)
    upstream_grad1 = upstream_grad.clone().detach()
    
    grad_lut = torch.load('accurate_grad.pth', weights_only=True).cuda()
    
    grad_A_my, grad_B_my = at.approx_gemm.ops.gemm_int8_gradient(A, B, upstream_grad, grad_lut)
    



    c = torch.matmul(a, b)
    c.backward(upstream_grad1)
    grad_A_official = a.grad
    grad_B_official = b.grad
    
    print("A", A)
    print("B", B)
    print("upstream_grad", upstream_grad)
    print('grad_A_my', grad_A_my)
    print('grad_A_official', grad_A_official)
    print('grad_B_my', grad_B_my)
    print('grad_B_official', grad_B_official)
    
    print('grad_A_my - grad_A_official', grad_A_my - grad_A_official)
    print('grad_B_my - grad_B_official', grad_B_my - grad_B_official)
    
if __name__ == '__main__':
    # test_gradient_fetch()
    test_cuda_gradient()