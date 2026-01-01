import torch
import approxtorch as at
import pandas as pd


def read_coefficients(file_path):
    df = pd.read_csv(file_path)
    cols = ['p10', 'p01', 'p20', 'p02', 'p11']
    result = df.set_index('name')[cols].apply(tuple, axis=1).to_dict()
    return result

def test_costum_grad():
    qmethod = ('dynamic', 'tensor', 'tensor')
    lut = at.load_lut('exact.txt').cuda()
    df = read_coefficients('./fit_data.csv')  
    feature = torch.randn(1, 1, 2, 2).cuda()
    weight = torch.randn(1, 1, 2, 2).cuda()
    print("feature:", feature)
    print("weight:", weight)
    stride = 2
    padding = 0
    dilation = 1
    groups = 1
    
    coefficients = df['1kx5']
    
    
    feature.requires_grad_(True)
    weight.requires_grad_(True)
    output = at.nn.conv2d_int8_custom(feature, 
                                      weight, 
                                      lut, 
                                      coefficients, 
                                      qmethod,
                                      None,
                                      None,
                                      bias=None,
                                      stride=stride, 
                                      padding=padding, 
                                      dilation=dilation)
    print("output:", output)
    upstream_grad = torch.ones_like(output)
    output.backward(upstream_grad)

    print("feature.grad:", feature.grad)
    print("weight.grad:", weight.grad)
    
    # calculate the right answer
    f = feature.clone().detach().cpu()
    w = weight.clone().detach().cpu()
    f = f.view(-1)
    w = w.view(-1)
    p10, p01, p20, p02, p11 = coefficients
    grad_f = p10 + 2 * p20 * f + p11 * w
    grad_w = p01 + 2 * p02 * w + p11 * f
    
    print("grad_f:", grad_f)
    print("grad_w:", grad_w)
    
    
def test_custom_conv():
    qmethod = ('dynamic', 'tensor', 'tensor')
    lut = at.load_lut('exact.txt').cuda()
    df = read_coefficients('./fit_data.csv')  
    feature = torch.randn(128, 6, 8, 8).cuda()
    weight = torch.randn(12, 6, 3, 3).cuda()
    feature.requires_grad_(True)
    weight.requires_grad_(True)
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    coefficients = df['1kx5']
    
    y = at.nn.conv2d_int8_custom(feature, 
                                 weight, 
                                 lut, 
                                 coefficients, 
                                 qmethod, 
                                 None, 
                                 None, 
                                 bias=None, 
                                 stride=stride, 
                                 padding=padding, 
                                 dilation=dilation)

if __name__ == "__main__":
    # test_costum_grad()
    test_custom_conv()