import torch
import numpy as np
import scipy


def fit(y):
    x = np.arange(-128, 128)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return slope, intercept


def generate_socc_est_grad(multiplier, save_path):
    lut = np.loadtxt(multiplier, dtype=np.float32)
    grad_lut = np.zeros((256, 2), dtype=np.float32)
    
    # z = x*w
    # compute dz/dx = w (realated to w)
    for w in range(-128, 128):
        line = lut[:, w]
        slope, intercept = fit(line)
        grad_lut[w, 0] = slope
        
    # compute dz/dw = x (related to x)
    for x in range(256):
        line = lut[x, :]
        slope, intercept = fit(line)
        grad_lut[x, 1] = slope
    
    np.savetxt(save_path, grad_lut, fmt='%.4f')
    
    return grad_lut


if __name__ == '__main__':
    lut = generate_socc_est_grad('../example/exact.txt')
