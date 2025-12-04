import torch
import numpy as np
import scipy


def fit_int8(y):
    x = np.arange(-128, 128)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return slope, intercept


def fit_uint8(y):
    x = np.arange(0, 256)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return slope, intercept


def generate_socc_lre_grad(multiplier, qtype, save_path):
    lut = np.loadtxt(multiplier, dtype=np.float32)
    grad_lut = np.zeros((256, 2), dtype=np.float32)
    
    # z = x*w
    # compute dz/dx = w (realated to w)
    
    
    match qtype:
        case 'int8':
            # compute dz/dx, related to w
            for w in range(-128, 128):
                line = lut[:, w+128]
                slope, intercept = fit_int8(line)
                grad_lut[w+128, 0] = slope
                
            # compute dz/dw, related to x
            for x in range(-128, 128):
                line = lut[x+128, :]
                slope, intercept = fit_int8(line)
                grad_lut[x+128, 1] = slope
        
        case 'uint8':
            # compute dz/dx, related to w
            for w in range(0, 256):
                line = lut[:, w]
                slope, intercept = fit_uint8(line)
                grad_lut[w, 0] = slope
                
            # compute dz/dw, related to x
            for x in range(0, 256):
                line = lut[x, :]
                slope, intercept = fit_uint8(line)
                grad_lut[x, 1] = slope
    
    
    np.savetxt(save_path, grad_lut)
    return grad_lut

