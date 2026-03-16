import torch
import numpy as np
import scipy.optimize as optimize
from scipy.stats import linregress
from scipy.optimize import curve_fit

def poly2d(X, p00, p01, p10, p11, p20, p02):
    x, y = X
    return (p00
            + p01 * y
            + p10 * x
            + p11 * x * y
            + p20 * x**2
            + p02 * y**2)

def fit_linear(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope

def load_txt_lut(file_path):
    lut = np.loadtxt(file_path, dtype=np.float32)
    lut = lut.reshape(256, 256)
    return lut


def lre(multiplier, qtype, save_path):
    lut = load_txt_lut(multiplier)
    grad_a = np.zeros((256), dtype=np.float32)
    grad_b = np.zeros((256), dtype=np.float32)
    match qtype:
        case 'int8':
            # compute z = x*y
            # compute dz/dx = at point y
            for y in range(-128, 128):
                line = lut[:, y+128]
                x = np.arange(-128, 128)
                k = fit_linear(x, line)
                grad_a[y+128] = k
            
            for x in range(-128, 128):
                line = lut[x+128, :]
                y = np.arange(-128, 128)
                k = fit_linear(y, line)
                grad_b[x+128] = k

            if save_path is not None:
                np.savetxt(f'{save_path}_lre_grad_a.txt', grad_a, fmt='%.4f')
                np.savetxt(f'{save_path}_lre_grad_b.txt', grad_b, fmt='%.4f')
            return grad_a, grad_b

                
def BQSG(multiplier, size = 64, save_path = None):
    lut = load_txt_lut(multiplier)
    
    grad_a = np.zeros((256, 256), dtype=np.float32)
    grad_b = np.zeros((256, 256), dtype=np.float32) # 对应 Col (j) 的梯度
    fitted = np.zeros((256, 256), dtype=np.float32)
    k = int(256 / size)
    for i in range(k):
        for j in range(k):
            # 1. 获取切片
            row_start, row_end = i*size, (i+1)*size
            col_start, col_end = j*size, (j+1)*size
            sub_lut = lut[row_start:row_end, col_start:col_end]
            
            # 2. 生成坐标网格 (关键修正: 使用 indexing='ij')
            # a 对应行索引 (Row/Y), b 对应列索引 (Col/X)
            a = np.arange(row_start, row_end)
            b = np.arange(col_start, col_end)
            A_grid, B_grid = np.meshgrid(a, b, indexing='ij')
            
            # 3. 展平以供 curve_fit 使用
            a_flat = A_grid.flatten()
            b_flat = B_grid.flatten()
            y_flat = sub_lut.flatten()

            # 4. 拟合
            # 初始猜测 p0=None 即可，或者全0
            try:
                popt, pcov = curve_fit(poly2d, (a_flat, b_flat), y_flat)
            except RuntimeError:
                print(f"Fit failed at block {i},{j}, using zeros.")
                popt = np.zeros(6)

            p00, p01, p10, p11, p20, p02 = popt
            # 5. 向量化计算 Fitted 值 (修正: 移除慢速循环)
            # fitted_block = poly2d((A_grid, B_grid), *popt)
            # fitted[row_start:row_end, col_start:col_end] = fitted_block
            
            # for x in range(row_start, row_end):
            #     for y in range(col_start, col_end):
            #         fitted[x, y] = poly2d((x, y), p00, p01, p10, p11, p20, p02)
            
            for x in range(row_start, row_end):
                for y in range(col_start, col_end):
                    grad_a[x, y] = p10 + 2 * p20 * x + p11 * y
                    grad_b[x, y] = p01 + 2 * p02 * y + p11 * x
            
    
    # E = fitted - lut
    # MAE = np.abs(E).sum() / (256 * 256)
    # MSE = np.square(E).sum() / (256 * 256)
    # print("MAE: ", MAE)
    # print("MSE: ", MSE)
    if save_path is not None:
        np.savetxt(f'{save_path}_bqsg_{size}_grad_a.txt', grad_a)
        np.savetxt(f'{save_path}_bqsg_{size}_grad_b.txt', grad_b)
    return grad_a, grad_b
            
    
    
################BQSG #################

import numpy as np


def _smooth_axis(lut, half_ws, axis):
    """Smooth LUT along the given axis using moving average (Eq. 4)."""
    size = lut.shape[axis]
    smoothed = np.zeros((256, 256), dtype=np.float32)
 
    if axis == 1:
        # Fix a (row), smooth along b (col)
        for k in range(256):
            line = lut[k, :]
            for i in range(size):
                if i >= half_ws and i <= 255 - half_ws:
                    smoothed[k, i] = line[i - half_ws: i + half_ws + 1].mean()
                else:
                    smoothed[k, i] = line[i]
 
    elif axis == 0:
        # Fix b (col), smooth along a (row)
        for k in range(256):
            line = lut[:, k]
            for i in range(size):
                if i >= half_ws and i <= 255 - half_ws:
                    smoothed[i, k] = line[i - half_ws: i + half_ws + 1].mean()
                else:
                    smoothed[i, k] = line[i]
 
    return smoothed
 
 
def _diff_grad(lut, smooth, half_ws, axis):
    """Difference-based gradient (Eqs. 5 & 6).
    
    Args:
        lut: original (unsmoothed) LUT, used for boundary fallback.
        smooth: already smoothed LUT from _smooth_axis.
        half_ws: half window size.
        axis: 0 for grad_a (d/da), 1 for grad_b (d/db).
    """
    grad = np.zeros((256, 256), dtype=np.float32)
 
    if axis == 1:
        # grad_b: derivative along b (col direction)
        for a in range(256):
            line = smooth[a, :]
            max_val = np.max(lut[a, :])
            min_val = np.min(lut[a, :])
            fallback_grad = (max_val - min_val) / 256
            if a <=128:
                fallback_grad = - fallback_grad
            for b in range(256):
                if b > half_ws and b < 255 - half_ws:
                    # Eq. 5: central difference
                    grad[a, b] = (line[b + 1] - line[b - 1]) / 2.0
                else:
                    # Eq. 6: boundary fallback
                    grad[a, b] = fallback_grad
 
    elif axis == 0:
        # grad_a: derivative along a (row direction)
        for b in range(256):
            line = smooth[:, b]
            max_val = np.max(lut[:, b])
            min_val = np.min(lut[:, b])
            fallback_grad = (max_val - min_val) / 256
            if b <=128:
                fallback_grad = - fallback_grad
            for a in range(256):
                if a > half_ws and a < 255 - half_ws:
                    # Eq. 5: central difference
                    grad[a, b] = (line[a + 1] - line[a - 1]) / 2.0
                else:
                    # Eq. 6: boundary fallback
                    grad[a, b] = fallback_grad
 
    return grad
 
 
def DATE(multiplier, half_ws=32, save_path=None):
    """
    Generate gradient LUTs using the smoothing + difference-based method.
    (Paper: "Gradient Approximation of Approximate Multipliers for
     High-Accuracy Deep Neural Network Retraining", Eqs. 4-6)
 
    Args:
        multiplier: path to a 256x256 LUT txt file, or a (256, 256) numpy array.
        half_ws: half window size for moving average smoothing.
        save_path: if not None, save grad_a and grad_b to files.
 
    Returns:
        grad_a: (256, 256) float32, gradient w.r.t. a (first index, row).
        grad_b: (256, 256) float32, gradient w.r.t. b (second index, col).
    """
    if isinstance(multiplier, np.ndarray):
        lut = multiplier.astype(np.float32)
    else:
        lut = np.loadtxt(multiplier).astype(np.float32)
 
    assert lut.shape == (256, 256), f"Expected (256, 256), got {lut.shape}"
 
    # Step 1: Smooth (Eq. 4)
    smooth_a = _smooth_axis(lut, half_ws, axis=0)
    smooth_b = _smooth_axis(lut, half_ws, axis=1)
 
    # Step 2: Difference-based gradient (Eqs. 5 & 6)
    # No double smoothing — pass the already-smoothed data directly
    grad_a = _diff_grad(lut, smooth_a, half_ws, axis=0)
    grad_b = _diff_grad(lut, smooth_b, half_ws, axis=1)
 
    if save_path is not None:
        np.savetxt(f'{save_path}_date_{half_ws}_grad_a.txt', grad_a)
        np.savetxt(f'{save_path}_date_{half_ws}_grad_b.txt', grad_b)
 
    return grad_a, grad_b
    