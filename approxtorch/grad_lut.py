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
            
            
            