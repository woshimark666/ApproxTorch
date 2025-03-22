import numpy as np
import torch

def load_lut(file_path):
    # Load the txt file as a numpy array
    lut_array = np.loadtxt(file_path, dtype=np.float32).reshape(256, 256)
    
    # Convert the numpy array to a PyTorch tensor
    lut_tensor = torch.tensor(lut_array)
    lut_tensor = lut_tensor.view(65536).to(torch.int)
    
    return lut_tensor



if __name__ == "__main__":
    lut_tensor = load_lut("/home/make/code/cnn_approx/lut/signed/txt/exact.txt")
    print(lut_tensor)
    print(lut_tensor.shape)