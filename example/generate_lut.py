import numpy as np

def generate_multiplier_lut():
    # Create a 256x256 LUT for uint8 multiplication
    lut = np.zeros((256, 256), dtype=np.int32)
    
    # Fill the LUT with multiplication results
    for i in range(256):
        for j in range(256):
            lut[i, j] = i * j
    
    return lut

def save_lut_to_txt(lut, filename="exact_uint8.txt"):
    """Save the LUT to a text file in 2D format"""
    with open(filename, 'w') as f:
        # Write header with column numbers
        f.write("    ")  # Space for row numbers
        for j in range(256):
            f.write(f"{j:5d}")
        f.write("\n")
        
        # Write each row with row number
        for i in range(256):
            f.write(f"{i:3d} ")  # Row number
            for j in range(256):
                f.write(f"{lut[i,j]:5d}")
            f.write("\n")
    print(f"LUT saved to {filename}")


def generate_exact_gradient_lut():
    lut = np.zeros((256, 2), dtype=np.float32)
    
    # dim0 is dx, related to y
    for y in range(-128, 128):
        lut[y+128, 0] = y
        
    # dim1 is dy, related to x
    for x in range(-128, 128):
        lut[x+128, 1] = x
    return lut

def main():
    gradient_lut = generate_exact_gradient_lut()
    np.savetxt("exact_gradient.txt", gradient_lut, fmt='%.2f')
    
    # print(gradient_lut)


if __name__ == "__main__":
    main()
