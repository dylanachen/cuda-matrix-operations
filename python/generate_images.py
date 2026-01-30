"""
Generate test images for convolution benchmarks

Creates grayscale images of different sizes with patterns for
demonstrating convolution filters (edges, gradients, shapes).
"""

import numpy as np
import os

def create_gradient_image(size):
    # Create a diagonal gradient pattern
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            img[i, j] = (i + j) % 256
    return img

def create_shapes_image(size):
    # Create an image with geometric shapes for edge detection
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Background gradient
    for i in range(size):
        for j in range(size):
            img[i, j] = 50
    
    # Rectangle in center
    rect_start = size // 4
    rect_end = 3 * size // 4
    img[rect_start:rect_end, rect_start:rect_end] = 200
    
    # Circle
    center = size // 2
    radius = size // 6
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 < radius**2:
                img[i, j] = 30
    
    # Diagonal line
    for i in range(size // 8, 7 * size // 8):
        j = i
        if j < size:
            img[i, max(0, j-2):min(size, j+3)] = 255
    
    return img

def create_checkerboard_image(size, block_size=None):
    # Create a checkerboard pattern
    if block_size is None:
        block_size = size // 8
    
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Fill in checkerboard pattern
    for i in range(size):
        for j in range(size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                img[i, j] = 255
            else:
                img[i, j] = 0
    return img

def create_noise_image(size, seed=42):
    # Create a random noise image
    np.random.seed(seed)
    return np.random.randint(0, 256, (size, size), dtype=np.uint8)

def save_pgm(img_array, filepath):
    # Save numpy array as PGM
    height, width = img_array.shape
    with open(filepath, 'wb') as f:
        f.write(f"P5\n{width} {height}\n255\n".encode())
        f.write(img_array.tobytes())
    print(f"Created: {filepath} ({width}x{height})")

def main():
    # Create output directory
    output_dir = "data/images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Image sizes to generate (M values)
    sizes = [256, 512, 1024]
    
    # Image types
    image_types = {
        'gradient': create_gradient_image,
        'shapes': create_shapes_image,
        'checkerboard': create_checkerboard_image,
        'noise': create_noise_image
    }
    
    print("Generating test images for convolution benchmarks")
    
    # Generate and save images
    for size in sizes:
        for img_type, create_func in image_types.items():
            img = create_func(size)
            filepath = os.path.join(output_dir, f"{img_type}_{size}.pgm")
            save_pgm(img, filepath)
    
    print(f"Generated {len(sizes) * len(image_types)} test images in {output_dir}/")
    print("\nImage sizes (M): ", sizes)
    print("Image types: ", list(image_types.keys()))

if __name__ == "__main__":
    main()