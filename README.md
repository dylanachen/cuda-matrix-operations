# cuda-matrix-operations

## Overview

Implementation of CPU and GPU matrix multiplication with performance analysis, shared library creation, and image convolution processing.

1. **Matrix Multiplication**: CPU, Naïve CUDA, Tiled CUDA (shared memory), and cuBLAS
2. **Image Convolution**: CPU and CUDA implementations with various filters
3. **Python Integration**: Shared libraries callable from Python via ctypes

## Project Structure

```
cuda-matrix-operations/
├── src/
│   ├── matrix_cpu.c          # CPU matrix multiplication
│   ├── matrix_naive.cu       # Naïve CUDA kernel
│   ├── matrix_tiled.cu       # Optimized CUDA with shared memory tiling
│   ├── matrix_cublas.cu      # cuBLAS implementation
│   ├── matrix_lib.cu         # Matrix multiplication shared library
│   ├── convolution_cpu.c     # CPU convolution
│   ├── convolution_cuda.cu   # CUDA convolution
│   └── convolution_lib.cu    # Convolution shared library
├── python/
│   ├── test_matrix_lib.py    # Test matrix multiplication library
│   ├── test_convolution_lib.py   # Test convolution library
│   ├── benchmark_convolution.py  # Compare CPU vs CUDA vs Python+CUDA
│   ├── generate_images.py    # Generate test images
│   └── generate_graphs.py    # Create performance graphs
├── data/
│   ├── images/               # Test images for convolution
│   └── output/               # Convolution output images
├── graphs/                   # Performance graphs for report
├── .gitignore
└── README.md
```

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (tested on RTX 3080 Ti)

### Software
- Windows 11
- CUDA Toolkit 13.x (or 12.x)
- Visual Studio 2022 with "Desktop development with C++" workload
- Python 3.8+ with packages: `numpy`, `matplotlib`

## Setup

### 1. Verify CUDA Installation
```powershell
nvcc --version
nvidia-smi
```

### 2. Install Python Dependencies
```powershell
pip install numpy matplotlib pillow scipy
```

### 3. Open x64 Native Tools Command Prompt
**Important**: Use "x64 Native Tools Command Prompt for VS 2022" for all compilation to avoid toolchain conflicts.

## Compilation

### Matrix Multiplication Programs
```cmd
cd src

# CPU version
nvcc matrix_cpu.c -o matrix_cpu.exe -O2

# CUDA versions
nvcc matrix_naive.cu -o matrix_naive.exe -O2
nvcc matrix_tiled.cu -o matrix_tiled.exe -O2
nvcc matrix_cublas.cu -o matrix_cublas.exe -O2 -lcublas

# Shared library for Python
nvcc --shared -o matrix_lib.dll matrix_lib.cu
```

## Usage

### Matrix Multiplication
```cmd
# Run with different matrix sizes
matrix_cpu.exe 512
matrix_cpu.exe 1024
matrix_cpu.exe 2048

matrix_naive.exe 512
matrix_tiled.exe 1024
matrix_cublas.exe 2048
```

### Python Tests
```powershell
cd python

# Generate test images
python generate_images.py

# Test matrix library
python test_matrix_lib.py
```

## Shared Library API

### Matrix Multiplication (`matrix_lib.dll`)

## Troubleshooting

### `cudafe++` ACCESS_VIOLATION Error
Use "x64 Native Tools Command Prompt for VS 2022" instead of regular PowerShell/CMD.

### Library Not Found
Ensure DLL files are in the working directory or add to PATH.

### CUDA Device Not Found
Check `nvidia-smi` works and drivers are installed.
