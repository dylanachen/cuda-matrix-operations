import ctypes
import numpy as np
import time
import os

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), '..', 'matrix_lib.dll')
lib = ctypes.CDLL(lib_path)

# Define the argument and return types of the C functions
lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int
]

# Set seed for reproducibility
rng = np.random.default_rng(seed=42)
# Create random matrices A and B
N = 1024  # Size of the NxN matrix
A = rng.random((N, N)).astype(np.float32)
B = rng.random((N, N)).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# Call the GPU matrix multiplication function and time it
start = time.time()
lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N)
end = time.time()

# Print the time elapsed
print(f"Python call to CUDA library completed in {end - start:.6f} seconds")
