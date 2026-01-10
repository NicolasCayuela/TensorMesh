
import torch
import time

def check_speed():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        
    N = 10000
    # Float32 test
    A32 = torch.randn(N, N, dtype=torch.float32)
    b32 = torch.randn(N, 1, dtype=torch.float32)
    t0 = time.time()
    torch.matmul(A32, b32)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"FP32 Matmul (10k): {time.time() - t0:.4f}s")
    
    # Float64 test
    A64 = torch.randn(N, N, dtype=torch.float64)
    b64 = torch.randn(N, 1, dtype=torch.float64)
    t0 = time.time()
    torch.matmul(A64, b64)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"FP64 Matmul (10k): {time.time() - t0:.4f}s")

if __name__ == "__main__":
    check_speed()

