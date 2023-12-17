
import torch 
try:
    import cupy as cp
    is_cupy_available = True
except ImportError:
    is_cupy_available = False 


def synchronize(index=None):
    """
    Synchronize all devices
    """
    torch.cuda.synchronize()
    if is_cupy_available:
        if index is None:
            for index in range(cp.cuda.runtime.getDeviceCount()):
                cp.cuda.Device(index).synchronize()
        else:
            cp.cuda.Device(index).synchronize()