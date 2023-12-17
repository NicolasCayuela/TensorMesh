import torch 
import os 

if "TORCH_FEM_USE_PETSC" not in os.environ or os.environ["TORCH_FEM_USE_PETSC"] == "true":
    try:
        import petsc4py
        is_petsc_available = True
    except ImportError:
        is_petsc_available = False
else:
    is_petsc_available = False
    
if "TORCH_FEM_USE_CUPY" not in os.environ or os.environ["TORCH_FEM_USE_CUPY"] == "true":
    try:
        import cupy 
        import cupy as cp
        is_cupy_available = True
    except ImportError:
        is_cupy_available = False
else:
    is_cupy_available = False

def tensor2cupy(tensor):
    """turn torch.Tensor to cupy.ndarray
    Parameters
    ----------
    tensor : torch.Tensor
        the input tensor
    Returns
    -------
    cupy.ndarray
        the output cupy array
    """
    assert is_cupy_available, "cupy is not available"
    assert tensor.device.type == "cuda", "the device of tensor must be cuda"
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))
def cupy2tensor(cupy):
    """turn cupy.ndarray to torch.Tensor
    
    Parameters
    ----------
    cupy : cupy.ndarray
        the input cupy array
    Returns
    -------
    torch.Tensor
        the output tensor
    
    """
    return torch.utils.dlpack.from_dlpack(cupy.toDlpack())
def shapeT(shape):
    """transpose the shape
    
    Parameters
    ----------
    shape : Tuple[int, int]
        the input shape
    Returns
    -------
    Tuple[int, int]
        the output shape
    """
    return (shape[1], shape[0])