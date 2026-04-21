
def check_gpu_env():
    import torch
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available. Please check your GPU environment.")
    else:
        print(f"CUDA is available. Found {torch.cuda.device_count()} CUDA device(s).")
    
check_gpu_env()

from .backend_gpu import BackendGPU

