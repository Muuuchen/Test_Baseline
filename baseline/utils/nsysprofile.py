import ctypes

_cudart = None

def enable_profile(use_profile):
    if use_profile:
        global _cudart
        _cudart = ctypes.CDLL('libcudart.so')
    else:
        raise NotImplementedError

def profile_start(use_profile):
    if use_profile:
        ret = _cudart.cudaProfilerStart()
        if ret != 0:
            raise Exception("cudaProfilerStart() returned %d" % ret)
    else:
        raise NotImplementedError


def profile_stop(use_profile):
    if use_profile:
        ret = _cudart.cudaProfilerStop()
        if ret != 0:
            raise Exception("cudaProfilerStop() returned %d" % ret)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    use_profile(True)
    profile_start(True)
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = torch.rand(5,5,device=device)
    n = torch.rand(5,5,device=device)
    res = torch.mul(m,n).to(device)
    print(res)
    profile_stop(True)



