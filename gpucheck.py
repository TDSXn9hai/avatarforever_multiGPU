import torch
print("TORCH_VERSION", torch.__version__)
print("CUDA_BUILD", torch.version.cuda)
avail = torch.cuda.is_available()
print("CUDA_AVAILABLE", avail)
print("DEVICE_COUNT", torch.cuda.device_count())
if avail:
    print("DEVICE_NAME", torch.cuda.get_device_name(0))
    try:
        x = torch.randn(1024, device="cuda")
        print("CUDA_TENSOR_OK", float((x * 2).sum().item()))
    except Exception as e:
        print("CUDA_TENSOR_FAIL", repr(e))
