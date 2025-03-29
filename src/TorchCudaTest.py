import torch
import sys
import subprocess
import platform
import os


def check_torch_cuda():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Check environment variables
    print("\nEnvironment variables:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    # Check system GPU
    if platform.system() == "Windows":
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                print("\nNVIDIA-SMI output:")
                print(result.stdout[:500])  # Print first 500 chars of output
            else:
                print("\nNVIDIA-SMI not available or failed")
        except FileNotFoundError:
            print("\nNVIDIA-SMI not found - NVIDIA drivers may not be installed")

    if torch.cuda.is_available():
        print(f"\nCUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} capability: {torch.cuda.get_device_capability(i)}")
            print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")

        # Test tensor creation on GPU
        print("\nTesting GPU tensor creation:")
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
            print(f"Successfully created tensor on GPU: {x}")
        except Exception as e:
            print(f"Error creating tensor on GPU: {e}")
    else:
        print("\nNo CUDA device detected. PyTorch will use CPU only.")
        print("Possible reasons:")
        print("1. Your system doesn't have a compatible NVIDIA GPU")
        print("2. CUDA drivers are not installed or outdated")
        print("3. PyTorch was installed without CUDA support")

        # Check if torch was built with CUDA
        print(f"\nPyTorch CUDA build info:")
        print(f"Built with CUDA: {torch.backends.cudnn.enabled if hasattr(torch.backends, 'cudnn') else 'Unknown'}")
        print("\nTo install PyTorch with CUDA support, visit: https://pytorch.org/get-started/locally/")


if __name__ == "__main__":
    check_torch_cuda()