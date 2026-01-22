"""cgpu - Quick CUDA/GPU status for ML engineers."""

import sys

__version__ = "0.2.1"

def cgpu() -> str:
    """
    Print a colorful summary of CUDA/GPU status and return the device string.

    Returns:
        str: 'cuda' if CUDA is available, otherwise 'cpu'
    """
    try:
        from colorama import init, Fore, Style
        init()
    except ImportError:
        # Fallback if colorama not installed
        class Fore:
            GREEN = YELLOW = RED = CYAN = MAGENTA = WHITE = RESET = ""
        class Style:
            BRIGHT = RESET_ALL = ""

    # Use ASCII-safe characters for Windows compatibility
    LINE = "=" * 39
    CHECK = "[OK]" if sys.platform == "win32" else "✓"
    CROSS = "[X]" if sys.platform == "win32" else "✗"

    # Check PyTorch CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
    except ImportError:
        print(f"{Fore.RED}PyTorch not installed{Style.RESET_ALL}")
        return "cpu"

    # Header
    print(f"{Style.BRIGHT}{Fore.CYAN}{LINE}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.CYAN}          GPU Status Summary{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.CYAN}{LINE}{Style.RESET_ALL}")

    # CUDA availability
    if cuda_available:
        print(f"{Fore.GREEN}{CHECK} CUDA Available{Style.RESET_ALL}")
        print(f"{Fore.WHITE}  Device: {Fore.GREEN}{device}{Style.RESET_ALL}")

        # GPU details via PyTorch
        gpu_count = torch.cuda.device_count()
        print(f"{Fore.WHITE}  GPU Count: {Fore.MAGENTA}{gpu_count}{Style.RESET_ALL}")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            vram_total = props.total_memory / (1024 ** 3)  # GB

            print(f"{Fore.WHITE}  [{i}] {Fore.YELLOW}{gpu_name}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}      VRAM: {Fore.MAGENTA}{vram_total:.1f} GB{Style.RESET_ALL}")

            # Current memory usage
            if torch.cuda.is_initialized() or True:
                try:
                    mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    print(f"{Fore.WHITE}      Allocated: {Fore.CYAN}{mem_allocated:.2f} GB{Style.RESET_ALL}")
                    print(f"{Fore.WHITE}      Reserved: {Fore.CYAN}{mem_reserved:.2f} GB{Style.RESET_ALL}")
                except:
                    pass

        # CUDA version info
        print(f"{Fore.WHITE}  CUDA Version: {Fore.CYAN}{torch.version.cuda}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}  cuDNN Version: {Fore.CYAN}{torch.backends.cudnn.version()}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}  PyTorch: {Fore.CYAN}{torch.__version__}{Style.RESET_ALL}")

    else:
        print(f"{Fore.RED}{CROSS} CUDA Not Available{Style.RESET_ALL}")
        print(f"{Fore.WHITE}  Device: {Fore.YELLOW}{device}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}  PyTorch: {Fore.CYAN}{torch.__version__}{Style.RESET_ALL}")

    print(f"{Style.BRIGHT}{Fore.CYAN}{LINE}{Style.RESET_ALL}")

    return device


# Allow calling the module directly
if __name__ == "__main__":
    cgpu()
