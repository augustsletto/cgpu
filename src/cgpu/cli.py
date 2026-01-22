"""CLI for cgpu - GPU status and PyTorch installation helper."""

import argparse
import subprocess
import sys


CUDA_INDEX_URLS = {
    "12.1": "https://download.pytorch.org/whl/cu121",
    "12.4": "https://download.pytorch.org/whl/cu124",
    "11.8": "https://download.pytorch.org/whl/cu118",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

TORCH_PACKAGES = ["torch", "torchvision", "torchaudio"]


def detect_package_manager():
    """Detect if uv or pip should be used."""
    try:
        subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            check=True,
        )
        return "uv"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "pip"


def install_torch(cuda_version=None, package_manager=None):
    """Install PyTorch with optional CUDA version specification."""
    if package_manager is None:
        package_manager = detect_package_manager()

    if cuda_version and cuda_version not in CUDA_INDEX_URLS:
        print(f"Error: Unknown CUDA version '{cuda_version}'")
        print(f"Supported versions: {', '.join(CUDA_INDEX_URLS.keys())}")
        return 1

    # Build the install command
    if package_manager == "uv":
        cmd = ["uv", "pip", "install"]
    else:
        cmd = [sys.executable, "-m", "pip", "install"]

    cmd.extend(TORCH_PACKAGES)

    if cuda_version:
        index_url = CUDA_INDEX_URLS[cuda_version]
        cmd.extend(["--index-url", index_url])
        print(f"Installing PyTorch with CUDA {cuda_version}...")
    else:
        print("Installing PyTorch from PyPI (includes CUDA support)...")

    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInstallation cancelled.")
        return 1


def show_status():
    """Show GPU status."""
    from cgpu import cgpu
    return cgpu()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="cgpu",
        description="Quick CUDA/GPU status and PyTorch installation helper for ML engineers",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Install subcommand
    install_parser = subparsers.add_parser(
        "install",
        help="Install PyTorch with optional CUDA version",
    )
    install_parser.add_argument(
        "--cuda",
        type=str,
        choices=list(CUDA_INDEX_URLS.keys()),
        help=f"CUDA version ({', '.join(CUDA_INDEX_URLS.keys())})",
    )
    install_parser.add_argument(
        "--pip",
        action="store_true",
        help="Force use of pip instead of uv",
    )

    # Status subcommand (also the default)
    subparsers.add_parser(
        "status",
        help="Show GPU status (default command)",
    )

    # Version
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version",
    )

    args = parser.parse_args()

    if args.version:
        from cgpu import __version__
        print(f"cgpu-info {__version__}")
        return 0

    if args.command == "install":
        pm = "pip" if args.pip else None
        return install_torch(cuda_version=args.cuda, package_manager=pm)

    # Default: show status
    show_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
