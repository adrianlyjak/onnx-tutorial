import sys
from importlib.metadata import version, PackageNotFoundError
import torch
import onnx
import onnxruntime
import webbrowser
import torch.nn as nn
import torch.nn.init as init

# Some standard imports
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx


def check_package_version(package_name):
    """Check if a package is installed and return its version."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def verify_setup():
    """Verify that all required packages are installed and print their versions."""
    required_packages = {
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
    }

    print("\n=== Environment Setup Verification ===")
    print(f"Python version: {sys.version.split()[0]}")

    for package, version in required_packages.items():
        print(f"{package} version: {version}")

    # Check CUDA availability for PyTorch
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")


def open_netron():
    """Open Netron website in the default web browser."""
    netron_url = "https://netron.app"
    print(f"\nOpening {netron_url} in your default web browser...")
    webbrowser.open(netron_url)


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor**2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv2.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv3.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv4.weight)


def main():
    print("Day 1: ONNX Environment Setup")
    print("-----------------------------")

    verify_setup()

    print("\nTo visualize ONNX models, you can use Netron.")
    response = input("Would you like to open Netron in your web browser? (y/n): ")

    if response.lower() == "y":
        open_netron()

    print("\nSetup verification complete!")
    print("\nNext steps:")
    print("1. Review the PyTorch ONNX Export Tutorial:")
    print(
        "   https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html"
    )
    print("2. Proceed to Day 2: Implementing a simple MLP model")


if __name__ == "__main__":
    main()
