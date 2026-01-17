#!/usr/bin/env python3
"""
Setup script for Open Domain Question Answering System

This script helps set up the development environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error {description}: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported.")
        print("Please use Python 3.8 or higher.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible!")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    # Install dependencies
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "logs",
        "cache",
        "data/samples",
        "models/cache",
        "test_outputs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    return True


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("⚠️ pytest not installed, skipping tests")
        return True
    
    return run_command(
        f"{sys.executable} -m pytest tests/ -v",
        "Running test suite"
    )


def check_gpu_support():
    """Check GPU support."""
    print("🖥️ Checking GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available! Found {gpu_count} GPU(s)")
            print(f"  GPU 0: {gpu_name}")
        else:
            print("ℹ️ CUDA not available, will use CPU")
    except ImportError:
        print("⚠️ PyTorch not installed yet")
    
    return True


def main():
    """Main setup function."""
    print("🚀 Setting up Open Domain Question Answering System")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    if not create_directories():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Check GPU support
    check_gpu_support()
    
    # Run tests
    if not run_tests():
        print("⚠️ Some tests failed, but setup can continue")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📚 Next steps:")
    print("  1. Try the simple example: python example.py")
    print("  2. Run the CLI interface: python src/cli.py --interactive")
    print("  3. Launch the web interface: streamlit run web_app/streamlit_app.py")
    print("  4. Read the documentation: README.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
