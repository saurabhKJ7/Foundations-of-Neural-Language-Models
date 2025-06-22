#!/usr/bin/env python3
"""
Setup script for Q1: Tokenization & Fill-in-the-Blank
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "tokenizers>=0.13.0",
        "sentencepiece>=0.1.99",
        "numpy>=1.21.0",
        "accelerate>=0.20.0"
    ]
    
    print("Installing required packages...")
    for req in requirements:
        print(f"Installing {req}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ {req} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")
            return False
    
    return True

def check_installation():
    """Check if packages are properly installed"""
    packages = [
        "torch",
        "transformers", 
        "tokenizers",
        "sentencepiece",
        "numpy"
    ]
    
    print("\nChecking installations...")
    all_good = True
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is not available")
            all_good = False
    
    return all_good

def main():
    """Main setup function"""
    print("=" * 60)
    print("Q1 SETUP: TOKENIZATION & FILL-IN-THE-BLANK")
    print("=" * 60)
    
    # Install requirements
    if install_requirements():
        print("\n✅ All packages installed successfully!")
    else:
        print("\n❌ Some packages failed to install")
        return
    
    # Check installation
    if check_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nYou can now run:")
        print("  python simple_tokenization_demo.py")
        print("  python tokenization_and_fill_mask.py")
    else:
        print("\n❌ Setup verification failed")

if __name__ == "__main__":
    main()