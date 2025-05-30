#!/usr/bin/env python3
"""
Test script to verify the MNIST application components work correctly.
"""

import sys
import torch
import torchvision
from PyQt5.QtWidgets import QApplication
from model import MNISTNet
from trainer import MNISTTrainer

def test_model():
    """Test the neural network model."""
    print("Testing MNISTNet model...")
    model = MNISTNet(hidden_size=128, dropout_rate=0.2)
    
    # Test with dummy input
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    print("✓ Model test passed!")

def test_trainer():
    """Test the trainer components."""
    print("Testing MNISTTrainer...")
    trainer = MNISTTrainer()
    
    # Test data loading
    assert trainer.train_dataset is not None, "Train dataset not loaded"
    assert trainer.test_dataset is not None, "Test dataset not loaded"
    assert len(trainer.train_dataset) > 0, "Train dataset is empty"
    assert len(trainer.test_dataset) > 0, "Test dataset is empty"
    
    print(f"✓ Train dataset size: {len(trainer.train_dataset)}")
    print(f"✓ Test dataset size: {len(trainer.test_dataset)}")
    print("✓ Trainer test passed!")

def test_pyqt():
    """Test PyQt5 availability."""
    print("Testing PyQt5...")
    app = QApplication([])
    print("✓ PyQt5 test passed!")

def main():
    print("Running MNIST Application Tests...")
    print("=" * 40)
    
    try:
        # Test PyTorch
        print(f"PyTorch version: {torch.__version__}")
        print(f"TorchVision version: {torchvision.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
        
        # Run tests
        test_model()
        test_trainer()
        test_pyqt()
        
        print("=" * 40)
        print("✓ All tests passed! The application should work correctly.")
        print("\nTo run the application, use: python main.py")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
