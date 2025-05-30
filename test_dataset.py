#!/usr/bin/env python3
"""
Test script to verify MNIST dataset downloading functionality
"""

import os
from trainer import MNISTTrainer

def test_dataset_download():
    print("Testing MNIST dataset download functionality...")
    
    # Check if dataset directory exists
    data_dir = './data'
    train_path = os.path.join(data_dir, 'MNIST', 'raw', 'train-images-idx3-ubyte')
    test_path = os.path.join(data_dir, 'MNIST', 'raw', 't10k-images-idx3-ubyte')
    
    print(f"Data directory: {data_dir}")
    print(f"Train data exists: {os.path.exists(train_path)}")
    print(f"Test data exists: {os.path.exists(test_path)}")
    
    # Initialize trainer (this will download dataset if needed)
    trainer = MNISTTrainer()
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(trainer.train_dataset)}")
    print(f"Test samples: {len(trainer.test_dataset)}")
    
    # Test a simple model creation
    model = trainer.create_model(hidden_size=64, dropout_rate=0.1)
    print(f"Model created successfully: {type(model).__name__}")
    
    # Test single prediction (should work even with untrained model)
    test_image, test_label = trainer.test_dataset[0]
    prediction, confidence = trainer.predict_single(test_image)
    print(f"Test prediction - Predicted: {prediction}, Actual: {test_label}, Confidence: {confidence:.2%}")

if __name__ == "__main__":
    test_dataset_download()
