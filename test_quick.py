#!/usr/bin/env python3
"""
Teste rapido do MNIST - Para verificar se tudo funciona
Execute: python test_quick.py
"""

import torch
from model import MNISTNet
from trainer import MNISTTrainer

def test_model():
    """Teste rapido do modelo"""
    print("🧪 Testando modelo MNIST...")
    
    # Criar modelo
    model = MNISTNet()
    print(f"✅ Modelo criado: {model}")
    
    # Teste forward pass
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    print(f"✅ Forward pass OK: {output.shape}")
    
    # Teste trainer
    trainer = MNISTTrainer()
    print(f"✅ Trainer criado no device: {trainer.device}")
    
    # Verificar dataset
    print(f"✅ Dataset MNIST carregado:")
    print(f"   Train: {len(trainer.train_dataset)} amostras")
    print(f"   Test: {len(trainer.test_dataset)} amostras")
    
    print("\n🎉 Tudo funcionando! Execute 'python main.py' para usar a interface.")

if __name__ == "__main__":
    test_model()
