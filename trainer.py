import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import MNISTNet
import os
from PIL import Image as PILImage

class MNISTTrainer:
    """
    Trainer simples para MNIST - sem drama desnecessario
    Agora com MNIST de verdade, nao EMNIST que ninguem pediu
    """
    def __init__(self, model_save_path='mnist_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MNISTNet(num_classes=10).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer simples - Adam funciona bem
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=0.001,
            weight_decay=1e-4
        )
        
        # Scheduler opcional
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1
        )
        
        self.model_save_path = model_save_path
        
        # Transformacoes SIMPLES - sem rotacao maluca
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST padrao
        ])
        
        # Transform identico para teste e inferencia
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.inference_transform = self.test_transform  # Mesmo transform!
        
        # Carregar MNIST (finalmente!)
        self.train_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=self.train_transform
        )
        
        self.test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False, 
            download=True, 
            transform=self.test_transform
        )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)
        
        # Historico de treinamento
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
    
    def train_epoch(self):
        """Treina uma epoca"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def test_epoch(self):
        """Testa o modelo"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, epochs=10, progress_callback=None):
        """Treina o modelo - versao simplificada"""
        print(f"Treinando modelo MNIST por {epochs} epocas no {self.device}")
        
        for epoch in range(epochs):
            # Treinar
            train_loss, train_acc = self.train_epoch()
            
            # Testar
            test_loss, test_acc = self.test_epoch()
            
            # Salvar historico
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            
            # Callback para UI
            if progress_callback:
                progress_callback(epoch + 1, epochs, train_loss, train_acc, test_loss, test_acc)
            
            print(f'Epoca {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            
            # Atualizar scheduler
            self.scheduler.step()
        
        # Salvar modelo automaticamente
        self.save_model()
        print(f"Modelo salvo em {self.model_save_path}")
    
    def save_model(self):
        """Salva o modelo"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies,
        }, self.model_save_path)
    
    def load_model(self, model_path=None):
        """Carrega o modelo"""
        if model_path is None:
            model_path = self.model_save_path
            
        if not os.path.exists(model_path):
            print(f"Modelo nao encontrado: {model_path}")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Carregar historico se disponivel
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
                self.train_accuracies = checkpoint['train_accuracies']
                self.test_losses = checkpoint['test_losses']
                self.test_accuracies = checkpoint['test_accuracies']
            
            print(f"Modelo carregado: {model_path}")
            return True
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return False
    
    def predict(self, image):
        """Prediz um digito - versao simplificada"""
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                # Converter numpy para PIL
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = PILImage.fromarray(image, mode='L')
                image_tensor = self.inference_transform(pil_image)
            else:
                image_tensor = image
            
            # Adicionar dimensoes se necessario
            if len(image_tensor.shape) == 2:
                image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
            elif len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            image_tensor = image_tensor.to(self.device)
            output = self.model(image_tensor)
            
            # Probabilidades
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), confidence.item()
    
    def plot_training_history(self):
        """Plota o historico de treinamento"""
        if not self.train_losses:
            print("Nenhum historico de treinamento disponivel")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoca')
        plt.ylabel('Loss')
        plt.title('Loss durante Treinamento')
        plt.legend()
        
        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoca')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy durante Treinamento')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Teste simples
    trainer = MNISTTrainer()
    trainer.train(epochs=5)
    trainer.plot_training_history()
