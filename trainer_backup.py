import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import EMNISTNet
import os

class EMNISTTrainer:    def __init__(self, model_save_path='emnist_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EMNISTNet(num_classes=26).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # Use a more conservative learning rate with weight decay
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0003, weight_decay=0.01)
        # More gradual learning rate decay
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, factor=0.5, verbose=True
        )
        self.model_save_path = model_save_path
          # Data transformations for EMNIST letters
        # EMNIST letters need to be rotated and flipped to match standard orientation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.rot90(x, k=3, dims=[1, 2])),  # Rotate 270 degrees
            transforms.Lambda(lambda x: torch.flip(x, dims=[2])),  # Flip horizontally
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Transform for user drawn images (without rotation/flip)
        self.inference_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load datasets
        self.train_dataset = torchvision.datasets.EMNIST(
            root='./data', 
            split='letters',
            train=True, 
            download=True, 
            transform=self.transform
        )
        
        self.test_dataset = torchvision.datasets.EMNIST(
            root='./data',
            split='letters', 
            train=False, 
            download=True, 
            transform=self.transform
        )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Adjust target labels (EMNIST letters are 1-26, we need 0-25)
            target = target - 1
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def test_epoch(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Adjust target labels
                target = target - 1
                
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total
        
        return test_loss, test_acc
    
    def train(self, epochs=20, progress_callback=None):
        print(f'Training on {self.device}')
        print(f'Training set size: {len(self.train_dataset)}')
        print(f'Test set size: {len(self.test_dataset)}')
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Testing
            test_loss, test_acc = self.test_epoch()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(epoch + 1, epochs, train_loss, train_acc, test_loss, test_acc)
        
        # Save the model
        self.save_model()
        print(f'Model saved to {self.model_save_path}')
    
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies
        }, self.model_save_path)
    
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_save_path
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training history if available
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
                self.train_accuracies = checkpoint['train_accuracies']
                self.test_losses = checkpoint['test_losses']
                self.test_accuracies = checkpoint['test_accuracies']
            
            print(f'Model loaded from {model_path}')
            return True
        else:
            print(f'No model found at {model_path}')
            return False
    
    def plot_training_history(self):
        if not self.train_losses:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.test_losses, label='Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Test Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.test_accuracies, label='Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Test Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
      def predict(self, image):
        """Predict a single image"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image for consistent preprocessing
                from PIL import Image as PILImage
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = PILImage.fromarray(image, mode='L')
                
                # Apply the inference transform (without rotation/flip)
                image_tensor = self.inference_transform(pil_image)
            else:
                image_tensor = image
            
            if len(image_tensor.shape) == 2:
                image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            elif len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            image_tensor = image_tensor.to(self.device)
            output = self.model(image_tensor)
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Convert back to letter (0-25 -> A-Z)
            letter_idx = predicted.cpu().numpy()[0]
            letter = chr(ord('A') + letter_idx)
            confidence_value = confidence.cpu().numpy()[0]
            
            return letter, confidence_value

if __name__ == "__main__":
    trainer = EMNISTTrainer()
    trainer.train(epochs=20)
