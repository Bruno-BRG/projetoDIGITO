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
import cv2
from PIL import Image as PILImage

class EMNISTTrainer:
    def __init__(self, model_save_path='emnist_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EMNISTNet(num_classes=26).to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
        
        # Better optimizer with improved parameters
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.001, 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        self.model_save_path = model_save_path
        
        # Enhanced data transformations for EMNIST letters with better augmentation
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.rot90(x, k=3, dims=[1, 2])),  # Rotate 270 degrees
            transforms.Lambda(lambda x: torch.flip(x, dims=[2])),  # Flip horizontally
            # More aggressive augmentation for better generalization
            transforms.RandomRotation(degrees=15, fill=0),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.15, 0.15), 
                scale=(0.85, 1.15),
                shear=10,
                fill=0
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.1),
            transforms.RandomApply([
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)  # Add noise
            ], p=0.1),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Test transform without augmentation
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.rot90(x, k=3, dims=[1, 2])),  # Rotate 270 degrees
            transforms.Lambda(lambda x: torch.flip(x, dims=[2])),  # Flip horizontally
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Transform for user drawn images with better preprocessing
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
            transform=self.train_transform
        )
        
        self.test_dataset = torchvision.datasets.EMNIST(
            root='./data',
            split='letters', 
            train=False, 
            download=True, 
            transform=self.test_transform
        )
        
        # Increase batch size for better gradient estimates
        self.train_loader = DataLoader(self.train_dataset, batch_size=256, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=256, shuffle=False, num_workers=2)
        
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
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
    
    def train(self, epochs=30, progress_callback=None):
        print(f'Training on {self.device}')
        print(f'Training set size: {len(self.train_dataset)}')
        print(f'Test set size: {len(self.test_dataset)}')
        
        best_test_acc = 0.0
        patience_counter = 0
        patience = 8  # Early stopping patience
        
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
            print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model and early stopping
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
                self.save_model()
                print(f'  New best model saved with test accuracy: {best_test_acc:.2f}%')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(epoch + 1, epochs, train_loss, train_acc, test_loss, test_acc)
        
        print(f'Training completed. Best test accuracy: {best_test_acc:.2f}%')
    
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
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
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
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
    
    def preprocess_user_image(self, image_array):
        """Enhanced preprocessing for user-drawn images"""
        # Ensure the image is in the right format
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Apply stronger preprocessing
        # 1. Noise reduction
        image_array = cv2.medianBlur(image_array, 3)
        
        # 2. Morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        image_array = cv2.morphologyEx(image_array, cv2.MORPH_CLOSE, kernel)
        image_array = cv2.morphologyEx(image_array, cv2.MORPH_OPEN, kernel)
        
        # 3. Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        image_array = clahe.apply(image_array)
        
        # 4. Center the character using moments
        moments = cv2.moments(image_array)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Calculate shift to center
            h, w = image_array.shape
            shift_x = w // 2 - cx
            shift_y = h // 2 - cy
            
            # Apply translation
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            image_array = cv2.warpAffine(image_array, M, (w, h))
        
        # 5. Normalize thickness - make strokes more consistent
        image_array = cv2.GaussianBlur(image_array, (2, 2), 0)
        
        return image_array
    
    def predict(self, image):
        """Predict a single image with enhanced preprocessing"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                # Enhanced preprocessing for user images
                processed_image = self.preprocess_user_image(image)
                
                # Convert to PIL Image for consistent preprocessing
                if processed_image.dtype != np.uint8:
                    processed_image = (processed_image * 255).astype(np.uint8)
                pil_image = PILImage.fromarray(processed_image, mode='L')
                
                # Apply the inference transform
                image_tensor = self.inference_transform(pil_image)
            else:
                image_tensor = image
            
            if len(image_tensor.shape) == 2:
                image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            elif len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            image_tensor = image_tensor.to(self.device)
            
            # Test-time augmentation for better accuracy
            predictions = []
            augmentations = [
                lambda x: x,  # Original
                lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # Rotate 90°
                lambda x: torch.rot90(x, k=-1, dims=[2, 3]), # Rotate -90°
                lambda x: torch.flip(x, dims=[3]),  # Horizontal flip
            ]
            
            for aug in augmentations:
                aug_tensor = aug(image_tensor)
                output = self.model(aug_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predictions.append(probabilities)
            
            # Average predictions from all augmentations
            avg_probabilities = torch.mean(torch.stack(predictions), dim=0)
            confidence, predicted = torch.max(avg_probabilities, 1)
            
            # Convert back to letter (0-25 -> A-Z)
            letter_idx = predicted.cpu().numpy()[0]
            letter = chr(ord('A') + letter_idx)
            confidence_value = confidence.cpu().numpy()[0]
            
            return letter, confidence_value

if __name__ == "__main__":
    trainer = EMNISTTrainer()
    trainer.train(epochs=30)
