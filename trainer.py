import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import numpy as np
import os
from model import MNISTNet

class MNISTTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        self.load_data()
    
    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Check if dataset already exists
        data_dir = './data'
        train_exists = os.path.exists(os.path.join(data_dir, 'MNIST', 'raw', 'train-images-idx3-ubyte'))
        test_exists = os.path.exists(os.path.join(data_dir, 'MNIST', 'raw', 't10k-images-idx3-ubyte'))
        
        # Only download if not already present
        download_needed = not (train_exists and test_exists)
        
        if download_needed:
            print("MNIST dataset not found. Downloading...")
        else:
            print("MNIST dataset found. Loading existing data...")
        
        self.train_dataset = datasets.MNIST(data_dir, train=True, download=download_needed, transform=transform)
        self.test_dataset = datasets.MNIST(data_dir, train=False, download=download_needed, transform=transform)
        
        print(f"Dataset loaded successfully. Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)}")
    
    def create_model(self, hidden_size=128, dropout_rate=0.2):
        self.model = MNISTNet(hidden_size, dropout_rate).to(self.device)
        return self.model
    
    def train_epoch(self, model, train_loader, optimizer, epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        return avg_loss, accuracy
    
    def validate(self, model, val_loader):
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                val_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_loss /= total
        accuracy = 100. * correct / total
        return val_loss, accuracy
    
    def train_with_kfold(self, k_folds=5, epochs=10, learning_rate=0.001, 
                        batch_size=64, hidden_size=128, dropout_rate=0.2, 
                        progress_callback=None):
        
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        results = []
        
        # Get indices for k-fold split
        dataset_size = len(self.train_dataset)
        indices = list(range(dataset_size))
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
            if progress_callback:
                progress_callback(f"Starting fold {fold + 1}/{k_folds}")
            
            # Create data samplers and loaders for this fold
            train_sampler = SubsetRandomSampler(train_ids)
            val_sampler = SubsetRandomSampler(val_ids)
            
            train_loader = DataLoader(self.train_dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(self.train_dataset, batch_size=batch_size, sampler=val_sampler)
            
            # Create new model for this fold
            model = MNISTNet(hidden_size, dropout_rate).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            fold_results = {
                'fold': fold + 1,
                'train_losses': [],
                'train_accuracies': [],
                'val_losses': [],
                'val_accuracies': []
            }
            
            # Training loop for this fold
            for epoch in range(epochs):
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, epoch)
                val_loss, val_acc = self.validate(model, val_loader)
                
                fold_results['train_losses'].append(train_loss)
                fold_results['train_accuracies'].append(train_acc)
                fold_results['val_losses'].append(val_loss)
                fold_results['val_accuracies'].append(val_acc)
                
                if progress_callback:
                    progress_callback(f"Fold {fold + 1}, Epoch {epoch + 1}: "
                                    f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            results.append(fold_results)
            
            # Save the model from the last fold as the final model
            if fold == k_folds - 1:
                self.model = model
        
        return results
    
    def test_model(self, model=None):
        if model is None:
            model = self.model
        
        if model is None:
            return None, None
        test_loader = DataLoader(self.test_dataset, batch_size=1000, shuffle=False)
        test_loss, test_accuracy = self.validate(model, test_loader)
        return test_loss, test_accuracy
    
    def predict_single(self, image_tensor, model=None):
        if model is None:
            model = self.model
        
        if model is None:
            return None, None
        
        model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            output = model(image_tensor)
            prediction = output.argmax(dim=1, keepdim=True).item()
            confidence = F.softmax(output, dim=1).max().item()
            
        return prediction, confidence
    
    def save_model(self, filepath):
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'hidden_size': self.model.fc1.out_features,
                    'dropout_rate': 0.2  # Default value, could be made configurable
                }
            }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        config = checkpoint.get('model_config', {'hidden_size': 128, 'dropout_rate': 0.2})
        
        self.model = MNISTNet(config['hidden_size'], config['dropout_rate']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.model
