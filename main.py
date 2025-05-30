import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import torch
from torchvision import transforms
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                           QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
                           QSpinBox, QDoubleSpinBox, QPushButton, QTextEdit,
                           QProgressBar, QFileDialog, QMessageBox, QSlider,
                           QFrame, QGroupBox, QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QPixmap, QFont
from trainer import MNISTTrainer

class TrainingThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(list)
    
    def __init__(self, trainer, params):
        super().__init__()
        self.trainer = trainer
        self.params = params
    
    def run(self):
        results = self.trainer.train_with_kfold(
            k_folds=self.params['k_folds'],
            epochs=self.params['epochs'],
            learning_rate=self.params['learning_rate'],
            batch_size=self.params['batch_size'],
            hidden_size=self.params['hidden_size'],
            dropout_rate=self.params['dropout_rate'],
            progress_callback=self.progress_signal.emit
        )
        self.finished_signal.emit(results)

class TrainingTab(QWidget):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.training_thread = None
        self.init_ui()
    
    def init_ui(self):
        layout = QHBoxLayout()
        
        # Left panel - Hyperparameters
        left_panel = QGroupBox("Hyperparameters")
        left_layout = QVBoxLayout()
        
        # K-Folds
        kfolds_layout = QHBoxLayout()
        kfolds_layout.addWidget(QLabel("K-Folds:"))
        self.kfolds_spin = QSpinBox()
        self.kfolds_spin.setRange(2, 10)
        self.kfolds_spin.setValue(5)
        kfolds_layout.addWidget(self.kfolds_spin)
        left_layout.addLayout(kfolds_layout)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        left_layout.addLayout(epochs_layout)
        
        # Learning Rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.0001)
        lr_layout.addWidget(self.lr_spin)
        left_layout.addLayout(lr_layout)
        
        # Batch Size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_combo = QComboBox()
        self.batch_combo.addItems(["16", "32", "64", "128", "256"])
        self.batch_combo.setCurrentText("64")
        batch_layout.addWidget(self.batch_combo)
        left_layout.addLayout(batch_layout)
        
        # Hidden Size
        hidden_layout = QHBoxLayout()
        hidden_layout.addWidget(QLabel("Hidden Size:"))
        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(32, 512)
        self.hidden_spin.setValue(128)
        self.hidden_spin.setSingleStep(32)
        hidden_layout.addWidget(self.hidden_spin)
        left_layout.addLayout(hidden_layout)
        
        # Dropout Rate
        dropout_layout = QHBoxLayout()
        dropout_layout.addWidget(QLabel("Dropout Rate:"))
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setValue(0.2)
        self.dropout_spin.setSingleStep(0.1)
        dropout_layout.addWidget(self.dropout_spin)
        left_layout.addLayout(dropout_layout)
        
        # Train Button
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        left_layout.addWidget(self.train_button)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)
        
        # Save/Load Model Buttons
        save_load_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Model")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        save_load_layout.addWidget(self.save_button)
        
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        save_load_layout.addWidget(self.load_button)
        left_layout.addLayout(save_load_layout)
        
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(300)
        
        # Right panel - Results
        right_panel = QGroupBox("Training Results")
        right_layout = QVBoxLayout()
        
        # Training Log
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        right_layout.addWidget(QLabel("Training Log:"))
        right_layout.addWidget(self.log_text)
          # Plots
        self.figure = Figure(figsize=(16, 12))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        
        right_panel.setLayout(right_layout)
        
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)
        self.setLayout(layout)
    
    def start_training(self):
        if self.training_thread and self.training_thread.isRunning():
            return
        
        params = {
            'k_folds': self.kfolds_spin.value(),
            'epochs': self.epochs_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'batch_size': int(self.batch_combo.currentText()),
            'hidden_size': self.hidden_spin.value(),
            'dropout_rate': self.dropout_spin.value()
        }
        
        self.train_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.log_text.clear()
        
        self.training_thread = TrainingThread(self.trainer, params)
        self.training_thread.progress_signal.connect(self.update_progress)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()
    
    def update_progress(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def training_finished(self, results):
        self.train_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        
        self.plot_results(results)
        
        # Test the final model
        test_loss, test_accuracy = self.trainer.test_model()
        if test_loss is not None:
            self.log_text.append(f"\nFinal Test Results:")
            self.log_text.append(f"Test Loss: {test_loss:.4f}")
            self.log_text.append(f"Test Accuracy: {test_accuracy:.2f}%")
    def plot_results(self, results):
        self.figure.clear()
        
        # Create a larger figure with more subplots (3x3 grid)
        gs = self.figure.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Calculate mean metrics across folds
        mean_train_losses, mean_val_losses = self.calculate_mean_metrics(results, 'losses')
        mean_train_accs, mean_val_accs = self.calculate_mean_metrics(results, 'accuracies')
        
        epochs = range(1, len(mean_train_losses) + 1)
        
        # 1. Individual fold training losses
        ax1 = self.figure.add_subplot(gs[0, 0])
        for result in results:
            fold = result['fold']
            epochs_fold = range(1, len(result['train_losses']) + 1)
            ax1.plot(epochs_fold, result['train_losses'], alpha=0.7, label=f'Fold {fold}')
        ax1.plot(epochs, mean_train_losses, 'k-', linewidth=3, label='Mean')
        ax1.set_title('Training Loss by Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Individual fold validation losses
        ax2 = self.figure.add_subplot(gs[0, 1])
        for result in results:
            fold = result['fold']
            epochs_fold = range(1, len(result['val_losses']) + 1)
            ax2.plot(epochs_fold, result['val_losses'], alpha=0.7, label=f'Fold {fold}')
        ax2.plot(epochs, mean_val_losses, 'k-', linewidth=3, label='Mean')
        ax2.set_title('Validation Loss by Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Mean loss comparison
        ax3 = self.figure.add_subplot(gs[0, 2])
        ax3.plot(epochs, mean_train_losses, 'b-', linewidth=2, label='Train Mean')
        ax3.plot(epochs, mean_val_losses, 'r-', linewidth=2, label='Val Mean')
        ax3.fill_between(epochs, mean_train_losses, alpha=0.3, color='blue')
        ax3.fill_between(epochs, mean_val_losses, alpha=0.3, color='red')
        ax3.set_title('Mean Loss Comparison')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Individual fold training accuracies
        ax4 = self.figure.add_subplot(gs[1, 0])
        for result in results:
            fold = result['fold']
            epochs_fold = range(1, len(result['train_accuracies']) + 1)
            ax4.plot(epochs_fold, result['train_accuracies'], alpha=0.7, label=f'Fold {fold}')
        ax4.plot(epochs, mean_train_accs, 'k-', linewidth=3, label='Mean')
        ax4.set_title('Training Accuracy by Epoch')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Individual fold validation accuracies
        ax5 = self.figure.add_subplot(gs[1, 1])
        for result in results:
            fold = result['fold']
            epochs_fold = range(1, len(result['val_accuracies']) + 1)
            ax5.plot(epochs_fold, result['val_accuracies'], alpha=0.7, label=f'Fold {fold}')
        ax5.plot(epochs, mean_val_accs, 'k-', linewidth=3, label='Mean')
        ax5.set_title('Validation Accuracy by Epoch')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Accuracy (%)')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Mean accuracy comparison
        ax6 = self.figure.add_subplot(gs[1, 2])
        ax6.plot(epochs, mean_train_accs, 'b-', linewidth=2, label='Train Mean')
        ax6.plot(epochs, mean_val_accs, 'r-', linewidth=2, label='Val Mean')
        ax6.fill_between(epochs, mean_train_accs, alpha=0.3, color='blue')
        ax6.fill_between(epochs, mean_val_accs, alpha=0.3, color='red')
        ax6.set_title('Mean Accuracy Comparison')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Accuracy (%)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Final fold performance summary
        ax7 = self.figure.add_subplot(gs[2, 0])
        final_train_accs = [result['train_accuracies'][-1] for result in results]
        final_val_accs = [result['val_accuracies'][-1] for result in results]
        fold_nums = [result['fold'] for result in results]
        
        x = np.arange(len(fold_nums))
        width = 0.35
        ax7.bar(x - width/2, final_train_accs, width, label='Train Acc', alpha=0.8)
        ax7.bar(x + width/2, final_val_accs, width, label='Val Acc', alpha=0.8)
        ax7.set_title('Final Accuracy by Fold')
        ax7.set_xlabel('Fold')
        ax7.set_ylabel('Accuracy (%)')
        ax7.set_xticks(x)
        ax7.set_xticklabels([f'F{f}' for f in fold_nums])
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Hyperparameter summary
        ax8 = self.figure.add_subplot(gs[2, 1])
        ax8.axis('off')
        params = self.get_current_hyperparameters()
        param_text = self.format_hyperparameters(params, results)
        ax8.text(0.1, 0.9, param_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax8.set_title('Training Configuration & Results')
        
        # 9. Learning curves with error bands
        ax9 = self.figure.add_subplot(gs[2, 2])
        val_accs_std = self.calculate_std_metrics(results, 'val_accuracies')
        ax9.plot(epochs, mean_val_accs, 'r-', linewidth=2, label='Val Mean')
        ax9.fill_between(epochs, 
                        np.array(mean_val_accs) - np.array(val_accs_std),
                        np.array(mean_val_accs) + np.array(val_accs_std),
                        alpha=0.3, color='red', label='±1 Std Dev')
        ax9.set_title('Validation Accuracy with Error Bands')
        ax9.set_xlabel('Epoch')
        ax9.set_ylabel('Accuracy (%)')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def calculate_mean_metrics(self, results, metric_type):
        """Calculate mean metrics across all folds"""
        if metric_type == 'losses':
            train_key, val_key = 'train_losses', 'val_losses'
        else:  # accuracies
            train_key, val_key = 'train_accuracies', 'val_accuracies'
        
        # Get the minimum number of epochs across all folds
        min_epochs = min(len(result[train_key]) for result in results)
        
        # Calculate means
        mean_train = []
        mean_val = []
        
        for epoch in range(min_epochs):
            train_values = [result[train_key][epoch] for result in results]
            val_values = [result[val_key][epoch] for result in results]
            mean_train.append(np.mean(train_values))
            mean_val.append(np.mean(val_values))
        
        return mean_train, mean_val
    
    def calculate_std_metrics(self, results, metric_key):
        """Calculate standard deviation of metrics across folds"""
        min_epochs = min(len(result[metric_key]) for result in results)
        std_values = []
        
        for epoch in range(min_epochs):
            values = [result[metric_key][epoch] for result in results]
            std_values.append(np.std(values))
        
        return std_values
    
    def get_current_hyperparameters(self):
        """Get current hyperparameter values from UI"""
        return {
            'k_folds': self.kfolds_spin.value(),
            'epochs': self.epochs_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'batch_size': int(self.batch_combo.currentText()),
            'hidden_size': self.hidden_spin.value(),
            'dropout_rate': self.dropout_spin.value()
        }
    
    def format_hyperparameters(self, params, results):
        """Format hyperparameters and results for display"""
        # Calculate summary statistics
        final_val_accs = [result['val_accuracies'][-1] for result in results]
        final_train_accs = [result['train_accuracies'][-1] for result in results]
        
        text = "HYPERPARAMETERS:\n"
        text += f"K-Folds: {params['k_folds']}\n"
        text += f"Epochs: {params['epochs']}\n"
        text += f"Learning Rate: {params['learning_rate']:.4f}\n"
        text += f"Batch Size: {params['batch_size']}\n"
        text += f"Hidden Size: {params['hidden_size']}\n"
        text += f"Dropout Rate: {params['dropout_rate']:.2f}\n\n"
        
        text += "RESULTS SUMMARY:\n"
        text += f"Mean Val Acc: {np.mean(final_val_accs):.2f}±{np.std(final_val_accs):.2f}%\n"
        text += f"Mean Train Acc: {np.mean(final_train_accs):.2f}±{np.std(final_train_accs):.2f}%\n"
        text += f"Best Val Acc: {np.max(final_val_accs):.2f}%\n"
        text += f"Worst Val Acc: {np.min(final_val_accs):.2f}%\n"
        text += f"Variance: {np.var(final_val_accs):.2f}\n"
        
        return text
    
    def save_model(self):
        if self.trainer.model is None:
            QMessageBox.warning(self, "Warning", "No trained model to save!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "", "PyTorch Model (*.pth)"
        )
        if filename:
            try:
                self.trainer.save_model(filename)
                QMessageBox.information(self, "Success", f"Model saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "PyTorch Model (*.pth)"
        )
        if filename:
            try:
                self.trainer.load_model(filename)
                QMessageBox.information(self, "Success", f"Model loaded from {filename}")
                self.save_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")

class TestTab(QWidget):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.current_image_idx = 0
        self.test_images = []
        self.test_labels = []
        self.init_ui()
        self.load_test_images()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Top section - Image display and controls
        top_layout = QHBoxLayout()
        
        # Left side - Image display
        image_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setMinimumSize(280, 280)
        self.image_label.setStyleSheet("border: 2px solid black; background-color: white;")
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label)
        
        # Image carousel controls
        carousel_layout = QHBoxLayout()
        self.prev_button = QPushButton("◀ Previous")
        self.prev_button.clicked.connect(self.prev_image)
        carousel_layout.addWidget(self.prev_button)
        
        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.valueChanged.connect(self.slider_changed)
        carousel_layout.addWidget(self.image_slider)
        
        self.next_button = QPushButton("Next ▶")
        self.next_button.clicked.connect(self.next_image)
        carousel_layout.addWidget(self.next_button)
        
        image_layout.addLayout(carousel_layout)
        
        # Image info
        self.image_info_label = QLabel("Image: 0/0")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_info_label)
        
        top_layout.addLayout(image_layout)
        
        # Right side - Prediction results
        results_layout = QVBoxLayout()
        results_group = QGroupBox("Prediction Results")
        results_inner_layout = QVBoxLayout()
        
        # Test button
        self.test_button = QPushButton("Test Current Image")
        self.test_button.clicked.connect(self.test_current_image)
        results_inner_layout.addWidget(self.test_button)
        
        # Results display
        self.prediction_label = QLabel("Prediction: -")
        self.prediction_label.setFont(QFont("Arial", 14, QFont.Bold))
        results_inner_layout.addWidget(self.prediction_label)
        
        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setFont(QFont("Arial", 12))
        results_inner_layout.addWidget(self.confidence_label)
        
        self.actual_label = QLabel("Actual: -")
        self.actual_label.setFont(QFont("Arial", 12))
        results_inner_layout.addWidget(self.actual_label)
        
        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 12, QFont.Bold))
        results_inner_layout.addWidget(self.result_label)
        
        # Load custom image button
        self.load_image_button = QPushButton("Load Custom Image")
        self.load_image_button.clicked.connect(self.load_custom_image)
        results_inner_layout.addWidget(self.load_image_button)
        
        results_group.setLayout(results_inner_layout)
        results_layout.addWidget(results_group)
        results_layout.addStretch()
        
        top_layout.addLayout(results_layout)
        
        layout.addLayout(top_layout)
        
        # Bottom section - Probability distribution
        prob_group = QGroupBox("Probability Distribution")
        prob_layout = QVBoxLayout()
        
        self.prob_figure = Figure(figsize=(10, 4))
        self.prob_canvas = FigureCanvas(self.prob_figure)
        prob_layout.addWidget(self.prob_canvas)
        
        prob_group.setLayout(prob_layout)
        layout.addWidget(prob_group)
        
        self.setLayout(layout)
    
    def load_test_images(self):
        # Load some test images from the MNIST test set
        test_dataset = self.trainer.test_dataset
        
        # Take first 100 test images for the carousel
        for i in range(min(100, len(test_dataset))):
            image, label = test_dataset[i]
            self.test_images.append(image)
            self.test_labels.append(label)
        
        if self.test_images:
            self.image_slider.setRange(0, len(self.test_images) - 1)
            self.image_slider.setValue(0)
            self.update_image_display()
    
    def update_image_display(self):
        if not self.test_images:
            return
        
        # Get current image
        image_tensor = self.test_images[self.current_image_idx]
        
        # Convert to numpy and scale to 0-255
        image_np = image_tensor.squeeze().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # Convert to QPixmap
        height, width = image_np.shape
        q_image = torch.zeros((height, width, 3), dtype=torch.uint8)
        q_image[:, :, 0] = torch.tensor(image_np)
        q_image[:, :, 1] = torch.tensor(image_np)
        q_image[:, :, 2] = torch.tensor(image_np)
        
        # Scale up the image for better visibility
        q_image = q_image.numpy()
        pil_image = Image.fromarray(q_image)
        pil_image = pil_image.resize((224, 224), Image.NEAREST)
        
        # Convert to QPixmap
        pil_image.save("temp_image.png")
        pixmap = QPixmap("temp_image.png")
        self.image_label.setPixmap(pixmap)
        
        # Update info
        actual_label = self.test_labels[self.current_image_idx]
        self.image_info_label.setText(f"Image: {self.current_image_idx + 1}/{len(self.test_images)}")
        self.actual_label.setText(f"Actual: {actual_label}")
        
        # Clear previous predictions
        self.prediction_label.setText("Prediction: -")
        self.confidence_label.setText("Confidence: -")
        self.result_label.setText("")
        self.clear_probability_plot()
    
    def prev_image(self):
        if self.test_images and self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.image_slider.setValue(self.current_image_idx)
            self.update_image_display()
    
    def next_image(self):
        if self.test_images and self.current_image_idx < len(self.test_images) - 1:
            self.current_image_idx += 1
            self.image_slider.setValue(self.current_image_idx)
            self.update_image_display()
    
    def slider_changed(self, value):
        self.current_image_idx = value
        self.update_image_display()
    
    def test_current_image(self):
        if not self.test_images or self.trainer.model is None:
            QMessageBox.warning(self, "Warning", "No model loaded or no image selected!")
            return
        
        # Get current image
        image_tensor = self.test_images[self.current_image_idx]
        
        # Make prediction
        prediction, confidence = self.trainer.predict_single(image_tensor)
        
        if prediction is not None:
            actual = self.test_labels[self.current_image_idx]
            
            self.prediction_label.setText(f"Prediction: {prediction}")
            self.confidence_label.setText(f"Confidence: {confidence:.2%}")
            
            if prediction == actual:
                self.result_label.setText("✓ CORRECT")
                self.result_label.setStyleSheet("color: green;")
            else:
                self.result_label.setText("✗ INCORRECT")
                self.result_label.setStyleSheet("color: red;")
            
            # Plot probability distribution
            self.plot_probabilities(image_tensor)
    
    def plot_probabilities(self, image_tensor):
        if self.trainer.model is None:
            return
        
        self.trainer.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.trainer.device)
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            output = self.trainer.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        self.prob_figure.clear()
        ax = self.prob_figure.add_subplot(1, 1, 1)
        
        classes = list(range(10))
        bars = ax.bar(classes, probabilities)
        
        # Highlight the predicted class
        max_idx = np.argmax(probabilities)
        bars[max_idx].set_color('red')
        
        ax.set_xlabel('Digit Class')
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')
        ax.set_xticks(classes)
        ax.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            if prob > 0.01:  # Only show labels for probabilities > 1%
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{prob:.1%}', ha='center', va='bottom', fontsize=8)
        
        self.prob_figure.tight_layout()
        self.prob_canvas.draw()
    
    def clear_probability_plot(self):
        self.prob_figure.clear()
        ax = self.prob_figure.add_subplot(1, 1, 1)
        ax.set_xlabel('Digit Class')
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')
        ax.set_xticks(list(range(10)))
        ax.grid(True, alpha=0.3)
        self.prob_canvas.draw()
    
    def load_custom_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if filename:
            try:
                # Load and preprocess the image
                image = Image.open(filename).convert('L')  # Convert to grayscale
                image = image.resize((28, 28))  # Resize to MNIST size
                
                # Convert to tensor and normalize
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                
                image_tensor = transform(image)
                
                # Add to the beginning of test images
                self.test_images.insert(0, image_tensor)
                self.test_labels.insert(0, -1)  # Unknown label for custom image
                
                # Update slider range and display
                self.image_slider.setRange(0, len(self.test_images) - 1)
                self.current_image_idx = 0
                self.image_slider.setValue(0)
                self.update_image_display()
                
                # Update actual label for custom image
                self.actual_label.setText("Actual: Custom Image")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

class MNISTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.trainer = MNISTTrainer()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("MNIST Neural Network Classifier")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create tabs
        self.training_tab = TrainingTab(self.trainer)
        self.test_tab = TestTab(self.trainer)
        
        # Add tabs
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.test_tab, "Testing")
        
        self.setCentralWidget(self.tabs)

def main():
    app = QApplication(sys.argv)
    window = MNISTApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
