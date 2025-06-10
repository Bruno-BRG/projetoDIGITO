import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QTextEdit, QTabWidget,
                             QProgressBar, QGroupBox, QGridLayout, QSpinBox,
                             QCheckBox, QMessageBox, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt5.QtGui import QPainter, QPen, QPixmap, QFont, QColor
import torch
from model import EMNISTNet
from trainer_improved import EMNISTTrainer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import string

class TrainingThread(QThread):
    progress_signal = pyqtSignal(int, int, float, float, float, float)
    finished_signal = pyqtSignal()
    
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.trainer = None
    
    def run(self):
        self.trainer = EMNISTTrainer()
        self.trainer.train(epochs=self.epochs, progress_callback=self.emit_progress)
        self.finished_signal.emit()
    
    def emit_progress(self, epoch, total_epochs, train_loss, train_acc, test_loss, test_acc):
        self.progress_signal.emit(epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)

class DrawingCanvas(QWidget):
    def __init__(self, width=280, height=280):
        super().__init__()
        self.width = width
        self.height = height
        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: white; border: 2px solid black;")
        
        # Drawing parameters
        self.drawing = False
        self.brush_size = 15
        self.last_point = QPoint()
        
        # Create pixmap for drawing
        self.pixmap = QPixmap(width, height)
        self.pixmap.fill(Qt.white)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
    
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.drawing:
            painter = QPainter(self.pixmap)
            painter.setPen(QPen(Qt.black, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
    
    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawPixmap(self.rect(), self.pixmap, self.pixmap.rect())
      def clear_canvas(self):
        self.pixmap.fill(Qt.white)
        self.update()
    
    def get_image_array(self):
        # Convert pixmap to numpy array
        image = self.pixmap.toImage()
        width = image.width()
        height = image.height()
        
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # RGBA
        
        # Convert to grayscale
        gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        
        # Invert colors (black on white to white on black)
        gray = 255 - gray
        
        # Enhanced preprocessing for better recognition
        # 1. Apply Gaussian blur to smooth the image
        gray = cv2.GaussianBlur(gray, (2, 2), 0)
        
        # 2. Threshold to create clean binary image
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # 3. Find contours to get the main character
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (main character)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding around the character
            padding = max(20, min(w, h) // 4)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(width - x, w + 2 * padding)
            h = min(height - y, h + 2 * padding)
            
            # Crop to bounding rectangle
            cropped = binary[y:y+h, x:x+w]
            
            # Make it square by padding with zeros
            max_dim = max(h, w)
            square = np.zeros((max_dim, max_dim), dtype=np.uint8)
            
            # Center the character in the square
            start_y = (max_dim - h) // 2
            start_x = (max_dim - w) // 2
            square[start_y:start_y+h, start_x:start_x+w] = cropped
            
        else:
            # If nothing is drawn, return a black square
            square = np.zeros((100, 100), dtype=np.uint8)
        
        # Resize to 28x28 for EMNIST with anti-aliasing
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Apply morphological operations to improve quality
        kernel = np.ones((2, 2), np.uint8)
        resized = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)
        
        # Final smoothing
        resized = cv2.GaussianBlur(resized, (1, 1), 0)
        
        return resized

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Initialize empty plots
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True)
        
        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy (%)')
        self.ax2.grid(True)
        
        self.fig.tight_layout()
    
    def update_plots(self, train_losses, train_accs, test_losses, test_accs):
        self.ax1.clear()
        self.ax2.clear()
        
        epochs = range(1, len(train_losses) + 1)
        
        # Plot losses
        self.ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        self.ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Plot accuracies
        self.ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
        self.ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy')
        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy (%)')
        self.ax2.legend()
        self.ax2.grid(True)
        
        self.fig.tight_layout()
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMNIST Letter Recognition App")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize model
        self.model = None
        self.trainer = EMNISTTrainer()
        self.load_existing_model()
        
        # Create main widget and tabs
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QVBoxLayout(main_widget)
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_training_tab()
        self.create_recognition_tab()
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #0078d4;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
    
    def create_training_tab(self):
        training_tab = QWidget()
        self.tab_widget.addTab(training_tab, "Training")
        
        layout = QVBoxLayout(training_tab)
        
        # Training controls
        controls_group = QGroupBox("Training Controls")
        controls_layout = QGridLayout(controls_group)
        
        # Epochs selection
        controls_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setMinimum(1)
        self.epochs_spinbox.setMaximum(100)
        self.epochs_spinbox.setValue(20)
        controls_layout.addWidget(self.epochs_spinbox, 0, 1)
        
        # Training button
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        controls_layout.addWidget(self.train_button, 0, 2)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        controls_layout.addWidget(self.progress_bar, 1, 0, 1, 3)
        
        layout.addWidget(controls_group)
        
        # Training info
        info_group = QGroupBox("Training Information")
        info_layout = QVBoxLayout(info_group)
        
        self.training_info = QTextEdit()
        self.training_info.setMaximumHeight(150)
        self.training_info.setReadOnly(True)
        info_layout.addWidget(self.training_info)
        
        layout.addWidget(info_group)
        
        # Plot canvas
        plot_group = QGroupBox("Training Progress")
        plot_layout = QVBoxLayout(plot_group)
        
        self.plot_canvas = PlotCanvas()
        plot_layout.addWidget(self.plot_canvas)
        
        layout.addWidget(plot_group)
    
    def create_recognition_tab(self):
        recognition_tab = QWidget()
        self.tab_widget.addTab(recognition_tab, "Letter Recognition")
        
        layout = QHBoxLayout(recognition_tab)
        
        # Left side - Drawing area
        left_group = QGroupBox("Draw Here")
        left_layout = QVBoxLayout(left_group)
        
        self.drawing_canvas = DrawingCanvas()
        left_layout.addWidget(self.drawing_canvas, alignment=Qt.AlignCenter)
        
        # Controls for drawing
        controls_layout = QHBoxLayout()
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_drawing)
        controls_layout.addWidget(self.clear_button)
        
        self.recognize_button = QPushButton("Recognize Letter")
        self.recognize_button.clicked.connect(self.recognize_letter)
        controls_layout.addWidget(self.recognize_button)
        
        left_layout.addLayout(controls_layout)
        layout.addWidget(left_group)
        
        # Right side - Results
        right_group = QGroupBox("Recognition Results")
        right_layout = QVBoxLayout(right_group)
        
        # Prediction result
        self.prediction_label = QLabel("Draw a letter and click 'Recognize Letter'")
        self.prediction_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("color: #0078d4; padding: 20px; border: 2px solid #0078d4; border-radius: 8px;")
        right_layout.addWidget(self.prediction_label)
        
        # Confidence
        self.confidence_label = QLabel("")
        self.confidence_label.setFont(QFont("Arial", 14))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.confidence_label)
        
        # Word formation
        word_group = QGroupBox("Word Formation")
        word_layout = QVBoxLayout(word_group)
        
        self.word_display = QLabel("")
        self.word_display.setFont(QFont("Arial", 18, QFont.Bold))
        self.word_display.setAlignment(Qt.AlignCenter)
        self.word_display.setStyleSheet("border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9;")
        word_layout.addWidget(self.word_display)
        
        word_controls = QHBoxLayout()
        
        self.add_letter_button = QPushButton("Add to Word")
        self.add_letter_button.clicked.connect(self.add_letter_to_word)
        self.add_letter_button.setEnabled(False)
        word_controls.addWidget(self.add_letter_button)
        
        self.clear_word_button = QPushButton("Clear Word")
        self.clear_word_button.clicked.connect(self.clear_word)
        word_controls.addWidget(self.clear_word_button)
        
        word_layout.addLayout(word_controls)
        right_layout.addWidget(word_group)
        
        # Model status
        self.model_status_label = QLabel()
        self.update_model_status()
        right_layout.addWidget(self.model_status_label)
        
        layout.addWidget(right_group)
        
        # Initialize word tracking
        self.current_word = ""
        self.last_prediction = ""
    
    def load_existing_model(self):
        """Try to load existing model"""
        if self.trainer.load_model():
            self.model = self.trainer.model
            return True
        return False
    
    def update_model_status(self):
        if self.model is not None:
            self.model_status_label.setText("✅ Model loaded and ready")
            self.model_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.model_status_label.setText("❌ No trained model available. Please train the model first.")
            self.model_status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def start_training(self):
        epochs = self.epochs_spinbox.value()
        
        # Disable training button
        self.train_button.setEnabled(False)
        self.epochs_spinbox.setEnabled(False)
        
        # Clear training info
        self.training_info.clear()
        self.training_info.append("Starting training...")
        
        # Reset progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(epochs)
        
        # Start training thread
        self.training_thread = TrainingThread(epochs)
        self.training_thread.progress_signal.connect(self.update_training_progress)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()
    
    def update_training_progress(self, epoch, total_epochs, train_loss, train_acc, test_loss, test_acc):
        # Update progress bar
        self.progress_bar.setValue(epoch)
        
        # Update training info
        info_text = f"Epoch {epoch}/{total_epochs}:\n"
        info_text += f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n"
        info_text += f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n"
        
        self.training_info.append(info_text)
        self.training_info.ensureCursorVisible()
        
        # Update plots if we have training history
        if hasattr(self.training_thread.trainer, 'train_losses'):
            trainer = self.training_thread.trainer
            self.plot_canvas.update_plots(
                trainer.train_losses,
                trainer.train_accuracies,
                trainer.test_losses,
                trainer.test_accuracies
            )
    
    def training_finished(self):
        # Re-enable training controls
        self.train_button.setEnabled(True)
        self.epochs_spinbox.setEnabled(True)
        
        # Update model
        self.model = self.training_thread.trainer.model
        self.trainer = self.training_thread.trainer
        
        # Update model status
        self.update_model_status()
        
        # Show completion message
        self.training_info.append("\n✅ Training completed successfully!")
        self.training_info.append("Model saved and ready for recognition.")
        
        QMessageBox.information(self, "Training Complete", "Model training completed successfully!")
    
    def clear_drawing(self):
        self.drawing_canvas.clear_canvas()
        self.prediction_label.setText("Draw a letter and click 'Recognize Letter'")
        self.confidence_label.setText("")
        self.add_letter_button.setEnabled(False)
    
    def recognize_letter(self):
        if self.model is None:
            QMessageBox.warning(self, "No Model", "Please train the model first before recognition.")
            return
        
        # Get image from canvas
        image_array = self.drawing_canvas.get_image_array()
        
        # Debug: Save the preprocessed image to see what the model receives
        debug_path = "debug_preprocessed.png"
        cv2.imwrite(debug_path, image_array)
        print(f"Debug: Preprocessed image saved to {debug_path}")
        
        try:
            # Predict letter
            letter, confidence = self.trainer.predict(image_array)
            
            # Only show prediction if confidence is reasonable
            if confidence > 0.1:  # 10% minimum confidence
                self.prediction_label.setText(f"Predicted Letter: {letter}")
                self.confidence_label.setText(f"Confidence: {confidence:.2%}")
                
                # Color code based on confidence
                if confidence > 0.7:
                    color = "#28a745"  # Green for high confidence
                elif confidence > 0.4:
                    color = "#ffc107"  # Yellow for medium confidence
                else:
                    color = "#dc3545"  # Red for low confidence
                
                self.prediction_label.setStyleSheet(f"color: {color}; padding: 20px; border: 2px solid {color}; border-radius: 8px;")
                
                # Store prediction for word formation
                self.last_prediction = letter
                self.add_letter_button.setEnabled(True)
            else:
                self.prediction_label.setText("Unable to recognize - try drawing clearer")
                self.confidence_label.setText(f"Confidence too low: {confidence:.2%}")
                self.prediction_label.setStyleSheet("color: #dc3545; padding: 20px; border: 2px solid #dc3545; border-radius: 8px;")
                self.add_letter_button.setEnabled(False)
                
        except Exception as e:
            QMessageBox.critical(self, "Recognition Error", f"Error during recognition: {str(e)}")
            print(f"Recognition error: {str(e)}")
    
    def add_letter_to_word(self):
        if self.last_prediction:
            self.current_word += self.last_prediction
            self.word_display.setText(self.current_word)
    
    def clear_word(self):
        self.current_word = ""
        self.word_display.setText("")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
