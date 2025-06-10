#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST Digit Recognition - Versao Ultra Simplificada
Porque as vezes simples e melhor que complexo

Criado por um senior que sabe que menos e mais
"""

import sys
import numpy as np
import cv2

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QPushButton, QLabel, QTabWidget,
    QProgressBar, QGroupBox, QGridLayout, QSpinBox,
    QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QPainter, QPen, QPixmap, QFont

# Nossos modulos
from model import MNISTNet
from trainer import MNISTTrainer

# Matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TrainingThread(QThread):
    """Thread para treinamento - nao trava a UI"""
    progress_signal = pyqtSignal(int, int, float, float, float, float)
    finished_signal = pyqtSignal()
    
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.trainer = None
    
    def run(self):
        """Executa o treinamento em thread separada"""
        try:
            self.trainer = MNISTTrainer()
            self.trainer.train(epochs=self.epochs, progress_callback=self.emit_progress)
            self.finished_signal.emit()
        except Exception as e:
            print(f"Erro no treinamento: {e}")
    
    def emit_progress(self, epoch, total_epochs, train_loss, train_acc, test_loss, test_acc):
        """Emite sinal de progresso"""
        self.progress_signal.emit(epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)


class DrawingCanvas(QWidget):
    """Canvas para desenhar digitos - o mais simples possivel"""
    
    def __init__(self, width=280, height=280):
        super().__init__()
        self.width = width
        self.height = height
        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: white; border: 2px solid black;")
        
        # Estados do desenho
        self.drawing = False
        self.brush_size = 15
        self.last_point = QPoint()
        
        # Pixmap para desenho
        self.pixmap = QPixmap(width, height)
        self.pixmap.fill(Qt.white)
    
    def mousePressEvent(self, event):
        """Inicia o desenho"""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
    
    def mouseMoveEvent(self, event):
        """Desenha durante o movimento"""
        if event.buttons() & Qt.LeftButton and self.drawing:
            painter = QPainter(self.pixmap)
            painter.setPen(QPen(Qt.black, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Para o desenho"""
        if event.button() == Qt.LeftButton:
            self.drawing = False
    
    def paintEvent(self, event):
        """Renderiza o canvas"""
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap, self.pixmap.rect())
    
    def clear_canvas(self):
        """Limpa o canvas"""
        self.pixmap.fill(Qt.white)
        self.update()
    
    def get_image_array(self):
        """Retorna imagem como array numpy 28x28 - preprocessado para MNIST"""
        # Converter QPixmap para QImage
        qimg = self.pixmap.toImage()
        width = qimg.width()
        height = qimg.height()
        
        # Converter para array numpy
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # RGBA
        
        # Converter para grayscale
        gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        
        # Inverter cores (MNIST tem fundo preto, digito branco)
        gray = 255 - gray
        
        # Resize para 28x28
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalizar
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized


class PlotCanvas(FigureCanvas):
    """Canvas para plots do treinamento"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Criar subplots
        self.loss_ax = self.fig.add_subplot(121)
        self.acc_ax = self.fig.add_subplot(122)
        
        # Dados dos plots
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
    
    def update_plots(self, train_losses, train_accs, test_losses, test_accs):
        """Atualiza os graficos de treinamento"""
        # Atualizar dados
        self.train_losses = train_losses
        self.train_accs = train_accs
        self.test_losses = test_losses
        self.test_accs = test_accs
        
        # Limpar plots anteriores
        self.loss_ax.clear()
        self.acc_ax.clear()
        
        # Plot Loss
        epochs = range(1, len(train_losses) + 1)
        self.loss_ax.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
        self.loss_ax.plot(epochs, test_losses, 'r-', label='Test', linewidth=2)
        self.loss_ax.set_title('Loss', fontsize=12, fontweight='bold')
        self.loss_ax.set_xlabel('Epoca')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.legend()
        self.loss_ax.grid(True, alpha=0.3)
        
        # Plot Accuracy
        self.acc_ax.plot(epochs, train_accs, 'b-', label='Train', linewidth=2)
        self.acc_ax.plot(epochs, test_accs, 'r-', label='Test', linewidth=2)
        self.acc_ax.set_title('Accuracy', fontsize=12, fontweight='bold')
        self.acc_ax.set_xlabel('Epoca')
        self.acc_ax.set_ylabel('Accuracy (%)')
        self.acc_ax.legend()
        self.acc_ax.grid(True, alpha=0.3)
        
        # Ajustar layout e redesenhar
        self.fig.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    """Interface principal - limpa e funcional"""
    
    def __init__(self):
        super().__init__()
        
        # Configurar janela principal
        self.setWindowTitle("🎯 MNIST Digit Recognition - Versao Ultra Simples")
        self.setGeometry(100, 100, 1000, 700)
        
        # Inicializar componentes
        self.model = None
        self.trainer = MNISTTrainer(model_save_path='mnist_model.pth')
        self.current_prediction = None
        self.digit_sequence = []
        
        # Tentar carregar modelo existente
        self.load_existing_model()
        
        # Criar interface
        self.setup_ui()
        
        # Aplicar estilo
        self.setStyleSheet(self.get_stylesheet())
    
    def setup_ui(self):
        """Configura a interface do usuario"""
        # Widget principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Layout principal
        layout = QVBoxLayout(main_widget)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Criar abas
        self.create_training_tab()
        self.create_recognition_tab()
    
    def get_stylesheet(self):
        """CSS simples e bonito"""
        return """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #ccc;
                background-color: white;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 3px solid #007acc;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #005999;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                font-size: 11px;
            }
        """
    
    def create_training_tab(self):
        """Cria a aba de treinamento"""
        training_tab = QWidget()
        self.tab_widget.addTab(training_tab, "🚀 Treinamento")
        
        layout = QVBoxLayout(training_tab)
        
        # Grupo de controles
        controls_group = QGroupBox("Parametros de Treinamento")
        controls_layout = QGridLayout(controls_group)
        
        # Numero de epocas
        controls_layout.addWidget(QLabel("Epocas:"), 0, 0)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 50)
        self.epochs_spinbox.setValue(10)
        controls_layout.addWidget(self.epochs_spinbox, 0, 1)
        
        # Status do modelo
        self.model_status_label = QLabel()
        self.update_model_status()
        controls_layout.addWidget(self.model_status_label, 1, 0, 1, 2)
        
        # Botao de treinamento
        self.train_button = QPushButton("Iniciar Treinamento")
        self.train_button.clicked.connect(self.start_training)
        controls_layout.addWidget(self.train_button, 2, 0, 1, 2)
        
        layout.addWidget(controls_group)
        
        # Barra de progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Status de treinamento
        self.training_status = QLabel("Pronto para treinar")
        layout.addWidget(self.training_status)
        
        # Canvas de plots
        self.plot_canvas = PlotCanvas()
        layout.addWidget(self.plot_canvas)
    
    def create_recognition_tab(self):
        """Cria a aba de reconhecimento"""
        recognition_tab = QWidget()
        self.tab_widget.addTab(recognition_tab, "🎨 Reconhecimento")
        
        layout = QHBoxLayout(recognition_tab)
        
        # Grupo do canvas de desenho
        canvas_group = QGroupBox("Desenhe um digito (0-9)")
        canvas_layout = QVBoxLayout(canvas_group)
        
        # Canvas de desenho
        self.drawing_canvas = DrawingCanvas()
        canvas_layout.addWidget(self.drawing_canvas)
        
        # Botoes do canvas
        canvas_buttons = QHBoxLayout()
        
        clear_button = QPushButton("🗑️ Limpar")
        clear_button.clicked.connect(self.clear_drawing)
        canvas_buttons.addWidget(clear_button)
        
        recognize_button = QPushButton("🔍 Reconhecer")
        recognize_button.clicked.connect(self.recognize_digit)
        canvas_buttons.addWidget(recognize_button)
        
        canvas_layout.addLayout(canvas_buttons)
        layout.addWidget(canvas_group)
        
        # Grupo de resultados
        results_group = QGroupBox("Resultado")
        results_layout = QVBoxLayout(results_group)
        
        # Label de predicao
        self.prediction_label = QLabel("Desenhe um digito e clique em 'Reconhecer'")
        self.prediction_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("color: #333; padding: 20px;")
        results_layout.addWidget(self.prediction_label)
        
        # Label de confianca
        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setFont(QFont("Arial", 12))
        results_layout.addWidget(self.confidence_label)
        
        # Sequencia de digitos
        self.word_label = QLabel("Digitos: ")
        self.word_label.setFont(QFont("Arial", 16))
        self.word_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        results_layout.addWidget(self.word_label)
        
        # Botoes da sequencia
        word_buttons = QHBoxLayout()
        
        add_digit_button = QPushButton("➕ Adicionar")
        add_digit_button.clicked.connect(self.add_digit_to_sequence)
        word_buttons.addWidget(add_digit_button)
        
        clear_word_button = QPushButton("🗑️ Limpar Sequencia")
        clear_word_button.clicked.connect(self.clear_sequence)
        word_buttons.addWidget(clear_word_button)
        
        results_layout.addLayout(word_buttons)
        layout.addWidget(results_group)
    
    def load_existing_model(self):
        """Carrega modelo existente se disponivel"""
        try:
            if self.trainer.load_model():
                self.model = self.trainer.model
                print("✅ Modelo carregado com sucesso!")
            else:
                print("ℹ️ Nenhum modelo salvo encontrado")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
    
    def update_model_status(self):
        """Atualiza o status do modelo"""
        if self.model is not None:
            self.model_status_label.setText("✅ Modelo carregado e pronto")
            self.model_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.model_status_label.setText("❌ Nenhum modelo treinado")
            self.model_status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def start_training(self):
        """Inicia o treinamento"""
        epochs = self.epochs_spinbox.value()
        
        # Desabilitar botao
        self.train_button.setEnabled(False)
        self.training_status.setText(f"Iniciando treinamento por {epochs} epocas...")
        
        # Criar e iniciar thread de treinamento
        self.training_thread = TrainingThread(epochs)
        self.training_thread.progress_signal.connect(self.update_training_progress)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()
    
    def update_training_progress(self, epoch, total_epochs, train_loss, train_acc, test_loss, test_acc):
        """Atualiza o progresso do treinamento"""
        # Atualizar barra de progresso
        progress = int((epoch / total_epochs) * 100)
        self.progress_bar.setValue(progress)
        
        # Atualizar status
        status = f"Epoca {epoch}/{total_epochs} - "
        status += f"Train: {train_acc:.1f}% ({train_loss:.4f}) - "
        status += f"Test: {test_acc:.1f}% ({test_loss:.4f})"
        self.training_status.setText(status)
        
        # Atualizar plots se disponivel
        if hasattr(self.training_thread.trainer, 'train_losses'):
            self.plot_canvas.update_plots(
                self.training_thread.trainer.train_losses,
                self.training_thread.trainer.train_accuracies,
                self.training_thread.trainer.test_losses,
                self.training_thread.trainer.test_accuracies
            )
    
    def training_finished(self):
        """Finaliza o treinamento"""
        # Reabilitar botao
        self.train_button.setEnabled(True)
        self.training_status.setText("✅ Treinamento concluido!")
        self.progress_bar.setValue(100)
        
        # Atualizar referencias do modelo
        self.trainer = self.training_thread.trainer
        self.model = self.trainer.model
        self.update_model_status()
        
        # Mostrar mensagem de sucesso
        QMessageBox.information(self, "Sucesso", "Treinamento concluido com sucesso!")
    
    def clear_drawing(self):
        """Limpa o canvas de desenho"""
        self.drawing_canvas.clear_canvas()
        self.prediction_label.setText("Desenhe um digito e clique em 'Reconhecer'")
        self.confidence_label.setText("")
    
    def recognize_digit(self):
        """Reconhece o digito desenhado"""
        if self.model is None:
            QMessageBox.warning(
                self, 
                "Erro", 
                "Nenhum modelo treinado!\n\nVa para a aba 'Treinamento' e treine o modelo primeiro."
            )
            return
        
        # Obter imagem do canvas
        try:
            image = self.drawing_canvas.get_image_array()
            
            # Fazer predicao
            predicted_digit, confidence = self.trainer.predict(image)
            
            # Armazenar predicao atual
            self.current_prediction = predicted_digit
            
            # Mostrar resultado
            self.prediction_label.setText(f"Digito: {predicted_digit}")
            self.confidence_label.setText(f"Confianca: {confidence:.1%}")
            
            # Colorir baseado na confianca
            if confidence > 0.8:
                color = "green"
                emoji = "🎯"
            elif confidence > 0.5:
                color = "orange"
                emoji = "⚠️"
            else:
                color = "red"
                emoji = "❓"
            
            self.prediction_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            self.confidence_label.setText(f"{emoji} Confianca: {confidence:.1%}")
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro na predicao:\n{str(e)}")
    
    def add_digit_to_sequence(self):
        """Adiciona o digito atual a sequencia"""
        if self.current_prediction is not None:
            self.digit_sequence.append(str(self.current_prediction))
            sequence_text = "".join(self.digit_sequence)
            self.word_label.setText(f"Digitos: {sequence_text}")
        else:
            QMessageBox.information(self, "Info", "Reconheca um digito primeiro!")
    
    def clear_sequence(self):
        """Limpa a sequencia de digitos"""
        self.digit_sequence = []
        self.word_label.setText("Digitos: ")


def main():
    """Funcao principal"""
    print("🚀 Iniciando MNIST Digit Recognition...")
    
    # Criar aplicacao
    app = QApplication(sys.argv)
    
    # Configurar fonte padrao
    font = QFont("Arial", 10)
    app.setFont(font)
    
    # Criar e mostrar janela principal
    try:
        window = MainWindow()
        window.show()
        print("✅ Interface carregada com sucesso!")
        
        # Executar aplicacao
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"❌ Erro ao inicializar: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
