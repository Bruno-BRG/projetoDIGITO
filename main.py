#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST Digit Recognition com Aprendizado por Reforco
Sistema de feedback interativo para melhoria continua

Desenvolvido por um senior que entende que IA real precisa de feedback humano
"""

import sys
import numpy as np
import cv2
import json
import os
from datetime import datetime

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QPushButton, QLabel, QTabWidget,
    QProgressBar, QGroupBox, QGridLayout, QSpinBox,
    QMessageBox, QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QTimer
from PyQt5.QtGui import QPainter, QPen, QPixmap, QFont

# Nossos modulos
from model import MNISTNet
from trainer import MNISTTrainer

# Matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class FeedbackManager:
    """Gerencia o feedback interativo do usuario para aprendizado continuo"""
    
    def __init__(self, feedback_file="user_feedback.json"):
        self.feedback_file = feedback_file
        self.feedback_data = []
        self.load_feedback()
    
    def load_feedback(self):
        """Carrega feedback do arquivo JSON"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, "r", encoding='utf-8') as f:
                    self.feedback_data = json.load(f)
                print(f"📁 Carregado {len(self.feedback_data)} exemplos de feedback")
            except Exception as e:
                print(f"❌ Erro ao carregar feedback: {e}")
                self.feedback_data = []
        else:
            self.feedback_data = []
    
    def save_feedback(self):
        """Salva feedback no arquivo JSON"""
        try:
            with open(self.feedback_file, "w", encoding='utf-8') as f:
                json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Feedback salvo: {len(self.feedback_data)} exemplos")
        except Exception as e:
            print(f"❌ Erro ao salvar feedback: {e}")
    
    def add_feedback(self, image_array, predicted_label, actual_label, confidence=0.0):
        """Adiciona feedback ao conjunto de dados"""
        timestamp = datetime.now().isoformat()
        feedback_entry = {
            "timestamp": timestamp,
            "image": image_array.tolist(),
            "predicted_label": int(predicted_label),
            "actual_label": int(actual_label),
            "confidence": float(confidence),
            "correct": predicted_label == actual_label
        }
        
        self.feedback_data.append(feedback_entry)
        self.save_feedback()
        
        status = "✅ ACERTOU" if predicted_label == actual_label else "❌ ERROU"
        print(f"📝 Feedback: Pred={predicted_label}, Real={actual_label} - {status}")
    
    def get_stats(self):
        """Retorna estatisticas do feedback"""
        if not self.feedback_data:
            return {
                "total": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": 0.0,
                "recent_accuracy": 0.0
            }
        
        total = len(self.feedback_data)
        correct = sum(1 for f in self.feedback_data if f['correct'])
        wrong = total - correct
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        
        # Acuracia dos ultimos 20 exemplos
        recent_data = self.feedback_data[-20:] if len(self.feedback_data) >= 20 else self.feedback_data
        recent_correct = sum(1 for f in recent_data if f['correct'])
        recent_accuracy = (recent_correct / len(recent_data)) * 100 if recent_data else 0.0
        
        return {
            "total": total,
            "correct": correct,
            "wrong": wrong,
            "accuracy": accuracy,
            "recent_accuracy": recent_accuracy
        }
    
    def get_feedback_for_training(self):
        """Retorna todos os dados de feedback para fine-tuning"""
        return self.feedback_data
    
    def get_wrong_predictions(self):
        """Retorna apenas as predicoes erradas para fine-tuning"""
        return [f for f in self.feedback_data if not f['correct']]
    
    def clear_feedback(self):
        """Limpa todos os dados de feedback"""
        self.feedback_data = []
        self.save_feedback()
        print("🗑️ Feedback limpo!")


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


class FeedbackTrainingThread(QThread):
    """Thread para fine-tuning com feedback do usuario"""
    finished_signal = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, trainer, feedback_data):
        super().__init__()
        self.trainer = trainer
        self.feedback_data = feedback_data
    
    def run(self):
        """Executa fine-tuning com dados de feedback"""
        try:
            if len(self.feedback_data) < 5:
                self.finished_signal.emit(False, "Precisa de pelo menos 5 exemplos para fine-tuning")
                return
            
            success = self.trainer.fine_tune_with_feedback(self.feedback_data, epochs=10, lr=0.0001)
            
            if success:
                # Validar melhorias
                accuracy = self.trainer.validate_on_feedback(self.feedback_data)
                message = f"Fine-tuning concluido! Acuracia no feedback: {accuracy:.1f}%"
                self.finished_signal.emit(True, message)
            else:
                self.finished_signal.emit(False, "Erro durante fine-tuning")
                
        except Exception as e:
            self.finished_signal.emit(False, f"Erro: {str(e)}")


class DrawingCanvas(QWidget):
    """Canvas para desenhar digitos - otimizado para reconhecimento"""
    
    def __init__(self, width=280, height=280):
        super().__init__()
        self.width = width
        self.height = height
        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: white; border: 2px solid black;")
        
        # Estados do desenho
        self.drawing = False
        self.brush_size = 20  # Pincel maior para melhor reconhecimento
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
    """Canvas para plots do treinamento e estatisticas de feedback"""
    
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
    
    def update_feedback_stats(self, stats):
        """Atualiza graficos com estatisticas de feedback"""
        self.loss_ax.clear()
        self.acc_ax.clear()
        
        # Grafico de acuracia do feedback
        if stats["total"] > 0:
            labels = ['Corretos', 'Errados']
            sizes = [stats["correct"], stats["wrong"]]
            colors = ['#4CAF50', '#F44336']
            
            self.loss_ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            self.loss_ax.set_title(f'Feedback Total: {stats["total"]} exemplos')
            
            # Grafico de evolucao da acuracia
            self.acc_ax.text(0.5, 0.6, f'Acuracia Geral: {stats["accuracy"]:.1f}%', 
                           transform=self.acc_ax.transAxes, fontsize=16, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            self.acc_ax.text(0.5, 0.4, f'Acuracia Recente: {stats["recent_accuracy"]:.1f}%', 
                           transform=self.acc_ax.transAxes, fontsize=14, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            
            self.acc_ax.set_title('Estatisticas de Aprendizado')
            self.acc_ax.axis('off')
        
        self.fig.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    """Interface principal - foco em aprendizado interativo"""
    
    def __init__(self):
        super().__init__()
        
        # Configurar janela principal
        self.setWindowTitle("🎯 MNIST com Aprendizado por Reforco")
        self.setGeometry(100, 100, 1200, 800)
        
        # Inicializar componentes
        self.model = None
        self.trainer = MNISTTrainer(model_save_path='mnist_model.pth')
        self.feedback_manager = FeedbackManager()
        
        # Estados da predicao atual
        self.current_prediction = None
        self.current_confidence = 0.0
        self.current_image = None
        
        # Tentar carregar modelo existente
        self.load_existing_model()
        
        # Criar interface
        self.setup_ui()
        
        # Aplicar estilo
        self.setStyleSheet(self.get_stylesheet())
        
        # Atualizar estatisticas iniciais
        self.update_feedback_stats()
    
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
        self.create_recognition_tab()
        self.create_feedback_tab()
        self.create_training_tab()
    
    def get_stylesheet(self):
        """CSS moderno e responsivo"""
        return """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #ccc;
                background-color: white;
                border-radius: 8px;
            }            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 16px 28px;
                margin-right: 3px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 3px solid #2196F3;
            }            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 16px 28px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QPushButton#correctButton {
                background-color: #4CAF50;
            }
            QPushButton#correctButton:hover {
                background-color: #45a049;
            }
            QPushButton#wrongButton {
                background-color: #F44336;
            }
            QPushButton#wrongButton:hover {
                background-color: #da190b;
            }            QGroupBox {
                font-weight: bold;
                border: 2px solid #ccc;
                border-radius: 8px;
                margin: 8px;
                padding-top: 18px;
                font-size: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                background-color: white;
            }            QLabel {
                font-size: 16px;
            }
        """
    
    def create_recognition_tab(self):
        """Cria a aba principal de reconhecimento com feedback"""
        recognition_tab = QWidget()
        self.tab_widget.addTab(recognition_tab, "🎨 Reconhecimento + Feedback")
        
        layout = QHBoxLayout(recognition_tab)
        
        # === LADO ESQUERDO: Canvas e Controles ===
        left_group = QGroupBox("📝 Desenhe um Digito")
        left_layout = QVBoxLayout(left_group)
        
        # Canvas de desenho
        self.drawing_canvas = DrawingCanvas()
        left_layout.addWidget(self.drawing_canvas)
        
        # Controles do canvas
        canvas_controls = QHBoxLayout()
        
        clear_button = QPushButton("🗑️ Limpar")
        clear_button.clicked.connect(self.clear_drawing)
        canvas_controls.addWidget(clear_button)
        
        recognize_button = QPushButton("🔍 Reconhecer")
        recognize_button.clicked.connect(self.recognize_digit)
        canvas_controls.addWidget(recognize_button)
        
        left_layout.addLayout(canvas_controls)
        
        # Instrucoes
        instructions = QLabel("💡 Dica: Desenhe numeros grandes e centralizados para melhor reconhecimento")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; font-style: italic; padding: 10px;")
        left_layout.addWidget(instructions)
        
        layout.addWidget(left_group)
        
        # === LADO DIREITO: Resultado e Feedback ===
        right_group = QGroupBox("🎯 Resultado e Feedback")
        right_layout = QVBoxLayout(right_group)
          # Resultado da predicao
        self.prediction_label = QLabel("Desenhe um digito e clique em 'Reconhecer'")
        self.prediction_label.setFont(QFont("Arial", 36, QFont.Bold))
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("color: #333; padding: 20px; background-color: #f8f9fa; border-radius: 8px;")
        right_layout.addWidget(self.prediction_label)
          # Confianca da predicao
        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setFont(QFont("Arial", 18))
        right_layout.addWidget(self.confidence_label)
        
        # === SISTEMA DE FEEDBACK ===
        feedback_group = QGroupBox("🎓 Sistema de Aprendizado")
        feedback_layout = QVBoxLayout(feedback_group)
          # Pergunta para feedback
        self.feedback_question = QLabel("A predicao esta correta?")
        self.feedback_question.setAlignment(Qt.AlignCenter)
        self.feedback_question.setFont(QFont("Arial", 18, QFont.Bold))
        self.feedback_question.setStyleSheet("color: #2196F3; margin: 10px;")
        feedback_layout.addWidget(self.feedback_question)
        
        # Botoes de feedback
        feedback_buttons = QHBoxLayout()
        
        self.correct_button = QPushButton("✅ ACERTOU")
        self.correct_button.setObjectName("correctButton")
        self.correct_button.clicked.connect(self.feedback_correct)
        feedback_buttons.addWidget(self.correct_button)
        
        self.wrong_button = QPushButton("❌ ERROU")
        self.wrong_button.setObjectName("wrongButton")
        self.wrong_button.clicked.connect(self.feedback_wrong)
        feedback_buttons.addWidget(self.wrong_button)
        
        feedback_layout.addLayout(feedback_buttons)
        
        # Selecao do digito correto (aparece quando erra)
        self.correct_digit_group = QGroupBox("Qual e o digito correto?")
        correct_digit_layout = QHBoxLayout(self.correct_digit_group)
        
        self.digit_buttons = []
        for digit in range(10):
            btn = QPushButton(str(digit))
            btn.clicked.connect(lambda checked, d=digit: self.set_correct_digit(d))
            btn.setMinimumSize(40, 40)
            correct_digit_layout.addWidget(btn)
            self.digit_buttons.append(btn)
        
        feedback_layout.addWidget(self.correct_digit_group)
        self.correct_digit_group.setVisible(False)  # Inicialmente oculto
        
        right_layout.addWidget(feedback_group)
        
        # Desabilitar botoes inicialmente
        self.set_feedback_enabled(False)
        
        layout.addWidget(right_group)
    
    def create_feedback_tab(self):
        """Cria aba de estatisticas e gerenciamento de feedback"""
        feedback_tab = QWidget()
        self.tab_widget.addTab(feedback_tab, "📊 Estatisticas de Aprendizado")
        
        layout = QVBoxLayout(feedback_tab)
        
        # Controles superiores
        controls_group = QGroupBox("🛠️ Controles de Aprendizado")
        controls_layout = QHBoxLayout(controls_group)
        
        # Estatisticas em tempo real
        self.stats_label = QLabel("Nenhum feedback ainda")
        self.stats_label.setFont(QFont("Arial", 12))
        controls_layout.addWidget(self.stats_label)
        
        controls_layout.addStretch()
        
        # Botao de fine-tuning
        self.finetune_button = QPushButton("🎯 Fazer Fine-tuning")
        self.finetune_button.clicked.connect(self.start_feedback_training)
        controls_layout.addWidget(self.finetune_button)
        
        # Botao para limpar feedback
        clear_feedback_button = QPushButton("🗑️ Limpar Feedback")
        clear_feedback_button.clicked.connect(self.clear_all_feedback)
        controls_layout.addWidget(clear_feedback_button)
        
        layout.addWidget(controls_group)
        
        # Canvas para estatisticas
        self.stats_canvas = PlotCanvas()
        layout.addWidget(self.stats_canvas)
    
    def create_training_tab(self):
        """Cria aba de treinamento inicial (opcional)"""
        training_tab = QWidget()
        self.tab_widget.addTab(training_tab, "🚀 Treinamento Inicial")
        
        layout = QVBoxLayout(training_tab)
        
        # Info sobre treinamento
        info_label = QLabel("""
        <h3>🎓 Treinamento Inicial vs Aprendizado Continuo</h3>
        <p><b>Treinamento Inicial:</b> Use apenas se nao houver modelo salvo.<br>
        <b>Aprendizado Continuo:</b> O sistema aprende com seu feedback na aba de Reconhecimento!</p>
        <p><i>Recomendacao: Use o modelo pre-treinado e foque no feedback interativo.</i></p>
        """)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("padding: 20px; background-color: #e3f2fd; border-radius: 8px;")
        layout.addWidget(info_label)
        
        # Controles de treinamento
        controls_group = QGroupBox("Parametros de Treinamento Inicial")
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
        self.train_button = QPushButton("Iniciar Treinamento Inicial")
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
                background-color: #2196F3;
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
            self.model_status_label.setText("✅ Modelo carregado e pronto para aprendizado!")
            self.model_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.model_status_label.setText("❌ Nenhum modelo - treine ou use feedback")
            self.model_status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def clear_drawing(self):
        """Limpa o canvas de desenho e reseta interface"""
        self.drawing_canvas.clear_canvas()
        self.prediction_label.setText("Desenhe um digito e clique em 'Reconhecer'")
        self.confidence_label.setText("")
        self.set_feedback_enabled(False)
        self.correct_digit_group.setVisible(False)
        
        # Reset das variaveis de estado
        self.current_prediction = None
        self.current_confidence = 0.0
        self.current_image = None
    
    def recognize_digit(self):
        """Reconhece o digito desenhado"""
        if self.model is None:
            QMessageBox.warning(
                self, 
                "Erro", 
                "Nenhum modelo disponivel!\n\nOpcoes:\n1. Treine um modelo inicial\n2. Use o feedback para aprender"
            )
            return
        
        try:
            # Obter imagem do canvas
            image = self.drawing_canvas.get_image_array()
            
            # Fazer predicao
            predicted_digit, confidence = self.trainer.predict(image)
            
            # Armazenar para feedback
            self.current_prediction = predicted_digit
            self.current_confidence = confidence
            self.current_image = image
            
            # Mostrar resultado
            self.prediction_label.setText(f"Digito: {predicted_digit}")
            
            # Colorir baseado na confianca
            if confidence > 0.8:
                color = "green"
                emoji = "🎯"
                status = "Alta confianca"
            elif confidence > 0.5:
                color = "orange"
                emoji = "⚠️"
                status = "Media confianca"
            else:
                color = "red"
                emoji = "❓"
                status = "Baixa confianca"
            
            self.prediction_label.setStyleSheet(f"color: {color}; background-color: #f8f9fa; border-radius: 8px; padding: 20px;")
            self.confidence_label.setText(f"{emoji} {status}: {confidence:.1%}")
            
            # Habilitar sistema de feedback
            self.set_feedback_enabled(True)
            self.feedback_question.setText("A predicao esta correta?")
            
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro na predicao:\n{str(e)}")
    
    def set_feedback_enabled(self, enabled):
        """Habilita/desabilita botoes de feedback"""
        self.correct_button.setEnabled(enabled)
        self.wrong_button.setEnabled(enabled)
    
    def feedback_correct(self):
        """Usuario confirma que a predicao esta correta"""
        if self.current_prediction is not None and self.current_image is not None:
            # Adicionar feedback positivo
            self.feedback_manager.add_feedback(
                self.current_image,
                self.current_prediction,
                self.current_prediction,  # mesma coisa quando acerta
                self.current_confidence
            )
            
            # Feedback visual
            self.feedback_question.setText("✅ Obrigado! O modelo esta aprendendo...")
            self.feedback_question.setStyleSheet("color: green; margin: 10px;")
            
            # Desabilitar botoes
            self.set_feedback_enabled(False)
            self.correct_digit_group.setVisible(False)
            
            # Atualizar estatisticas
            self.update_feedback_stats()
            
            # Auto-limpar apos 2 segundos
            QTimer.singleShot(2000, self.clear_drawing)
    
    def feedback_wrong(self):
        """Usuario indica que a predicao esta errada"""
        self.feedback_question.setText("❌ Qual e o digito correto?")
        self.feedback_question.setStyleSheet("color: red; margin: 10px;")
        self.correct_digit_group.setVisible(True)
        self.set_feedback_enabled(False)
    
    def set_correct_digit(self, correct_digit):
        """Usuario especifica qual e o digito correto"""
        if self.current_prediction is not None and self.current_image is not None:
            # Adicionar feedback negativo com correcao
            self.feedback_manager.add_feedback(
                self.current_image,
                self.current_prediction,
                correct_digit,
                self.current_confidence
            )
            
            # Feedback visual
            self.feedback_question.setText(f"✅ Anotado! Era {correct_digit}, nao {self.current_prediction}")
            self.feedback_question.setStyleSheet("color: green; margin: 10px;")
            
            # Ocultar selecao de digitos
            self.correct_digit_group.setVisible(False)
            
            # Atualizar estatisticas
            self.update_feedback_stats()
            
            # Auto-limpar apos 3 segundos
            QTimer.singleShot(3000, self.clear_drawing)
    
    def update_feedback_stats(self):
        """Atualiza estatisticas de feedback na interface"""
        stats = self.feedback_manager.get_stats()
        
        # Atualizar label de estatisticas
        if stats["total"] > 0:
            self.stats_label.setText(
                f"📊 Total: {stats['total']} | ✅ Corretos: {stats['correct']} | "
                f"❌ Errados: {stats['wrong']} | 🎯 Acuracia: {stats['accuracy']:.1f}%"
            )
            
            # Habilitar fine-tuning se tiver dados suficientes
            self.finetune_button.setEnabled(stats["total"] >= 5)
            
            # Atualizar graficos
            self.stats_canvas.update_feedback_stats(stats)
        else:
            self.stats_label.setText("📊 Nenhum feedback ainda - comece a usar o reconhecimento!")
            self.finetune_button.setEnabled(False)
    
    def start_feedback_training(self):
        """Inicia fine-tuning com dados de feedback"""
        feedback_data = self.feedback_manager.get_feedback_for_training()
        
        if len(feedback_data) < 5:
            QMessageBox.information(
                self, 
                "Info", 
                f"Precisa de pelo menos 5 exemplos para fine-tuning.\nVoce tem: {len(feedback_data)}"
            )
            return
        
        # Confirmar fine-tuning
        reply = QMessageBox.question(
            self,
            "Confirmar Fine-tuning",
            f"Iniciar fine-tuning com {len(feedback_data)} exemplos de feedback?\n\n"
            f"Isto ira melhorar o modelo baseado no seu feedback.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Desabilitar botao
            self.finetune_button.setEnabled(False)
            self.finetune_button.setText("🔄 Treinando...")
            
            # Iniciar thread de fine-tuning
            self.feedback_training_thread = FeedbackTrainingThread(self.trainer, feedback_data)
            self.feedback_training_thread.finished_signal.connect(self.feedback_training_finished)
            self.feedback_training_thread.start()
    
    def feedback_training_finished(self, success, message):
        """Callback quando fine-tuning termina"""
        # Reabilitar botao
        self.finetune_button.setEnabled(True)
        self.finetune_button.setText("🎯 Fazer Fine-tuning")
        
        # Mostrar resultado
        if success:
            QMessageBox.information(self, "Sucesso", message)
            # Atualizar referencia do modelo
            self.model = self.trainer.model
            self.update_model_status()
        else:
            QMessageBox.warning(self, "Erro", message)
    
    def clear_all_feedback(self):
        """Limpa todos os dados de feedback"""
        reply = QMessageBox.question(
            self,
            "Confirmar",
            "Limpar todos os dados de feedback?\n\nEsta acao nao pode ser desfeita.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.feedback_manager.clear_feedback()
            self.update_feedback_stats()
            QMessageBox.information(self, "Info", "Feedback limpo com sucesso!")
    
    # === METODOS DE TREINAMENTO INICIAL ===
    
    def start_training(self):
        """Inicia o treinamento inicial (opcional)"""
        epochs = self.epochs_spinbox.value()
        
        reply = QMessageBox.question(
            self,
            "Confirmar Treinamento",
            f"Iniciar treinamento inicial por {epochs} epocas?\n\n"
            f"Isso pode demorar varios minutos.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
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
        self.training_status.setText("✅ Treinamento inicial concluido!")
        self.progress_bar.setValue(100)
        
        # Atualizar referencias do modelo
        self.trainer = self.training_thread.trainer
        self.model = self.trainer.model
        self.update_model_status()
        
        # Mostrar mensagem de sucesso
        QMessageBox.information(
            self, 
            "Sucesso", 
            "Treinamento inicial concluido!\n\nAgora voce pode usar o reconhecimento e feedback para melhorar o modelo."
        )


def main():
    """Funcao principal"""
    print("🚀 Iniciando MNIST com Aprendizado por Reforco...")
    
    # Criar aplicacao
    app = QApplication(sys.argv)
      # Configurar fonte padrao
    font = QFont("Arial", 14)
    app.setFont(font)
    
    # Criar e mostrar janela principal
    try:
        window = MainWindow()
        window.show()
        print("✅ Interface carregada com sucesso!")
        print("💡 Use a aba 'Reconhecimento + Feedback' para ensinar o modelo!")
        
        # Executar aplicacao
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"❌ Erro ao inicializar: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
