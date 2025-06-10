# EMNIST Letter Recognition App

A PyTorch + PyQt5 application for training neural networks on the EMNIST dataset and recognizing handwritten letters.

## Features

1. **Neural Network Training**: Train a CNN model on the EMNIST letters dataset
2. **Interactive Drawing**: Draw letters with your mouse on a digital canvas
3. **Real-time Recognition**: Recognize drawn letters using the trained model
4. **Word Formation**: Combine recognized letters to form words
5. **Training Visualization**: View training progress with real-time plots

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:

```bash
python main.py
```

2. **Training Tab**:
   - Set the number of epochs (default: 20)
   - Click "Start Training" to train the model on EMNIST dataset
   - Monitor training progress with real-time loss and accuracy plots
   - The trained model will be automatically saved

3. **Letter Recognition Tab**:
   - Draw a letter on the white canvas using your mouse
   - Click "Recognize Letter" to predict the drawn letter
   - View the prediction confidence
   - Add recognized letters to form words
   - Clear the canvas or word as needed

## Model Architecture

The neural network uses a CNN architecture with:
- 3 Convolutional layers with batch normalization
- Max pooling and dropout for regularization
- 3 Fully connected layers
- Optimized for 26-class letter classification (A-Z)

## Dataset

The application uses the EMNIST Letters dataset, which contains:
- 145,600 training samples
- 24,000 test samples
- 28x28 grayscale images of handwritten letters

## Files Structure

- `main.py`: Main application with PyQt5 GUI
- `model.py`: Neural network architecture definition
- `trainer.py`: Training logic and model management
- `requirements.txt`: Required Python packages
- `emnist_model.pth`: Saved model weights (created after training)

## Requirements

- Python 3.7+
- PyTorch
- PyQt5
- NumPy
- OpenCV
- Matplotlib
- scikit-learn

## Tips for Best Results

1. **Drawing**: Draw letters clearly and try to center them in the canvas
2. **Training**: More epochs generally lead to better accuracy (try 20-50 epochs)
3. **Recognition**: The model works best with letters similar to the EMNIST training style
4. **Word Formation**: Letters are combined left-to-right as you recognize them

## Troubleshooting

- If CUDA is available, training will use GPU acceleration
- The EMNIST dataset will be downloaded automatically on first run
- Model weights are saved automatically after training completion
