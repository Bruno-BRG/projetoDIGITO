# MNIST Neural Network Classifier

A PyQt5 application for training and testing neural networks on the MNIST digit recognition dataset using PyTorch.

## Key Features

- **Smart Dataset Management**: Automatically checks if MNIST dataset exists and downloads only if needed
- **K-Fold Cross Validation**: Robust model evaluation with configurable folds
- **Interactive GUI**: Professional PyQt5 interface with real-time updates
- **Image Carousel**: Browse and test individual images with confidence visualization

## Features

### Training Tab
- **Hyperparameter Adjustment**: Configure K-folds, epochs, learning rate, batch size, hidden layer size, and dropout rate
- **K-Fold Cross Validation**: Implements K-fold cross-validation for robust model evaluation
- **Real-time Training Progress**: Live updates of training progress and metrics
- **Results Visualization**: Interactive plots showing training/validation loss and accuracy across all folds
- **Model Save/Load**: Save trained models and load them for later use

### Testing Tab
- **Image Carousel**: Browse through MNIST test images using slider or navigation buttons
- **Real-time Prediction**: Test the trained model on selected images
- **Confidence Visualization**: View prediction confidence and probability distribution
- **Custom Image Support**: Load and test your own handwritten digit images
- **Correctness Indication**: Visual feedback showing if predictions are correct or incorrect

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

Required packages:
- torch>=2.0.0
- torchvision>=0.15.0
- PyQt5>=5.15.0
- numpy>=1.21.0
- matplotlib>=3.5.0
- scikit-learn>=1.0.0
- Pillow>=8.3.0

## Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python main.py
   ```

3. **Training a Model**:
   - Go to the "Training" tab
   - Adjust hyperparameters as desired
   - Click "Start Training"
   - Monitor progress and view results
   - Save the trained model

4. **Testing the Model**:
   - Go to the "Testing" tab
   - Use the carousel to select test images
   - Click "Test Current Image" to see predictions
   - View probability distributions and confidence scores
   - Load custom images for testing

## Model Architecture

The neural network uses a CNN architecture with:
- 2 Convolutional layers (32 and 64 filters)
- Max pooling and dropout for regularization
- 2 Fully connected layers
- Configurable hidden layer size and dropout rate

## File Structure

- `main.py`: Main application with PyQt5 GUI
- `model.py`: Neural network model definition
- `trainer.py`: Training logic with K-fold cross-validation
- `requirements.txt`: Required Python packages
- `data/`: MNIST dataset (downloaded automatically)

## Features in Detail

### K-Fold Cross Validation
The application implements K-fold cross-validation to provide more robust model evaluation. The training data is split into K folds, and the model is trained K times, each time using a different fold as validation data.

### Hyperparameter Tuning
Users can adjust:
- **K-Folds**: Number of folds for cross-validation (2-10)
- **Epochs**: Number of training epochs (1-100)
- **Learning Rate**: Adam optimizer learning rate (0.0001-1.0)
- **Batch Size**: Training batch size (16, 32, 64, 128, 256)
- **Hidden Size**: Size of the hidden layer (32-512)
- **Dropout Rate**: Dropout probability for regularization (0.0-0.9)

### Interactive Testing
The testing interface provides:
- Image carousel with 100 test images
- Real-time prediction with confidence scores
- Probability distribution visualization
- Support for custom image upload
- Visual feedback for correct/incorrect predictions

## GPU Support

The application automatically detects and uses CUDA if available, falling back to CPU if not.

## License

MIT License - see LICENSE file for details.
