# 🧠 Neural Network with Python

A custom implementation of a neural network from scratch using Python, designed for educational purposes and machine learning experimentation.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Results](#results)
- [Customization](#customization)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements a neural network from scratch without relying on high-level deep learning frameworks like TensorFlow or PyTorch. The implementation provides a clear understanding of the fundamental concepts behind neural networks, including:

- Forward propagation
- Backpropagation
- Gradient descent optimization
- Activation functions
- Loss computation

The neural network is trained on CSV datasets and can be used for classification or regression tasks.

## ✨ Features

- **Pure Python Implementation**: Built from scratch using NumPy for matrix operations
- **Modular Architecture**: Clean, well-structured code that's easy to understand and modify
- **Training & Testing Pipeline**: Separate datasets for model training and evaluation
- **Customizable Network**: Easily adjust layers, neurons, learning rate, and epochs
- **Performance Metrics**: Track accuracy, loss, and other performance indicators
- **CSV Data Support**: Works with standard CSV format datasets

## 📁 Project Structure

```
Neural_Network_With_Python/
│
├── app.py                 # Main neural network implementation
├── train_X.csv           # Training features dataset
├── train_label.csv       # Training labels dataset
├── test_X.csv            # Testing features dataset
├── test_label.csv        # Testing labels dataset
└── README.md             # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/Ujjwal-Kumar6/Neural_Network_With_Python.git
cd Neural_Network_With_Python
```

2. **Install required dependencies**

```bash
pip install numpy pandas matplotlib
```

Or use a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

Run the neural network training:

```bash
python app.py
```

### Customizing Parameters

You can modify the following parameters in `app.py`:

```python
# Network architecture
hidden_layers = [128, 64, 32]  # Number of neurons in each hidden layer
learning_rate = 0.01           # Learning rate for gradient descent
epochs = 1000                  # Number of training iterations
batch_size = 32                # Mini-batch size

# Activation functions
activation = 'relu'            # Options: 'relu', 'sigmoid', 'tanh'
output_activation = 'softmax'  # For classification: 'softmax' or 'sigmoid'
```

### Example Code Snippet

```python
# Initialize the neural network
nn = NeuralNetwork(input_size=784, 
                   hidden_layers=[128, 64], 
                   output_size=10,
                   learning_rate=0.01)

# Train the model
nn.train(train_X, train_labels, epochs=100, batch_size=32)

# Make predictions
predictions = nn.predict(test_X)

# Evaluate accuracy
accuracy = nn.evaluate(test_X, test_labels)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
```

## 📊 Dataset

The project includes four CSV files:

- **train_X.csv**: Training features (input data)
- **train_label.csv**: Training labels (expected outputs)
- **test_X.csv**: Testing features (for model evaluation)
- **test_label.csv**: Testing labels (ground truth for testing)

### Dataset Format

- All data should be in CSV format
- Features should be normalized (scaled between 0 and 1 or standardized)
- Labels should be encoded appropriately (one-hot for multi-class classification)

### Using Your Own Dataset

Replace the CSV files with your own data, ensuring the format matches:

```python
# Load your custom dataset
import pandas as pd

train_X = pd.read_csv('your_train_features.csv')
train_y = pd.read_csv('your_train_labels.csv')
test_X = pd.read_csv('your_test_features.csv')
test_y = pd.read_csv('your_test_labels.csv')
```

## 🔧 How It Works

### 1. **Forward Propagation**

The input data flows through the network:

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → ... → Output Layer
```

Each layer applies:
- Linear transformation: `Z = W·X + b`
- Activation function: `A = activation(Z)`

### 2. **Loss Calculation**

The network computes the difference between predictions and actual values:
- For classification: Cross-entropy loss
- For regression: Mean squared error (MSE)

### 3. **Backpropagation**

Gradients are computed backward through the network using the chain rule:
- Calculate output layer gradients
- Propagate gradients to hidden layers
- Compute weight and bias gradients

### 4. **Weight Update**

Weights are updated using gradient descent:

```
W = W - learning_rate × ∂Loss/∂W
b = b - learning_rate × ∂Loss/∂b
```

## 📈 Results

After training, the model outputs:

- **Training Accuracy**: Performance on the training dataset
- **Testing Accuracy**: Performance on unseen test data
- **Loss Curve**: Visualization of loss reduction over epochs
- **Confusion Matrix**: Detailed breakdown of predictions (for classification)

### Example Output

```
Epoch 1/100 - Loss: 2.3456 - Accuracy: 45.23%
Epoch 10/100 - Loss: 1.2345 - Accuracy: 67.89%
Epoch 50/100 - Loss: 0.5678 - Accuracy: 85.45%
Epoch 100/100 - Loss: 0.2345 - Accuracy: 92.34%

Final Test Accuracy: 91.23%
```

## 🎨 Customization

### Adding New Activation Functions

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
```

### Implementing Different Optimizers

```python
# Adam optimizer
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        # Implementation details...
```

### Adding Regularization

```python
# L2 Regularization
def compute_loss_with_regularization(predictions, targets, weights, lambda_reg=0.01):
    data_loss = cross_entropy_loss(predictions, targets)
    reg_loss = lambda_reg * np.sum([np.sum(w**2) for w in weights])
    return data_loss + reg_loss
```

## 📦 Dependencies

Create a `requirements.txt` file:

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0  # Optional, for data preprocessing
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Contribution Ideas

- Add support for more activation functions
- Implement advanced optimizers (Adam, RMSprop)
- Add data augmentation capabilities
- Create visualization tools for network architecture
- Improve documentation and code comments
- Add unit tests
- Implement regularization techniques (Dropout, L1/L2)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- Inspired by the fundamentals of deep learning and neural network theory
- Built for educational purposes to understand the mathematics behind neural networks
- Thanks to the open-source community for Python libraries like NumPy and Pandas

## 📧 Contact

**Ujjwal Kumar** - [@Ujjwal-Kumar6](https://github.com/Ujjwal-Kumar6)

Project Link: [https://github.com/Ujjwal-Kumar6/Neural_Network_With_Python](https://github.com/Ujjwal-Kumar6/Neural_Network_With_Python)

---

## 🎓 Learning Resources

If you're new to neural networks, check out these resources:

- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [Deep Learning Book by Ian Goodfellow](https://www.deeplearningbook.org/)
- [Neural Networks and Deep Learning by Michael Nielsen](http://neuralnetworksanddeeplearning.com/)

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

---

**Made with ❤️ and Python**
