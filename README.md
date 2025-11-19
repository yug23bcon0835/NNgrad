# NNGrad: A NumPy-Based Automatic Differentiation Engine

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-green.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A lightweight, pure NumPy implementation of an automatic differentiation engine with PyTorch-like API for deep learning research and education.**

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Architecture & Components](#architecture--components)
- [Installation & Setup](#installation--setup)
- [Quick Start Guide](#quick-start-guide)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)

---

## Overview

**NNGrad** is a fundamental deep learning library built entirely on NumPy that provides the essential components needed for building and training simple deep learning models. It implements:

1. **Automatic Differentiation (Autograd)**: A computation graph-based system for automatic gradient calculation
2. **Tensor Class**: A NumPy-backed data structure with gradient tracking capabilities
3. **Neural Network Modules**: A modular, extensible API for building neural network architectures
4. **Activation Functions**: Common activation functions (ReLU, Tanh, Sigmoid, Softmax)
5. **Loss Functions**: Standard loss functions (L1Loss, MSELoss, BCELoss)
6. **Regularization**: Dropout layer for preventing overfitting
7. **Normalization**: Layer normalization for stabilizing training

This project is ideal for:
- **Educational purposes**: Understanding how deep learning frameworks work internally
- **Research**: Experimenting with novel architectures and gradient computation strategies
- **Rapid prototyping**: Quick implementation of simple neural networks without external GPU dependencies

---

## Key Features

✅ **Pure NumPy Implementation**
- No external deep learning dependencies (no TensorFlow, PyTorch)
- Lightweight and easy to understand
- Transparent gradient computation

✅ **Automatic Differentiation**
- Dynamic computation graphs
- Backpropagation with topological sorting
- Cycle detection in computational graphs
- Custom backward passes for each operation

✅ **PyTorch-like API**
- Familiar `Module` base class for layer definitions
- Sequential model composition
- Modular containers (ModuleList, ModuleDict)
- Parameter management and state serialization

✅ **Rich Operation Set**
- Element-wise operations: exp, log, transpose, reshape
- Array operations: sum, mean, broadcast, concat, indexing
- Activation functions: relu, tanh, sigmoid, softmax
- Loss functions: L1Loss, MSELoss, BCELoss

✅ **Advanced Features**
- Gradient clipping for numerical stability
- Masked fill for selective gradient flow
- Layer normalization for training stability
- Dropout for regularization

---

## Project Structure

```
nngrad/
├── README.md                 # This file
├── __init__.py              # Package root
├── core/                    # Core engine and modules
│   ├── __init__.py
│   ├── engine.py           # Tensor class and autograd implementation
│   ├── utils.py            # Utility functions
│   ├── optimizers.py       # Optimizer definitions
│   └── nn/                 # Neural network components
│       ├── __init__.py
│       ├── main.py         # Module, ModuleList, ModuleDict base classes
│       ├── linear.py       # Linear/Dense layer
│       ├── activation.py   # Activation functions
│       ├── losses.py       # Loss function implementations
│       ├── dropout.py      # Dropout regularization
│       ├── layernorm.py    # Layer normalization
│       └── sequential.py   # Sequential model container
```

---

## Architecture & Components

### 1. **Core Engine** (`core/engine.py`)

#### Tensor Class
The `Tensor` class is the fundamental data structure that stores:
- **data**: NumPy array containing the numerical values
- **grad**: Accumulated gradients (same shape as data)
- **requires_grad**: Boolean flag indicating if gradients should be computed
- **prev**: Set of child tensors in the computation graph
- **op**: String identifier for the operation that created this tensor
- **grad_fn**: Backward function for custom gradient computation

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `backward()` | Initiates backpropagation through the entire computation graph |
| `reset_grad()` | Clears accumulated gradients |
| `set_requires_grad(val)` | Sets whether gradients should be tracked |

#### Supported Operations

**Array Manipulation:**
- `sum(axis, keepdims)` - Sum reduction along axes
- `T(axes)` - Transpose with custom axis ordering
- `reshape(shape)` - Reshape tensor to new dimensions
- `broadcast(shape)` - Expand tensor to new shape
- `concat(others, dim)` - Concatenate tensors along dimension
- `__getitem__` - Indexing with gradient support
- `__setitem__` - Element assignment with gradient support

**Element-wise Operations:**
- `exp()` - Exponential function
- `log()` - Natural logarithm
- `mean(axis, keepdims)` - Mean reduction

**Utilities:**
- `clip(min, max, clip_grad)` - Data and gradient clipping for numerical stability
- `masked_fill(mask, value)` - Fill masked elements while stopping gradient flow

**Graph Features:**
- Topological sorting for correct backward pass order
- Cycle detection to prevent infinite loops
- Automatic gradient accumulation

### 2. **Neural Network Modules** (`core/nn/`)

#### Module Base Class (`main.py`)

The foundation for all neural network layers:

```python
class Module:
    def forward(self, *args, **kwargs): 
        # Override in subclasses
    
    def __call__(self, *args, **kwargs): 
        # Calls forward method
    
    def train(self):  
        # Set training mode
    
    def eval(self):   
        # Set evaluation mode
    
    def parameters(self): 
        # Get all learnable parameters
    
    def zero_grad(self): 
        # Reset all gradients to zero
    
    def state_dict(self, prefix=''): 
        # Get model state as dictionary
```

#### Model Containers

**Sequential** (`sequential.py`)
- Chains modules sequentially
- Output of module `i` becomes input to module `i+1`

```python
model = Sequential(
    Linear(10, 64),
    relu,
    Linear(64, 32),
    relu,
    Linear(32, 1)
)
```

**ModuleList** (`main.py`)
- List-like container for modules
- Supports indexing, iteration, appending, extending
- Useful for flexible forward pass logic

**ModuleDict** (`main.py`)
- Dictionary-like container for modules
- Access modules by string keys
- Useful for named, branching architectures

#### Linear Layer (`linear.py`)

Fully connected linear transformation:
```python
class Linear(Module):
    def __init__(self, in_features, out_features, use_bias=True):
        self.weight  # Shape: (out_features, in_features)
        self.bias    # Shape: (out_features,) if use_bias=True
    
    def forward(self, x):
        return x @ self.weight.T() + self.bias
```

#### Activation Functions (`activation.py`)

- **ReLU**: Rectified Linear Unit (f(x) = max(0, x))
- **Tanh**: Hyperbolic tangent activation
- **Sigmoid**: Logistic sigmoid function
- **Softmax**: Softmax for multi-class distributions

#### Loss Functions (`losses.py`)

| Loss Function | Use Case |
|--------------|----------|
| **L1Loss** | Sparse errors, robust to outliers |
| **MSELoss** | Regression, squared error penalty |
| **BCELoss** | Binary classification with numerical stability |

#### Dropout (`dropout.py`)

Regularization technique:
- Randomly drops activations during training
- Disabled during evaluation
- Scales remaining activations by 1/(1-p)

```python
dropout = Dropout(p=0.5)  # Drop 50% of activations
```

#### Layer Normalization (`layernorm.py`)

Normalizes activations across features:
- Reduces internal covariate shift
- Learns affine parameters (weight, bias)
- Stabilizes training of deep networks

```python
norm = LayerNorm(normalized_shape=512, eps=1e-5)
```

### 3. **Utilities** (`core/utils.py`)

**broadcast_axis**: Determines which axes need reduction when broadcasting two tensors of different shapes.

### 4. **Optimizers** (`core/optimizers.py`)

Currently a placeholder for optimizer implementations (SGD, Adam, etc.)

---

## Installation & Setup

### Requirements
- Python 3.7+
- NumPy 1.19+

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nngrad.git
cd nngrad

# Install dependencies
pip install numpy>=1.19.0
```

### Verification

```python
import sys
sys.path.insert(0, '/path/to/nngrad')

from core import Tensor
from core.nn import Linear, Sequential, relu

# Quick test
x = Tensor([[1, 2], [3, 4]], requires_grad=True)
y = x.sum()
y.backward()
print(x.grad)  # Should show accumulated gradients
```

---

## Quick Start Guide

### 1. Creating and Computing Gradients

```python
from core import Tensor
import numpy as np

# Create a tensor
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

# Perform operations
y = x.sum()

# Backward pass
y.backward()

# Access gradients
print(x.grad)  # [1. 1. 1.]
```

### 2. Building a Simple Neural Network

```python
from core.nn import Linear, Sequential, relu
from core import Tensor

# Create a simple MLP
model = Sequential(
    Linear(10, 64),
    Linear(64, 32),
    Linear(32, 1)
)

# Forward pass
x = Tensor(np.random.randn(5, 10), requires_grad=True)
output = model(x)
print(output.shape)  # (5, 1)
```

### 3. Training Loop Example

```python
from core.nn import Linear, MSELoss
from core import Tensor
import numpy as np

# Create simple dataset
X = Tensor(np.random.randn(100, 10), requires_grad=True)
y = Tensor(np.random.randn(100, 1), requires_grad=True)

# Create model and loss
model = Sequential(
    Linear(10, 64),
    Linear(64, 1)
)
loss_fn = MSELoss(reduction='mean')

# Training step
output = model(X)
loss = loss_fn(output, y)
loss.backward()

# Update parameters (manual SGD)
learning_rate = 0.01
for param in model.parameters():
    param.data -= learning_rate * param.grad
    param.reset_grad()
```

### 4. Working with Computation Graphs

```python
from core import Tensor

# Create a computation graph
x = Tensor(2.0, requires_grad=True)
y = Tensor(3.0, requires_grad=True)

# Compose operations
z = (x ** 2) + (y ** 3)

# Automatic differentiation
z.backward()

print(f"dz/dx = {x.grad}")  # 4.0
print(f"dz/dy = {y.grad}")  # 27.0
```

---

## API Reference

### Tensor

#### Core Methods
- `backward()`: Perform backpropagation
- `reset_grad()`: Zero out gradients
- `set_requires_grad(val)`: Enable/disable gradient tracking

#### Tensor Operations

| Operation | Method | Example |
|-----------|--------|---------|
| Sum | `sum(axis, keepdims)` | `x.sum(axis=0)` |
| Mean | `mean(axis, keepdims)` | `x.mean(axis=1)` |
| Transpose | `T(axes)` | `x.T()` |
| Reshape | `reshape(shape)` | `x.reshape((2, -1))` |
| Broadcast | `broadcast(shape)` | `x.broadcast((5, 10))` |
| Concatenate | `concat(others, dim)` | `x.concat([y, z], dim=0)` |
| Indexing | `__getitem__` | `x[0:5, :]` |
| Assignment | `__setitem__` | `x[0] = new_val` |
| Exponential | `exp()` | `x.exp()` |
| Logarithm | `log()` | `x.log()` |
| Clip | `clip(min, max)` | `x.clip(0, 1)` |
| Masked Fill | `masked_fill(mask, val)` | `x.masked_fill(x < 0, 0)` |

#### Tensor Attributes
- `data`: NumPy array containing values
- `grad`: NumPy array containing gradients
- `requires_grad`: Boolean flag
- `shape`: Shape tuple
- `ndim`: Number of dimensions
- `dtype`: Data type

### Module

#### Core Methods
- `forward(*args, **kwargs)`: Forward pass (must be overridden)
- `__call__(*args, **kwargs)`: Calls forward method
- `train()`: Set to training mode
- `eval()`: Set to evaluation mode
- `parameters()`: Generator of learnable parameters
- `zero_grad()`: Reset all gradients
- `state_dict(prefix)`: Get model state as dictionary

### Common Layers

#### Linear(in_features, out_features, use_bias=True)
Fully connected layer: `y = xW^T + b`

#### Sequential(*modules)
Chains modules sequentially.

#### ModuleList(modules)
List container for modules.

#### ModuleDict(modules)
Dictionary container for modules.

### Activation Functions

#### relu(x)
ReLU activation: f(x) = max(0, x)

#### tanh(x)
Hyperbolic tangent: f(x) = tanh(x)

#### sigmoid(x)
Sigmoid function: f(x) = 1 / (1 + e^(-x))

#### softmax(t, axis=-1)
Softmax normalization for probability distributions.

### Loss Functions

#### L1Loss(reduction='sum')
Mean absolute error loss.

#### MSELoss(reduction='sum')
Mean squared error loss.

#### BCELoss(eps=1e-6)
Binary cross-entropy loss with numerical stability.

### Regularization

#### Dropout(p=0.5)
Randomly drops activations during training.

#### LayerNorm(normalized_shape, eps=1e-5)
Layer normalization across features.

---

## Usage Examples

### Example 1: Simple Linear Regression

```python
from core.nn import Linear, MSELoss
from core import Tensor
import numpy as np

# Synthetic data
np.random.seed(42)
X = Tensor(np.random.randn(50, 1), requires_grad=True)
y_true = 2 * X.data + 3 + 0.1 * np.random.randn(50, 1)
y = Tensor(y_true, requires_grad=True)

# Model
model = Linear(1, 1)
loss_fn = MSELoss(reduction='mean')

# Training
for epoch in range(100):
    # Forward pass
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Update parameters
    learning_rate = 0.01
    for param in model.parameters():
        param.data -= learning_rate * param.grad

print(f"Weight: {model.weight.data}")
print(f"Bias: {model.bias.data}")
```

### Example 2: Multi-layer Perceptron

```python
from core.nn import Linear, Sequential, relu, Dropout
from core import Tensor
import numpy as np

# Create MLP with dropout and normalization
model = Sequential(
    Linear(784, 256),
    Dropout(p=0.2),
    Linear(256, 128),
    Dropout(p=0.2),
    Linear(128, 10)
)

# Forward pass
x = Tensor(np.random.randn(32, 784))
output = model(x)
print(output.shape)  # (32, 10)
```

### Example 3: Custom Module

```python
from core.nn import Module, Linear
from core import Tensor
import numpy as np

class CustomNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(10, 64)
        self.fc2 = Linear(64, 32)
        self.fc3 = Linear(32, 1)
    
    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Usage
model = CustomNet()
x = Tensor(np.random.randn(5, 10), requires_grad=True)
output = model(x)
```

### Example 4: Gradient Computation and Visualization

```python
from core import Tensor
import numpy as np

# Create a simple computation graph
x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = (x ** 2).sum()

# Backward pass
y.backward()

# View gradients
print("Input:")
print(x.data)
print("\nGradient:")
print(x.grad)  # [2x, 4x, 6x, 8x] - derivative of sum(x^2) = 2x
```

---

## Design Patterns & Best Practices

### 1. **Tensor Creation**
- Always specify `requires_grad=True` for tensors that need gradient computation
- Use appropriate dtypes (default: float32) for memory efficiency

### 2. **Model Definition**
- Inherit from `Module` for custom layers
- Define learnable parameters in `__init__`
- Implement `forward` method for computation logic
- Use `Sequential` for simple linear architectures
- Use `ModuleList` for flexible layer management

### 3. **Training**
- Call `model.train()` before training
- Call `zero_grad()` before backward pass to prevent gradient accumulation
- Monitor loss to detect divergence
- Use appropriate learning rates (typically 0.001 - 0.1)

### 4. **Evaluation**
- Call `model.eval()` before inference
- Disable gradient computation for inference: `Tensor.grad_is_enabled = False`
- Use `state_dict()` to save/load model weights

### 5. **Numerical Stability**
- Use loss functions with built-in stability (e.g., BCELoss)
- Apply gradient clipping for unstable gradients
- Normalize input data
- Use layer normalization for deep networks

---

## Known Limitations

1. **No GPU Support**: Pure NumPy implementation, runs on CPU only
2. **Performance**: Not optimized for production use; use PyTorch/TensorFlow for large-scale training
3. **Incomplete Implementations**: Some features (e.g., optimizers) are placeholders
4. **Limited Broadcasting**: Broadcasting logic simpler than NumPy's full implementation
5. **No Batch Normalization**: Missing important normalization technique for deep networks

---

## Contributing

Contributions are welcome! Areas for improvement:

- Implement optimizers (SGD, Adam, RMSprop)
- Add batch normalization layer
- Optimize gradient computation
- Add more activation functions (GELU, Swish)
- Implement convolution operations
- Add comprehensive unit tests
- Performance benchmarking

### Guidelines
1. Follow existing code style and conventions
2. Add docstrings for new methods
3. Test thoroughly before submitting
4. Update documentation for new features

---

## Related Resources

### Understanding Automatic Differentiation
- [Autograd: Automatic Differentiation](https://github.com/HIPS/autograd)
- [The Mechanics of Neural Networks - 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk)
- [Backpropagation Explained](http://neuralnetworksanddeeplearning.com/)

### PyTorch References
- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### Deep Learning Theory
- [Deep Learning Book - Goodfellow et al.](https://www.deeplearningbook.org/)
- [Stanford CS231n - Convolutional Neural Networks](http://cs231n.stanford.edu/)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

NNGrad is inspired by:
- **PyTorch**: For the modular API design
- **Autograd**: For the automatic differentiation framework
- **Micrograd**: For demonstrating core concepts clearly

---

## Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Create a discussion for feature requests
- Submit pull requests for contributions

---

<div align="center">

Made with ❤️ for learning deep learning from first principles.

⭐ If you find this helpful, please consider starring the repository!

</div>
