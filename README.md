# NNGrad

NNgrad: A NumPy-Based Automatic Differentiation Engine
This repository contains NNgrad, a fundamental deep learning library and PyTorch-like automatic differentiation (Autograd) engine built entirely on NumPy. It provides the essential components—a Tensor data structure, core arithmetic/array operations with custom backward passes, and a modular Neural Network API—necessary for building and training simple deep learning models.

⚙️ Core Architecture and Features
1. The Autograd Engine (core/engine.py)
The heart of NNgrad is the Tensor class, which serves as the central data structure for all computations.

Automatic Differentiation: The Tensor maintains its numerical data, an optional gradient grad, a reference to its parent tensors (prev), and the operation (op) that created it.

Backpropagation: The backward() method initiates the gradient flow. It first performs a topological sort on the computation graph to determine the correct order of operations and includes logic for cycle detection. It then iterates through the nodes in reverse order, applying the custom gradient functions (grad_fn) to accumulate gradients for each operation.

Supported Operations (with custom backward passes):

Array Manipulation: sum(), T() (transpose), reshape(), broadcast(), concat(), array indexing (__getitem__), and element assignment (__setitem__).

Element-wise: exp() (exponential), log() (natural logarithm).

Utilities: clip() (data and gradient clipping) and masked_fill() (allows masking out gradient flow).

2. Neural Network Modules (core/nn)
The framework provides a modular and extensible system for defining neural network layers and models.

Base Module: The foundation for all layers. It handles parameter tracking (parameters()), gradient clearing (zero_grad()), and model state management (train(), eval(), state_dict()).

Model Containers: Facilitates complex model construction:

  Sequential: Chains modules together, passing the output of one as input to the next.

  ModuleList & ModuleDict: Provide list-like and dictionary-like containers for modules.
