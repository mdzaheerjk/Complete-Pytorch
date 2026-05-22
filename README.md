# 🔥 PyTorch: Complete Notes
### From Absolute Beginner → Industry-Ready Production Engineer

> **Who is this for?** ML/DL students, AI/ML engineers, researchers, and anyone who wants to go from zero to production with PyTorch.  
> **How to use:** Read sequentially. Every section builds on the last. Code every example yourself.  
> **Philosophy:** PyTorch = "Define-by-Run" — dynamic computation graphs, pythonic, research-first.

---

## 📋 Table of Contents

1. [What is PyTorch?](#1-what-is-pytorch)
2. [Installation & Setup](#2-installation--setup)
3. [Core Concepts: Tensors](#3-core-concepts-tensors)
4. [Autograd: Automatic Differentiation](#4-autograd-automatic-differentiation)
5. [Building Neural Networks with nn.Module](#5-building-neural-networks-with-nnmodule)
6. [Loss Functions](#6-loss-functions)
7. [Optimizers](#7-optimizers)
8. [Training Loop (The PyTorch Way)](#8-training-loop-the-pytorch-way)
9. [Data Loading: Dataset & DataLoader](#9-data-loading-dataset--dataloader)
10. [Convolutional Neural Networks (CNNs)](#10-convolutional-neural-networks-cnns)
11. [Recurrent Neural Networks (RNNs, LSTM, GRU)](#11-recurrent-neural-networks-rnns-lstm-gru)
12. [Transfer Learning & Fine-Tuning](#12-transfer-learning--fine-tuning)
13. [Regularization Techniques](#13-regularization-techniques)
14. [Learning Rate Schedulers](#14-learning-rate-schedulers)
15. [Saving & Loading Models](#15-saving--loading-models)
16. [GPU Acceleration (CUDA)](#16-gpu-acceleration-cuda)
17. [TorchVision (Computer Vision)](#17-torchvision-computer-vision)
18. [TorchText & NLP Pipelines](#18-torchtext--nlp-pipelines)
19. [Transformers & Attention](#19-transformers--attention)
20. [Custom Layers & Functions](#20-custom-layers--functions)
21. [Mixed Precision Training](#21-mixed-precision-training)
22. [Distributed Training](#22-distributed-training)
23. [TorchScript & Model Export](#23-torchscript--model-export)
24. [ONNX Export & Deployment](#24-onnx-export--deployment)
25. [PyTorch Lightning (Production Framework)](#25-pytorch-lightning-production-framework)
26. [Debugging & Profiling](#26-debugging--profiling)
27. [Performance Optimization](#27-performance-optimization)
28. [Real-World Project Checklist](#28-real-world-project-checklist)
29. [Quick Reference Cheat Sheet](#29-quick-reference-cheat-sheet)

---

## 1. What is PyTorch?

PyTorch is an **open-source deep learning framework** developed by Meta AI (Facebook), released in 2016. It's the dominant framework in research and increasingly in production.

### Why PyTorch?
- **Dynamic computation graphs** — graph built on-the-fly, easy to debug
- **Pythonic** — feels like native Python/NumPy
- **Research-friendly** — easy to prototype novel architectures
- **Production-ready** — TorchScript, ONNX, TorchServe, ExecuTorch
- **Massive ecosystem** — HuggingFace, Lightning, timm, MMDetection, etc.

### PyTorch vs TensorFlow 2.x

| Aspect | PyTorch | TensorFlow 2.x |
|---|---|---|
| Execution | Eager (dynamic graph) | Eager + static via @tf.function |
| Debugging | Easy (native Python) | Medium |
| Research Use | Dominant (80%+ papers) | Less common |
| Production | TorchServe, ONNX, TorchScript | TF Serving, TFLite (more mature) |
| Mobile | ExecuTorch | TFLite (more mature) |
| NLP/LLM Ecosystem | Dominant (HuggingFace) | Limited |
| Learning Curve | Moderate | Moderate |

### PyTorch Ecosystem
```
torch          — Core tensor library + autograd
torch.nn       — Neural network layers and losses
torch.optim    — Optimizers
torch.utils.data — Dataset, DataLoader
torchvision    — Computer vision (datasets, models, transforms)
torchaudio     — Audio processing
torchtext      — Text processing
torch.distributed — Distributed training
torch.jit      — TorchScript (model compilation)
torch.onnx     — ONNX export
```

---

## 2. Installation & Setup

### System Requirements
- Python 3.8 – 3.12
- pip 21.0+
- (Optional) NVIDIA GPU with CUDA 11.8+ or 12.x

### Installation

```bash
# CPU only
pip install torch torchvision torchaudio

# GPU (CUDA 12.1) — visit pytorch.org for exact command
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Via conda (recommended for CUDA management)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Standard Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Device setup — always do this!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### Reproducibility

```python
import torch, random, numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 3. Core Concepts: Tensors

A **tensor** is the fundamental data structure in PyTorch — a multi-dimensional array, similar to NumPy's ndarray but with GPU support and autograd.

### Creating Tensors

```python
import torch

# From Python data
a = torch.tensor(42)                            # Scalar
b = torch.tensor([1.0, 2.0, 3.0])              # 1D tensor
c = torch.tensor([[1, 2, 3], [4, 5, 6]])        # 2D tensor
d = torch.tensor([[[1, 2], [3, 4]],
                  [[5, 6], [7, 8]]])             # 3D tensor

# Dtype inference from Python types
torch.tensor([1, 2, 3])          # torch.int64
torch.tensor([1.0, 2.0, 3.0])   # torch.float32
torch.tensor([True, False])      # torch.bool

# Factory functions
torch.zeros(3, 4)                    # All zeros, shape (3,4)
torch.ones(2, 3)                     # All ones
torch.eye(4)                         # 4x4 identity matrix
torch.empty(3, 3)                    # Uninitialized memory
torch.arange(0, 10, step=2)         # [0, 2, 4, 6, 8]
torch.linspace(0, 1, steps=5)       # [0.0, 0.25, 0.5, 0.75, 1.0]
torch.full((3, 3), fill_value=7)    # All 7s

# Random tensors
torch.rand(3, 4)                     # Uniform [0, 1)
torch.randn(3, 4)                    # Normal (mean=0, std=1)
torch.randint(0, 10, size=(3, 4))   # Random integers [0, 10)
torch.rand_like(c.float())           # Same shape as c

# From NumPy (shares memory!)
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)           # Shares memory with arr
t = torch.tensor(arr)               # Copies data

# To NumPy
arr = t.numpy()                     # On CPU only
arr = t.detach().cpu().numpy()      # Safe version (works always)
```

### Tensor Properties

```python
t = torch.randn(3, 4)

t.shape         # torch.Size([3, 4])
t.size()        # torch.Size([3, 4])
t.ndim          # 2
t.dtype         # torch.float32
t.device        # device(type='cpu')
t.requires_grad # False
t.is_cuda       # False
t.numel()       # 12 (total elements)
t.element_size()# 4 (bytes per element for float32)
t.is_contiguous()  # True/False — memory layout
```

### Data Types

```python
torch.float16   # Half precision (FP16)
torch.float32   # Single precision (default for weights)
torch.float64   # Double precision
torch.int8
torch.int16
torch.int32
torch.int64     # Default for integer tensors
torch.bool
torch.complex64
torch.bfloat16  # Brain float (training on TPUs/A100s)

# Type conversion
x = torch.tensor([1, 2, 3])          # int64
x = x.float()                         # float32
x = x.to(torch.float16)              # float16
x = x.type(torch.DoubleTensor)       # float64
```

### Tensor Operations

```python
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])

# Arithmetic (element-wise)
a + b           # or torch.add(a, b)
a - b           # or torch.sub(a, b)
a * b           # or torch.mul(a, b)  — element-wise!
a / b           # or torch.div(a, b)
a ** 2          # or torch.pow(a, 2)
torch.sqrt(a)
torch.exp(a)
torch.log(a)

# Matrix operations
torch.mm(a, b)          # Matrix multiply (2D only)
torch.matmul(a, b)      # Matrix multiply (any dim), or a @ b
torch.bmm(A, B)         # Batched matrix multiply: (B,n,m) @ (B,m,k)
torch.dot(v1, v2)       # Dot product (1D only)
torch.outer(v1, v2)     # Outer product

# Reduction
torch.sum(a)              # Sum all
torch.sum(a, dim=0)       # Sum along rows
torch.mean(a)
torch.std(a)
torch.min(a)
torch.max(a)
torch.argmax(a, dim=1)   # Index of max along dim 1
torch.topk(a, k=2)       # Top-k values and indices

# Shape operations
a.reshape(4, 1)           # New shape, shares data when possible
a.view(4, 1)              # Like reshape but requires contiguous
a.contiguous()            # Make contiguous in memory
a.squeeze()               # Remove dims of size 1
a.unsqueeze(0)            # Add dim at position 0
a.expand(3, 2, 2)         # Expand without copying memory
a.repeat(2, 3)            # Repeat along dimensions (copies memory)
a.permute(1, 0)           # Reorder dimensions
a.transpose(0, 1)         # Swap two dimensions
a.flatten()               # Flatten to 1D
a.flatten(start_dim=1)    # Flatten from dim 1 onwards

# Concatenation
torch.cat([a, b], dim=0)   # Concatenate along existing dim
torch.stack([a, b], dim=0) # Stack along NEW dim
torch.split(a, 1, dim=0)   # Split into chunks

# In-place operations (modifies tensor, use carefully with autograd)
a.add_(b)       # a += b  — trailing _ means in-place
a.mul_(2)       # a *= 2
a.zero_()       # a = 0

# Comparison
a == b           # Element-wise equality
torch.eq(a, b)
a > 2
torch.where(a > 2, a, torch.zeros_like(a))  # Conditional select

# Indexing & Slicing (NumPy-like)
t = torch.arange(12).reshape(3, 4)
t[0]            # Row 0
t[:, 1]         # Column 1
t[1:, :2]       # Submatrix
t[[0, 2]]       # Rows 0 and 2 (fancy indexing)
t[t > 5]        # Boolean indexing
```

### Broadcasting Rules

```python
# Tensors broadcast if trailing dimensions are compatible
a = torch.ones(3, 1)    # shape (3,1)
b = torch.ones(1, 4)    # shape (1,4)
(a + b).shape           # (3,4) — broadcast!

# Rules:
# 1. Align shapes from the right
# 2. Dimensions must match OR one of them must be 1
```

---

## 4. Autograd: Automatic Differentiation

PyTorch's **autograd** engine computes gradients automatically by recording operations on tensors with `requires_grad=True`.

### Basic Gradient Computation

```python
# Create tensor with gradient tracking
x = torch.tensor(3.0, requires_grad=True)

# Forward pass — PyTorch builds computation graph
y = x ** 2 + 2 * x + 1   # y = x² + 2x + 1

# Backward pass — compute gradients
y.backward()

# Access gradient: dy/dx = 2x + 2 = 2*3 + 2 = 8
print(x.grad)   # tensor(8.)
```

### Gradients with Multiple Variables

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()   # y = x1² + x2² + x3²

y.backward()
print(x.grad)        # tensor([2., 4., 6.])  → dy/dxi = 2xi
```

### Gradient Accumulation

```python
# PyTorch ACCUMULATES gradients — always zero before backward!
x = torch.tensor(2.0, requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()
    print(x.grad)    # 4, 8, 12 — keeps accumulating!
    # x.grad.zero_()  ← you must call this in training loops
```

### Controlling Gradient Flow

```python
# 1. No gradient tracking (inference / preprocessing)
with torch.no_grad():
    y = model(x)   # No graph built, saves memory

# 2. Disable gradient for specific tensors
x = x.detach()    # Detach from computation graph
# or
x.requires_grad_(False)

# 3. Only compute gradients for subset
for param in model.parameters():
    param.requires_grad = False  # Freeze layer

# 4. Retain graph for multiple backward passes
y.backward(retain_graph=True)   # Keeps graph after backward

# 5. Gradient of non-scalar output (use Jacobian-vector product)
v = torch.tensor([1.0, 1.0, 1.0])
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
y.backward(gradient=v)   # Equivalent to (y * v).sum().backward()
```

### Higher-Order Derivatives

```python
x = torch.tensor(2.0, requires_grad=True)

# First derivative
dy_dx = torch.autograd.grad(x**3, x, create_graph=True)[0]  # 3x²=12

# Second derivative
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]   # 6x=12
print(d2y_dx2)   # tensor(12.)
```

### How Autograd Works (Mental Model)

```
Forward pass:  x → [op1] → [op2] → [op3] → loss
               PyTorch records each op in a DAG

Backward pass: loss.backward()
               PyTorch traverses DAG in reverse
               Applies chain rule at each node
               Accumulates gradient into .grad attribute
```

---

## 5. Building Neural Networks with nn.Module

`nn.Module` is the **base class** for all neural networks in PyTorch.

### The nn.Module Contract

```python
class MyLayer(nn.Module):
    def __init__(self):
        super().__init__()   # ALWAYS call super().__init__()
        # Define learnable parameters / sub-modules here

    def forward(self, x):
        # Define the forward computation here
        return x
```

### Common nn Layers

```python
import torch.nn as nn

# Linear / Fully Connected
nn.Linear(in_features=128, out_features=64, bias=True)

# Convolutional
nn.Conv1d(in_channels, out_channels, kernel_size)
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
nn.Conv3d(in_channels, out_channels, kernel_size)
nn.ConvTranspose2d(in_channels, out_channels, kernel_size)  # Upsampling

# Pooling
nn.MaxPool2d(kernel_size=2, stride=2)
nn.AvgPool2d(kernel_size=2)
nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Output fixed size regardless of input
nn.AdaptiveMaxPool2d(output_size=(4, 4))

# Normalization
nn.BatchNorm1d(num_features)
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)
nn.GroupNorm(num_groups, num_channels)
nn.InstanceNorm2d(num_features)

# Recurrent
nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.0)
nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

# Embedding
nn.Embedding(num_embeddings, embedding_dim, padding_idx=None)
nn.EmbeddingBag(num_embeddings, embedding_dim, mode='mean')

# Dropout
nn.Dropout(p=0.5)
nn.Dropout2d(p=0.5)    # Zero whole channels
nn.AlphaDropout(p=0.5)

# Activation functions
nn.ReLU()
nn.LeakyReLU(negative_slope=0.01)
nn.ELU(alpha=1.0)
nn.GELU()              # Used in Transformers
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=-1)
nn.LogSoftmax(dim=-1)
nn.Mish()
nn.SiLU()              # Swish activation

# Attention
nn.MultiheadAttention(embed_dim, num_heads)

# Transformer
nn.TransformerEncoderLayer(d_model, nhead)
nn.TransformerEncoder(encoder_layer, num_layers)

# Flatten
nn.Flatten(start_dim=1)

# Containers
nn.Sequential(...)        # Stack layers sequentially
nn.ModuleList([...])      # List of modules (registered properly)
nn.ModuleDict({...})      # Dict of modules (registered properly)
```

### Model Architecture Patterns

#### Pattern 1: Sequential (Simple Stacks)

```python
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)
)
```

#### Pattern 2: nn.Module Subclass (Standard)

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x):
        x = self.hidden(x)
        return self.output(x)

model = MLP(784, [512, 256, 128], 10)
```

#### Pattern 3: Functional API (nn.functional)

```python
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)  # respects train/eval mode
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.fc3(x)   # Raw logits (no softmax — use CrossEntropyLoss)
```

### Model Inspection

```python
model = MLP(784, [512, 256], 10)

# Print architecture
print(model)

# Count parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")

# Iterate parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

# Iterate modules
for name, module in model.named_modules():
    print(f"{name}: {type(module).__name__}")

# Access specific layer
model.hidden[0].weight    # First Linear layer's weight
model.hidden[0].bias

# Train / Eval mode (affects Dropout, BatchNorm)
model.train()   # Training mode
model.eval()    # Inference mode
```

---

## 6. Loss Functions

```python
import torch.nn as nn
import torch.nn.functional as F

# --- CLASSIFICATION LOSSES ---

# Binary Cross-Entropy (with sigmoid built-in)
nn.BCEWithLogitsLoss()      # Input: raw logits; PREFERRED over BCELoss
nn.BCELoss()                # Input: probabilities (after sigmoid)

# Multi-class Cross-Entropy (with softmax built-in)
nn.CrossEntropyLoss()       # Input: raw logits, Target: class indices (int)
                            # = log_softmax + NLLLoss
nn.NLLLoss()                # Input: log-probabilities, Target: class indices

# --- REGRESSION LOSSES ---
nn.MSELoss()                # Mean Squared Error: L = mean((y - ŷ)²)
nn.L1Loss()                 # Mean Absolute Error: L = mean(|y - ŷ|)
nn.HuberLoss(delta=1.0)     # Smooth L1 — robust to outliers
nn.SmoothL1Loss()           # Same as Huber with delta=1

# --- RANKING / SIMILARITY ---
nn.CosineEmbeddingLoss()    # For cosine similarity tasks
nn.TripletMarginLoss()      # For triplet networks
nn.MarginRankingLoss()      # For ranking tasks

# --- USAGE ---
criterion = nn.CrossEntropyLoss()

# For class imbalance — weight each class
weights = torch.tensor([1.0, 5.0, 2.0])   # class 1 is rare, weight it more
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

# Label smoothing (regularization)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Reduction options
nn.MSELoss(reduction='mean')   # Default: average over batch
nn.MSELoss(reduction='sum')    # Sum over batch
nn.MSELoss(reduction='none')   # Per-element loss (for custom weighting)

# Functional equivalents
F.cross_entropy(logits, targets)
F.mse_loss(preds, targets)
F.binary_cross_entropy_with_logits(logits, targets)
```

### Custom Loss

```python
class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)                          # p(correct class)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

---

## 7. Optimizers

```python
import torch.optim as optim

# --- STANDARD OPTIMIZERS ---
optim.SGD(model.parameters(), lr=0.01)
optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

optim.Adam(model.parameters(), lr=1e-3)
optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)  # Adam + decoupled weight decay
optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
optim.Adagrad(model.parameters(), lr=0.01)
optim.Adadelta(model.parameters())
optim.NAdam(model.parameters(), lr=1e-3)   # Nesterov Adam

# --- PER-LAYER LEARNING RATES ---
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},   # Pretrained: small LR
    {'params': model.head.parameters(),     'lr': 1e-3},   # New head: larger LR
], lr=1e-3)

# --- OPTIMIZER OPERATIONS ---
optimizer.zero_grad()        # Clear accumulated gradients (call before backward!)
optimizer.step()             # Update weights using current gradients
optimizer.state_dict()       # Save optimizer state
optimizer.load_state_dict(state_dict)  # Restore optimizer state

# --- GRADIENT CLIPPING (before optimizer.step) ---
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # Clip by norm
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5) # Clip by value
```

### Optimizer Selection Guide

| Problem | Optimizer | Notes |
|---|---|---|
| General | Adam | Best default choice |
| Vision (training from scratch) | SGD + momentum | Often beats Adam on final accuracy |
| Vision (fine-tuning) | AdamW | Stable fine-tuning |
| NLP / Transformers | AdamW | Standard for BERT, GPT |
| Large batch training | LARS / LAMB | Distributed training |
| When overfitting | AdamW + weight_decay | Regularization built-in |

---

## 8. Training Loop (The PyTorch Way)

The **training loop** is explicit in PyTorch — you write it yourself. This gives full control.

### The Standard Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()   # IMPORTANT: sets Dropout/BN to training mode
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        # Move to device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 1. Zero gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        outputs = model(inputs)
        
        # 3. Compute loss
        loss = criterion(outputs, targets)
        
        # 4. Backward pass
        loss.backward()
        
        # 5. (Optional) Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 6. Update weights
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()   # Disable gradient computation for validation
def evaluate(model, loader, criterion, device):
    model.eval()   # IMPORTANT: disables Dropout, uses BN running stats
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100.0 * correct / total


# --- FULL TRAINING PIPELINE ---

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# Training loop
EPOCHS = 50
best_val_acc = 0.0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
    
    scheduler.step(val_loss)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")
```

### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

# Usage
early_stopping = EarlyStopping(patience=10, mode='min')
for epoch in range(EPOCHS):
    val_loss, _ = evaluate(model, val_loader, criterion, device)
    early_stopping(val_loss)
    if early_stopping.should_stop:
        print("Early stopping triggered!")
        break
```

---

## 9. Data Loading: Dataset & DataLoader

### Custom Dataset

```python
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Template for any custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)          # Total number of samples
    
    def __getitem__(self, idx):        # Fetch single sample by index
        sample = self.data[idx]
        label  = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


# CSV Dataset Example
class CSVDataset(Dataset):
    def __init__(self, csv_path, feature_cols, target_col, transform=None):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_col].values, dtype=torch.long)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


# Image Dataset Example
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
```

### DataLoader

```python
# Create datasets
train_dataset = CustomDataset(X_train, y_train)
val_dataset   = CustomDataset(X_val, y_val)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,          # Shuffle training data every epoch
    num_workers=4,         # Parallel data loading (0 = main thread)
    pin_memory=True,       # Faster GPU transfer (use with CUDA)
    drop_last=True,        # Drop last incomplete batch
    persistent_workers=True  # Keep workers alive between epochs
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,        # Can use larger batch for validation
    shuffle=False,         # Don't shuffle validation
    num_workers=4,
    pin_memory=True
)

# Splitting a dataset
from torch.utils.data import random_split

full_dataset = CustomDataset(X, y)
n_train = int(0.8 * len(full_dataset))
n_val   = len(full_dataset) - n_train
train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
```

### TorchVision Transforms

```python
import torchvision.transforms as transforms
import torchvision.transforms.v2 as T  # New v2 API (recommended)

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),                               # PIL → Tensor [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# Validation/Test transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Reverse normalization (for visualization)
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(3, 1, 1)
    std  = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean
```

### Built-in Datasets (torchvision)

```python
import torchvision.datasets as datasets

# Standard datasets
datasets.MNIST(root='./data', train=True, download=True, transform=transform)
datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
datasets.ImageNet(root='./data', split='train', transform=transform)

# From folder (directory structure: root/class_a/img1.jpg, root/class_b/img2.jpg)
dataset = datasets.ImageFolder(root='./data/train', transform=train_transform)
print(dataset.classes)        # ['cat', 'dog', ...]
print(dataset.class_to_idx)  # {'cat': 0, 'dog': 1, ...}
```

---

## 10. Convolutional Neural Networks (CNNs)

### CNN Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Reusable conv block: Conv → BN → ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class CIFAR_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: 3 → 64 channels
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),    # 32x32 → 16x16
            nn.Dropout2d(0.25),
            
            # Block 2: 64 → 128 channels
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),    # 16x16 → 8x8
            nn.Dropout2d(0.25),
            
            # Block 3: 128 → 256 channels
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.AdaptiveAvgPool2d((2, 2)),  # Any input → 2x2 output
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# --- TRAINING ---
import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform_train)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
testloader  = DataLoader(testset,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CIFAR_CNN(num_classes=10).to(device)
```

### ResNet-Style Skip Connections

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
        # Shortcut projection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)   # Skip connection
        return F.relu(out)
```

---

## 11. Recurrent Neural Networks (RNNs, LSTM, GRU)

### LSTM for Text Classification

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,         # input shape: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x):
        # x shape: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))   # (batch, seq, embed_dim)
        
        output, (hidden, cell) = self.lstm(embedded)
        # output: (batch, seq, hidden*2)
        # hidden: (num_layers*2, batch, hidden)
        
        # Use last hidden state from both directions
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)   # (batch, hidden*2)
        
        return self.classifier(self.dropout(hidden))


# Using packed sequences for variable-length inputs
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PackedLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        
        # Pack padded sequences
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (hidden, _) = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)  # Unpack
        
        return self.fc(hidden[-1])  # Use last hidden state
```

### Time Series with LSTM

```python
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use only the last time step's output
        return self.head(lstm_out[:, -1, :])
```

---

## 12. Transfer Learning & Fine-Tuning

### Loading Pretrained Models

```python
import torchvision.models as models

# Load pretrained model
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

# Available models (torchvision 0.14+)
# models.alexnet, vgg16, resnet18/34/50/101/152
# models.densenet121, inception_v3, googlenet
# models.mobilenet_v2/v3, efficientnet_b0-b7
# models.vit_b_16, vit_b_32, vit_l_16
# models.swin_t, swin_s, swin_b
# models.convnext_tiny/small/base/large
```

### Strategy 1: Feature Extraction (Freeze Backbone)

```python
# Method: Replace classifier, freeze backbone
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final classification head
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)
# Only model.fc parameters have requires_grad=True

model = model.to(device)

# Only optimize new head parameters
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
```

### Strategy 2: Fine-Tuning (Gradual Unfreezing)

```python
# Phase 1: Train head only
model = models.efficientnet_b0(weights='DEFAULT')

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, num_classes)
)

optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
train(model, optimizer, epochs=5)   # Train head only

# Phase 2: Unfreeze and fine-tune with smaller LR
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW([
    {'params': model.features.parameters(), 'lr': 1e-5},   # Backbone: tiny LR
    {'params': model.classifier.parameters(), 'lr': 1e-4}, # Head: small LR
], weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
train(model, optimizer, scheduler, epochs=20)
```

---

## 13. Regularization Techniques

```python
# 1. DROPOUT
nn.Dropout(p=0.5)         # Standard dropout
nn.Dropout2d(p=0.5)       # Drop entire channels (for CNNs)

# 2. WEIGHT DECAY (L2 Regularization)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# 3. BATCH NORMALIZATION
nn.BatchNorm2d(64)    # After Conv2d
nn.BatchNorm1d(256)   # After Linear
# Benefits: acts as regularizer, allows higher LR, reduces sensitivity to init

# 4. DATA AUGMENTATION (best regularizer for vision)
transforms.RandomHorizontalFlip()
transforms.RandomCrop(32, padding=4)
transforms.ColorJitter(brightness=0.2, contrast=0.2)
transforms.RandomErasing(p=0.5)    # CutOut
transforms.RandomGrayscale(p=0.1)

# MixUp augmentation
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# CutMix augmentation
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return x, y, y[idx], lam

# 5. LABEL SMOOTHING
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 6. STOCHASTIC DEPTH (DropPath) — for ResNets/ViTs
class StochasticDepth(nn.Module):
    def __init__(self, survival_prob=0.8):
        super().__init__()
        self.survival_prob = survival_prob
    
    def forward(self, x):
        if not self.training:
            return x
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return x / self.survival_prob * binary_tensor
```

---

## 14. Learning Rate Schedulers

```python
import torch.optim.lr_scheduler as lr_scheduler

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- STEP-BASED ---
# Reduce LR by gamma every step_size epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Reduce LR by gamma at specific milestones
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

# Multiply LR by gamma every epoch
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# --- ADAPTIVE ---
# Reduce LR when metric stops improving
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5,
    min_lr=1e-7, verbose=True
)

# --- COSINE ANNEALING ---
# Oscillate LR between max and min
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Cosine annealing with warm restarts (SGDR)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# --- LINEAR WARMUP + COSINE DECAY (Transformer standard) ---
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer, max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.1,     # 10% of training for warmup
    anneal_strategy='cos'
)

# --- CYCLIC ---
scheduler = lr_scheduler.CyclicLR(
    optimizer, base_lr=1e-5, max_lr=1e-2,
    step_size_up=2000, mode='triangular2'
)

# --- CUSTOM LAMBDA ---
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

# --- USAGE IN TRAINING LOOP ---
for epoch in range(EPOCHS):
    train(...)
    val_loss = validate(...)
    
    # For most schedulers:
    scheduler.step()
    
    # For ReduceLROnPlateau — pass the metric:
    # scheduler.step(val_loss)
    
    # For OneCycleLR — call inside batch loop:
    # scheduler.step() inside the for batch loop

# Check current LR
print(optimizer.param_groups[0]['lr'])
```

### Warmup + Cosine Decay (Manual Implementation)

```python
def get_lr_with_warmup(step, d_model, warmup_steps=4000):
    """Transformer-style learning rate schedule (Vaswani et al.)"""
    if step == 0:
        step = 1
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

scheduler = lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: get_lr_with_warmup(step, d_model=512)
)
```

---

## 15. Saving & Loading Models

### Save/Load State Dict (Recommended)

```python
# SAVE — only weights (portable, no class definition needed at save time)
torch.save(model.state_dict(), 'model_weights.pth')

# LOAD — need model class definition
model = MyModel()
model.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
model.eval()   # Set to eval mode!

# Save with map_location for GPU→CPU compatibility
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
```

### Save/Load Full Checkpoint (During Training)

```python
# SAVE complete training state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_acc': val_acc,
    'config': {'lr': 1e-3, 'batch_size': 64}
}
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')

# LOAD and resume training
checkpoint = torch.load('checkpoint_epoch_10.pth', map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1

for epoch in range(start_epoch, EPOCHS):
    # Continue training...
```

### Save Full Model (Not Recommended for Production)

```python
# Save entire model (includes class definition — brittle)
torch.save(model, 'full_model.pth')

# Load (requires same class to be importable)
model = torch.load('full_model.pth')
```

### Best Model Saving Pattern

```python
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(...)
    val_loss   = evaluate(...)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }, 'best_checkpoint.pth')
        print(f"✓ Saved best model (val_loss={val_loss:.4f})")
```

---

## 16. GPU Acceleration (CUDA)

### Device Management

```python
# Check GPU availability
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0))

# Memory info
print(torch.cuda.memory_allocated(0) / 1024**2, "MB allocated")
print(torch.cuda.memory_reserved(0) / 1024**2,  "MB reserved")

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Move model to GPU
model = model.to(device)
model = model.cuda()   # Equivalent, shorthand

# Move tensors to GPU
x = x.to(device)
x = x.cuda()

# Move back to CPU (e.g., for visualization)
x_cpu = x.cpu()
x_numpy = x.detach().cpu().numpy()
```

### Multi-GPU with DataParallel (Simple)

```python
# Wrap model for data-parallel training
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

model = model.to(device)

# Access underlying model (to save weights)
if isinstance(model, nn.DataParallel):
    torch.save(model.module.state_dict(), 'weights.pth')
else:
    torch.save(model.state_dict(), 'weights.pth')
```

### Memory Optimization

```python
# Clear cache between experiments
torch.cuda.empty_cache()

# Gradient checkpointing (trade compute for memory)
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def forward(self, x):
        # Instead of:   x = self.heavy_layer(x)
        # Use:
        x = checkpoint(self.heavy_layer, x)   # Recompute activations in backward
        return x

# Reduce batch size / use gradient accumulation
accumulation_steps = 4  # Simulate batch_size * 4

optimizer.zero_grad()
for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs.to(device))
    loss = criterion(outputs, targets.to(device)) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 17. TorchVision (Computer Vision)

### Image Classification Pipeline

```python
import torchvision
import torchvision.transforms.v2 as T

# Modern transforms (v2 API)
train_transform = T.Compose([
    T.RandomResizedCrop(224, antialias=True),
    T.RandomHorizontalFlip(),
    T.TrivialAugmentWide(),        # State-of-the-art augmentation
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# AugMix (robust to corruption)
train_transform = T.Compose([
    T.RandomResizedCrop(224, antialias=True),
    T.AugMix(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### Object Detection with torchvision

```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

# Load pretrained Faster R-CNN
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
model.eval()

# Inference
from torchvision.transforms.functional import to_tensor
from PIL import Image

image = to_tensor(Image.open('image.jpg'))

with torch.no_grad():
    predictions = model([image])

boxes  = predictions[0]['boxes']    # (N, 4) — x1,y1,x2,y2
labels = predictions[0]['labels']   # (N,)
scores = predictions[0]['scores']   # (N,)

# Filter by confidence
mask = scores > 0.5
boxes  = boxes[mask]
labels = labels[mask]
```

### Semantic Segmentation

```python
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
model.eval()

with torch.no_grad():
    output = model(image_batch)['out']   # (B, num_classes, H, W)
    predicted = output.argmax(1)          # (B, H, W) class per pixel
```

---

## 18. TorchText & NLP Pipelines

### Text Classification Pipeline

```python
# Modern approach: use torchtext or HuggingFace tokenizers
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS

# Tokenizer
tokenizer = get_tokenizer('basic_english')

# Build vocabulary
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

train_iter = AG_NEWS(split='train')
vocab = build_vocab_from_iterator(
    yield_tokens(train_iter),
    specials=["<unk>", "<pad>"],
    min_freq=2
)
vocab.set_default_index(vocab["<unk>"])

# Text pipeline
text_pipeline  = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1   # 1-indexed to 0-indexed

# Collate function for DataLoader
def collate_batch(batch):
    labels, texts = [], []
    for label, text in batch:
        labels.append(label_pipeline(label))
        texts.append(torch.tensor(text_pipeline(text), dtype=torch.long))
    
    labels = torch.tensor(labels, dtype=torch.long)
    texts  = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    return texts, labels

loader = DataLoader(dataset, batch_size=64, collate_fn=collate_batch)
```

### HuggingFace Transformers (Industry Standard for NLP)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pretrained BERT
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
).to(device)

# Tokenize
texts = ["This movie is great!", "This movie is terrible."]
encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

# Inference
model.eval()
with torch.no_grad():
    outputs = model(**{k: v.to(device) for k, v in encodings.items()})
    logits = outputs.logits
    predictions = logits.argmax(dim=-1)

# Fine-tuning
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
# Training loop is identical to standard PyTorch
```

---

## 19. Transformers & Attention

### Self-Attention from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.d_k)
        return x.transpose(1, 2)   # (B, heads, T, d_k)
    
    def forward(self, x, mask=None):
        B, T, _ = x.shape
        
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        
        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = self.dropout(F.softmax(scores, dim=-1))
        out  = attn @ V
        
        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-norm (modern style)
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class GPT_Mini(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len, ff_dim, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_size, bias=False)
        self.drop  = nn.Dropout(dropout)
    
    def forward(self, idx):
        B, T = idx.shape
        
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        
        # Causal mask (lower triangular)
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        return self.head(x)   # (B, T, vocab_size)
```

### Using PyTorch's Built-in Transformer

```python
# Encoder-only (BERT-like)
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True
)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# Encoder-Decoder (Translation-like)
transformer = nn.Transformer(
    d_model=512, nhead=8,
    num_encoder_layers=6, num_decoder_layers=6,
    dim_feedforward=2048, dropout=0.1,
    batch_first=True
)

src = torch.rand(32, 10, 512)   # (batch, src_seq, d_model)
tgt = torch.rand(32, 20, 512)   # (batch, tgt_seq, d_model)
out = transformer(src, tgt)     # (32, 20, 512)
```

---

## 20. Custom Layers & Functions

### Custom Layer

```python
class SelfNormalizingLinear(nn.Module):
    """SELU activation + alpha dropout for self-normalizing networks."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Initialize for SELU
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return F.selu(self.linear(x))


# Custom weight initialization
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

model.apply(init_weights)
```

### Custom Autograd Function

```python
class StraightThroughEstimator(torch.autograd.Function):
    """Quantization with straight-through gradient estimator."""
    @staticmethod
    def forward(ctx, x):
        return x.round()    # Quantize (non-differentiable)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output   # Pass gradient straight through
    
quantize = StraightThroughEstimator.apply


class SwishFunction(torch.autograd.Function):
    """Custom Swish activation with hand-written backward pass."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sig = torch.sigmoid(x)
        return grad_output * (sig + x * sig * (1 - sig))

swish = SwishFunction.apply
```

### Parameter-Free Layers

```python
class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, _, _ = x.shape
        scale = self.excite(self.squeeze(x)).view(B, C, 1, 1)
        return x * scale
```

---

## 21. Mixed Precision Training

Train with **float16** for 2-3x speedup and 50% memory reduction on modern GPUs (Volta+).

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler (prevents float16 underflow)
scaler = GradScaler()

def train_one_epoch_amp(model, loader, criterion, optimizer, device):
    model.train()
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass in float16
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Scale loss, backward, unscale, step
        scaler.scale(loss).backward()
        
        # Optional: gradient clipping with AMP
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
    
    return loss.item()

# BFloat16 (on A100/H100 — more stable than float16)
with autocast(dtype=torch.bfloat16):
    outputs = model(inputs)
```

---

## 22. Distributed Training

### DistributedDataParallel (DDP) — Recommended

```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',      # NCCL for GPU (gloo for CPU)
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_worker(rank, world_size, dataset):
    setup(rank, world_size)
    
    # Build model on this GPU
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Distributed sampler — each GPU sees different data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)  # Reshuffle each epoch
        for inputs, targets in loader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
    
    # Save from rank 0 only
    if rank == 0:
        torch.save(model.module.state_dict(), 'ddp_model.pth')
    
    cleanup()

# Launch
world_size = torch.cuda.device_count()
mp.spawn(train_worker, args=(world_size, dataset), nprocs=world_size, join=True)
```

### torchrun (Modern Way to Launch DDP)

```bash
# Single machine, 4 GPUs
torchrun --standalone --nproc_per_node=4 train.py

# Multi-machine
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d \
         --rdzv_endpoint=master_addr:29500 train.py
```

---

## 23. TorchScript & Model Export

TorchScript compiles PyTorch models to a static, serializable representation for production.

### Tracing (Simple, No Control Flow)

```python
model.eval()

# Trace model with example input
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Save
traced_model.save('model_traced.pt')

# Load (no Python needed!)
loaded = torch.jit.load('model_traced.pt')
output = loaded(example_input)
```

### Scripting (Full Python Support)

```python
# Script model — handles if/else, loops, dynamic shapes
scripted_model = torch.jit.script(model)

scripted_model.save('model_scripted.pt')
loaded = torch.jit.load('model_scripted.pt')

# Add type annotations for scripting
class ScriptableModel(nn.Module):
    def forward(self, x: torch.Tensor, use_dropout: bool = True) -> torch.Tensor:
        if use_dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        return self.fc(x)
```

---

## 24. ONNX Export & Deployment

**ONNX** (Open Neural Network Exchange) — deploy PyTorch models to any runtime (TensorRT, OpenVINO, ONNXRuntime, CoreML).

### Export to ONNX

```python
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    export_params=True,        # Store weights in file
    opset_version=17,          # ONNX opset version
    do_constant_folding=True,  # Optimize constants
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},    # Variable batch size
        'output': {0: 'batch_size'}
    }
)

# Verify ONNX model
import onnx
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")
```

### Run with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Create inference session
session = ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Get I/O info
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = session.run([output_name], {input_name: input_data})
print(result[0].shape)
```

### TorchServe (Production Serving)

```bash
# Install
pip install torchserve torch-model-archiver

# Create model archive
torch-model-archiver \
  --model-name my_model \
  --version 1.0 \
  --serialized-file model_scripted.pt \
  --handler image_classifier \
  --export-path model_store/

# Start server
torchserve --start --model-store model_store/ --models my_model=my_model.mar

# Inference via REST
curl http://localhost:8080/predictions/my_model \
  -T image.jpg

# Stop server
torchserve --stop
```

---

## 25. PyTorch Lightning (Production Framework)

PyTorch Lightning removes boilerplate while keeping PyTorch flexibility. Industry standard for production training.

```bash
pip install lightning
```

### LightningModule — The Core

```python
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

class LitClassifier(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_classes, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()   # Saves all __init__ args to self.hparams
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc',  acc,  prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc',  acc,  prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(1) == y).float().mean()
        self.log('test_acc', acc)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}
        }


# LightningDataModule — encapsulates all data logic
class LitDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = MyDataset(self.data_dir, split='train')
            self.val_ds   = MyDataset(self.data_dir, split='val')
        if stage == 'test' or stage is None:
            self.test_ds  = MyDataset(self.data_dir, split='test')
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4)


# Trainer — handles everything else
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

trainer = L.Trainer(
    max_epochs=100,
    accelerator='auto',           # CPU/GPU/TPU — automatically detected
    devices='auto',               # Use all available devices
    precision='16-mixed',         # Mixed precision training
    accumulate_grad_batches=4,    # Gradient accumulation
    gradient_clip_val=1.0,        # Gradient clipping
    log_every_n_steps=10,
    callbacks=[
        EarlyStopping('val_loss', patience=10),
        ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=3),
        LearningRateMonitor('epoch')
    ],
    logger=L.pytorch.loggers.TensorBoardLogger('tb_logs', name='my_model'),
)

model     = LitClassifier(input_dim=784, hidden_dim=512, num_classes=10)
datamodule = LitDataModule('data/', batch_size=128)

trainer.fit(model, datamodule)
trainer.test(model, datamodule)

# Multi-GPU DDP — just change devices!
trainer = L.Trainer(accelerator='gpu', devices=4, strategy='ddp')
```

---

## 26. Debugging & Profiling

### Debugging Tips

```python
# 1. Check tensor shapes at every step
print(f"Input:  {x.shape}")
print(f"After conv1: {x.shape}")

# 2. Detect NaN/Inf
torch.autograd.set_detect_anomaly(True)   # Detailed NaN traceback (slow)
torch.isnan(x).any()
torch.isinf(x).any()

# 3. Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}, "
              f"param_norm={param.data.norm():.4f}")

# 4. Register hooks for intermediate activations
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.layer1.register_forward_hook(get_activation('layer1'))
output = model(x)
print(activations['layer1'].shape)

# 5. Gradient hooks
def grad_hook(grad):
    print(f"Gradient: min={grad.min():.4f}, max={grad.max():.4f}, norm={grad.norm():.4f}")
    return grad

x.register_hook(grad_hook)

# 6. Run on small data first
x_small = x[:4]   # Test entire pipeline on 4 samples before full run
```

### Profiling

```python
# PyTorch Profiler (built-in)
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("model_inference"):
        output = model(inputs)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export for TensorBoard
prof.export_chrome_trace("trace.json")

# Simple timer
import time
start = time.perf_counter()
for _ in range(100):
    with torch.no_grad():
        out = model(x)
torch.cuda.synchronize()   # Wait for GPU
elapsed = time.perf_counter() - start
print(f"100 forward passes: {elapsed*1000:.2f}ms")
```

### Common Errors & Solutions

| Error | Cause | Solution |
|---|---|---|
| `RuntimeError: Expected all tensors on same device` | Mixing CPU/GPU | `.to(device)` all tensors/model |
| `RuntimeError: CUDA out of memory` | Batch too large / memory leak | Reduce batch, `torch.cuda.empty_cache()`, check for missing `no_grad` |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Wrong Linear layer size | Check `.shape` at each step |
| `Loss is NaN` | Exploding gradients / bad data | Clip gradients, check for zeros in log |
| Gradients are None | `no_grad` context or wrong `requires_grad` | Check wrapping, ensure `requires_grad=True` |
| Loss doesn't decrease | LR too high/low, bad initialization | Try 1e-3 Adam, check data normalization |
| Validation acc >> Train acc | Train augmentation too aggressive | Reduce augmentation |

---

## 27. Performance Optimization

### Compilation (torch.compile)

```python
# PyTorch 2.0+ — massive speedup via Triton kernels
model = torch.compile(model)                         # Default mode
model = torch.compile(model, mode='reduce-overhead') # For repeated calls (training)
model = torch.compile(model, mode='max-autotune')    # Maximum optimization (slow compile)
model = torch.compile(model, fullgraph=True)         # Whole-graph compilation

# Works transparently
output = model(x)   # First call: compiles; subsequent calls: fast
```

### Flash Attention (for Transformers)

```python
# PyTorch 2.0+ has built-in Flash Attention
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    output = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0)

# Or use with nn.MultiheadAttention — automatically uses Flash Attention when available
```

### DataLoader Optimization

```python
# Optimal DataLoader settings
DataLoader(
    dataset,
    batch_size=64,
    num_workers=os.cpu_count(),    # Use all CPU cores
    pin_memory=True,               # Faster CPU→GPU transfer
    persistent_workers=True,       # Don't re-spawn workers each epoch
    prefetch_factor=2,             # Prefetch batches per worker
    drop_last=True                 # Consistent batch size (good for BN)
)

# For datasets that fit in memory: cache them
class CachedDataset(Dataset):
    def __init__(self, dataset):
        self.cache = [dataset[i] for i in range(len(dataset))]
    def __len__(self):
        return len(self.cache)
    def __getitem__(self, idx):
        return self.cache[idx]
```

### Memory Optimization

```python
# 1. Use in-place operations where safe
F.relu(x, inplace=True)
x.add_(1)

# 2. Delete large tensors
del large_tensor
torch.cuda.empty_cache()

# 3. Gradient checkpointing
import torch.utils.checkpoint as cp
x = cp.checkpoint(self.expensive_block, x)   # Recomputes activations in backward

# 4. Half precision inference
model.half()   # Convert weights to float16
x = x.half()
output = model(x)

# 5. 8-bit inference (bitsandbytes)
# pip install bitsandbytes
import bitsandbytes as bnb
linear = bnb.nn.Linear8bitLt(in_features, out_features, has_fp16_weights=False)
```

---

## 28. Real-World Project Checklist

### End-to-End Production ML with PyTorch

```
1. PROBLEM & DATA
   □ Define problem type + success metric + baseline
   □ EDA: distributions, missing values, class balance
   □ Feature engineering + preprocessing pipeline
   □ Train/Val/Test split (stratified if classification)
   □ Check for data leakage

2. DATASET & DATALOADER
   □ Custom Dataset class with __len__ and __getitem__
   □ DataLoader with num_workers, pin_memory, shuffle
   □ Appropriate transforms/augmentation
   □ Handle class imbalance (weighted sampler or loss weights)

3. MODEL
   □ Start with proven architecture (ResNet, EfficientNet, BERT)
   □ Use pretrained weights when possible
   □ Add appropriate head for your task
   □ Initialize weights carefully for scratch training

4. TRAINING LOOP
   □ model.train() / model.eval() in correct places
   □ optimizer.zero_grad() before backward
   □ Loss → backward → (clip) → step
   □ Track train/val loss and metrics
   □ Log to TensorBoard / W&B

5. REGULARIZATION & OPTIMIZATION
   □ Appropriate optimizer (AdamW for transformers, SGD for CNNs)
   □ Learning rate schedule (cosine decay or ReduceLROnPlateau)
   □ Dropout / BatchNorm / weight decay
   □ Data augmentation
   □ Early stopping

6. EVALUATION
   □ Evaluate on held-out test set ONCE at the end
   □ Confusion matrix, per-class accuracy
   □ Analyze failure cases
   □ Check model fairness/bias

7. EXPERIMENT TRACKING
   □ Use Weights & Biases or MLflow
   □ Log hyperparameters, metrics, model artifacts
   □ Version datasets with DVC

8. DEPLOYMENT
   □ Export via TorchScript or ONNX
   □ Benchmark inference speed
   □ Set up TorchServe or FastAPI endpoint
   □ Monitor predictions in production
   □ Plan for model updates and rollback
```

### Experiment Tracking with Weights & Biases

```python
import wandb

# Initialize run
wandb.init(
    project='my_project',
    config={
        'lr': 1e-3, 'batch_size': 64,
        'epochs': 100, 'architecture': 'ResNet50'
    }
)

# In training loop
wandb.log({
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_accuracy': val_acc,
    'lr': optimizer.param_groups[0]['lr'],
    'epoch': epoch
})

# Log model
wandb.save('best_model.pth')

# Log images
wandb.log({'predictions': [wandb.Image(img, caption=f"Label: {label}") 
                            for img, label in zip(images[:8], preds[:8])]})

# Watch model (log gradients and weights)
wandb.watch(model, log='all', log_freq=100)
```

### Hyperparameter Tuning with Optuna

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 1, 5)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    model = MyModel(n_layers=n_layers, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(20):
        train_loss = train_one_epoch(model, loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)
        
        # Pruning: stop if not promising
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return val_acc

# Run optimization
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100, n_jobs=1)

print("Best params:", study.best_params)
print("Best val_acc:", study.best_value)
```

---

## 29. Quick Reference Cheat Sheet

### Architecture Selection Guide

| Task | Architecture | Library |
|---|---|---|
| Image Classification | EfficientNet, ConvNeXt, ViT | torchvision |
| Object Detection | YOLOv8, Faster R-CNN, DETR | torchvision, ultralytics |
| Image Segmentation | U-Net, SegFormer, Mask R-CNN | torchvision, segmentation-models |
| Text Classification | BERT, RoBERTa, DistilBERT | HuggingFace |
| Text Generation | GPT-2, Llama, Mistral | HuggingFace |
| Translation/Seq2Seq | T5, mBART | HuggingFace |
| Time Series | LSTM, Temporal Fusion Transformer | pytorch-forecasting |
| Tabular Data | TabNet, NODE, MLP | pytorch-tabular |
| Point Cloud | PointNet, PointNet++ | torch-geometric |
| Graph Neural Net | GCN, GAT, GraphSAGE | PyTorch Geometric |

### The 6 Lines Every PyTorch Training Step

```python
optimizer.zero_grad()      # 1. Clear old gradients
outputs = model(inputs)    # 2. Forward pass
loss = criterion(outputs, targets)  # 3. Compute loss
loss.backward()            # 4. Compute gradients
optimizer.step()           # 5. Update weights
scheduler.step()           # 6. Update learning rate
```

### Tensor Operations Cheat Sheet

```python
# Shape
x.shape                    # torch.Size([B, C, H, W])
x.reshape(B, -1)           # Flatten C×H×W → (B, C*H*W)
x.unsqueeze(0)             # Add batch dim: (C,H,W) → (1,C,H,W)
x.squeeze(0)               # Remove batch dim: (1,C,H,W) → (C,H,W)
x.permute(0, 2, 3, 1)     # BCHW → BHWC
x.transpose(1, 2)          # Swap dims 1 and 2
x.contiguous()             # Make contiguous after permute/transpose

# Device
x.to(device)               # Move to device
x.cuda()                   # Move to GPU
x.cpu()                    # Move to CPU
x.detach().cpu().numpy()  # Tensor → NumPy (safe)

# Dtype
x.float()                  # → float32
x.half()                   # → float16
x.long()                   # → int64
x.bool()                   # → bool
x.to(torch.float32)        # Explicit cast

# Math
x @ y                      # Matrix multiply
x * y                      # Element-wise multiply
x.sum(dim=1)              # Sum along dim 1
x.mean(dim=0)             # Mean along dim 0
x.argmax(dim=1)           # Index of max along dim 1
x.softmax(dim=-1)         # Softmax along last dim
```

### The Golden Rules of PyTorch

```
1. 🔑 ALWAYS set model.train() before training, model.eval() before evaluation
2. 🔑 ALWAYS call optimizer.zero_grad() before loss.backward()
3. 🔑 ALWAYS move model AND data to the same device
4. 🔑 Use detach().cpu().numpy() to convert GPU tensors to NumPy
5. 🔑 Use torch.no_grad() for inference — saves memory and is faster
6. 🔑 Normalize your inputs — neural networks hate raw pixel values
7. 🔑 Gradients accumulate — zero them unless you're doing it on purpose
8. 🔑 Use weight_decay in AdamW, not a separate L2 penalty
9. 🔑 Clip gradients when training RNNs or Transformers
10. 🔑 torch.compile() your model for free 2x speedup (PyTorch 2.0+)
11. 🔑 Save checkpoints often — use val metric, not just latest epoch
12. 🔑 Profile before optimizing — don't guess the bottleneck
```

---

## 📚 Resources & Ecosystem

### Official Resources
- [PyTorch Docs](https://pytorch.org/docs/stable/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Blog](https://pytorch.org/blog/)
- [PyTorch Forum](https://discuss.pytorch.org/)

### Essential Libraries
```
torchvision          — CV datasets, models, transforms
torchaudio           — Audio processing
torchtext            — Text processing
torchmetrics         — 100+ ML metrics
pytorch-lightning    — Production training framework
timm                 — 700+ pretrained vision models (pip install timm)
transformers         — HuggingFace: BERT, GPT, T5, Llama
diffusers            — HuggingFace: Stable Diffusion, DDPM
torch-geometric      — Graph neural networks
detectron2           — Facebook's object detection
mmdetection          — OpenMMLab detection framework
segmentation-models  — U-Net variants for segmentation
kornia               — Differentiable computer vision
albumentations       — Fast image augmentation
wandb                — Experiment tracking
optuna               — Hyperparameter optimization
onnxruntime          — Cross-platform inference
```

### Recommended Learning Path
```
Week 1-2:   Tensors, Autograd, nn.Module, training loop
Week 3-4:   CNNs on CIFAR-10, Data pipelines, callbacks
Week 5-6:   Transfer learning, LR schedulers, regularization
Week 7-8:   RNNs/LSTMs, NLP tasks, HuggingFace basics
Week 9-10:  Transformers from scratch, Attention mechanisms
Week 11-12: Mixed precision, Distributed training, TorchScript
Week 13+:   PyTorch Lightning, ONNX, production deployment

Projects to build:
  1. MNIST classifier (get basics right)
  2. CIFAR-10 CNN with augmentation + LR schedule
  3. Transfer learning for custom image dataset
  4. Sentiment analysis with LSTM and BERT
  5. Image segmentation with U-Net
  6. Deploy a model as a REST API with FastAPI + ONNX
  7. Fine-tune an LLM with LoRA
```

---

*These notes cover PyTorch from first principles to production-grade deployment.*  
*Every concept links to the next. Write every code snippet yourself — understanding comes from doing.*

> **"In theory, theory and practice are the same. In practice, they are not." — train more models.**

---
*Last updated: 2025 | PyTorch 2.x | Python 3.10+ | CUDA 12.x*