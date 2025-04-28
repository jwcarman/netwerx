# Netwerx

A lightweight, extensible deep learning library for Java.  
Designed for learning, exploration, and full control over the internals of neural network training.

Netwerx provides a clean, modular structure inspired by frameworks like PyTorch and TensorFlow, while maintaining simplicity for hands-on experimentation.

---

## ✨ Features

- Fully connected feed-forward neural networks
- Activation-specific weight initialization (Xavier, He, etc.)
- Activation-specific bias initialization (e.g., positive bias for ReLU)
- Pluggable activation functions (Sigmoid, ReLU, LeakyReLU, Tanh, Softmax)
- Pluggable optimizers (SGD, Momentum)
- Pluggable loss functions (Binary Cross Entropy, Weighted Binary Cross Entropy)
- Modular design with fluent builder APIs
- Full batch training with optional support for mini-batch training (coming soon)
- Customizable random initialization (per-layer and per-task)
- Training observer pattern for metrics and early stopping
- Designed for extensibility and clarity — perfect for learning and research

---

## 📦 Installation
 
For now, clone the repository and import into your Java project directly.

```bash
git clone https://github.com/yourusername/netwerx.git