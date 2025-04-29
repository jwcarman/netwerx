# Netwerx

![Java](https://img.shields.io/badge/Java-23%2B-blue)
![License](https://img.shields.io/badge/License-Apache_2.0-blue)

---

**Netwerx** is a lightweight, extensible deep learning library for Java 23+.

It is designed to give you full control over neural network internals while maintaining simplicity, modularity, and clean API design — ideal for learning, prototyping, and research.

Built for Java developers who want to **understand** and **customize** neural networks without the complexity of heavyweight frameworks.

---

## ✨ Features

- Fully connected feed-forward neural networks
- Activation-specific weight initialization (Xavier, He, etc.)
- Activation-specific bias initialization
- Modular activation functions
- Pluggable optimizers
- Rich set of loss functions
- Fluent, clean API for building networks
- Training observer pattern (loss tracking, early stopping)
- Reproducible random initialization with customizable `Random` sources
- Lightweight — no external heavy dependencies (only EJML for efficient matrix math)

---

## 🚀 Quickstart

```java
var rand = new Random(42); // For reproducibility

var classifier = NeuralNetwork.builder(inputFeatureCount)
    .random(rand)
    .optimizer(() -> Optimizers.momentum(0.001, 0.9))
    .layer(layer -> layer
        .units(8)
        .activation(Activations.relu())
    )
    .layer(layer -> layer
        .units(4)
        .activation(Activations.relu())
    )
    .binaryClassifier(bc -> bc
        .loss(Losses.weightedBce(positiveWeight, negativeWeight))
    );

// Train for up to 300 epochs
classifier.train(inputs, labels, (epoch, loss, predictions, groundTruth) -> {
    System.out.println("Epoch " + epoch + " - Loss: " + loss);
    return epoch < 300;
});

// Make predictions
boolean[] testPredictions = classifier.predict(testInputs);
```

✅ Simple, readable, powerful!

---

## 📚 Core Concepts

| Concept | Description |
|:--------|:------------|
| **Layer** | Fully connected layer with customizable activation, weight optimizer, bias optimizer. |
| **Activation Functions** | Modular: ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, Linear. |
| **Weight Initialization** | Automatically based on activation (e.g., Xavier, He initialization). |
| **Optimizers** | SGD, Momentum, Adam, RMSProp — pluggable, configurable, and layer-specific if needed. |
| **Loss Functions** | Wide variety for classification and regression tasks. |
| **Training Observer** | Hook to monitor and control training behavior dynamically. |
| **Fluent API** | `NeuralNetworkBuilder` allows clean, expressive, type-safe building. |

---

## ⚡ Supported Activation Functions

| Activation | Description |
|:-----------|:-------------|
| **Sigmoid** | Squashes input into (0, 1). Used in binary classifiers. |
| **Tanh** | Squashes input into (-1, 1). Zero-centered; often preferred over Sigmoid in hidden layers. |
| **ReLU** (Rectified Linear Unit) | Outputs 0 for negative inputs, identity for positive. Most common hidden layer activation. |
| **Leaky ReLU** | Like ReLU, but allows small negative values (small slope) to avoid "dying" neurons. |
| **Softmax** | Converts logits into probabilities across multiple classes. Used for multi-class classification outputs. |
| **Linear** | Identity function. Typically used for regression outputs. |

---

## ⚙️ Supported Optimizers

| Optimizer | Description |
|:----------|:------------|
| **SGD (Stochastic Gradient Descent)** | Standard gradient descent. |
| **Momentum** | Adds momentum to SGD to accelerate convergence and smooth out updates. |
| **Adam** | Adaptive Moment Estimation — popular, combines momentum and adaptive learning rates. |
| **RMSProp** | Adaptive learning rate based on recent gradients, good for non-stationary objectives. |

---

## 🎯 Supported Loss Functions

| Loss Function | Best For | Notes |
|:--------------|:---------|:------|
| **Binary Cross Entropy (BCE)** | Binary classification | |
| **Weighted BCE** | Imbalanced binary classification | |
| **Categorical Cross Entropy (CCE)** | Multi-class classification | |
| **Mean Squared Error (MSE)** | Regression | Sensitive to outliers |
| **Mean Absolute Error (MAE)** | Regression | More robust to outliers |
| **Huber Loss** | Regression | Smooth combination of MSE and MAE |
| **Log-Cosh Loss** | Regression | Smooth approximation to MAE |
| **Hinge Loss** | Binary classification (SVM-style margin) | |

---

## 📊 Metrics Supported

| Model Type | Metrics |
|:-----------|:--------|
| **Binary Classifier** | Accuracy, Precision, Recall, F1 Score |
| **Multi-Class Classifier** | Accuracy, Precision, Recall, F1 Score |
| **Regression Model** | Mean Squared Error (MSE), Mean Absolute Error (MAE), R² |

---

## 📈 Example: Titanic Dataset (Survival Prediction)

Netwerx has been successfully used to model Titanic survival prediction:

- 6 input features (ticket class, age, sex, fare, parents/children, siblings/spouses)
- Two hidden layers (8 neurons, then 4 neurons) with ReLU activation
- Weighted binary cross-entropy loss to handle survivor imbalance
- Momentum optimizer for smoother convergence
- 300–500 epochs training

**Results:**
- Predicts realistic survival rates (~120 survivors out of 418 test samples)
- Matches known Titanic dataset survival statistics
- Demonstrates correct learning and generalization dynamics

✅ Netwerx handles real-world noisy datasets without needing heavyweight frameworks.

---

## 🔧 Extending Netwerx

Adding new functionality is easy:

| To Add | Implement |
|:-------|:----------|
| New activation function | `Activation` interface |
| New optimizer | `Optimizer` interface |
| New loss function | `Loss` interface |
| Custom training observers | `TrainingObserver` |
| Custom model types (e.g., ensembles) | Extend `NeuralNetwork` or build wrappers |

✅ Designed for maximum extensibility with minimum ceremony.

---

## 🛤 Roadmap

- [x] Adam and RMSProp optimizer support
- [x] Expanded loss functions
- [ ] Mini-batch training
- [ ] Learning rate schedulers (decay, warm-up)
- [ ] Early stopping support
- [ ] Model saving/loading (serialization)
- [ ] Multi-threaded training (batch parallelism)
- [ ] Visualization utilities (training curves, metrics)

---

## 🤝 Contributing

Contributions are welcome!

Feel free to:
- Open issues
- Submit pull requests
- Suggest new optimizers, activations, or loss functions
- Add examples and benchmarks

Contribution guidelines (CONTRIBUTING.md) coming soon.

---

## 📄 License

Netwerx is open-sourced under the [Apache License 2.0](LICENSE).

You may freely use, modify, and distribute it under the terms of the license.

---

## 🙏 Acknowledgements

- Inspired by PyTorch, Keras, TensorFlow — but rebuilt for lightweight simplicity in Java.
- Designed for curiosity, learning, and practical prototyping.
- Created and maintained by [James Carman](https://github.com/jwcarman).

---

# 🚀 Build neural networks, understand them deeply — with **Netwerx**.