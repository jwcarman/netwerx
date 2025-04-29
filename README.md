# Netwerx

![Java](https://img.shields.io/badge/Java-23%2B-blue)
![License](https://img.shields.io/badge/License-Apache_2.0-blue)

---

**Netwerx** is a lightweight, extensible deep learning library for Java.

It is designed to give you full control over neural network internals while maintaining simplicity, modularity, and clean API design — ideal for learning, prototyping, and research.

Built for Java developers who want to **understand** and **customize** neural networks without the complexity of heavyweight frameworks.

---

## ✨ Features

- Fully connected feed-forward neural networks
- Activation-specific weight initialization (Xavier, He, custom strategies)
- Activation-specific bias initialization
- Modular activation functions (Sigmoid, ReLU, LeakyReLU, Tanh, Softmax, Linear)
- Pluggable optimizers: SGD, Momentum, Adam, RMSProp
- Pluggable loss functions: Binary Cross Entropy, Weighted BCE, Categorical Cross Entropy, Mean Squared Error
- Fluent builder API for easy model creation
- Training observer pattern (loss tracking, early stopping)
- Lightweight — no external heavy dependencies (only EJML for efficient matrix math)
- Reproducible random initialization with customizable Random sources
- Java 23+ modern style (records, Math.clamp, lambdas)

---

## 🚀 Quickstart

### Binary Classification Example

```java
var rand = new Random(42);

var classifier = NeuralNetwork.builder(inputFeatureCount)
    .optimizer(() -> Optimizers.momentum(0.01, 0.9))
    .layer(layer -> layer
        .units(8)
        .activation(Activations.relu())
        .random(rand)
    )
    .layer(layer -> layer
        .units(4)
        .activation(Activations.relu())
        .random(rand)
    )
    .binaryClassifier(bc -> bc
        .random(rand)
        .loss(Losses.weightedBce(positiveWeight, negativeWeight))
    );

// Train up to 300 epochs
classifier.train(inputs, labels, (epoch, loss, predictions, targets) -> {
    System.out.println("Epoch " + epoch + " - Loss: " + loss);
    return epoch < 300;
});

// Make predictions
boolean[] predictions = classifier.predict(testInputs);
```

✅ Simple, readable, powerful!

---

## 📚 Core Concepts

| Concept | Description |
|:--------|:------------|
| **Layer** | Fully connected layer with customizable activation, weight optimizer, and bias optimizer. |
| **Activation Functions** | ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, and Linear activations. |
| **Weight Initialization** | Automatically matched to activation (Xavier, He, or custom). |
| **Optimizers** | SGD, Momentum, Adam, and RMSProp optimizers available and pluggable. |
| **Losses** | Binary Cross Entropy, Weighted BCE, Categorical Cross Entropy, and Mean Squared Error. |
| **TrainingObserver** | Hook to monitor and control training process (early stopping, metrics). |
| **Fluent Builder API** | `NeuralNetwork.builder()` provides a clean, composable setup. |

---

## 📈 Example: Titanic Dataset (Survival Prediction)

Netwerx has been successfully used to model Titanic survival prediction:

- 6 input features (ticket class, age, sex, fare, parents/children, siblings/spouses)
- Two hidden layers (8 neurons → 4 neurons) with ReLU activations
- Weighted binary cross-entropy loss to handle class imbalance
- Momentum optimizer for smoother convergence
- 300–500 epochs training

**Results:**
- Predicts realistic survival rates (~120 survivors out of 418)
- Matches known Titanic dataset averages
- Shows proper learning dynamics and generalization

✅ Netwerx handles real-world noisy datasets gracefully.

---

## 🧩 Extending Netwerx

Extending the library is easy:

| To add | Implement |
|:-------|:----------|
| New activation function | Implement the `Activation` interface |
| New optimizer | Implement the `Optimizer` interface |
| New loss function | Implement the `Loss` interface |
| New training metrics | Implement a `TrainingObserver` |
| Custom layer types | Extend the `Layer` class |

✅ Netwerx is designed for *plug and play* extensibility.

---

## 🛤 Roadmap

- [x] Add Adam optimizer
- [x] Add RMSProp optimizer
- [ ] Support mini-batch training
- [ ] Learning rate scheduling (decay, warm restarts)
- [ ] Save/load model parameters
- [ ] Dropout regularization support
- [ ] Training curve visualization tools
- [ ] Example projects and datasets (Titanic, Wine, Concrete Strength, etc.)

---

## 🤝 Contributing

Contributions are welcome!

Ways you can contribute:
- Open issues for bugs or feature requests
- Submit pull requests
- Improve documentation and examples
- Suggest new activation functions, optimizers, or loss functions

Please see [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon) for contribution guidelines.

---

## 📄 License

Netwerx is open-sourced under the [Apache License 2.0](LICENSE).

You may use, modify, and distribute this software under the terms of the Apache 2.0 license.  
See the [LICENSE](LICENSE) file for full legal details.

---

## 🙏 Acknowledgements

- Inspired by the flexibility of PyTorch, Keras, TensorFlow — but focused on **clarity** and **control**.
- Built for Java developers who want to truly **understand** machine learning.
- Created and maintained by [James Carman](https://github.com/jwcarman).

---

# 🚀 Build neural networks, understand them deeply — with **Netwerx**.