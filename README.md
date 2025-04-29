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
- Activation-specific weight initialization (Xavier, He, and custom strategies)
- Activation-specific bias initialization
- Modular activation functions (Sigmoid, ReLU, LeakyReLU, Tanh, Softmax)
- Pluggable optimizers (SGD, Momentum)
- Pluggable loss functions (Binary Cross Entropy, Weighted BCE)
- Fluent API for building networks
- Training observer pattern (loss tracking, early stopping)
- Designed for extensibility and transparency
- Reproducible random initialization with customizable Random sources
- Lightweight — no external heavy dependencies (only EJML for efficient matrix math)

---

## 🚀 Quickstart

```java
var rand = new Random(42); // For reproducibility

var classifier = new NeuralNetworkBuilder(inputFeatureCount)
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

// Train for up to 300 epochs
classifier.train(inputs, targets, (epoch, loss, predictions, labels) -> {
    System.out.println("Epoch " + epoch + " - Loss: " + loss);
    return epoch < 300;
});

// Make predictions
var testPredictions = classifier.predict(testInputs);
```

✅ Simple, readable, powerful!

---

## 📚 Core Concepts

| Concept | Description |
|:--------|:------------|
| **Layer** | Fully connected layer with customizable activation, weight optimizer, bias optimizer. |
| **Activation Functions** | Modular activation classes: ReLU, Sigmoid, Tanh, LeakyReLU, Softmax. |
| **Weight Initialization** | Each activation defines its preferred weight and bias initialization. |
| **Optimizers** | SGD and Momentum optimizers available; pluggable per-layer. |
| **Loss Functions** | BCE (Binary Cross Entropy) and Weighted BCE included. |
| **TrainingObserver** | Hook to monitor progress and optionally stop training early. |
| **Fluent API** | `NeuralNetworkBuilder` allows clean, simple, readable network definitions. |

---

## 📈 Example: Titanic Dataset (Survival Prediction)

Netwerx has been successfully used to model Titanic survival prediction:

- 6 input features (ticket class, age, sex, fare, number of parents/children, siblings/spouses)
- Two hidden layers (8 neurons, then 4 neurons) with ReLU activation
- Weighted binary cross-entropy loss to handle survivor imbalance
- Momentum optimizer for smoother convergence
- 300–500 epochs training

**Results:**
- Predicts realistic survival rates (~120 survivors out of 418 test samples)
- Matches known Titanic dataset average survival rates
- Demonstrates correct learning dynamics and generalization

✅ Netwerx handles real-world noisy datasets with no tricks.

---

## 🔧 Extending Netwerx

Adding new functionality is simple!

| To add | Implement |
|:-------|:----------|
| New activation function | `Activation` interface |
| New optimizer | `Optimizer` interface |
| New loss function | `LossFunction` interface |
| New metrics during training | `TrainingObserver` |
| Custom layer types | Extend `Layer` class |

✅ Netwerx is designed for *plug and play* extensibility.

---

## 🛤 Roadmap

- [ ] Add Adam and RMSProp optimizers
- [ ] Support mini-batch training
- [ ] Learning rate decay scheduling
- [ ] Multi-class classification (Softmax output support)
- [ ] Save/load model parameters
- [ ] Training curve visualization
- [ ] Example projects and tutorials

---

## 🤝 Contributing

Contributions are welcome!

Feel free to:
- Open issues
- Submit pull requests
- Suggest improvements
- Add examples and tutorials

Please see [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon) for contribution guidelines.

---

## 📄 License

Netwerx is open-sourced under the [Apache License 2.0](LICENSE).

You may use, modify, and distribute this software under the terms of the Apache 2.0 license.
See the [LICENSE](LICENSE) file for full details.

---

## 🙏 Acknowledgements

- Inspired by PyTorch, Keras, TensorFlow, and the academic principles behind deep learning.
- Built for curiosity, education, and exploration.
- Created and maintained by [James Carman](https://github.com/jwcarman).

---

# 🚀 Build neural networks, understand them deeply — with **Netwerx**.