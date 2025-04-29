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
- Modular activation functions (Sigmoid, ReLU, LeakyReLU, Tanh, Softmax, Linear)
- Pluggable optimizers (SGD, Momentum, Adam)
- Pluggable loss functions (Binary Cross Entropy, Weighted BCE, Categorical Cross Entropy, Mean Squared Error)
- Fluent API for building networks
- Training observer pattern (loss tracking, early stopping)
- Lightweight — no external heavy dependencies (only EJML for efficient matrix math)
- Reproducible random initialization with customizable Random sources
- Java 23+ modern style (`Records`, `Math.clamp`, lambdas)

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

// Train for up to 300 epochs
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
| **Layer** | Fully connected layer with customizable activation, weight optimizer, bias optimizer. |
| **Activation Functions** | Modular activation classes: ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, Linear. |
| **Weight Initialization** | Each activation defines its preferred weight and bias initialization (Xavier, He, etc.). |
| **Optimizers** | SGD, Momentum, and Adam optimizers available; fully pluggable. |
| **Losses** | BCE, Weighted BCE, Categorical Cross Entropy, MSE (for regression tasks). |
| **TrainingObserver** | Hook to monitor loss, custom metrics, and support early stopping. |
| **Fluent API** | `NeuralNetwork.builder()` for clean, composable network definitions. |

---

## 📈 Example: Titanic Dataset (Survival Prediction)

Netwerx has been successfully used to model Titanic survival prediction:

- 6 input features (ticket class, age, sex, fare, parents/children, siblings/spouses)
- Two hidden layers (8 neurons and 4 neurons) with ReLU activation
- Weighted Binary Cross-Entropy loss to account for class imbalance
- Momentum optimizer to improve convergence
- 300–500 epochs of training

**Results:**
- Predicts realistic survival rates (~120 survivors out of 418 passengers)
- Matches known Titanic dataset survival averages
- Demonstrates correct learning dynamics and generalization

✅ Netwerx handles real-world noisy datasets without needing complex tricks.

---

## 🧩 Extending Netwerx

Adding new functionality is simple!

| To add | Implement |
|:-------|:----------|
| New activation function | `Activation` interface |
| New optimizer | `Optimizer` interface |
| New loss function | `Loss` interface |
| New metrics during training | `TrainingObserver` |
| Custom layer types | Extend the `Layer` class |

✅ Netwerx is designed for *plug and play* extensibility.

---

## 🛤 Roadmap

- [ ] Add RMSProp optimizer
- [ ] Support mini-batch training
- [ ] Learning rate decay scheduling
- [ ] Save/load model parameters
- [ ] Training curve visualization
- [ ] Dropout regularization support
- [ ] Early stopping utilities
- [ ] Learning Rate Finder utilities
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

- Inspired by the spirit of PyTorch, Keras, TensorFlow, but built for **understanding**, **experimentation**, and **control**.
- Special thanks to the academic deep learning community.
- Created and maintained by [James Carman](https://github.com/jwcarman).

---

# 🚀 Build neural networks, understand them deeply — with **Netwerx**.