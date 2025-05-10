# Netwerx

[![CI with Maven](https://github.com/jwcarman/netwerx/actions/workflows/maven.yml/badge.svg)](https://github.com/jwcarman/netwerx/actions/workflows/maven.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=jwcarman_netwerx&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=jwcarman_netwerx)
![License](https://img.shields.io/badge/License-Apache_2.0-blue)

[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=jwcarman_netwerx&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=jwcarman_netwerx)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=jwcarman_netwerx&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=jwcarman_netwerx)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=jwcarman_netwerx&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=jwcarman_netwerx)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=jwcarman_netwerx&metric=vulnerabilities)](https://sonarcloud.io/summary/new_code?id=jwcarman_netwerx)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=jwcarman_netwerx&metric=coverage)](https://sonarcloud.io/summary/new_code?id=jwcarman_netwerx)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=jwcarman_netwerx&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=jwcarman_netwerx)

---

**Netwerx** is a lightweight, extensible deep learning library for Java 23+.

It’s designed for learning, prototyping, and research — with full transparency into what your neural network is doing under the hood. No magic, no black boxes.

---

## 📑 Table of Contents

* [✨ Features](#-features)
* [🚀 Quickstart](#-quickstart)
* [📚 Core Concepts](#-core-concepts)
* [🔌 Activation Functions](#-activation-functions)
* [⚙️ Optimizers](#️-optimizers)
* [🎯 Loss Functions](#-loss-functions)
* [🧪 Training Executors](#-training-executors)
* [📊 Scoring Functions](#-scoring-functions)
* [⏹ Early Stopping (Stopping Advisors)](#-early-stopping-stopping-advisors)
* [🛡 Regularization](#-regularization)
* [🎛 Parameter Initialization](#-parameter-initialization)
* [📣 Training Listeners](#-training-listeners)
* [🧠 Model Types](#-model-types)
* [🧮 Matrix Abstraction](#-matrix-abstraction)
* [🧪 Titanic Example](#-titanic-example)
* [🔧 Extending Netwerx](#-extending-netwerx)
* [🛤 Roadmap](#-roadmap)
* [🤝 Contributing](#-contributing)
* [📄 License](#-license)
* [🙏 Acknowledgements](#-acknowledgements)

---

## ✨ Features

* Fully connected feed-forward networks
* Binary/multi-class classifiers, regressors, autoencoders
* Mini-batch or full-batch training
* Dropout support for regularization
* Modular components (optimizers, activations, loss functions, etc.)
* Pluggable matrix backend
* Lightweight — depends only on EJML
* Training listeners and early stopping
* Reproducibility with pluggable random sources

---

## 🚀 Quickstart

```java
var trainer = new DefaultNeuralNetworkTrainerBuilder<>(factory, 5)
        .defaultOptimizer(() -> Optimizers.adam(0.01))
        .denseLayer(layer -> layer.units(8).activationFunction(ActivationFunctions.relu()))
        .denseLayer(layer -> layer.units(1).activationFunction(ActivationFunctions.sigmoid()))
        .buildBinaryClassifierTrainer();

var trainFeatures = factory.filled(5, 10, 0.5);
var trainLabels = factory.filled(1, 10, 1.0);
var dataset = new Dataset<>(trainFeatures, trainLabels);
var network = trainer.train(dataset);
```

---

## 📚 Core Concepts

* **Activation Functions** — introduce non-linearity into your network
* **Regularization** — penalize model complexity
* **Loss Functions** — define how wrong your model is
* **Optimizers** — control how weights are updated
* **Training Executors** — manage batching and parallel execution
* **Scoring Functions** — evaluate training progress
* **Stopping Advisors** — control when training halts
* **Parameter Initialization** — choose sensible starting points for weights and biases
* **Training Listeners** — observe and respond to training events
* **Matrix Abstraction** — plug in your own backend

---

## 🔌 Activation Functions

Netwerx supports ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, and Linear. You can plug in your own:

```java
var relu = ActivationFunctions.relu();
var custom = (ActivationFunction) (input) -> ...
```

---

## ⚙️ Optimizers

Optimizers update your weights each step:

* **SGD** — basic gradient descent
* **Momentum** — adds inertia
* **Adam** — adaptive learning rate + momentum
* **RMSProp** — adaptive learning rate only

```java
Optimizers.adam(0.01, 0.9, 0.999, 1e-8);
```

---

## 🎯 Loss Functions

Use a loss function suited to your task:

* **MSE** — regression
* **MAE** — regression
* **Binary Cross Entropy** — binary classification
* **Categorical Cross Entropy** — multi-class
* **Hinge** — SVM-style classifiers

```java
LossFunctions.bce();
```

---

## 🧪 Training Executors

Training executors handle how training samples are fed to the network:

* **Full Batch**: use entire dataset each epoch
* **Mini Batch**: configurable size, shuffling, and parallelism

```java
TrainingExecutors.miniBatch(32, new Random(), Executors.newFixedThreadPool(4));
```

---

## 📊 Scoring Functions

Scoring functions monitor progress, typically by evaluating validation loss or accuracy. Use one of ours or create your own:

```java
ScoringFunctions.validationLoss();
```

---

## ⏹ Early Stopping (Stopping Advisors)

Stop training when it's no longer improving:

* **Max Epochs**
* **Score Threshold**
* **Patience**

```java
StoppingAdvisors.patience(10, 1e-4);
```

---

## 🛡 Regularization

Avoid overfitting by penalizing weights:

* **L1** (sparsity)
* **L2** (shrinkage)
* **Elastic Net** (combines both)

```java
Regularizations.l2(1e-4);
```

---

## 🎛 Parameter Initialization

Choose how weights and biases are initialized:

```java
ParameterInitializers.heUniform();
ParameterInitializers.zeros();
```

---

## 📣 Training Listeners

Attach listeners to monitor progress:

```java
TrainingListeners.logging(logger, 100);
```

Custom listeners can log metrics, write to disk, update UIs, etc.

---

## 🧠 Model Types

Netwerx supports:

* **BinaryClassifierTrainer** — one output, sigmoid, BCE loss
* **MultiClassifierTrainer** — softmax, categorical loss
* **RegressionTrainer** — identity output, MSE/MAE
* **AutoencoderTrainer** — encoder/decoder pattern with MSE loss

---

## 🧮 Matrix Abstraction

All computations are built on a pluggable matrix abstraction:

```java
Matrix<M> matrix = factory.random(rows, cols);
```

Plug in your own backend (e.g., EJML, ND4J) by implementing `Matrix<M>`.

---

## 🧪 Titanic Example

A binary classifier predicts Titanic survival:

* Input: class, age, sex, fare, family members
* Layers: \[8 → 4 → 1]
* Activation: ReLU + Sigmoid
* Loss: Binary Cross Entropy
* Optimizer: SGD

```java
Accuracy: ~83%, F1 Score: 0.75
```

---

## 🔧 Extending Netwerx

| You want to...         | Implement...                          |
| ---------------------- | ------------------------------------- |
| Add an activation      | `ActivationFunction`                  |
| Add a loss function    | `LossFunction`                        |
| Create an optimizer    | `Optimizer`                           |
| Add scoring/early stop | `ScoringFunction` / `StoppingAdvisor` |
| Monitor training       | `TrainingListener`                    |

---

## 🛤 Roadmap

* [x] Binary/Multi/Regression/Autoencoder Trainers
* [x] Dropout support
* [x] Mini-batch + parallel execution
* [x] Early stopping (patience, score threshold)
* [x] Adam, RMSProp, Momentum
* [x] Xavier, He initialization
* [ ] Model serialization
* [ ] Learning rate schedulers
* [ ] CNN, RNN layer support
* [ ] Visual training dashboards

---

## 🤝 Contributing

Have an idea? Found a bug? Contributions are welcome!

* Fork, branch, submit PRs
* Add your own trainers, layers, components
* Suggest improvements via Issues

---

## 📄 License

Licensed under [Apache License 2.0](LICENSE)

---

## 🙏 Acknowledgements

Inspired by:

* PyTorch
* Keras
* TensorFlow

Built from scratch for Java developers who want to deeply understand what’s happening in a neural network.

---

# 🧠 Build neural networks. Understand them deeply. With **Netwerx**.
