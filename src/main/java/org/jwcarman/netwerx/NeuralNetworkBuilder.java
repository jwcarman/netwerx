package org.jwcarman.netwerx;

import org.jwcarman.netwerx.classification.binary.BinaryClassifier;
import org.jwcarman.netwerx.classification.multi.MultiClassifier;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.regression.RegressionModel;
import org.jwcarman.netwerx.util.Customizer;

import java.util.Random;
import java.util.function.Supplier;

public interface NeuralNetworkBuilder {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Sets the bias optimizer for the neural network.
     *
     * @param biasOptimizer a supplier that provides the optimizer to use for bias parameters
     * @return the current instance of NeuralNetworkBuilder for method chaining
     */
    NeuralNetworkBuilder biasOptimizer(Supplier<Optimizer> biasOptimizer);

    /**
     * Creates a binary classifier using the neural network. This classifier is designed for binary classification
     * tasks, where the output is either true or false (1 or 0).
     *
     * @return a BinaryClassifier instance that uses the neural network for binary classification tasks
     */
    BinaryClassifier binaryClassifier();

    /**
     * Creates a binary classifier using the neural network with a custom configuration.
     *
     * @param customizer a Customizer that allows for configuration of the BinaryClassifier
     * @return a BinaryClassifier instance that uses the neural network for binary classification tasks
     */
    BinaryClassifier binaryClassifier(Customizer<BinaryClassifierConfig> customizer);

    /**
     * Builds the neural network with the specified configuration.
     *
     * @return a NeuralNetwork instance that has been configured and built according to the provided settings
     */
    NeuralNetwork build();

    /**
     * Adds a layer to the neural network with the specified configuration.
     *
     * @param customizer a Customizer that allows for configuration of the LayerConfig
     * @return the current instance of NeuralNetworkBuilder for method chaining
     */
    NeuralNetworkBuilder layer(Customizer<LayerConfig> customizer);

    /**
     * Creates a multi-class classifier using the neural network.
     *
     * @return a MultiClassifier instance that uses the neural network for multi-class classification tasks
     */
    MultiClassifier multiClassifier();

    /**
     * Creates a multi-class classifier using the neural network with a custom configuration.
     *
     * @param customizer a Customizer that allows for configuration of the MultiClassifierConfig
     * @return a MultiClassifier instance that uses the neural network for multi-class classification tasks
     */
    MultiClassifier multiClassifier(Customizer<MultiClassifierConfig> customizer);

    /**
     * Sets the optimizer for the neural network. This optimizer is responsible for adjusting the weights and biases
     * during training to minimize the loss function.
     *
     * @param optimizer a supplier that provides the optimizer to use for training the neural network
     * @return the current instance of NeuralNetworkBuilder for method chaining
     */
    NeuralNetworkBuilder optimizer(Supplier<Optimizer> optimizer);

    /**
     * Sets the random number generator for the neural network. This is typically used for initializing weights.
     *
     * @param random the Random instance to use for random number generation
     * @return the current instance of NeuralNetworkBuilder for method chaining
     */
    NeuralNetworkBuilder random(Random random);

    /**
     * Creates a regression model using the neural network. This model is designed for regression tasks,
     * where the output is a continuous value rather than a class label.
     *
     * @return a RegressionModel instance that uses the neural network for regression tasks
     */
    RegressionModel regressionModel();

    /**
     * Creates a regression model using the neural network with a custom configuration.
     *
     * @param customizer a Customizer that allows for configuration of the RegressionModelConfig
     * @return a RegressionModel instance that uses the neural network for regression tasks
     */
    RegressionModel regressionModel(Customizer<RegressionModelConfig> customizer);

    /**
     * Sets the weight optimizer for the neural network. This optimizer is responsible for adjusting the weights
     * during training to minimize the loss function.
     *
     * @param weightOptimizer a supplier that provides the optimizer to use for weight parameters
     * @return the current instance of NeuralNetworkBuilder for method chaining
     */
    NeuralNetworkBuilder weightOptimizer(Supplier<Optimizer> weightOptimizer);

}
