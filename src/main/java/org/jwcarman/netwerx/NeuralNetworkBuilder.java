package org.jwcarman.netwerx;

import org.jwcarman.netwerx.activation.Activations;
import org.jwcarman.netwerx.classification.binary.BinaryClassifier;
import org.jwcarman.netwerx.classification.binary.BinaryClassifierConfig;
import org.jwcarman.netwerx.classification.binary.DefaultBinaryClassifier;
import org.jwcarman.netwerx.classification.multi.DefaultMultiClassifier;
import org.jwcarman.netwerx.classification.multi.MultiClassifier;
import org.jwcarman.netwerx.classification.multi.MultiClassifierConfig;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.regression.DefaultRegressionModel;
import org.jwcarman.netwerx.regression.RegressionModel;
import org.jwcarman.netwerx.regression.RegressionModelConfig;
import org.jwcarman.netwerx.util.Customizer;
import org.jwcarman.netwerx.util.Randoms;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

public class NeuralNetworkBuilder {

// ------------------------------ FIELDS ------------------------------

    private final List<LayerConfig> layerConfigs = new LinkedList<>();
    private Supplier<Optimizer> weightOptimizer = Optimizers::sgd;
    private Supplier<Optimizer> biasOptimizer = Optimizers::sgd;
    private Random random = Randoms.defaultRandom();
    private int inputSize;

// --------------------------- CONSTRUCTORS ---------------------------

    /**
     * Creates a NeuralNetworkBuilder with the specified input size.
     *
     * @param inputSize the size of the input layer
     */
    public NeuralNetworkBuilder(int inputSize) {
        this.inputSize = inputSize;
    }

// -------------------------- OTHER METHODS --------------------------

    public NeuralNetworkBuilder biasOptimizer(Supplier<Optimizer> biasOptimizer) {
        this.biasOptimizer = biasOptimizer;
        return this;
    }

    /**
     * Creates a binary classifier with default configuration.
     *
     * @return a BinaryClassifier instance with default settings
     */
    public BinaryClassifier binaryClassifier() {
        return binaryClassifier(Customizer.noop());
    }

    /**
     * Creates a binary classifier with the given customizer.
     *
     * @param customizer a customizer to configure the binary classifier
     * @return a BinaryClassifier instance configured with the provided customizer
     */
    public BinaryClassifier binaryClassifier(Customizer<BinaryClassifierConfig> customizer) {
        BinaryClassifierConfig config = new BinaryClassifierConfig();
        config.random(random);
        config.weightOptimizer(weightOptimizer.get());
        config.biasOptimizer(biasOptimizer.get());
        customizer.customize(config);

        layer(layer -> layer
                .units(1)
                .activation(Activations.sigmoid())
                .weightOptimizer(config.getWeightOptimizer())
                .biasOptimizer(config.getBiasOptimizer())
                .random(config.getRandom()));

        return new DefaultBinaryClassifier(build(), config.getLoss());
    }

    /**
     * Adds a layer to the neural network with the specified configuration.
     *
     * @param customizer a customizer to configure the layer
     * @return this NeuralNetworkBuilder instance for method chaining
     */
    public NeuralNetworkBuilder layer(Customizer<LayerConfig> customizer) {
        LayerConfig config = new LayerConfig(inputSize);
        config.random(random);
        config.weightOptimizer(weightOptimizer.get());
        config.biasOptimizer(biasOptimizer.get());
        customizer.customize(config);
        layerConfigs.add(config);
        this.inputSize = config.getUnits();
        return this;
    }

    /**
     * Builds the neural network with the configured layers.
     *
     * @return a NeuralNetwork instance containing the configured layers
     */
    public NeuralNetwork build() {
        return new NeuralNetwork(layerConfigs.stream().map(Layer::new).toList());
    }

    /**
     * Creates a multi-class classifier with default configuration.
     *
     * @return a MultiClassifier instance with default settings
     */
    public MultiClassifier multiClassifier() {
        return multiClassifier(Customizer.noop());
    }

    /**
     * Creates a multi-class classifier with the given customizer.
     *
     * @param customizer a customizer to configure the multi-class classifier
     * @return a MultiClassifier instance configured with the provided customizer
     */
    public MultiClassifier multiClassifier(Customizer<MultiClassifierConfig> customizer) {
        var config = new MultiClassifierConfig();
        config.random(random);
        config.weightOptimizer(weightOptimizer.get());
        config.biasOptimizer(biasOptimizer.get());
        customizer.customize(config);

        layer(layer -> layer
                .units(config.getOutputClasses())
                .activation(Activations.softmax())
                .weightOptimizer(config.getWeightOptimizer())
                .biasOptimizer(config.getBiasOptimizer())
                .random(config.getRandom())
        );

        return new DefaultMultiClassifier(build(), config.getLoss(), config.getOutputClasses());
    }

    public NeuralNetworkBuilder optimizer(Supplier<Optimizer> optimizer) {
        this.weightOptimizer = optimizer;
        this.biasOptimizer = optimizer;
        return this;
    }

    public NeuralNetworkBuilder random(Random random) {
        this.random = random;
        return this;
    }

    /**
     * Creates a regression model with default configuration.
     *
     * @return a RegressionModel instance with default settings
     */
    public RegressionModel regressionModel() {
        return regressionModel(Customizer.noop());
    }

    /**
     * Creates a regression model with the given customizer.
     *
     * @param customizer a customizer to configure the regression model
     * @return a RegressionModel instance configured with the provided customizer
     */
    public RegressionModel regressionModel(Customizer<RegressionModelConfig> customizer) {
        RegressionModelConfig config = new RegressionModelConfig();
        config.random(random);
        config.weightOptimizer(weightOptimizer.get());
        config.biasOptimizer(biasOptimizer.get());
        customizer.customize(config);

        layer(layer -> layer
                .units(1)
                .activation(Activations.linear())
                .weightOptimizer(config.getWeightOptimizer())
                .biasOptimizer(config.getBiasOptimizer())
                .random(config.getRandom())
        );

        return new DefaultRegressionModel(build(), config.getLoss());
    }

    public NeuralNetworkBuilder weightOptimizer(Supplier<Optimizer> weightOptimizer) {
        this.weightOptimizer = weightOptimizer;
        return this;
    }

}
