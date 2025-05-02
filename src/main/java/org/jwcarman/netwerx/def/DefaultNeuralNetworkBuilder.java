package org.jwcarman.netwerx.def;

import org.jwcarman.netwerx.LayerConfig;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.NeuralNetworkBuilder;
import org.jwcarman.netwerx.activation.Activations;
import org.jwcarman.netwerx.classification.binary.BinaryClassifier;
import org.jwcarman.netwerx.classification.binary.BinaryClassifierConfig;
import org.jwcarman.netwerx.classification.binary.DefaultBinaryClassifier;
import org.jwcarman.netwerx.classification.multi.DefaultMultiClassifier;
import org.jwcarman.netwerx.classification.multi.MultiClassifier;
import org.jwcarman.netwerx.classification.multi.MultiClassifierConfig;
import org.jwcarman.netwerx.matrix.Matrix;
import org.jwcarman.netwerx.matrix.MatrixFactory;
import org.jwcarman.netwerx.regression.DefaultRegressionModel;
import org.jwcarman.netwerx.regression.RegressionModel;
import org.jwcarman.netwerx.regression.RegressionModelConfig;
import org.jwcarman.netwerx.util.Customizer;
import org.jwcarman.netwerx.util.Randoms;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class DefaultNeuralNetworkBuilder<M extends Matrix<M>> implements NeuralNetworkBuilder<M> {

// ------------------------------ FIELDS ------------------------------

    private final MatrixFactory<M> factory;
    private final List<DefaultLayerConfig> layerConfigs = new LinkedList<>();
    private Random random = Randoms.defaultRandom();
    private int inputSize;

// --------------------------- CONSTRUCTORS ---------------------------

    /**
     * Creates a NeuralNetworkBuilder with the specified input size.
     *
     * @param inputSize the size of the input layer
     */
    public DefaultNeuralNetworkBuilder(MatrixFactory<M> factory, int inputSize) {
        this.factory = factory;
        this.inputSize = inputSize;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface NeuralNetworkBuilder ---------------------

    /**
     * Creates a binary classifier with default configuration.
     *
     * @return a BinaryClassifier instance with default settings
     */
    @Override
    public BinaryClassifier<M> binaryClassifier() {
        return binaryClassifier(Customizer.noop());
    }

    /**
     * Creates a binary classifier with the given customizer.
     *
     * @param customizer a customizer to configure the binary classifier
     * @return a BinaryClassifier instance configured with the provided customizer
     */
    @Override
    public BinaryClassifier<M> binaryClassifier(Customizer<BinaryClassifierConfig> customizer) {
        DefaultBinaryClassifierConfig config = new DefaultBinaryClassifierConfig();
        customizer.customize(config);

        layer(layer -> layer
                .units(1)
                .activation(Activations.sigmoid())
        );

        return new DefaultBinaryClassifier<>(build(), config.getLoss());
    }

    /**
     * Builds the neural network with the configured layers.
     *
     * @return a NeuralNetwork instance containing the configured layers
     */
    @Override
    public NeuralNetwork<M> build() {
        return new DefaultNeuralNetwork<>(layerConfigs.stream().map(config -> new Layer<>(factory, random, config)).toList());
    }

    /**
     * Adds a layer to the neural network with the specified configuration.
     *
     * @param customizer a customizer to configure the layer
     * @return this NeuralNetworkBuilder instance for method chaining
     */
    @Override
    public DefaultNeuralNetworkBuilder<M> layer(Customizer<LayerConfig> customizer) {
        DefaultLayerConfig config = new DefaultLayerConfig(inputSize);
        customizer.customize(config);
        layerConfigs.add(config);
        this.inputSize = config.getUnits();
        return this;
    }

    /**
     * Creates a multi-class classifier with default configuration.
     *
     * @return a MultiClassifier instance with default settings
     */
    @Override
    public MultiClassifier<M> multiClassifier() {
        return multiClassifier(Customizer.noop());
    }

    /**
     * Creates a multi-class classifier with the given customizer.
     *
     * @param customizer a customizer to configure the multi-class classifier
     * @return a MultiClassifier instance configured with the provided customizer
     */
    @Override
    public MultiClassifier<M> multiClassifier(Customizer<MultiClassifierConfig> customizer) {
        var config = new DefaultMultiClassifierConfig();
        customizer.customize(config);

        layer(layer -> layer
                .units(config.getOutputClasses())
                .activation(Activations.softmax())
        );

        return new DefaultMultiClassifier<>(build(), config.getLoss(), config.getOutputClasses());
    }

    @Override
    public DefaultNeuralNetworkBuilder<M> random(Random random) {
        this.random = random;
        return this;
    }

    /**
     * Creates a regression model with default configuration.
     *
     * @return a RegressionModel instance with default settings
     */
    @Override
    public RegressionModel<M> regressionModel() {
        return regressionModel(Customizer.noop());
    }

    /**
     * Creates a regression model with the given customizer.
     *
     * @param customizer a customizer to configure the regression model
     * @return a RegressionModel instance configured with the provided customizer
     */
    @Override
    public RegressionModel<M> regressionModel(Customizer<RegressionModelConfig> customizer) {
        DefaultRegressionModelConfig config = new DefaultRegressionModelConfig();
        customizer.customize(config);

        layer(layer -> layer
                .units(1)
                .activation(Activations.linear())
        );

        return new DefaultRegressionModel<>(build(), config.getLoss());
    }

}
