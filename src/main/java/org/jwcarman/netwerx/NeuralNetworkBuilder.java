package org.jwcarman.netwerx;

import org.jwcarman.netwerx.activation.Activations;
import org.jwcarman.netwerx.util.Customizer;

import java.util.LinkedList;
import java.util.List;

public class NeuralNetworkBuilder {

// ------------------------------ FIELDS ------------------------------

    private final List<LayerConfiguration> layerConfigurations = new LinkedList<>();

    private int inputSize;

// --------------------------- CONSTRUCTORS ---------------------------

    public NeuralNetworkBuilder(int inputSize) {
        this.inputSize = inputSize;
    }

// -------------------------- OTHER METHODS --------------------------

    public BinaryClassifier binaryClassifier(Customizer<BinaryClassifierConfig> customizer) {
        BinaryClassifierConfig config = new BinaryClassifierConfig();
        customizer.customize(config);

        layer(layer -> layer
                .units(1)
                .activation(Activations.sigmoid())
                .weightOptimizer(config.getWeightOptimizer())
                .biasOptimizer(config.getBiasOptimizer())
                .random(config.getRandom()));

        return new NeuralNetworkBinaryClassifier(build(), config.getLossFunction());
    }

    public NeuralNetworkBuilder layer(Customizer<LayerConfiguration> customizer) {
        LayerConfiguration config = new LayerConfiguration(inputSize);
        customizer.customize(config);
        layerConfigurations.add(config);
        this.inputSize = config.getUnits();
        return this;
    }

    public NeuralNetwork build() {
        return new NeuralNetwork(layerConfigurations.stream().map(Layer::new).toList());
    }

}
