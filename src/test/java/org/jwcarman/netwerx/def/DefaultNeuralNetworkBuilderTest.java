package org.jwcarman.netwerx.def;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.classification.binary.BinaryClassifier;
import org.jwcarman.netwerx.classification.multi.MultiClassifier;
import org.jwcarman.netwerx.loss.Losses;
import org.jwcarman.netwerx.optimization.Optimizers;
import org.jwcarman.netwerx.regression.RegressionModel;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

class DefaultNeuralNetworkBuilderTest {

// ------------------------------ FIELDS ------------------------------

    private static final int INPUT_SIZE = 4;

// -------------------------- OTHER METHODS --------------------------

    @Test
    void shouldAllowLayerCustomization() {
        NeuralNetwork network = new DefaultNeuralNetworkBuilder(INPUT_SIZE)
                .layer(layer -> layer.units(6))
                .layer(layer -> layer.units(3))
                .build();
        assertThat(network).isNotNull();
    }

    @Test
    void shouldBuildBinaryClassifier() {
        BinaryClassifier classifier = new DefaultNeuralNetworkBuilder(INPUT_SIZE)
                .binaryClassifier(bc -> bc.loss(Losses.bce()));
        assertThat(classifier).isNotNull();
    }

    @Test
    void shouldBuildMultiClassifier() {
        MultiClassifier classifier = new DefaultNeuralNetworkBuilder(INPUT_SIZE)
                .multiClassifier(mc -> mc.outputClasses(3));
        assertThat(classifier).isNotNull();
    }

    @Test
    void shouldBuildRegressionModel() {
        RegressionModel model = new DefaultNeuralNetworkBuilder(INPUT_SIZE)
                .regressionModel(rm -> rm.loss(Losses.mse()));
        assertThat(model).isNotNull();
    }

    @Test
    void shouldSetSeparateWeightAndBiasOptimizers() {
        NeuralNetwork network = new DefaultNeuralNetworkBuilder(INPUT_SIZE)
                .weightOptimizer(Optimizers::adam)
                .biasOptimizer(Optimizers::rmsProp)
                .layer(layer -> layer.units(2))
                .build();
        assertThat(network).isNotNull();
    }

    @Test
    void shouldUseCustomRandomAndOptimizer() {
        Random seededRandom = new Random(42);
        NeuralNetwork network = new DefaultNeuralNetworkBuilder(INPUT_SIZE)
                .random(seededRandom)
                .optimizer(Optimizers::momentum)
                .layer(layer -> layer.units(5))
                .build();
        assertThat(network).isNotNull();
    }

}
