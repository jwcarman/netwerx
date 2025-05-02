package org.jwcarman.netwerx.def;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.NeuralNetwork;
import org.jwcarman.netwerx.classification.binary.BinaryClassifier;
import org.jwcarman.netwerx.classification.multi.MultiClassifier;
import org.jwcarman.netwerx.loss.Losses;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrix;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;
import org.jwcarman.netwerx.regression.RegressionModel;

import static org.assertj.core.api.Assertions.assertThat;

class DefaultNeuralNetworkBuilderTest {

// ------------------------------ FIELDS ------------------------------

    private static final int INPUT_SIZE = 4;

// -------------------------- OTHER METHODS --------------------------

    @Test
    void shouldAllowLayerCustomization() {
        NeuralNetwork<EjmlMatrix> network = new DefaultNeuralNetworkBuilder<>(new EjmlMatrixFactory(), INPUT_SIZE)
                .layer(layer -> layer.units(6))
                .layer(layer -> layer.units(3))
                .build();
        assertThat(network).isNotNull();
    }

    @Test
    void shouldBuildBinaryClassifier() {
        BinaryClassifier<EjmlMatrix> classifier = new DefaultNeuralNetworkBuilder<>(new EjmlMatrixFactory(), INPUT_SIZE)
                .binaryClassifier(bc -> bc.loss(Losses.bce()));
        assertThat(classifier).isNotNull();
    }

    @Test
    void shouldBuildMultiClassifier() {
        MultiClassifier<EjmlMatrix> classifier = new DefaultNeuralNetworkBuilder<>(new EjmlMatrixFactory(), INPUT_SIZE)
                .multiClassifier(mc -> mc.outputClasses(3));
        assertThat(classifier).isNotNull();
    }

    @Test
    void shouldBuildRegressionModel() {
        RegressionModel<EjmlMatrix> model = new DefaultNeuralNetworkBuilder<>(new EjmlMatrixFactory(), INPUT_SIZE)
                .regressionModel(rm -> rm.loss(Losses.mse()));
        assertThat(model).isNotNull();
    }

}
