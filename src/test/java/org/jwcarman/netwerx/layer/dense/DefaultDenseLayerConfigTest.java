package org.jwcarman.netwerx.layer.dense;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.activation.ActivationFunctions;
import org.jwcarman.netwerx.activation.ReLU;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrix;
import org.jwcarman.netwerx.optimization.Optimizer;
import org.jwcarman.netwerx.optimization.SgdOptimizer;
import org.jwcarman.netwerx.regularization.NoopRegularizationFunction;
import org.jwcarman.netwerx.regularization.RegularizationFunction;
import org.jwcarman.netwerx.regularization.Regularizations;

import java.util.function.Supplier;

import static org.assertj.core.api.Assertions.assertThat;

class DefaultDenseLayerConfigTest {
    @Test
    void testDefaultDenseLayerConfig() {
        var config = new DefaultDenseLayerConfig<EjmlMatrix>(5);
        assertThat(config.getUnits()).isEqualTo(8);
        assertThat(config.getInputSize()).isEqualTo(5);
        assertThat(config.getActivationFunction()).isInstanceOf(ReLU.class);
        assertThat(config.getWeightOptimizerSupplier()).isNotNull();
        assertThat(config.getWeightOptimizerSupplier().get()).isInstanceOf(SgdOptimizer.class);
        assertThat(config.getBiasOptimizerSupplier()).isNotNull();
        assertThat(config.getBiasOptimizerSupplier().get()).isInstanceOf(SgdOptimizer.class);
        assertThat(config.getRegularizationFunction()).isNotNull();
        assertThat(config.getRegularizationFunction()).isInstanceOf(NoopRegularizationFunction.class);
    }

    @Test
    void testActivationFunction() {
        var config = new DefaultDenseLayerConfig<EjmlMatrix>(5);
        var activationFunction = ActivationFunctions.sigmoid();
        config.activationFunction(activationFunction);
        assertThat(config.getActivationFunction()).isSameAs(activationFunction);
    }

    @Test
    void testUnits() {
        var config = new DefaultDenseLayerConfig<EjmlMatrix>(5);
        config.units(10);
        assertThat(config.getUnits()).isEqualTo(10);
    }

    @Test
    void testWeightOptimizer() {
        var config = new DefaultDenseLayerConfig<EjmlMatrix>(5);
        Supplier<Optimizer<EjmlMatrix>> optimizer = () -> new SgdOptimizer<>(0.01);
        config.weightOptimizer(optimizer);
        assertThat(config.getWeightOptimizerSupplier()).isSameAs(optimizer);
    }

    @Test
    void testBiasOptimizer() {
        var config = new DefaultDenseLayerConfig<EjmlMatrix>(5);
        Supplier<Optimizer<EjmlMatrix>> optimizer = () -> new SgdOptimizer<>(0.01);
        config.biasOptimizer(optimizer);
        assertThat(config.getBiasOptimizerSupplier()).isSameAs(optimizer);
    }

    @Test
    void testRegularizationFunction() {
        var config = new DefaultDenseLayerConfig<EjmlMatrix>(5);
        RegularizationFunction<EjmlMatrix> regularizationFunction = Regularizations.l2(0.01);
        config.regularizationFunction(regularizationFunction);
        assertThat(config.getRegularizationFunction()).isSameAs(regularizationFunction);
    }

    @Test
    void testOptimizers() {
        var config = new DefaultDenseLayerConfig<EjmlMatrix>(5);
        Supplier<Optimizer<EjmlMatrix>> optimizer = () -> new SgdOptimizer<>(0.01);
        config.optimizers(optimizer);
        assertThat(config.getWeightOptimizerSupplier()).isSameAs(optimizer);
        assertThat(config.getBiasOptimizerSupplier()).isSameAs(optimizer);
    }


}