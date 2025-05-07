package org.jwcarman.netwerx.layer.dropout;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrix;
import org.jwcarman.netwerx.util.Randoms;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class DropoutLayerTrainerTest {

    @Test
    void testConstructor() {
        var trainer = new DropoutLayerTrainer<EjmlMatrix>(10, 0.75, Randoms.defaultRandom());
        assertThat(trainer.outputSize()).isEqualTo(10);
        assertThat(trainer.inputSize()).isEqualTo(10);
    }

    @Test
    void createLayerShouldThrowException() {
        var trainer = new DropoutLayerTrainer<EjmlMatrix>(10, 0.75, Randoms.defaultRandom());

        assertThatThrownBy(trainer::createLayer)
                .isInstanceOf(UnsupportedOperationException.class);
    }
}