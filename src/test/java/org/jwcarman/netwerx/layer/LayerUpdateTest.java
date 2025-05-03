package org.jwcarman.netwerx.layer;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrix;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class LayerUpdateTest {

    @Test
    void testAddGradient() {
        var factory = new EjmlMatrixFactory();
        var update = new LayerUpdate<EjmlMatrix>();
        var gradient = factory.zeros(2, 2);
        update.addGradient("foo", gradient);
        assertThat(update.gradient("foo")).isSameAs(gradient);
    }

    @Test
    void testGradientNotFound() {

        var update = new LayerUpdate<EjmlMatrix>();

        assertThatThrownBy(() -> update.gradient("notfound"))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void testNames() {
        var factory = new EjmlMatrixFactory();
        var update = new LayerUpdate<EjmlMatrix>();
        var gradient1 = factory.zeros(2, 2);
        var gradient2 = factory.zeros(3, 3);

        update.addGradient("foo", gradient1);
        update.addGradient("bar", gradient2);

        assertThat(update.gradientNames()).containsExactlyInAnyOrder("foo", "bar");
    }

}