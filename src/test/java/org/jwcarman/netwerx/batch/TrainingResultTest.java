package org.jwcarman.netwerx.batch;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrix;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TrainingResultTest {

    @Test
    void testAggregateWithEmptyResults() {
        List<TrainingResult<EjmlMatrix>> empty = List.of();
        assertThatThrownBy(() -> TrainingResult.aggregate(empty))
                .isInstanceOf(IllegalArgumentException.class);
    }
}