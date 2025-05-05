package org.jwcarman.netwerx.batch;

import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.matrix.Matrix;

import java.util.function.Function;

public interface TrainingExecutor<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    TrainingResult<M> execute(Dataset<M> dataset, Function<Dataset<M>, TrainingResult<M>> trainerFunction);

}
