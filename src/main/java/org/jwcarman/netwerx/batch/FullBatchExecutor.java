package org.jwcarman.netwerx.batch;

import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.matrix.Matrix;

import java.util.function.Function;

public class FullBatchExecutor<M extends Matrix<M>> implements TrainingExecutor<M> {

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface TrainingExecutor ---------------------

    @Override
    public TrainingResult<M> execute(Dataset<M> dataset, Function<Dataset<M>, TrainingResult<M>> trainerFunction) {
        return trainerFunction.apply(dataset);
    }

}
