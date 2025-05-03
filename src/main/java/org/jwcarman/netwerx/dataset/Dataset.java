package org.jwcarman.netwerx.dataset;

import org.jwcarman.netwerx.matrix.Matrix;

public record Dataset<M extends Matrix<M>>(M inputs, M outputs) {


}
