package org.jwcarman.netwerx;

import org.jwcarman.netwerx.matrix.Matrix;

public record Dataset<M extends Matrix<M>>(M inputs, M outputs) {

}
