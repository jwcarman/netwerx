package org.jwcarman.netwerx;

import org.jwcarman.netwerx.matrix.Matrix;

public interface DropoutLayerConfig<M extends Matrix<M>> {

    DropoutLayerConfig<M> dropoutRate(double dropoutRate);

}
