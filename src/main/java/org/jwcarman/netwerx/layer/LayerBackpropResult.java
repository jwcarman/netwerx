package org.jwcarman.netwerx.layer;

import org.jwcarman.netwerx.matrix.Matrix;

public record LayerBackpropResult<M extends Matrix<M>>(M outputGradient, LayerUpdate<M> layerUpdate) {
}
