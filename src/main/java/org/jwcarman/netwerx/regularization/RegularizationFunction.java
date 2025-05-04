package org.jwcarman.netwerx.regularization;

import org.jwcarman.netwerx.matrix.Matrix;

public interface RegularizationFunction<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Computes the gradient of the regularization term for the given matrix.
     *
     * @param matrix the matrix to compute the gradient for
     * @return the computed gradient
     */
    M gradient(M matrix);

    /**
     * Computes the penalty for the given matrix.
     *
     * @param matrix the matrix to compute the penalty for
     * @return the computed penalty
     */
    double penalty(M matrix);

}
