package org.jwcarman.netwerx.activation;

import org.jwcarman.netwerx.matrix.Matrix;

/**
 * Represents an activation function used in neural networks to introduce non-linearity.
 * <p>
 * Implementations must define how the function is applied element-wise and how its derivative is calculated
 * for use during backpropagation.
 */
public interface ActivationFunction {

// -------------------------- OTHER METHODS --------------------------

    /**
     * Applies the activation function element-wise to the pre-activation matrix <b>z</b>.
     *
     * @param input the input matrix <i>z</i> (typically the result of a linear transformation like <i>Wx + b</i>)
     * @param <M>   the matrix type, which must extend {@code Matrix<M>}
     * @return the output matrix <i>a</i>, where <i>a = f(z)</i>
     */
    <M extends Matrix<M>> M apply(M input);

    /**
     * Computes the derivative of the activation function with respect to the input matrix <b>z</b>,
     * typically used during backpropagation.
     *
     * @param input the input matrix <i>z</i> (pre-activation values)
     * @param <M>   the matrix type, which must extend {@code Matrix<M>}
     * @return the matrix ∂<i>a</i>/∂<i>z</i>, representing the element-wise derivatives
     */
    <M extends Matrix<M>> M derivative(M input);
}
