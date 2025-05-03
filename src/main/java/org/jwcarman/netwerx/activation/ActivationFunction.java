package org.jwcarman.netwerx.activation;

import org.jwcarman.netwerx.matrix.Matrix;

import java.util.Random;

import static org.jwcarman.netwerx.distribution.Distributions.xavierUniform;

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

    /**
     * Returns the initial bias value for a neuron using this activation function.
     * <p>
     * Most activation functions use 0.0 as the default bias. Some (e.g., ReLU) may benefit
     * from small non-zero initial bias values to reduce the likelihood of "dead" neurons.
     *
     * @param rand   the random number generator to use
     * @param fanIn  the number of input connections to the neuron
     * @param fanOut the number of output connections from the neuron
     * @return the initial bias value
     */
    default double initialBias(Random rand, int fanIn, int fanOut) {
        return 0.0;
    }

    /**
     * Returns the initial weight value for a connection governed by this activation function.
     * <p>
     * The default implementation uses Xavier (Glorot) uniform initialization:
     * <pre>
     *   W ∼ 𝒰 [−√(6 / (fanIn + fanOut)), √(6 / (fanIn + fanOut))]
     * </pre>
     *
     * @param rand   the random number generator to use
     * @param fanIn  the number of input connections to the neuron
     * @param fanOut the number of output connections from the neuron
     * @return an initial weight value sampled from an appropriate distribution
     */
    default double initialWeight(Random rand, int fanIn, int fanOut) {
        return xavierUniform(rand, fanIn, fanOut);
    }

}
