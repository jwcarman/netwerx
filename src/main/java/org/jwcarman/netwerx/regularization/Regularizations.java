package org.jwcarman.netwerx.regularization;

import org.jwcarman.netwerx.matrix.Matrix;

public class Regularizations {

// -------------------------- STATIC METHODS --------------------------

    public static <M extends Matrix<M>> RegularizationFunction<M> noop() {
        return new NoopRegularizationFunction<>();
    }

    public static <M extends Matrix<M>> RegularizationFunction<M> l1(double lambda) {
        return new L1RegularizationFunction<>(lambda);
    }

    public static <M extends Matrix<M>> RegularizationFunction<M> l2(double lambda) {
        return new L2RegularizationFunction<>(lambda);
    }

    public static <M extends Matrix<M>> RegularizationFunction<M> elasticNet(double lambda1, double lambda2) {
        return new ElasticNetRegularizationFunction<>(lambda1, lambda2);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private Regularizations() {
        // Prevent instantiation
    }

}
