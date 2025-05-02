package org.jwcarman.netwerx.optimization;

import org.jwcarman.netwerx.matrix.Matrix;

import java.util.function.Supplier;

public class Optimizers {

// -------------------------- STATIC METHODS --------------------------

    public static <M extends Matrix<M>> OptimizerProvider<M> uniform(Supplier<Optimizer<M>> supplier) {
        return new OptimizerProvider<>() {
            @Override
            public Optimizer<M> weightOptimizer(int layer) {
                return supplier.get();
            }

            @Override
            public Optimizer<M> biasOptimizer(int layer) {
                return supplier.get();
            }
        };
    }

    public static <M extends Matrix<M>> Optimizer<M> sgd() {
        return new SgdOptimizer<>();
    }

    public static <M extends Matrix<M>> Optimizer<M> sgd(double learningRate) {
        return new SgdOptimizer<>(learningRate);
    }

    public static <M extends Matrix<M>> Optimizer<M> momentum() {
        return new MomentumOptimizer<>();
    }

    public static <M extends Matrix<M>> Optimizer<M> momentum(double learningRate, double momentumFactor) {
        return new MomentumOptimizer<>(learningRate, momentumFactor);
    }

    public static <M extends Matrix<M>> Optimizer<M> adam() {
        return new AdamOptimizer<>();
    }

    public static <M extends Matrix<M>> Optimizer<M> adam(double learningRate, double beta1, double beta2, double epsilon) {
        return new AdamOptimizer<>(learningRate, beta1, beta2, epsilon);
    }

    public static <M extends Matrix<M>> Optimizer<M> rmsProp() {
        return new RmsPropOptimizer<>();
    }

    public static <M extends Matrix<M>> Optimizer<M> rmsProp(double learningRate, double beta, double epsilon) {
        return new RmsPropOptimizer<>(learningRate, beta, epsilon);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private Optimizers() {
        // Prevent instantiation
    }

}
