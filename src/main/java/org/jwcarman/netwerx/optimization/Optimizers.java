package org.jwcarman.netwerx.optimization;

import org.jwcarman.netwerx.learning.LearningRateProvider;
import org.jwcarman.netwerx.matrix.Matrix;

public class Optimizers {

// -------------------------- STATIC METHODS --------------------------

    public static <M extends Matrix<M>> Optimizer<M> sgd() {
        return new SgdOptimizer<>();
    }

    public static <M extends Matrix<M>> Optimizer<M> sgd(double learningRate) {
        return new SgdOptimizer<>(learningRate);
    }

    public static <M extends Matrix<M>> Optimizer<M> sgd(LearningRateProvider learningRateProvider) {
        return new SgdOptimizer<>(learningRateProvider);
    }


    public static <M extends Matrix<M>> Optimizer<M> momentum() {
        return new MomentumOptimizer<>();
    }

    public static <M extends Matrix<M>> Optimizer<M> momentum(double learningRate, double momentumFactor) {
        return new MomentumOptimizer<>(learningRate, momentumFactor);
    }

    public static <M extends Matrix<M>> Optimizer<M> momentum(LearningRateProvider learningRateProvider, double momentumFactor) {
        return new MomentumOptimizer<>(learningRateProvider, momentumFactor);
    }

    public static <M extends Matrix<M>> Optimizer<M> adam() {
        return new AdamOptimizer<>();
    }

    public static <M extends Matrix<M>> Optimizer<M> adam(double learningRate, double beta1, double beta2, double epsilon) {
        return new AdamOptimizer<>(learningRate, beta1, beta2, epsilon);
    }

    public static <M extends Matrix<M>> Optimizer<M> adam(LearningRateProvider learningRateProvider, double beta1, double beta2, double epsilon) {
        return new AdamOptimizer<>(learningRateProvider, beta1, beta2, epsilon);
    }

    public static <M extends Matrix<M>> Optimizer<M> rmsProp() {
        return new RmsPropOptimizer<>();
    }

    public static <M extends Matrix<M>> Optimizer<M> rmsProp(double learningRate, double beta, double epsilon) {
        return new RmsPropOptimizer<>(learningRate, beta, epsilon);
    }

    public static <M extends Matrix<M>> Optimizer<M> rmsProp(LearningRateProvider learningRateProvider, double beta, double epsilon) {
        return new RmsPropOptimizer<>(learningRateProvider, beta, epsilon);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private Optimizers() {
        // Prevent instantiation
    }

}
