package org.jwcarman.netwerx.activation;

public class LeakyReLU extends ScalarActivationFunction {

// ------------------------------ FIELDS ------------------------------

    public static final double DEFAULT_ALPHA = 0.01;

    public static final ActivationFunction DEFAULT = new LeakyReLU(DEFAULT_ALPHA);
    private final double alpha;

// --------------------------- CONSTRUCTORS ---------------------------

    public LeakyReLU(double alpha) {
        this.alpha = alpha;
    }

// -------------------------- OTHER METHODS --------------------------

    @Override
    public double apply(double x) {
        return x >= 0 ? x : alpha * x;
    }

    @Override
    public double derivative(double x) {
        return x >= 0 ? 1.0 : alpha;
    }

}
