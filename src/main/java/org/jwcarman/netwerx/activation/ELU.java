package org.jwcarman.netwerx.activation;

public class ELU extends ScalarActivationFunction {

// ------------------------------ FIELDS ------------------------------

    public static final double DEFAULT_ALPHA = 1.0;

    public static final ActivationFunction DEFAULT = new ELU(DEFAULT_ALPHA);

    private final double alpha;

// --------------------------- CONSTRUCTORS ---------------------------

    public ELU(double alpha) {
        this.alpha = alpha;
    }

// -------------------------- OTHER METHODS --------------------------

    @Override
    protected double apply(double x) {
        return x >= 0 ? x : alpha * (Math.exp(x) - 1);
    }

    @Override
    protected double derivative(double x) {
        return x >= 0 ? DEFAULT_ALPHA : alpha * Math.exp(x);
    }

}