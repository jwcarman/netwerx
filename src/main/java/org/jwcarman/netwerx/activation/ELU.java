package org.jwcarman.netwerx.activation;

public class ELU extends HeActivation {

// ------------------------------ FIELDS ------------------------------

    public static final double DEFAULT_ALPHA = 1.0;
    private final double alpha;

// --------------------------- CONSTRUCTORS ---------------------------

    public ELU() {
        this.alpha = DEFAULT_ALPHA;
    }

    public ELU(double initialBias, double alpha) {
        super(initialBias);
        this.alpha = alpha;
    }

// -------------------------- OTHER METHODS --------------------------

    @Override
    public double apply(double x) {
        return x >= 0 ? x : alpha * (Math.exp(x) - 1);
    }

    @Override
    public double derivative(double x) {
        return x >= 0 ? DEFAULT_ALPHA : alpha * Math.exp(x);
    }

}