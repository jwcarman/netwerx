package org.jwcarman.netwerx.activation;

public class LeakyReLU extends ReLU {

// ------------------------------ FIELDS ------------------------------

    public static final double DEFAULT_ALPHA = 0.01;
    private final double alpha;

// --------------------------- CONSTRUCTORS ---------------------------

    public LeakyReLU() {
        this.alpha = DEFAULT_ALPHA;
    }

    public LeakyReLU(double initialBias, double alpha) {
        super(initialBias);
        this.alpha = alpha;
    }

// -------------------------- OTHER METHODS --------------------------

    @Override
    public double apply(double input) {
        return input >= 0.0 ? input : alpha * input;
    }
}
