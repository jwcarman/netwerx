package org.jwcarman.netwerx;

import org.ejml.simple.SimpleMatrix;
import org.jwcarman.netwerx.activation.Activation;
import org.jwcarman.netwerx.util.Matrices;
import org.jwcarman.netwerx.optimization.Optimizer;

import static org.jwcarman.netwerx.util.Matrices.addColumnVector;

class Layer {

// ------------------------------ FIELDS ------------------------------

    private SimpleMatrix weights;
    private SimpleMatrix biases;
    private final Activation activation;
    private final Optimizer weightOptimizer;
    private final Optimizer biasOptimizer;

// -------------------------- STATIC METHODS --------------------------

    private static SimpleMatrix sumColumns(SimpleMatrix matrix) {
        SimpleMatrix sum = new SimpleMatrix(matrix.getNumRows(), 1);
        for (int row = 0; row < matrix.getNumRows(); row++) {
            sum.set(row, 0, matrix.extractVector(true, row).elementSum());
        }
        return sum;
    }

// --------------------------- CONSTRUCTORS ---------------------------

    public Layer(LayerConfig config) {
        this.weights = Matrices.filled(config.getUnits(), config.getInputSize(), () -> config.getActivation().generateInitialWeight(config.getRandom(), config.getInputSize(), config.getUnits()));
        this.biases = SimpleMatrix.filled(config.getUnits(), 1, config.getActivation().generateInitialBias());
        this.activation = config.getActivation();
        this.weightOptimizer = config.getWeightOptimizer();
        this.biasOptimizer = config.getBiasOptimizer();
    }

// -------------------------- OTHER METHODS --------------------------

    public Backprop forward(final SimpleMatrix aPrev) {
        final var z = addColumnVector(weights.mult(aPrev), biases);
        final var a = activation.apply(z);
        return new Backprop() {

            @Override
            public SimpleMatrix a() {
                return a;
            }

            @Override
            public SimpleMatrix apply(SimpleMatrix gradOutput) {
                final var dz = gradOutput.elementMult(activation.derivative(z));

                final var m = gradOutput.getNumCols();

                final var originalWeights = weights.copy();

                final var dw = dz.mult(aPrev.transpose()).divide(m);
                weights = weightOptimizer.optimize(weights, dw);

                final var db = sumColumns(dz).divide(m);
                biases = biasOptimizer.optimize(biases, db);

                return originalWeights.transpose().mult(dz);
            }

        };
    }

    public static interface Backprop {
        SimpleMatrix a();
        SimpleMatrix apply(SimpleMatrix gradOutput);
    }
}
