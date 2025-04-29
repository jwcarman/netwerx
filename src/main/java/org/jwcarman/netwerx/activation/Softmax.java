package org.jwcarman.netwerx.activation;

import org.ejml.simple.SimpleMatrix;

public class Softmax implements Activation {

    @Override
    public SimpleMatrix apply(SimpleMatrix input) {
        var output = new SimpleMatrix(input.getNumRows(), input.getNumCols());

        for (int col = 0; col < input.getNumCols(); col++) {
            var column = input.extractVector(false, col);

            var max = column.elementMaxAbs();
            var stabilized = column.minus(max);
            var exp = stabilized.elementExp();
            var sumExp = exp.elementSum();
            for (int row = 0; row < exp.getNumRows(); row++) {
                output.set(row, col, exp.get(row) / sumExp);
            }
        }

        return output;
    }

    @Override
    public SimpleMatrix derivative(SimpleMatrix input) {
        return SimpleMatrix.filled(input.getNumRows(), input.getNumCols(), 1.0);
    }
}
