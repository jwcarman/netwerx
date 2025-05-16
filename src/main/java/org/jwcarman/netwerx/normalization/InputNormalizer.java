package org.jwcarman.netwerx.normalization;

import org.jwcarman.netwerx.dataset.Dataset;
import org.jwcarman.netwerx.matrix.Matrix;

import java.util.Map;
import java.util.function.UnaryOperator;


public class InputNormalizer<M extends Matrix<M>> implements UnaryOperator<M> {

// ------------------------------ FIELDS ------------------------------

    private final NormalizationFunction[] normalizationFunctions;

// -------------------------- STATIC METHODS --------------------------

    public static <M extends Matrix<M>> UnaryOperator<M> empty() {
        return input -> input;
    }

    public static <M extends Matrix<M>> UnaryOperator<M> forDataset(NormalizationFunctionFactory defaultFactory, Map<Integer, NormalizationFunctionFactory> factories, Dataset<M> dataset) {
        var rowCount = dataset.features().rowCount();
        var normalizationFunctions = new NormalizationFunction[rowCount];
        for(int featureIndex = 0; featureIndex < rowCount; featureIndex++) {
            var factory = factories.getOrDefault(featureIndex, defaultFactory);
            normalizationFunctions[featureIndex] = factory.create(dataset.features().rowValues(featureIndex));
        }


        return new InputNormalizer<>(normalizationFunctions);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    public InputNormalizer(NormalizationFunction[]  normalizationFunctions) {
        this.normalizationFunctions = normalizationFunctions;
    }

// -------------------------- OTHER METHODS --------------------------


    @Override
    public M apply(M input) {
        return input.map((row, _, value) -> normalizationFunctions[row].normalize(value));
    }
}
