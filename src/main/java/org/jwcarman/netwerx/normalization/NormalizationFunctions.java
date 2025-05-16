package org.jwcarman.netwerx.normalization;

import java.util.stream.DoubleStream;

public class NormalizationFunctions {

// -------------------------- STATIC METHODS --------------------------

    public static NormalizationFunction identity() {
        return value -> value;
    }

    public static NormalizationFunction minMax(DoubleStream values) {
        return MinMaxNormalizationFunction.fit(values);
    }

    public static NormalizationFunction maxAbs(DoubleStream values) {
        return MaxAbsNormalizationFunction.fit(values);
    }

    public static NormalizationFunction l2(DoubleStream values) {
        return L2NormalizationFunction.fit(values);
    }

    public static NormalizationFunction zScore(DoubleStream values) {
        return ZScoreNormalizationFunction.fit(values);
    }

    public static NormalizationFunction robustScaling(DoubleStream values) {
        return RobustScalingNormalizationFunction.fit(values);
    }

    public static NormalizationFunction log(DoubleStream values) {
        return LogNormalizationFunction.fit(values);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private NormalizationFunctions() {
        // Prevent instantiation
    }

}
