package org.jwcarman.netwerx.normalization;

import java.util.Arrays;
import java.util.stream.DoubleStream;

public class RobustScalingNormalizationFunction implements NormalizationFunction {

// ------------------------------ FIELDS ------------------------------

    private final double median;
    private final double iqr;

// -------------------------- STATIC METHODS --------------------------

    public static RobustScalingNormalizationFunction fit(DoubleStream values) {
        var arr = values.toArray();
        if (arr.length == 0) {
            throw new IllegalArgumentException("Cannot fit normalizer on empty input");
        }

        Arrays.sort(arr);
        return new RobustScalingNormalizationFunction(calculateMedian(arr), calculateIQR(arr));
    }

    private static double calculateMedian(double[] sorted) {
        int n = sorted.length;
        return (n % 2 == 0)
                ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                : sorted[n / 2];
    }

    private static double calculateIQR(double[] sorted) {
        double q1 = interpolatePercentile(sorted, 0.25);
        double q3 = interpolatePercentile(sorted, 0.75);
        return q3 - q1;
    }

    private static double interpolatePercentile(double[] sorted, double p) {
        double pos = p * (sorted.length - 1);
        int lower = (int) Math.floor(pos);
        int upper = (int) Math.ceil(pos);
        if (lower == upper) return sorted[lower];

        double weight = pos - lower;
        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private RobustScalingNormalizationFunction(double median, double iqr) {
        this.median = median;
        this.iqr = iqr;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Normalizer ---------------------

    @Override
    public double normalize(double value) {
        return (iqr == 0.0) ? 0.0 : (value - median) / iqr;
    }

}
