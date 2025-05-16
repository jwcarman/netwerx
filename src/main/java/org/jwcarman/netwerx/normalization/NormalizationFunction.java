package org.jwcarman.netwerx.normalization;

@FunctionalInterface
public interface NormalizationFunction {
    double normalize(double value);
}
