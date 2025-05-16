package org.jwcarman.netwerx.normalization;

import java.util.stream.DoubleStream;

@FunctionalInterface
public interface NormalizationFunctionFactory {

// -------------------------- OTHER METHODS --------------------------

    NormalizationFunction create(DoubleStream values);

}
