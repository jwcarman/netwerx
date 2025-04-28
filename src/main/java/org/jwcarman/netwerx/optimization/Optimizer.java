package org.jwcarman.netwerx.optimization;

import org.ejml.simple.SimpleMatrix;

@FunctionalInterface
public interface Optimizer {
    SimpleMatrix optimize(SimpleMatrix parameter, SimpleMatrix gradient);
}
