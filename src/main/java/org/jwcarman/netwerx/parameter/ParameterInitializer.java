package org.jwcarman.netwerx.parameter;

import java.util.Random;

@FunctionalInterface
public interface ParameterInitializer {

// -------------------------- OTHER METHODS --------------------------

    double initialize(Random random, int fanIn, int fanOut);

}
