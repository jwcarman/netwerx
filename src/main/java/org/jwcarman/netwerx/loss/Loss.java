package org.jwcarman.netwerx.loss;

import org.ejml.simple.SimpleMatrix;

public interface Loss {

    double loss(SimpleMatrix output, SimpleMatrix target);

    SimpleMatrix gradient(SimpleMatrix output, SimpleMatrix target);
}