package org.jwcarman.netwerx;

import org.ejml.simple.SimpleMatrix;

public interface Backprop {
    SimpleMatrix a();
    SimpleMatrix apply(SimpleMatrix gradOutput);
}
