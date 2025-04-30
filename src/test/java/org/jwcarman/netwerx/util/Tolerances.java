package org.jwcarman.netwerx.util;

import org.assertj.core.data.Offset;

import static org.assertj.core.api.Assertions.within;

public class Tolerances {

// ------------------------------ FIELDS ------------------------------

    public static final double DEFAULT_TOLERANCE = 1e-6;

// -------------------------- STATIC METHODS --------------------------

    public static Offset<Double> withinTolerance() {
        return within(DEFAULT_TOLERANCE);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private Tolerances() {}

}
