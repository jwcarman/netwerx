package org.jwcarman.netwerx.util;

import java.util.Random;
import java.util.SplittableRandom;

public class Randoms {

// ------------------------------ FIELDS ------------------------------

    private static final Random DEFAULT_RANDOM = Random.from(new SplittableRandom());

// -------------------------- STATIC METHODS --------------------------

    public static Random defaultRandom() {
        return DEFAULT_RANDOM;
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private Randoms() {
        // Prevent instantiation
    }

}
