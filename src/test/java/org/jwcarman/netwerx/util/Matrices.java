package org.jwcarman.netwerx.util;

import org.jwcarman.netwerx.matrix.ejml.EjmlMatrix;
import org.jwcarman.netwerx.matrix.ejml.EjmlMatrixFactory;

import java.util.Random;

public class Matrices {

// -------------------------- STATIC METHODS --------------------------

    private static final EjmlMatrixFactory FACTORY = new EjmlMatrixFactory();

    public static EjmlMatrix of(double[][] data) {
        return FACTORY.from(data);
    }

    public static EjmlMatrix of(int rows, int cols) {
        return FACTORY.zeros(rows, cols);
    }

    public static EjmlMatrix random(int rows, int columns, double min, double max, Random random) {
        return FACTORY.random(rows, columns, min, max, random);
    }
}
