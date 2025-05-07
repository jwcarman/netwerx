package org.jwcarman.netwerx.layer.dropout;

import org.jwcarman.netwerx.DropoutLayerConfig;
import org.jwcarman.netwerx.matrix.Matrix;

public class DefaultDropoutLayerConfig<M extends Matrix<M>> implements DropoutLayerConfig<M> {

// ------------------------------ FIELDS ------------------------------

    private double dropoutRate = 0.5;

// --------------------- GETTER / SETTER METHODS ---------------------

    public double getDropoutRate() {
        return dropoutRate;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface DropoutLayerConfig ---------------------

    @Override
    public DropoutLayerConfig<M> dropoutRate(double dropoutRate) {
        this.dropoutRate = dropoutRate;
        return this;
    }

}
