package org.jwcarman.netwerx.def;

import org.jwcarman.netwerx.classification.binary.BinaryClassifierConfig;
import org.jwcarman.netwerx.loss.Loss;
import org.jwcarman.netwerx.loss.Losses;

class DefaultBinaryClassifierConfig implements BinaryClassifierConfig {

// ------------------------------ FIELDS ------------------------------

    private Loss loss = Losses.bce();

// --------------------- GETTER / SETTER METHODS ---------------------

    public Loss getLoss() {
        return loss;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface BinaryClassifierConfig ---------------------

    @Override
    public DefaultBinaryClassifierConfig loss(Loss loss) {
        this.loss = loss;
        return this;
    }

}
